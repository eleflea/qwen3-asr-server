from __future__ import annotations

import asyncio
import io
import logging
import logging.config
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import jieba
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from mlx_audio.audio_io import read as audio_read
from mlx_audio.stt import load
from pydantic import BaseModel
from scipy import signal

SAMPLE_RATE = 16000
QUEUE_MAX_SIZE = 10
ASR_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
ALIGNER_MODEL_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
CHINESE_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD = 0.4


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": "INFO", "handlers": ["console"]},
            "loggers": {
                "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
                "uvicorn.error": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "fastapi": {"level": "INFO", "handlers": ["console"], "propagate": False},
            },
        }
    )


configure_logging()
logger = logging.getLogger(__name__)


class TimestampItem(BaseModel):
    start_time: float
    end_time: float
    text: str


class ASRResponse(BaseModel):
    language: str
    text: str
    timestamps: list[TimestampItem]


class HealthResponse(BaseModel):
    status: str
    queue_length: int
    queue_max_size: int
    model_loaded: bool


@dataclass
class ASRTask:
    audio_bytes: bytes
    filename: Optional[str]
    language: Optional[str]
    context: Optional[str]
    future: asyncio.Future


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down, padtype="edge")


def load_audio_from_bytes(
    data: bytes,
    filename: Optional[str],
    sr: int = SAMPLE_RATE,
    dtype=np.float32,
) -> np.ndarray:
    buffer = io.BytesIO(data)
    # Keep filename for compatibility with callers; decoding is fully in-memory.
    if filename:
        buffer.name = filename

    audio, sample_rate = audio_read(buffer, always_2d=True)
    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)

    return np.array(audio, dtype=dtype).mean(axis=1)


def tokenize_context(context: Optional[str]) -> Optional[str]:
    if not context:
        return None

    tokens: list[str] = []
    cursor = 0

    for match in CHINESE_TEXT_RE.finditer(context):
        start, end = match.span()

        if start > cursor:
            non_chinese_chunk = context[cursor:start]
            tokens.extend(non_chinese_chunk.split())

        chinese_chunk = match.group(0)
        tokens.extend(token.strip() for token in jieba.cut(chinese_chunk) if token.strip())
        cursor = end

    if cursor < len(context):
        tail_chunk = context[cursor:]
        tokens.extend(tail_chunk.split())

    return " ".join(tokens) if tokens else None


def equal_timestamp_ratio(timestamps: list[TimestampItem]) -> float:
    if not timestamps:
        return 0.0

    equal_count = sum(
        1 for item in timestamps if abs(float(item.start_time) - float(item.end_time)) < 1e-6
    )
    return equal_count / len(timestamps)


async def asr_worker(app: FastAPI) -> None:
    queue: asyncio.Queue[ASRTask] = app.state.queue

    while True:
        task = await queue.get()
        started_at = time.perf_counter()
        try:
            audio = load_audio_from_bytes(task.audio_bytes, task.filename)
            normalized_context = tokenize_context(task.context)

            generate_kwargs = {}
            if normalized_context:
                generate_kwargs["system_prompt"] = normalized_context
            if task.language:
                generate_kwargs["language"] = task.language

            try:
                result = app.state.model.generate(audio, **generate_kwargs)
            except TypeError:
                # Some model builds may not support a language argument.
                generate_kwargs.pop("language", None)
                result = app.state.model.generate(audio, **generate_kwargs)

            result_language = task.language or getattr(result, "language", [None])[0] or "English"
            aligned = app.state.aligner.generate(
                audio=audio,
                text=result.text,
                language=result_language,
            )

            timestamps = [
                TimestampItem(
                    start_time=float(item.start_time),
                    end_time=float(item.end_time),
                    text=item.text,
                )
                for item in aligned
            ]

            # When context is provided, many zero-duration aligned tokens indicate likely hallucination.
            same_ts_ratio = equal_timestamp_ratio(timestamps)
            if normalized_context and same_ts_ratio > HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD:
                logger.warning(
                    "Suspected hallucination: equal_timestamp_ratio=%.2f%%",
                    same_ts_ratio * 100,
                )
                task.future.set_result(
                    ASRResponse(
                        language="None",
                        text="",
                        timestamps=[],
                    )
                )
            else:
                task.future.set_result(
                    ASRResponse(
                        language=result_language,
                        text=result.text,
                        timestamps=timestamps,
                    )
                )

            audio_seconds = float(len(audio)) / float(SAMPLE_RATE)
            elapsed_seconds = time.perf_counter() - started_at
            rtfx = (audio_seconds / elapsed_seconds) if elapsed_seconds > 0 else 0.0
            logger.info(
                "ASR perf: audio=%.2fs cost=%.2fs RTFx=%.2fx",
                audio_seconds,
                elapsed_seconds,
                rtfx,
            )
        except Exception as exc:
            task.future.set_exception(exc)
        finally:
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    app.state.model = load(ASR_MODEL_ID)
    app.state.aligner = load(ALIGNER_MODEL_ID)
    app.state.worker = asyncio.create_task(asr_worker(app))

    try:
        yield
    finally:
        app.state.worker.cancel()
        await asyncio.gather(app.state.worker, return_exceptions=True)


app = FastAPI(title="Qwen3 ASR Server", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        queue_length=app.state.queue.qsize(),
        queue_max_size=app.state.queue.maxsize,
        model_loaded=hasattr(app.state, "model") and hasattr(app.state, "aligner"),
    )


@app.post("/asr", response_model=ASRResponse)
async def asr(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    context: Optional[str] = Form(default=None),
) -> ASRResponse:
    if app.state.queue.full():
        raise HTTPException(status_code=429, detail="Queue is full")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    task = ASRTask(
        audio_bytes=audio_bytes,
        filename=audio.filename,
        language=language,
        context=context,
        future=future,
    )

    await app.state.queue.put(task)

    try:
        return await future
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}") from exc
