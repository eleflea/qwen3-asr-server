from __future__ import annotations

import asyncio
import io
import json
import logging
import logging.config
import re
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
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
MIN_AUDIO_RMS = 1e-3
CHINESE_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD = 0.1
HIGH_CONFIDENCE_EQUAL_TS_RATIO_THRESHOLD = 0.8
SAME_TS_RATIO_EXCLUDED_TOKENS = ["我"]
SAME_TS_SAMPLE_DIR = Path(__file__).resolve().parent / "same_ts_ratio_samples"
SAME_TS_SAMPLE_MAX_ENTRIES = 100
SAVE_SAME_TS_SAMPLES = False


def configure_logging() -> None:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "asr_server.log"

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
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "filename": str(log_file),
                    "maxBytes": 20 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                }
            },
            "root": {"level": "INFO", "handlers": ["console", "file"]},
            "loggers": {
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "fastapi": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
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


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0

    samples = audio.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(samples * samples)))


def set_future_result(future: asyncio.Future, response: ASRResponse) -> None:
    if future.done():
        logger.info("ASR future already done; dropping result")
        return

    future.set_result(response)


def set_future_exception(future: asyncio.Future, exc: Exception) -> None:
    if future.done():
        logger.info("ASR future already done; dropping exception")
        return

    future.set_exception(exc)


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

    effective_timestamps = [
        item for item in timestamps if item.text not in SAME_TS_RATIO_EXCLUDED_TOKENS
    ]
    if not effective_timestamps:
        return 0.0

    equal_count = sum(
        1
        for item in effective_timestamps
        if abs(float(item.start_time) - float(item.end_time)) < 1e-6
    )
    return equal_count / len(effective_timestamps)


def same_timestamp_tokens(timestamps: list[TimestampItem]) -> list[str]:
    return [
        item.text
        for item in timestamps
        if item.text not in SAME_TS_RATIO_EXCLUDED_TOKENS
        if abs(float(item.start_time) - float(item.end_time)) < 1e-6
    ]


def _prune_old_same_ts_samples(sample_dir: Path, max_entries: int) -> None:
    txt_files = sorted(sample_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    overflow = len(txt_files) - max_entries
    if overflow <= 0:
        return

    for txt_file in txt_files[:overflow]:
        wav_file = sample_dir / f"{txt_file.stem}.wav"
        try:
            txt_file.unlink(missing_ok=True)
            wav_file.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to prune same_ts_ratio sample: %s", txt_file.stem)


def save_same_ts_sample(
    audio: np.ndarray,
    response: ASRResponse,
    same_ts_ratio: float,
    *,
    filename: Optional[str],
    language: Optional[str],
    context: Optional[str],
    normalized_context: Optional[str],
    retried_without_context: bool,
    first_response_before_retry: Optional[ASRResponse],
) -> None:
    SAME_TS_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    ratio_str = f"{same_ts_ratio:.6f}"
    base_name = f"{timestamp}_{ratio_str}"

    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        wav_path = SAME_TS_SAMPLE_DIR / f"{name}.wav"
        txt_path = SAME_TS_SAMPLE_DIR / f"{name}.txt"
        if not wav_path.exists() and not txt_path.exists():
            break
        suffix += 1

    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = np.asarray(pcm * 32767.0, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())

    txt_payload = {
        "saved_at": timestamp,
        "same_ts_ratio": round(float(same_ts_ratio), 6),
        "audio": {
            "file": wav_path.name,
            "sample_rate": SAMPLE_RATE,
            "num_samples": int(len(audio)),
            "duration_seconds": float(len(audio)) / float(SAMPLE_RATE),
        },
        "request": {
            "filename": filename,
            "language": language,
            "context": context,
            "normalized_context": normalized_context,
        },
        "runtime": {
            "hallucination_equal_ts_ratio_threshold": HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD,
            "high_confidence_equal_ts_ratio_threshold": HIGH_CONFIDENCE_EQUAL_TS_RATIO_THRESHOLD,
            "same_ts_ratio_excluded_tokens": SAME_TS_RATIO_EXCLUDED_TOKENS,
            "retried_without_context": retried_without_context,
            "asr_model_id": ASR_MODEL_ID,
            "aligner_model_id": ALIGNER_MODEL_ID,
        },
    }

    final_response_payload = {
        "language": response.language,
        "text": response.text,
        "timestamps": [item.model_dump() for item in response.timestamps],
        "same_ts_token": same_timestamp_tokens(response.timestamps),
    }

    if retried_without_context and first_response_before_retry is not None:
        txt_payload["response_before_retry"] = {
            "language": first_response_before_retry.language,
            "text": first_response_before_retry.text,
            "timestamps": [item.model_dump() for item in first_response_before_retry.timestamps],
            "same_ts_token": same_timestamp_tokens(first_response_before_retry.timestamps),
        }
        txt_payload["response_after_retry"] = final_response_payload
    else:
        txt_payload["response"] = final_response_payload

    txt_path.write_text(json.dumps(txt_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _prune_old_same_ts_samples(SAME_TS_SAMPLE_DIR, SAME_TS_SAMPLE_MAX_ENTRIES)


async def asr_worker(app: FastAPI) -> None:
    queue: asyncio.Queue[ASRTask] = app.state.queue

    while True:
        task = await queue.get()
        started_at = time.perf_counter()
        if task.future.cancelled():
            logger.info("Skipping cancelled ASR task before processing")
            queue.task_done()
            continue

        if len(task.audio_bytes) < 400:
            logger.warning(
                "Audio payload too small (%d bytes); returning empty result",
                len(task.audio_bytes),
            )
            set_future_result(
                task.future,
                ASRResponse(
                    language="None",
                    text="",
                    timestamps=[],
                ),
            )
            queue.task_done()
            continue

        try:
            audio = load_audio_from_bytes(task.audio_bytes, task.filename)
            rms = audio_rms(audio)
            if rms < MIN_AUDIO_RMS:
                logger.warning(
                    "Audio RMS too low (%.8f < %.8f); returning empty result",
                    rms,
                    MIN_AUDIO_RMS,
                )
                set_future_result(
                    task.future,
                    ASRResponse(
                        language="None",
                        text="",
                        timestamps=[],
                    )
                )
                continue

            normalized_context = tokenize_context(task.context)

            generate_kwargs = {}
            if normalized_context:
                generate_kwargs["system_prompt"] = normalized_context
            if task.language:
                generate_kwargs["language"] = task.language

            try:
                logger.info("ASR generate started: audio=%.2fs rms=%.8f", len(audio) / SAMPLE_RATE, rms)
                result = app.state.model.generate(audio, **generate_kwargs)
            except TypeError:
                # Some model builds may not support a language argument.
                generate_kwargs.pop("language", None)
                result = app.state.model.generate(audio, **generate_kwargs)

            result_language = task.language or getattr(result, "language", [None])[0] or "English"
            logger.info("ASR align started: text_chars=%d language=%s", len(result.text), result_language)
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

            first_response = ASRResponse(
                language=result_language,
                text=result.text,
                timestamps=timestamps,
            )

            # When context is provided, many zero-duration aligned tokens indicate likely hallucination.
            same_ts_ratio = equal_timestamp_ratio(timestamps)
            retried_without_context = False
            if normalized_context and 0 < same_ts_ratio < HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD:
                logger.warning(
                    "Equal_timestamp_ratio=%.2f%%; result may be partially hallucinated",
                    same_ts_ratio * 100,
                )

            if normalized_context and same_ts_ratio > HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD:
                if same_ts_ratio > HIGH_CONFIDENCE_EQUAL_TS_RATIO_THRESHOLD:
                    logger.warning(
                        "High-confidence hallucination: equal_timestamp_ratio=%.2f%%; returning empty result",
                        same_ts_ratio * 100,
                    )
                    final_response = ASRResponse(
                        language="None",
                        text="",
                        timestamps=[],
                    )
                else:
                    logger.warning(
                        "Suspected hallucination: equal_timestamp_ratio=%.2f%%; retrying without context",
                        same_ts_ratio * 100,
                    )
                    retried_without_context = True

                    retry_generate_kwargs = {}
                    if task.language:
                        retry_generate_kwargs["language"] = task.language

                    try:
                        logger.info("ASR retry generate started without context")
                        retry_result = app.state.model.generate(audio, **retry_generate_kwargs)
                    except TypeError:
                        # Some model builds may not support a language argument.
                        retry_generate_kwargs.pop("language", None)
                        retry_result = app.state.model.generate(audio, **retry_generate_kwargs)

                    retry_language = (
                        task.language or getattr(retry_result, "language", [None])[0] or "English"
                    )
                    logger.info(
                        "ASR retry align started: text_chars=%d language=%s",
                        len(retry_result.text),
                        retry_language,
                    )
                    retry_aligned = app.state.aligner.generate(
                        audio=audio,
                        text=retry_result.text,
                        language=retry_language,
                    )
                    retry_timestamps = [
                        TimestampItem(
                            start_time=float(item.start_time),
                            end_time=float(item.end_time),
                            text=item.text,
                        )
                        for item in retry_aligned
                    ]

                    final_response = ASRResponse(
                        language=retry_language,
                        text=retry_result.text,
                        timestamps=retry_timestamps,
                    )
            else:
                final_response = first_response

            if SAVE_SAME_TS_SAMPLES and 0.05 < same_ts_ratio < HIGH_CONFIDENCE_EQUAL_TS_RATIO_THRESHOLD:
                try:
                    save_same_ts_sample(
                        audio,
                        final_response,
                        same_ts_ratio,
                        filename=task.filename,
                        language=task.language,
                        context=task.context,
                        normalized_context=normalized_context,
                        retried_without_context=retried_without_context,
                        first_response_before_retry=(
                            first_response if retried_without_context else None
                        ),
                    )
                except Exception:
                    logger.exception("Failed to save same_ts_ratio sample")

            set_future_result(task.future, final_response)

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
            set_future_exception(task.future, exc)
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
