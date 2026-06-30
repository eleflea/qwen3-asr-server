from __future__ import annotations

import argparse
import time
from pathlib import Path

from mlx_audio.stt import load

from asr_server import (
    HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD,
    HIGH_CONFIDENCE_EQUAL_TS_RATIO_THRESHOLD,
    SAMPLE_RATE,
    TimestampItem,
    audio_rms,
    equal_timestamp_ratio,
    load_audio_from_bytes,
    resolve_model_paths,
    tokenize_context,
)


def transcribe_one(model, aligner, audio, *, language: str | None, context: str | None):
    normalized_context = tokenize_context(context)

    generate_kwargs = {}
    if normalized_context:
        generate_kwargs["system_prompt"] = normalized_context
    if language:
        generate_kwargs["language"] = language

    try:
        result = model.generate(audio, **generate_kwargs)
    except TypeError:
        generate_kwargs.pop("language", None)
        result = model.generate(audio, **generate_kwargs)

    result_language = language or getattr(result, "language", [None])[0] or "English"
    aligned = aligner.generate(audio=audio, text=result.text, language=result_language)
    timestamps = [
        TimestampItem(
            start_time=float(item.start_time),
            end_time=float(item.end_time),
            text=item.text,
        )
        for item in aligned
    ]

    same_ts_ratio = equal_timestamp_ratio(timestamps)
    if normalized_context and same_ts_ratio > HALLUCINATION_EQUAL_TS_RATIO_THRESHOLD:
        if same_ts_ratio > HIGH_CONFIDENCE_EQUAL_TS_RATIO_THRESHOLD:
            return result_language, "", [], same_ts_ratio

        retry_kwargs = {}
        if language:
            retry_kwargs["language"] = language

        try:
            result = model.generate(audio, **retry_kwargs)
        except TypeError:
            retry_kwargs.pop("language", None)
            result = model.generate(audio, **retry_kwargs)

        result_language = language or getattr(result, "language", [None])[0] or "English"
        aligned = aligner.generate(audio=audio, text=result.text, language=result_language)
        timestamps = [
            TimestampItem(
                start_time=float(item.start_time),
                end_time=float(item.end_time),
                text=item.text,
            )
            for item in aligned
        ]

    return result_language, result.text, timestamps, same_ts_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark batchsize=1 ASR throughput.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--language", default=None)
    parser.add_argument("--context", default=None)
    args = parser.parse_args()

    audio_paths = sorted(args.data_dir.glob("*"))
    if not audio_paths:
        raise SystemExit(f"No audio files found in {args.data_dir}")

    asr_path, aligner_path = resolve_model_paths()
    print(f"ASR model: {asr_path}")
    print(f"Aligner model: {aligner_path}")

    model = load(asr_path)
    aligner = load(aligner_path)

    total_audio_seconds = 0.0
    total_elapsed_seconds = 0.0

    for path in audio_paths:
        audio = load_audio_from_bytes(path.read_bytes(), path.name)
        audio_seconds = len(audio) / SAMPLE_RATE
        total_audio_seconds += audio_seconds

        started_at = time.perf_counter()
        language, text, timestamps, same_ts_ratio = transcribe_one(
            model,
            aligner,
            audio,
            language=args.language,
            context=args.context,
        )
        elapsed_seconds = time.perf_counter() - started_at
        total_elapsed_seconds += elapsed_seconds

        rtfx = audio_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
        print(
            f"{path.name}\t"
            f"audio={audio_seconds:.2f}s\t"
            f"cost={elapsed_seconds:.2f}s\t"
            f"RTFx={rtfx:.2f}x\t"
            f"rms={audio_rms(audio):.6f}\t"
            f"lang={language}\t"
            f"tokens={len(timestamps)}\t"
            f"same_ts={same_ts_ratio:.2%}\t"
            f"text={text[:80]}"
        )

    aggregate_rtfx = (
        total_audio_seconds / total_elapsed_seconds if total_elapsed_seconds > 0 else 0.0
    )
    print(
        f"TOTAL\tfiles={len(audio_paths)}\t"
        f"audio={total_audio_seconds:.2f}s\t"
        f"cost={total_elapsed_seconds:.2f}s\t"
        f"RTFx={aggregate_rtfx:.2f}x"
    )


if __name__ == "__main__":
    main()
