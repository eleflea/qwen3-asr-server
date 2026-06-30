from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from mlx_audio.stt import load

from asr_server import SAMPLE_RATE, load_audio_from_bytes, resolve_model_paths
from benchmark import transcribe_one


@dataclass
class FileResult:
    file: str
    audio_seconds: float
    elapsed_seconds: float
    rtfx: float
    language: str
    text: str
    timestamps: list[dict]


@dataclass
class SuiteResult:
    name: str
    asr_model: str
    aligner_model: str
    files: list[FileResult]
    total_audio_seconds: float
    total_elapsed_seconds: float
    rtfx: float


def run_suite(
    *,
    name: str,
    asr_model: str,
    aligner_model: str,
    audio_paths: list[Path],
    language: str | None,
    context: str | None,
) -> SuiteResult:
    model = load(asr_model)
    aligner = load(aligner_model)

    files: list[FileResult] = []
    total_audio_seconds = 0.0
    total_elapsed_seconds = 0.0

    for path in audio_paths:
        audio = load_audio_from_bytes(path.read_bytes(), path.name)
        audio_seconds = len(audio) / SAMPLE_RATE
        started_at = time.perf_counter()
        result_language, text, timestamps, _same_ts_ratio = transcribe_one(
            model,
            aligner,
            audio,
            language=language,
            context=context,
        )
        elapsed_seconds = time.perf_counter() - started_at
        rtfx = audio_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0

        total_audio_seconds += audio_seconds
        total_elapsed_seconds += elapsed_seconds
        files.append(
            FileResult(
                file=path.name,
                audio_seconds=audio_seconds,
                elapsed_seconds=elapsed_seconds,
                rtfx=rtfx,
                language=result_language,
                text=text,
                timestamps=[item.model_dump() for item in timestamps],
            )
        )
        print(
            f"{name}\t{path.name}\t"
            f"audio={audio_seconds:.2f}s\tcost={elapsed_seconds:.2f}s\t"
            f"RTFx={rtfx:.2f}x\ttokens={len(timestamps)}\t{text[:70]}",
            flush=True,
        )

    del model
    del aligner
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass

    return SuiteResult(
        name=name,
        asr_model=asr_model,
        aligner_model=aligner_model,
        files=files,
        total_audio_seconds=total_audio_seconds,
        total_elapsed_seconds=total_elapsed_seconds,
        rtfx=(
            total_audio_seconds / total_elapsed_seconds
            if total_elapsed_seconds > 0
            else 0.0
        ),
    )


def compare_file(old: FileResult, new: FileResult) -> dict:
    text_similarity = SequenceMatcher(None, old.text, new.text).ratio()
    old_tokens = [item["text"] for item in old.timestamps]
    new_tokens = [item["text"] for item in new.timestamps]
    min_len = min(len(old.timestamps), len(new.timestamps))

    same_position_count = sum(
        1 for idx in range(min_len) if old_tokens[idx] == new_tokens[idx]
    )
    same_position_ratio = same_position_count / min_len if min_len else 1.0

    start_deltas = []
    end_deltas = []
    for idx in range(min_len):
        if old_tokens[idx] != new_tokens[idx]:
            continue
        start_deltas.append(
            abs(
                float(old.timestamps[idx]["start_time"])
                - float(new.timestamps[idx]["start_time"])
            )
        )
        end_deltas.append(
            abs(
                float(old.timestamps[idx]["end_time"])
                - float(new.timestamps[idx]["end_time"])
            )
        )

    return {
        "file": old.file,
        "text_equal": old.text == new.text,
        "text_similarity": text_similarity,
        "old_text": old.text,
        "new_text": new.text,
        "old_token_count": len(old.timestamps),
        "new_token_count": len(new.timestamps),
        "same_position_token_ratio": same_position_ratio,
        "mean_start_delta_seconds": mean(start_deltas) if start_deltas else None,
        "max_start_delta_seconds": max(start_deltas) if start_deltas else None,
        "mean_end_delta_seconds": mean(end_deltas) if end_deltas else None,
        "max_end_delta_seconds": max(end_deltas) if end_deltas else None,
        "old_elapsed_seconds": old.elapsed_seconds,
        "new_elapsed_seconds": new.elapsed_seconds,
        "old_rtfx": old.rtfx,
        "new_rtfx": new.rtfx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and current ASR models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--language", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--output", type=Path, default=Path("asr_model_compare.json"))
    args = parser.parse_args()

    audio_paths = sorted(args.data_dir.glob("*"))
    if not audio_paths:
        raise SystemExit(f"No audio files found in {args.data_dir}")

    new_asr, new_aligner = resolve_model_paths()
    suites = [
        run_suite(
            name="old",
            asr_model="mlx-community/Qwen3-ASR-1.7B-8bit",
            aligner_model="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            audio_paths=audio_paths,
            language=args.language,
            context=args.context,
        ),
        run_suite(
            name="new",
            asr_model=new_asr,
            aligner_model=new_aligner,
            audio_paths=audio_paths,
            language=args.language,
            context=args.context,
        ),
    ]

    old_by_file = {item.file: item for item in suites[0].files}
    new_by_file = {item.file: item for item in suites[1].files}
    comparisons = [
        compare_file(old_by_file[path.name], new_by_file[path.name])
        for path in audio_paths
    ]

    payload = {
        "suites": [
            {
                **asdict(suite),
                "files": [asdict(item) for item in suite.files],
            }
            for suite in suites
        ],
        "comparisons": comparisons,
        "summary": {
            "old_rtfx": suites[0].rtfx,
            "new_rtfx": suites[1].rtfx,
            "speedup": suites[1].rtfx / suites[0].rtfx if suites[0].rtfx else None,
            "all_text_equal": all(item["text_equal"] for item in comparisons),
            "mean_text_similarity": mean(
                item["text_similarity"] for item in comparisons
            ),
        },
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "SUMMARY\t"
        f"old={suites[0].rtfx:.2f}x\t"
        f"new={suites[1].rtfx:.2f}x\t"
        f"speedup={payload['summary']['speedup']:.2f}x\t"
        f"mean_text_similarity={payload['summary']['mean_text_similarity']:.4f}\t"
        f"all_text_equal={payload['summary']['all_text_equal']}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
