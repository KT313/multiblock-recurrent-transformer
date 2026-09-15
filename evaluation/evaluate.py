# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Sample generations and benchmark scores for one training checkpoint.

    uv run python evaluation/evaluate.py --checkpoint outputs/<run>/checkpoints/step-00000094-<run>.pth
    uv run python evaluation/evaluate.py --checkpoint <pth> --tasks arc_easy,hellaswag --limit 200 --recurrence 4,4,4 --recurrence 12,12,12

Writes `samples/step-XXXXXXXX.jsonl` and `benchmarks/step-XXXXXXXX.json` into the checkpoint's run directory
(`--out_dir` elsewhere). The tokenizer comes from the dataset config the checkpoint was trained with
(`--tokenizer_dir` overrides). Benchmarks need the `eval` extra (`uv sync --extra eval`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python evaluation/evaluate.py` from the repo root

from evaluation.cli import (
    load_checkpoint_evaluation,
    load_checkpoint_model,
    parse_arguments,
    parse_recurrences,
    run_checkpoint_benchmarks,
    run_checkpoint_samples,
    tokenizer_dir_of,
)


__all__ = [
    "load_checkpoint_model",
    "main",
    "parse_arguments",
    "parse_recurrences",
    "tokenizer_dir_of",
]


def main(argv: list[str] | None = None) -> int:
    # parse arguments and restore the checkpoint's evaluation inputs
    arguments = parse_arguments(argv)
    evaluation = load_checkpoint_evaluation(arguments)
    print(f"checkpoint {evaluation.checkpoint} (step {evaluation.step}) on {arguments.device}; output under {evaluation.out_dir}")

    # generate samples before running the requested benchmarks
    if not arguments.no_samples:
        run_checkpoint_samples(arguments, evaluation)
    tasks = [task for task in arguments.tasks.split(",") if task.strip()]
    if tasks:
        run_checkpoint_benchmarks(arguments, evaluation, tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
