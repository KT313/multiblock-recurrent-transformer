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

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from data_preparation.dataset_config import load_dataset_config
from data_preparation.layout import DatasetLayout
from evaluation.benchmarks import benchmarks_path, evaluate_on_benchmarks
from evaluation.prompts import load_prompts
from evaluation.samples import generate_and_save_samples, samples_path
from model.config import RecurrentConfig
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample generations and benchmark scores for a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="a training checkpoint (.pth)")
    parser.add_argument("--out_dir", default=None, help="default: the checkpoint's run directory")
    parser.add_argument("--tokenizer_dir", default=None, help="default: from the checkpoint's dataset config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_samples", action="store_true", help="skip the sample generations")
    parser.add_argument("--prompts_file", default=None, help="prompts file (see evaluation/prompts.py)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy")
    parser.add_argument("--tasks", default="", help="comma-separated lm-eval tasks; empty: no benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="examples per task")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--recurrence",
        action="append",
        default=None,
        help='recurrent steps per block, e.g. "4,12,4"; repeat the option for several settings (default: the mean recurrence)',
    )
    return parser.parse_args(argv)


def parse_recurrences(values: list[str] | None) -> list[list[int] | None]:
    """
    `["4,12,4", "8,8,8"]` as lists of ints; None or an empty list means the mean recurrence once.
    """

    if not values:
        return [None]
    return [[int(steps) for steps in value.split(",")] for value in values]


def load_checkpoint_model(state: dict[str, Any], device: str) -> RecurrentGPT:
    """
    The checkpoint's model on device (its stored `model_config` and `model` state dict).
    """

    known = {field.name for field in fields(RecurrentConfig)}
    config = RecurrentConfig(**{key: value for key, value in state["model_config"].items() if key in known})
    model = RecurrentGPT(config)
    model.load_state_dict(state["model"])
    return model.to(device)


def tokenizer_dir_of(state: dict[str, Any]) -> Path:
    """
    The tokenizer directory the checkpoint's run used: `dataset_dir/tokenizers/<name>` of its dataset config.
    """

    settings = state["settings"]
    dataset_config = load_dataset_config(settings["dataset_config"])
    return DatasetLayout(Path(settings["dataset_dir"])).tokenizer_dir(dataset_config.tokenizer.name)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    checkpoint = Path(arguments.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = int(state["step"])
    out_dir = Path(arguments.out_dir) if arguments.out_dir else checkpoint.parent.parent
    model = load_checkpoint_model(state, arguments.device)
    tokenizer = Tokenizer(arguments.tokenizer_dir or tokenizer_dir_of(state))
    recurrences = parse_recurrences(arguments.recurrence)
    print(f"checkpoint {checkpoint} (step {step}) on {arguments.device}; output under {out_dir}")

    if not arguments.no_samples:
        path = samples_path(out_dir, step)
        samples = generate_and_save_samples(
            model,
            tokenizer,
            path,
            step=step,
            prompts=load_prompts(file=arguments.prompts_file),
            max_new_tokens=arguments.max_new_tokens,
            temperature=arguments.temperature,
            recurrences=recurrences,
            batch_size=arguments.batch_size,
        )
        print(f"{len(samples)} samples written to {path}")
        for sample in samples:
            print(f"--- [{sample.kind}, recurrence {sample.recurrence or 'mean'}] {sample.prompt!r}\n{sample.completion}")

    tasks = [task for task in arguments.tasks.split(",") if task.strip()]
    if tasks:
        path = benchmarks_path(out_dir, step)
        metrics = evaluate_on_benchmarks(
            model,
            tokenizer,
            tasks,
            num_fewshot=arguments.num_fewshot,
            limit=arguments.limit,
            batch_size=arguments.batch_size,
            recurrences=recurrences,
            out_path=path,
            step=step,
        )
        print(f"benchmark results written to {path}")
        for name, value in metrics.items():
            print(f"  {name:40s} {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
