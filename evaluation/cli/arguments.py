# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Arguments and recurrence parsing for the checkpoint evaluation command."""

import argparse

import torch

from evaluation.benchmarks import TASK_DEFAULT_FEWSHOT
from model.execution import PRECISIONS


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample generations and benchmark scores for a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="a training checkpoint (.pth)")
    parser.add_argument("--out_dir", default=None, help="default: the checkpoint's run directory")
    parser.add_argument("--tokenizer_dir", default=None, help="default: from the checkpoint's dataset config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=PRECISIONS, default=None,
                        help="override stored run precision; old checkpoints preserve caller-controlled precision")
    parser.add_argument("--no_samples", action="store_true", help="skip the sample generations")
    parser.add_argument("--prompts_file", default=None, help="prompts file (see evaluation/prompts.py)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--no_sample_cache", action="store_true", help="legacy full-prefix latent resampling (default: fixed per-token latents and per-recurrence KV cache)")
    parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy")
    parser.add_argument("--tasks", default="", help="comma-separated lm-eval tasks; empty: no benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="examples per task")
    parser.add_argument(
        "--num_fewshot", type=int, default=TASK_DEFAULT_FEWSHOT,
        help="examples in the context of every task (-1: each task's own default, e.g. 5 for gsm8k)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="seeds the isolated RNG of both entry points")
    parser.add_argument(
        "--recurrence", action="append", default=None,
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
