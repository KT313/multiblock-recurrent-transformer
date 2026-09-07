# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Benchmark scores through lm-eval-harness (the `eval` extra), on the live model or a checkpoint.

`lm_eval` is imported on first use, so training without the extra works until a benchmark is requested. The task
datasets come from the HuggingFace Hub (network on first use).
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.wrapper import Recurrence, check_recurrence, hf_wrapper_around, isolated_inference, recurrence_label
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer
from training.settings import DEFAULT_BENCHMARK_TASKS

DEFAULT_TASKS: tuple[str, ...] = DEFAULT_BENCHMARK_TASKS  # one tuple, defined with the run setting that names it
BENCHMARKS_DIR = "benchmarks"  # under the run directory
METRIC_PREFIX = "benchmark"  # wandb keys: benchmark/<recurrence label>/<task>/<metric>
EVAL_EXTRA_HINT = "lm_eval is not installed: install the eval extra (uv sync --extra eval) to run benchmarks"
TASK_DEFAULT_FEWSHOT = -1  # num_fewshot: each task's own default (gsm8k is 5-shot), lm-eval's `num_fewshot=None`
BOOTSTRAP_ITERS = 100  # lm-eval's stderr resampling; its default of 100000 is the slowest part of a small run


def benchmarks_path(run_directory: Path, step: int) -> Path:
    return run_directory / BENCHMARKS_DIR / f"step-{step:08d}.json"


def evaluate_on_benchmarks(
    model: RecurrentGPT,
    tokenizer: Tokenizer,
    tasks: Sequence[str] = DEFAULT_TASKS,
    *,
    num_fewshot: int = TASK_DEFAULT_FEWSHOT,
    limit: int | None = None,
    batch_size: int = 8,
    recurrences: Sequence[Recurrence] = (None,),
    out_path: Path | None = None,
    step: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """
    Score the model on tasks with lm-eval-harness, once per recurrence setting (steps per core block, None: the
    mean recurrence), and return `benchmark/<recurrence label>/<task>/<metric>` floats (stderr entries left out).
    limit caps the examples per task, num_fewshot -1 leaves every task at its own default (`TASK_DEFAULT_FEWSHOT`).
    seed seeds the isolated RNG and lm-eval's own seeding of torch (it reseeds on every call, so passing it is the
    only way the seed reaches the scoring). With out_path the full lm-eval results per setting (plus step and the
    settings used) are written as JSON.
    """

    if not tasks:
        raise ValueError("no benchmark tasks given")
    if not recurrences:
        raise ValueError("no recurrence setting given (None stands for the mean recurrence)")
    if num_fewshot < TASK_DEFAULT_FEWSHOT:
        raise ValueError(f"num_fewshot must be >= {TASK_DEFAULT_FEWSHOT} ({TASK_DEFAULT_FEWSHOT}: each task's own default), got {num_fewshot}")
    for recurrence in recurrences:
        check_recurrence(recurrence, model)
    lm_eval, hf_models = _import_lm_eval()
    metrics: dict[str, float] = {}
    raw_results: dict[str, Any] = {}
    versions: dict[str, Any] = {}
    n_shot: dict[str, Any] = {}
    for recurrence in recurrences:
        with isolated_inference(model, recurrence, seed=seed):
            wrapper = hf_wrapper_around(model, tokenizer)
            language_model = hf_models.HFLM(  # BOS as in training and sampling; the table length caps the few-shot prompts
                pretrained=wrapper, tokenizer=tokenizer.processor, batch_size=batch_size, add_bos_token=True,
                max_length=model.config.model_max_sequence_length,
            )
            results: dict[str, Any] = lm_eval.simple_evaluate(  # log_samples: the per-sample logs are held in memory and never read
                model=language_model, tasks=list(tasks), limit=limit, log_samples=False,
                num_fewshot=None if num_fewshot == TASK_DEFAULT_FEWSHOT else num_fewshot,
                torch_random_seed=seed, bootstrap_iters=BOOTSTRAP_ITERS,
            )
        label = recurrence_label(recurrence)
        metrics |= flatten_results(results["results"], label)
        raw_results[label] = results["results"]
        versions = results.get("versions", versions)
        n_shot = results.get("n-shot", n_shot)
    if out_path is not None:
        record = {
            "step": step,
            "tasks": list(tasks),
            "num_fewshot": num_fewshot,
            "limit": limit,
            "seed": seed,
            "recurrences": [None if recurrence is None else list(recurrence) for recurrence in recurrences],
            "metrics": metrics,
            "results": raw_results,
            "versions": versions,
            "n-shot": n_shot,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return metrics


def flatten_results(results: Mapping[str, Mapping[str, Any]], label: str) -> dict[str, float]:
    """
    lm-eval's per-task metric dicts (`{"acc,none": 0.23, "acc_stderr,none": 0.01, "alias": ...}`) as
    `benchmark/<label>/<task>/<metric>` floats without the stderr entries; label names the recurrence setting. A
    metric under a filter other than `none` keeps it as a suffix (`exact_match,strict-match` -> `exact_match_strict-match`),
    so two filters of one metric stay two entries.
    """

    flat: dict[str, float] = {}
    for task, metrics in results.items():
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            name, _, metric_filter = key.partition(",")
            if name.endswith("_stderr"):
                continue
            if metric_filter and metric_filter != "none":
                name = f"{name}_{metric_filter}"
            flat[f"{METRIC_PREFIX}/{label}/{task}/{name}"] = float(value)
    return flat


def _import_lm_eval() -> tuple[Any, Any]:
    try:
        return importlib.import_module("lm_eval"), importlib.import_module("lm_eval.models.huggingface")
    except ImportError as error:
        raise ImportError(EVAL_EXTRA_HINT) from error
