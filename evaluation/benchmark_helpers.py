# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Validation, harness setup and result serialization for benchmark evaluation."""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evaluation.metadata import METADATA_VERSION, dependency_versions
from evaluation.session import InferenceSession
from evaluation.wrapper import Recurrence, check_recurrence
from model.execution import ExecutionPolicy
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer
from training.settings import DEFAULT_BENCHMARK_TASKS

if TYPE_CHECKING:
    from model.hf.modeling import RecurrentGPTForCausalLM

DEFAULT_TASKS: tuple[str, ...] = DEFAULT_BENCHMARK_TASKS  # one tuple, defined with the run setting that names it
BENCHMARKS_DIR = "benchmarks"  # under the run directory
METRIC_PREFIX = "benchmark"  # wandb keys: benchmark/<recurrence label>/<task>/<metric>
EVAL_EXTRA_HINT = "lm_eval is not installed: run uv sync to install the project dependencies and run benchmarks"
TASK_DEFAULT_FEWSHOT = -1  # num_fewshot: each task's own default (gsm8k is 5-shot), lm-eval's `num_fewshot=None`
HARNESS_RANDOM_SEED = 0  # preserve lm-eval defaults, independently of the model seed
HARNESS_NUMPY_SEED = 1234
HARNESS_FEWSHOT_SEED = 1234
BOOTSTRAP_ITERS = 100  # lm-eval's stderr resampling; its default of 100000 is the slowest part of a small run


def check_benchmark_requests(
    model: RecurrentGPT, tasks: Sequence[str], recurrences: Sequence[Recurrence], num_fewshot: int,
) -> None:
    if not tasks:
        raise ValueError("no benchmark tasks given")
    if not recurrences:
        raise ValueError("no recurrence setting given (None stands for the mean recurrence)")
    if num_fewshot < TASK_DEFAULT_FEWSHOT:
        raise ValueError(f"num_fewshot must be >= {TASK_DEFAULT_FEWSHOT} ({TASK_DEFAULT_FEWSHOT}: each task's own default), got {num_fewshot}")
    for recurrence in recurrences:
        check_recurrence(recurrence, model)


def build_benchmark_harness(
    session: InferenceSession, tokenizer: Tokenizer, hf_models: Any, batch_size: int | str, max_length: int,
) -> tuple[RecurrentGPTForCausalLM, Any]:
    wrapper = session.hf_wrapper(tokenizer)
    language_model = hf_models.HFLM(  # BOS as in training and sampling; the table length caps few-shot prompts
        pretrained=wrapper, tokenizer=tokenizer.processor, batch_size=batch_size, add_bos_token=True,
        max_length=max_length, mixed_precision_dtype=session.mixed_precision_dtype,
    )
    return wrapper, language_model


def run_benchmark_harness(
    lm_eval: Any, language_model: Any, tasks: Sequence[str], limit: int | None, num_fewshot: int,
) -> dict[str, Any]:
    """Use the caller's post-construction Torch seed and preserve the harness's other seed defaults.

    Reseeding immediately before this call matches simple_evaluate's former Torch seeding point: its preceding
    argument checks, Python/NumPy seeds and logging consume no Torch randomness. None opts out of the harness's
    all-device Torch seeding, including its pending lazy CUDA side effects.
    """
    results: dict[str, Any] = lm_eval.simple_evaluate(  # per-sample logs would be retained in memory but never read
        model=language_model, tasks=list(tasks), limit=limit, log_samples=False,
        num_fewshot=None if num_fewshot == TASK_DEFAULT_FEWSHOT else num_fewshot,
        random_seed=HARNESS_RANDOM_SEED, numpy_random_seed=HARNESS_NUMPY_SEED,
        torch_random_seed=None, fewshot_random_seed=HARNESS_FEWSHOT_SEED, bootstrap_iters=BOOTSTRAP_ITERS,
    )
    return results


def save_benchmark_results(
    out_path: Path, *, step: int | None, tasks: Sequence[str], num_fewshot: int, limit: int | None, seed: int,
    recurrences: Sequence[Recurrence], metrics: dict[str, float], raw_results: dict[str, Any],
    versions: dict[str, Any], n_shot: dict[str, Any], execution_settings: dict[str, Any],
    custom_kernels: bool, execution_policy: ExecutionPolicy | None,
) -> None:
    record: dict[str, Any] = {
        "step": step,
        "tasks": list(tasks),
        "num_fewshot": num_fewshot,
        "limit": limit,
        "seed": seed,
        "rng_seeds": {
            "random_seed": HARNESS_RANDOM_SEED, "numpy_random_seed": HARNESS_NUMPY_SEED,
            "torch_random_seed": seed, "fewshot_random_seed": HARNESS_FEWSHOT_SEED,
        },
        "torch_seed_owner": "evaluation_cpu_model_device",
        "recurrences": [None if recurrence is None else list(recurrence) for recurrence in recurrences],
        "metrics": metrics,
        "results": raw_results,
        "versions": versions,
        "n-shot": n_shot,
        "execution_metadata": {
            "schema_version": METADATA_VERSION,
            "dependencies": dependency_versions(custom_kernels=custom_kernels),
            "recurrences": execution_settings,
        },
    }
    if execution_policy is not None:
        record["execution_precision"] = execution_policy.precision
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


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


def import_lm_eval() -> tuple[Any, Any]:
    try:
        package = importlib.import_module("lm_eval")
        huggingface = importlib.import_module("lm_eval.models.huggingface")
    except ImportError as error:
        raise ImportError(EVAL_EXTRA_HINT) from error
    required = ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed")
    parameters = inspect.signature(package.simple_evaluate).parameters
    accepts_keywords = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    missing = [name for name in required if name not in parameters and not accepts_keywords]
    if missing:
        raise RuntimeError(f"lm-eval simple_evaluate lacks required RNG seed arguments: {', '.join(missing)}")
    return package, huggingface
