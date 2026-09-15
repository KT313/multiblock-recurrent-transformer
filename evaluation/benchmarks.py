# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Benchmark scores through lm-eval-harness (the `eval` extra), on the live model or a checkpoint.

`lm_eval` is imported on first use, so training without the extra works until a benchmark is requested. The task
datasets come from the HuggingFace Hub (network on first use).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation.benchmark_helpers import (
    BENCHMARKS_DIR,
    BOOTSTRAP_ITERS,
    DEFAULT_TASKS,
    EVAL_EXTRA_HINT,
    HARNESS_FEWSHOT_SEED,
    HARNESS_NUMPY_SEED,
    HARNESS_RANDOM_SEED,
    METRIC_PREFIX,
    TASK_DEFAULT_FEWSHOT,
    build_benchmark_harness,
    check_benchmark_requests,
    flatten_results,
    import_lm_eval as _import_lm_eval,
    run_benchmark_harness,
    save_benchmark_results,
)
from evaluation.metadata import benchmark_execution_metadata
from evaluation.rng import preserve_rng, seed_model_rng
from evaluation.session import inference_session
from evaluation.wrapper import Recurrence, recurrence_label
from model.execution import ExecutionPolicy
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer


__all__ = [
    "BENCHMARKS_DIR",
    "BOOTSTRAP_ITERS",
    "DEFAULT_TASKS",
    "EVAL_EXTRA_HINT",
    "HARNESS_FEWSHOT_SEED",
    "HARNESS_NUMPY_SEED",
    "HARNESS_RANDOM_SEED",
    "METRIC_PREFIX",
    "TASK_DEFAULT_FEWSHOT",
    "_import_lm_eval",
    "benchmarks_path",
    "evaluate_on_benchmarks",
    "flatten_results",
]


def benchmarks_path(run_directory: Path, step: int) -> Path:
    return run_directory / BENCHMARKS_DIR / f"step-{step:08d}.json"


def evaluate_on_benchmarks(
    model: RecurrentGPT,
    tokenizer: Tokenizer,
    tasks: Sequence[str] = DEFAULT_TASKS,
    *,
    num_fewshot: int = TASK_DEFAULT_FEWSHOT,
    limit: int | None = None,
    batch_size: int | str = 8,
    recurrences: Sequence[Recurrence] = (None,),
    out_path: Path | None = None,
    step: int | None = None,
    seed: int = 0,
    execution_policy: ExecutionPolicy | None = None,
) -> dict[str, float]:
    """
    Score the model on tasks with lm-eval-harness, once per recurrence setting (steps per core block, None: the
    mean recurrence), and return `benchmark/<recurrence label>/<task>/<metric>` floats (stderr entries left out).
    limit caps the examples per task, num_fewshot -1 leaves every task at its own default (`TASK_DEFAULT_FEWSHOT`).
    seed seeds the isolated CPU/model-device RNG again immediately before the harness call. The harness's
    all-device Torch seeding is disabled through its supported None option. Python/NumPy/few-shot seeds
    retain their historical defaults (0/1234/1234). With out_path the full lm-eval results per setting (plus step and the
    settings used) are written as JSON.
    """

    # validate requests before importing optional dependencies inside RNG isolation
    check_benchmark_requests(model, tasks, recurrences, num_fewshot)
    with preserve_rng(next(model.parameters()).device):
        lm_eval, hf_models = _import_lm_eval()  # optional imports must not leak global RNG draws

    # evaluate each recurrence in its own inference session
    metrics: dict[str, float] = {}
    raw_results: dict[str, Any] = {}
    versions: dict[str, Any] = {}
    n_shot: dict[str, Any] = {}
    execution_settings: dict[str, Any] = {}
    for recurrence in recurrences:
        with inference_session(model, recurrence, seed=seed, execution_policy=execution_policy) as session:
            wrapper, language_model = build_benchmark_harness(
                session, tokenizer, hf_models, batch_size, model.config.model_max_sequence_length,
            )
            seed_model_rng(seed, session.device)  # reseed after HFLM/wrapper setup; it may consume Torch RNG
            results = run_benchmark_harness(lm_eval, language_model, tasks, limit, num_fewshot)
            if out_path is not None:
                execution_settings[recurrence_label(recurrence)] = benchmark_execution_metadata(
                    session, language_model, wrapper, results, lm_eval.simple_evaluate, batch_size=batch_size,
                    context_cap=model.config.model_max_sequence_length, custom_kernels=model.config.use_custom_kernels,
                )
        label = recurrence_label(recurrence)
        metrics |= flatten_results(results["results"], label)
        raw_results[label] = results["results"]
        versions = results.get("versions", versions)
        n_shot = results.get("n-shot", n_shot)

    # save complete results and execution metadata when requested
    if out_path is not None:
        save_benchmark_results(
            out_path, step=step, tasks=tasks, num_fewshot=num_fewshot, limit=limit, seed=seed, recurrences=recurrences,
            metrics=metrics, raw_results=raw_results, versions=versions, n_shot=n_shot,
            execution_settings=execution_settings, custom_kernels=model.config.use_custom_kernels,
            execution_policy=execution_policy,
        )
    return metrics
