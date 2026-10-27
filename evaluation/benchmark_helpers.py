# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Validation, harness setup and result serialization for benchmark evaluation.

A sample selection (`BenchmarkSamples`, read by `load_benchmark_samples`) restricts a run to listed examples per
lm-eval leaf task, by position in the task's evaluation documents, the numbering `evaluation/contamination_check.py`
writes its `clean_samples.json` in. Every leaf task of the run must be listed (`check_samples_cover`), so no task
silently falls back to its full test set, and the result record names the file, its digest and the counts.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evaluation.harness_runtime import check_benchmark_results, use_serial_bootstrap
from evaluation.publication import open_atomic_output
from evaluation.metadata import METADATA_VERSION, dependency_versions
from evaluation.session import InferenceSession
from evaluation.wrapper import Recurrence, check_recurrence, recurrence_label
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
CODE_EXECUTION_ENV = "HF_ALLOW_CODE_EVAL"  # "1" lets tasks that execute generated code run (humaneval, mbpp)


@dataclass(frozen=True)
class BenchmarkSamples:
    """The examples to evaluate per lm-eval leaf task (positions in its evaluation documents), and their provenance."""

    indices: dict[str, list[int]]
    source: str
    digest: str  # sha256 of the file's bytes
    criterion: dict[str, Any]  # how the file was made, as written by its producer

    def record(self) -> dict[str, Any]:
        return {"source": self.source, "digest": self.digest, "criterion": self.criterion,
                "counts": {task: len(indices) for task, indices in sorted(self.indices.items())}}


def load_benchmark_samples(path: str | Path) -> BenchmarkSamples:
    """
    Read `{"samples": {leaf task: [doc index, ...]}, "criterion": {...}}`; every list non-empty, of distinct
    non-negative integers (a task without examples left cannot be scored, and lm-eval would fail on it later).
    """

    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("samples"), dict) or not data["samples"]:
        raise ValueError(f"{path}: expected a JSON object with a non-empty 'samples' mapping")
    indices: dict[str, list[int]] = {}
    for task, values in data["samples"].items():
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or value < 0 for value in values) or len(set(values)) != len(values)):
            raise ValueError(f"{path}: samples of {task!r} must be a non-empty list of distinct non-negative integers")
        indices[str(task)] = sorted(values)
    criterion = data.get("criterion", {})
    return BenchmarkSamples(indices, str(Path(path).resolve()), hashlib.sha256(raw).hexdigest(),
                            criterion if isinstance(criterion, dict) else {"value": criterion})


def benchmark_leaf_tasks(tasks: Sequence[str]) -> dict[str, int]:
    """{leaf task name: evaluation documents} of the requested tasks, groups expanded the way lm-eval runs them."""

    from lm_eval.tasks import TaskManager, get_task_dict

    leaves: dict[str, int] = {}

    def collect(mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                collect(value)
            else:
                leaves[str(key)] = len(value.eval_docs)

    collect(get_task_dict(list(tasks), TaskManager()))
    return leaves


def check_samples_cover(samples: BenchmarkSamples, leaves: Mapping[str, int]) -> None:
    """Every leaf task of the run is listed and every listed index exists; tasks of the file outside the run are fine."""

    missing = sorted(set(leaves) - set(samples.indices))
    if missing:
        raise ValueError(f"sample selection {samples.source} does not list {missing}; they would run in full")
    for task, count in leaves.items():
        beyond = [index for index in samples.indices[task] if index >= count]
        if beyond:
            raise ValueError(f"sample selection lists {task} documents {beyond[:5]} but the task has {count}")


def code_execution_allowed() -> bool:
    """
    Whether tasks marked unsafe_code may run. The same variable unlocks the `code_eval` metric those tasks score
    with, so one switch covers both; it is not a Settings field because it does not change any score.
    """

    return os.environ.get(CODE_EXECUTION_ENV) == "1"


def check_benchmark_requests(
    model: RecurrentGPT, tasks: Sequence[str], recurrences: Sequence[Recurrence], num_fewshot: int,
) -> None:
    if not tasks:
        raise ValueError("no benchmark tasks given")
    if not recurrences:
        raise ValueError("no recurrence setting given (None stands for the mean recurrence)")
    if num_fewshot < TASK_DEFAULT_FEWSHOT:
        raise ValueError(f"num_fewshot must be >= {TASK_DEFAULT_FEWSHOT} ({TASK_DEFAULT_FEWSHOT}: each task's own default), got {num_fewshot}")
    if len({recurrence_label(value) for value in recurrences}) != len(recurrences):
        raise ValueError("benchmark recurrence labels must be unique")
    for recurrence in recurrences:
        check_recurrence(recurrence, model)


def build_benchmark_harness(
    session: InferenceSession, tokenizer: Tokenizer, hf_models: Any, batch_size: int | str, max_length: int,
    apply_chat_template: bool = False, *, local_generation: bool = False,
) -> tuple[RecurrentGPTForCausalLM, Any]:
    wrapper = session.hf_wrapper(tokenizer)
    harness = hf_models.HFLM
    if tokenizer.contract is not None:
        from evaluation.literal_harness import create_literal_harness_class

        harness = create_literal_harness_class(harness, chat=apply_chat_template)
    elif apply_chat_template:
        raise ValueError("chat benchmarks require the literal chat tokenizer profile")
    if local_generation:
        from evaluation.local_harness import create_local_harness_class
        harness = create_local_harness_class(harness)
    language_model = harness(  # BOS as in training and sampling; the table length caps few-shot prompts
        pretrained=wrapper, tokenizer=tokenizer.processor, batch_size=batch_size, add_bos_token=True,
        max_length=max_length, mixed_precision_dtype=session.mixed_precision_dtype, logits_cache=True,
    )
    return wrapper, language_model


def run_benchmark_harness(
    lm_eval: Any, language_model: Any, tasks: Sequence[str], limit: int | None, num_fewshot: int,
    apply_chat_template: bool = False, samples: BenchmarkSamples | None = None,
) -> dict[str, Any]:
    """Use the caller's post-construction Torch seed and preserve the harness's other seed defaults.

    Reseeding immediately before this call matches simple_evaluate's former Torch seeding point: its preceding
    argument checks, Python/NumPy seeds and logging consume no Torch randomness. None opts out of the harness's
    all-device Torch seeding, including its pending lazy CUDA side effects.
    """
    if samples is not None and limit is not None:
        raise ValueError("a sample selection and benchmark_limit cannot be combined")
    with use_serial_bootstrap():
        results: dict[str, Any] = lm_eval.simple_evaluate(  # per-sample logs would be retained in memory but never read
            model=language_model, tasks=list(tasks), limit=limit, log_samples=False,
            num_fewshot=None if num_fewshot == TASK_DEFAULT_FEWSHOT else num_fewshot,
            random_seed=HARNESS_RANDOM_SEED, numpy_random_seed=HARNESS_NUMPY_SEED,
            torch_random_seed=None, fewshot_random_seed=HARNESS_FEWSHOT_SEED, bootstrap_iters=BOOTSTRAP_ITERS,
            fewshot_as_multiturn=True, use_cache=None, cache_requests=False,
            confirm_run_unsafe_code=code_execution_allowed(),
            **({"samples": {task: list(indices) for task, indices in samples.indices.items()}} if samples else {}),
            **({"apply_chat_template": True} if apply_chat_template else {}),
        )
    check_benchmark_results(results)
    return results


def save_benchmark_results(
    out_path: Path, *, step: int | None, tasks: Sequence[str], num_fewshot: int, limit: int | None, seed: int,
    recurrences: Sequence[Recurrence], metrics: dict[str, float], raw_results: dict[str, Any],
    versions: dict[str, Any], n_shot: dict[str, Any], execution_settings: dict[str, Any],
    custom_kernels: bool, execution_policy: ExecutionPolicy | None, samples: BenchmarkSamples | None = None,
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
    if samples is not None:
        record["samples"] = samples.record()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_atomic_output(out_path) as file:
        file.write(json.dumps(record, indent=2, default=str) + "\n")


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
