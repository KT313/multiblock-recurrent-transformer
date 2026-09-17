# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Run the stock evaluator once, using existing replicas for fixed inference jobs."""
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.benchmark_helpers import (
    build_benchmark_harness, check_benchmark_requests, flatten_results, import_lm_eval,
    run_benchmark_harness, save_benchmark_results,
)
from evaluation.distributed import EvaluationCancelled, agree_on_phase, exchange, finish_publication, poll_stop
from evaluation.metadata import benchmark_execution_metadata
from evaluation.rng import preserve_rng
from evaluation.session import inference_session
from evaluation.wrapper import Recurrence, recurrence_label
from model.model import RecurrentGPT
from training.backend.base import Backend
from training.data.tokenizer import Tokenizer
from training.failure import FatalHandler, fatal_errors
from training.stopping import StopController


@dataclass
class BenchmarkPhaseResult:
    completed: bool
    metrics: dict[str, float]


def evaluate_distributed_benchmarks(
    backend: Backend, model: RecurrentGPT, tokenizer: Tokenizer, tasks: Sequence[str], *, stop: StopController,
    recurrences: Sequence[Recurrence], out_path: Path, step: int, seed: int = 0, batch_size: int = 8,
    limit: int | None = None, num_fewshot: int = -1, apply_chat_template: bool = False,
    on_fatal_error: FatalHandler | None = None,
) -> BenchmarkPhaseResult:
    with fatal_errors(on_fatal_error):
        check_benchmark_requests(model, tasks, recurrences, num_fewshot)
        if type(batch_size) is not int or batch_size < 1 or (limit is not None and limit < 1):
            raise ValueError("distributed benchmarks require a positive fixed batch size and positive limit or null")
        agree_on_phase(backend, model, tokenizer, {
            "phase": "benchmarks", "step": step, "tasks": tasks, "recurrences": recurrences, "seed": seed,
            "batch_size": batch_size, "limit": limit, "num_fewshot": num_fewshot, "chat": apply_chat_template,
        })
        if poll_stop(stop, "before benchmark jobs"):
            return BenchmarkPhaseResult(False, {})
        with preserve_rng(backend.device, on_fatal_error):
            lm_eval, hf_models = import_lm_eval()
            from evaluation.benchmark_model import BenchmarkController, BenchmarkExecutor
            from evaluation.benchmark_jobs import PROTOCOL, TARGET_REQUESTS

        metrics: dict[str, float] = {}
        raw: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        versions: dict[str, Any] = {}
        shots: dict[str, Any] = {}
        for index, recurrence in enumerate(recurrences):
            cancelled = False
            controller: BenchmarkController | None = None
            with (
                inference_session(model, recurrence, seed=seed, execution_policy=backend.execution_policy,
                                  on_fatal_error=on_fatal_error) as session,
                fatal_errors(on_fatal_error),
            ):
                wrapper, worker = build_benchmark_harness(
                    session, tokenizer, hf_models, batch_size, model.config.model_max_sequence_length, apply_chat_template, local_generation=True,
                )
                executor = BenchmarkExecutor(backend, worker, stop, seed=seed, recurrence=index, on_fatal_error=on_fatal_error)
                exchange(backend, "ready")
                if backend.is_main:
                    controller = BenchmarkController(executor)
                    try:
                        results = run_benchmark_harness(lm_eval, controller, tasks, limit, num_fewshot, apply_chat_template)
                    except EvaluationCancelled:
                        cancelled = True
                    else:
                        label = recurrence_label(recurrence)
                        metrics.update(flatten_results(results["results"], label))
                        raw[label] = results["results"]
                        versions.update(results.get("versions", {}))
                        shots.update(results.get("n-shot", {}))
                        metadata[label] = benchmark_execution_metadata(
                            session, worker, wrapper, results, lm_eval.simple_evaluate, batch_size=batch_size,
                            context_cap=model.config.model_max_sequence_length, custom_kernels=model.config.use_custom_kernels,
                        )
                        metadata[label].update(tokenizer_contract=tokenizer.contract, apply_chat_template=apply_chat_template,
                                               evaluator_world_size=1, inference_world_size=backend.world_size,
                                               protocol=PROTOCOL, target_requests=TARGET_REQUESTS, serial_bootstrap=True,
                                               plan_digest=executor.plan_digest, group_subtasks=results.get("group_subtasks", {}),
                                               sample_counts=results.get("n-samples", {}),
                                               task_provenance={name: {key: config.get(key) for key in
                                                   ("dataset_path", "dataset_name", "output_type", "num_fewshot", "description",
                                                    "doc_to_text", "doc_to_target", "target_delimiter", "fewshot_delimiter")}
                                                   for name, config in results.get("configs", {}).items()})
                    executor.finish(cancelled)
                else:
                    cancelled = executor.serve()
            local_stats = asdict(executor.stats)
            controller = None
            del executor, worker, wrapper
            stats = exchange(backend, local_stats)
            if backend.is_main and not cancelled:
                metadata[recurrence_label(recurrence)]["workers"] = stats
            requested = poll_stop(stop, "after benchmark recurrence")
            if cancelled or (requested and index + 1 < len(recurrences)):
                return BenchmarkPhaseResult(False, {})

        if backend.is_main:
            save_benchmark_results(
                out_path, step=step, tasks=tasks, num_fewshot=num_fewshot, limit=limit, seed=seed, recurrences=recurrences,
                metrics=metrics, raw_results=raw, versions=versions, n_shot=shots, execution_settings=metadata,
                custom_kernels=model.config.use_custom_kernels, execution_policy=backend.execution_policy,
            )
        finish_publication(backend)
        poll_stop(stop, "after benchmark publication")
        return BenchmarkPhaseResult(True, metrics)
