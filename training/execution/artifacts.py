# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Optional sample, benchmark, and export artifacts for a training run."""

from __future__ import annotations

from pathlib import Path

from data_preparation.lib.log import get_logger
from evaluation.benchmarks import benchmarks_path
from evaluation.prompts import load_prompts
from evaluation.samples import GeneratedSample, samples_path
from training.data.tokenizer import Tokenizer
from training.execution.state import RunState
from training.logger import RunLogger
from training.failure import FatalHandler, fatal_errors
from training.stopping import StopController

log = get_logger("training.run")


def write_samples(
    state: RunState, logger: RunLogger, tokenizer: Tokenizer, *, stop: StopController | None = None,
    on_fatal_error: FatalHandler | None = None,
) -> list[GeneratedSample]:
    """Generate fixed batches on resident replicas; rank zero publishes complete samples."""
    from evaluation.distributed_samples import generate_distributed_samples

    settings, step = state.settings, state.progress.step
    path = samples_path(state.run_directory, step)
    with logger.working("generating samples"), fatal_errors(on_fatal_error):
        result = generate_distributed_samples(
            state.backend, state.backend.plain_model(state.model), tokenizer, path, step=step,
            prompts=load_prompts() if state.backend.is_main else [], recurrences=settings.sample_recurrences or [None],
            stop=stop or StopController(state.backend), batch_size=settings.sample_batch_size,
            max_new_tokens=settings.sample_max_new_tokens, temperature=settings.sample_temperature,
            use_cache=settings.sample_use_cache, on_fatal_error=on_fatal_error,
        )
        if result.completed and state.backend.is_main:
            logger.log_samples(path, result.samples)
            log.info("sample jobs per rank: %s", result.jobs_per_rank)
    return result.samples


def run_benchmarks(
    state: RunState, logger: RunLogger, tokenizer: Tokenizer, *, stop: StopController | None = None,
    on_fatal_error: FatalHandler | None = None,
) -> dict[str, float] | None:
    """Run the official evaluator once and distribute inference jobs on resident replicas."""
    from evaluation.distributed_benchmarks import evaluate_configured_benchmarks

    settings, step = state.settings, state.progress.step
    path = benchmarks_path(state.run_directory, step)
    with logger.working("benchmarking"), fatal_errors(on_fatal_error):
        result = evaluate_configured_benchmarks(
            state.backend, state.backend.plain_model(state.model), tokenizer, settings,
            stop=stop or StopController(state.backend), out_path=path, step=step, on_fatal_error=on_fatal_error,
        )
        if result.completed and state.backend.is_main:
            logger.log_benchmarks(result.metrics, path, step)
    return result.metrics if result.completed else None



def export_if_requested(state: RunState, logger: RunLogger) -> Path | None:
    """
    With `export_to_hf`: write the HuggingFace folder (`export_hf_path`, default `run_directory / hf_export`), tell
    the logger and return the folder; None otherwise.
    """

    settings = state.settings
    if not settings.export_to_hf:
        return None
    export_dir = Path(settings.export_hf_path) if settings.export_hf_path else state.run_directory / "hf_export"
    logger.status("exporting")
    trained_model = state.backend.plain_model(state.model)
    from model.hf import export_to_hf  # transformers behind it: imported when a run exports, not at start-up

    export_to_hf(
        trained_model, trained_model.config, export_dir, tokenizer_dir=state.dataset.tokenizer_dir,
        execution_policy=state.backend.execution_policy,
    )
    logger.log_export(export_dir)
    return export_dir
