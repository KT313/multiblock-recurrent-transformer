# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Optimizer-step, checkpoint, inference, and completion phases of the training loop."""

from __future__ import annotations

from training.checkpoint import is_checkpoint_step
from training.data.loader import RunDataloaders
from training.data.tokenizer import Tokenizer
from training.evaluation import evaluate, is_evaluation_step
from training.execution.artifacts import export_if_requested, run_benchmarks, write_samples
from training.execution.checkpoints import checkpoint_before_failed_step, save_run_checkpoint
from training.execution.state import RunState
from training.failure import FatalHandler, handle_fatal_error
from training.logger import RunLogger, TrainingReport
from training.steps import NonFiniteLossError, RankBatches
from training.step import run_one_optimizer_step
from training.stopping import StopController
from training.triggers import StepTriggers


def run_and_log_training_step(
    state: RunState, loaders: RunDataloaders, logger: RunLogger, batches: RankBatches,
    stop: StopController, on_fatal_error: FatalHandler | None,
) -> None:
    """Update, advance progress, validate when due, and log before considering later work."""

    settings, backend, model, optimizer = state.settings, state.backend, state.model, state.optimizer
    stage_manager, progress = state.stage_manager, state.progress
    try:
        result = run_one_optimizer_step(
            settings, backend, model, optimizer, stage_manager, batches, progress,
            on_micro_batch=logger.note_micro_batch,
        )  # fmt: skip
    except NonFiniteLossError as error:
        handle_fatal_error(on_fatal_error, error)  # fatal workers use the last regular checkpoint
        raise RuntimeError(f"{error}. Terminating; {checkpoint_before_failed_step(state, logger, batches)}") from None
    progress.advance()
    stop.poll("after optimizer step")
    if not stop.requested and is_evaluation_step(settings, progress.step, stage_manager):
        validation_loader = loaders.val_loaders[stage_manager.entering_stage_at(progress.step)]
        with logger.evaluating():
            result.validation = evaluate(settings, backend, model, validation_loader)
        stop.poll("after validation")
    logger.log_step(result, progress, data_wait=loaders.take_wait_seconds())


def save_checkpoint_if_due(state: RunState, logger: RunLogger, batches: RankBatches, stop: StopController) -> bool:
    """Publish a scheduled checkpoint and poll for stops; return whether it is fresh."""

    if not is_checkpoint_step(state.settings, state.progress.step, state.stage_manager):
        return False
    save_run_checkpoint(state, logger, batches)
    stop.poll("after checkpoint publication")
    return True


def run_scheduled_inference(
    state: RunState, logger: RunLogger, tokenizer: Tokenizer, sample_triggers: StepTriggers,
    benchmark_triggers: StepTriggers, stop: StopController, checkpoint_fresh: bool,
    on_fatal_error: FatalHandler | None = None,
) -> bool:
    """Run samples before benchmarks, polling after each phase and tracking checkpoint freshness."""

    if sample_triggers.due(state.progress.step):
        checkpoint_fresh = False  # inference can change third-party RNG state
        write_samples(state, logger, tokenizer, stop=stop, on_fatal_error=on_fatal_error)
        if stop.poll("after samples"):
            return checkpoint_fresh
    if benchmark_triggers.due(state.progress.step):
        checkpoint_fresh = False  # inference can change third-party RNG state
        run_benchmarks(state, logger, tokenizer, stop=stop, on_fatal_error=on_fatal_error)
        if stop.poll("after benchmarks"):
            return checkpoint_fresh
    return checkpoint_fresh


def finish_training_loop(
    state: RunState, logger: RunLogger, batches: RankBatches, stop: StopController, checkpoint_fresh: bool,
) -> TrainingReport:
    """Save completed state on a stop, or export normally, then close the report."""

    backend, stage_manager, progress = state.backend, state.stage_manager, state.progress
    if not stop.requested:
        stop.poll("before final export")
    stopped = stop.requested and progress.step < stage_manager.total_steps
    if stop.requested:
        logger.status("stopping, saving the completed state" if stopped else
                      "training updates completed; skipping optional work on request")
        if not checkpoint_fresh:
            save_run_checkpoint(state, logger, batches)
    export_dir = None if stop.requested or not backend.is_main else export_if_requested(state, logger)
    return logger.close(progress, export_dir, stopped=stopped)
