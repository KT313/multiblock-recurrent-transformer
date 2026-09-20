# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Coordinated setup phases and resource lifetimes for the training pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

from data_preparation.lib.abort import StopCheck
from data_preparation.lib.build.lock import DatasetLease
from training.backend.base import Backend
from training.data.dataset_resolver import ResolvedDataset
from training.data.loader import RunDataloaders
from training.data.ownership import main_rank_phase, training_dataset_access
from training.execution.checkpoints import resolve_resume_checkpoint
from training.execution.setup import build_stage_manager, get_run_directory
from training.execution.state import ResumePoint, RunState
from training.failure import FatalHandler, fatal_errors, handle_fatal_error
from training.logger import RunLogger
from training.provenance import check_fresh_run_directory
from training.settings import Settings
from training.stage_manager import StageManager
from training.steps import BatchStream, RankBatches
from training.triggers import StepTriggers


@contextmanager
def close_backend_on_exit(backend: Backend, on_fatal_error: FatalHandler | None) -> Iterator[None]:
    """Invoke the fatal policy before backend cleanup, including failures during setup."""

    try:
        yield
    except Exception as error:
        handle_fatal_error(on_fatal_error, error)
        raise
    finally:
        backend.shutdown()  # the process group, on every way out


@contextmanager
def open_training_dataset(settings: Settings, backend: Backend, on_fatal_error: FatalHandler | None) -> Iterator[DatasetLease | None]:
    """Keep dataset access and the run lock until readers close; fatal workers exit before release."""

    with (
        training_dataset_access(
            Path(settings.dataset_dir), get_run_directory(settings), backend, shared=not settings.auto_prepare,
        ) as lease,
        fatal_errors(on_fatal_error),
    ):
        yield lease


def select_resume_for_run(settings: Settings, run_directory: Path, backend: Backend) -> Path | None:
    """Select the checkpoint and coordinate refusal of fresh reuse before dataset resolution."""

    with main_rank_phase(backend, "run directory selection"):
        resume_path = resolve_resume_checkpoint(settings, run_directory)
        if backend.is_main and resume_path is None:
            check_fresh_run_directory(run_directory)
    return resume_path


def validate_resolved_schedule(settings: Settings, dataset: ResolvedDataset, backend: Backend, configured: StageManager) -> StageManager:
    """Confirm every rank resolved the same schedule that passed preflight."""

    with main_rank_phase(backend, "learning-rate schedule validation"):
        stage_manager = build_stage_manager(settings, dataset, backend.world_size)
        if configured.stages != [replace(stage, val_data=[]) for stage in dataset.stages]:
            raise ValueError(
                "The dataset stage plan changed after learning-rate schedule validation; "
                "restart training with a stable dataset configuration."
            )
    return stage_manager


def build_run_triggers(settings: Settings, total_steps: int) -> tuple[StepTriggers, StepTriggers]:
    """Resolve the sample and benchmark triggers before constructing loaders."""

    samples = StepTriggers.from_settings(settings.sample_step_interval, settings.sample_at_training_progress, total_steps)
    benchmarks = StepTriggers.from_settings(settings.benchmark_step_interval, settings.benchmark_at_training_progress, total_steps)
    return samples, benchmarks


@contextmanager
def close_loaders_on_exit(loaders: RunDataloaders, on_fatal_error: FatalHandler | None) -> Iterator[None]:
    """Invoke the fatal policy before stopping loader workers on any exit."""

    try:
        yield
    except Exception as error:
        handle_fatal_error(on_fatal_error, error)
        raise
    finally:
        loaders.close()  # stop workers before releasing dataset access


@contextmanager
def open_run_logger(state: RunState, started_at: float | None, keep_history: bool, on_fatal_error: FatalHandler | None) -> Iterator[RunLogger]:
    """Coordinate logger initialization and retain its original fatal-before-cleanup boundary."""

    with ExitStack() as stack, fatal_errors(on_fatal_error):
        with main_rank_phase(state.backend, "run logger initialization"):
            logger = stack.enter_context(RunLogger.open(
                state.settings, state.run_directory, state.dataset, state.model, state.stage_manager,
                state.progress, state.backend, setup_started=started_at, keep_history=keep_history,
            ))
        yield logger


def prepare_run_stream(
    state: RunState, loaders: RunDataloaders, logger: RunLogger, resume: ResumePoint | None,
    sample_triggers: StepTriggers, benchmark_triggers: StepTriggers,
) -> RankBatches:
    """Log startup and triggers, then construct and restore the rank views of the data stream."""

    settings, backend, stage_manager, progress = state.settings, state.backend, state.stage_manager, state.progress
    with main_rank_phase(backend, "run stream and logging setup"):
        if resume is None:
            logger.log_fresh_start()
        else:
            logger.log_resume(resume.checkpoint, progress.step)
        logger.log_triggers("samples", sample_triggers.listed(stage_manager.total_steps))
        logger.log_triggers("benchmarks", benchmark_triggers.listed(stage_manager.total_steps))
        stream = BatchStream(settings, loaders, stage_manager, progress) if backend.is_main else None  # main rank owns the stream
        batches = RankBatches(backend, stream, settings.tokens_per_micro_batch)
        if resume is not None and resume.data_stream is not None:
            batches.load_state_dict(resume.data_stream)
        logger.status("training")
    return batches


def is_stop_requested(should_stop: StopCheck | None) -> bool:
    """
    Whether the caller asked the run to stop (None: never).
    """

    return should_stop is not None and should_stop()
