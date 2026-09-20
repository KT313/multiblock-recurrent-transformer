# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Checkpoint selection, restoration, and publication for a training run."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from data_preparation.lib.log import get_logger
from training.checkpoint import (
    CheckpointMetadata, check_settings_unchanged, checkpoint_path, find_latest_checkpoint,
    load_training_checkpoint, save_training_checkpoint,
)
from training.data.dataset_resolver import check_dataset_unchanged
from training.execution.state import ResumePoint, RunState
from training.logger import RunLogger
from training.optim.sharding import prepare_optimizer_state, release_optimizer_state
from training.settings import Settings
from training.steps import RankBatches
from training.stopping import complete_main_phase
from tokenization.validation import check_model_vocabulary

log = get_logger("training.run")


def resolve_resume_checkpoint(settings: Settings, run_directory: Path) -> Path | None:
    """Select a checkpoint once; None requests fresh setup, whose destination the caller must validate."""
    if not settings.resume:
        return None
    path = Path(settings.resume_checkpoint_path) if settings.resume_checkpoint_path else find_latest_checkpoint(
        run_directory, settings.run_name
    )
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist or is not a file: {path}")
    return path



def restore_checkpoint_if_resuming(state: RunState) -> ResumePoint | None:
    """Select and restore a checkpoint for callers that already built their model."""
    path = resolve_resume_checkpoint(state.settings, state.run_directory)
    return restore_checkpoint(state, path) if path is not None else None



def restore_checkpoint(state: RunState, resume_path: Path) -> ResumePoint:
    """
    Restore the checkpoint selected before model construction and say where from.

    Restores model and optimizer state, verifies dataset and settings against the checkpoint, refuses a checkpoint
    written with another number of ranks (its RNG states are
    per rank), restores this rank's RNG state and sets `progress.step = progress.resume_step = checkpoint step`. The
    returned data-stream state goes into `BatchStream.load_state_dict` once the stream exists.
    """

    settings = state.settings
    log.info("loading the checkpoint %s (%.1f GB) into the model and the optimizer", resume_path, resume_path.stat().st_size / 1e9)
    started = time.monotonic()
    metadata = load_training_checkpoint(state.backend, resume_path, state.model, state.optimizer,
                                        tokenizer_contract=state.tokenizer_contract)
    log.info("checkpoint of step %d loaded in %.1fs", metadata.step, time.monotonic() - started)
    plain_model = state.backend.plain_model(state.model)
    check_model_vocabulary(plain_model.config, state.tokenizer_contract, plain_model)
    check_dataset_unchanged(metadata, state.dataset, settings.allow_dataset_change)
    model_config = state.backend.plain_model(state.model).config.to_dict()
    check_settings_unchanged(metadata, settings, model_config, settings.allow_settings_change)
    if metadata.world_size != state.backend.world_size:
        raise ValueError(
            f"{resume_path} was written by a run with {metadata.world_size} rank(s), this run has "
            f"{state.backend.world_size}: resume with the same number of ranks (its RNG states are per rank; "
            "continuing on another number of GPUs is not supported)"
        )
    state.progress.step = state.progress.resume_step = metadata.step
    state.backend.set_rng_state(metadata.rng_states[state.backend.rank])
    return ResumePoint(resume_path, metadata.data_stream, metadata)



def checkpoint_before_failed_step(state: RunState, logger: RunLogger, batches: RankBatches) -> str:
    """
    A step that produced a non-finite loss or gradient norm did not update the model (`optimizer.step` never ran),
    so the model and optimizer state are those of the completed steps: save them, unless no step completed yet.
    Returns the note for the error message.

    Written as `...-failed.pth`, beside (never over) the regular checkpoint of the same step, and skipped by
    `find_latest_checkpoint` so a plain `resume: true` continues from the last regular checkpoint. The data stream
    is the one AFTER the failed step: the step read all its micro-batches before the loss was checked, so a resume
    from this file re-runs the step on the NEXT documents and the failed step's documents are skipped. To train
    them, resume from a regular checkpoint instead.
    """

    if state.progress.step == 0:
        return "no checkpoint written (the first step failed)"
    state.optimizer.zero_grad(set_to_none=True)
    path = save_run_checkpoint(state, logger, batches, failed=True)
    return (
        f"the model before this step is checkpointed as {path} (resuming from it continues AFTER this step's "
        "documents, which are skipped; the regular checkpoints are untouched)"
    )



def save_run_checkpoint(state: RunState, logger: RunLogger, batches: RankBatches, failed: bool = False) -> Path:
    """
    Write the checkpoint of `state.progress.step` completed optimizer steps and tell the logger. Every rank calls
    this (the RNG states are gathered here); the backend writes the file on the main rank, whose `batches` hold the
    data stream.

    Named `step-{step:08d}-{run_name}.pth`, plus `-stage-{i}_end` after the last plain step of stage i; `stage` is
    the stage the run is heading for (`StageManager.entering_stage_at`). Called after evaluation and logging; the
    stored RNG states (one per rank, gathered) are those after the step, evaluation having drawn under
    `torch.random.fork_rng`. `failed`
    appends `-failed` to the name, for the checkpoint of a step that ended the run on a non-finite loss
    (`checkpoint_before_failed_step`).
    """

    settings, progress, stage_manager = state.settings, state.progress, state.stage_manager
    stage_end = stage_manager.stage_ending_at(progress.step - 1)
    path = checkpoint_path(state.run_directory, settings.run_name, progress.step, stage_end, failed=failed)
    metadata = CheckpointMetadata(
        step=progress.step,
        stage=stage_manager.entering_stage_at(progress.step),
        world_size=state.backend.world_size,
        rng_states=state.backend.all_gather_object(state.backend.rng_state()),
        settings=asdict(settings),
        model_config=state.backend.plain_model(state.model).config.to_dict(),
        dataset_config_hash=state.dataset.config_hash,
        validation_rows=state.dataset.validation_rows,
        source_rows=state.dataset.source_rows,
        dataset_build_id=state.dataset.dataset_build_id,
        tokenizer_contract=state.tokenizer_contract,
        data_stream=batches.state_dict(),
    )
    def publish() -> None:
        with logger.saving_checkpoint():
            save_training_checkpoint(state.backend, path, state.model, state.optimizer, metadata)
        logger.log_checkpoint(path)

    prepare_optimizer_state(state.optimizer)
    try:
        complete_main_phase(state.backend, "checkpoint publication", publish)
    finally:
        release_optimizer_state(state.optimizer)
    if not state.backend.is_main:
        logger.log_checkpoint(path)
    return path
