# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
`train()`: one training run as a readable entry function, plus the setup helpers it is made of.

    create_backend                    device, precision, torch flags; then `seed_everything`
    prepare_run_directory             out_dir/<run_name>/checkpoints, run_config.json
    run_lock                          one training run per out_dir (`data_preparation/lib/build/lock.py`)
    resolve_dataset                   verify / auto-prepare the dataset config, validation split, tokenizer dir
    build_stage_manager               token budgets -> optimizer-step boundaries, transitions, per-stage base LR/weights
    build_run_dataloaders             one train loader per SOURCE (whole run), one validation loader per stage
    build_run_model                   architecture yaml + overrides, block-size check, model_config.json, to device
    build_run_optimizer               parameter groups, optimizer, backend wrap
    RunState                          the objects above in one place for the helpers below
    restore_checkpoint_if_resuming    latest / explicit checkpoint -> model, optimizer, RNG state, progress
    loop                              run_one_optimizer_step -> advance -> evaluate -> log -> checkpoint (or stop)
    export_if_requested               the HuggingFace folder, once training finished

Steps are OPTIMIZER steps (one world batch each). `train()` touches no tensor, device, clock or `print`: numerics
live in `step.py` and `evaluation.py`, device code in `backend/`, console lines and the dashboard in `logger.py`.
The setup order is itself numerics: seed, dataset and loaders (no torch RNG draw), model (its init is the first RNG
consumer), optimizer, resume (restores the stored RNG state). The golden run in `test_run.py` fails on any change.

A resume continues the run exactly (the checkpoint carries `BatchStream.state_dict()`, the RNG states and the
optimizer state; the loaders seed their iterators from a private generator): on a deterministic backend the resumed
steps reproduce the uninterrupted run's numbers, on a GPU the usual nondeterminism applies.

The CLI around this is `training/train.py`; `TrainingReport` is defined next to `RunLogger` in `logger.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch.nn import Module
from torch.optim import Optimizer

from data_preparation.lib.abort import StopCheck
from model import RecurrentConfig, RecurrentGPT
from model.hf import export_to_hf
from training.backend import get_backend
from training.backend.base import Backend, plain_model
from training.checkpoint import (
    CheckpointMetadata,
    checkpoint_dir,
    check_settings_unchanged,
    checkpoint_path,
    find_latest_checkpoint,
    is_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
)
from training.data.collate import IGNORE_INDEX
from training.data.loader import build_run_dataloaders
from training.data.dataset_resolver import ResolvedDataset, check_dataset_unchanged, resolve_dataset
from training.evaluation import evaluate, is_evaluation_step
from training.logger import RunLogger, TrainingReport
from training.optim import build_optimizer, get_param_groups
from data_preparation.lib.build.lock import TRAIN_LOCK_NAME, run_lock
from training.settings import Settings
from training.stage_manager import StageManager
from training.step import BatchStream, TrainingProgress, run_one_optimizer_step


@dataclass(frozen=True)
class RunState:
    """
    The run once it is set up, handed to `restore_checkpoint_if_resuming`, `save_run_checkpoint` and
    `export_if_requested` as one argument. Every member keeps its identity for the whole run; only `progress` moves.
    """

    settings: Settings
    run_directory: Path
    backend: Backend
    model: Module
    optimizer: Optimizer
    dataset: ResolvedDataset
    stage_manager: StageManager
    progress: TrainingProgress


@dataclass(frozen=True)
class ResumePoint:
    """
    Where a resumed run continues from: the checkpoint it was restored from and the data-stream state stored in
    it (None in a checkpoint written before the stream existed), for `BatchStream.load_state_dict`.
    """

    checkpoint: Path
    data_stream: dict[str, Any] | None


def train(
    settings: Settings,
    *,
    backend: Backend | None = None,
    should_stop: StopCheck | None = None,
    started_at: float | None = None,
    keep_history: bool = False,
) -> TrainingReport:
    """
    Run the training run described by `settings` and return its report.

    `backend`: created from the settings unless given (tests inject the CPU backend); seeded here either way.
    `should_stop`: the run's stop request (the CLI's Ctrl-C), polled between build shards and after every optimizer
    step; the loop then saves a checkpoint, skips the export and returns with `report.stopped`.
    `started_at`: the caller's clock reading at the start of the run (`report.setup_seconds`).
    `keep_history`: a test knob; `report.history` then holds every log step's metric dict.
    `out_dir` is locked for the whole run: a second run on the same `out_dir` fails with `RunLocked`.

    Numerics: the setup order (module docstring) and the loop body (step, evaluation, then the checkpoint, so the
    stored RNG state includes the evaluation draws) are the thesis loop's; `test_golden_tiny_run` fails on any change.
    """

    backend = backend or create_backend(settings)
    backend.seed_everything(settings.seed)
    run_directory = prepare_run_directory(settings)
    with run_lock(Path(settings.out_dir) / TRAIN_LOCK_NAME, "training"):  # released on every way out, exception included
        dataset = resolve_dataset(settings, backend, should_stop=should_stop)
        stage_manager = build_stage_manager(settings, dataset, backend.world_size)
        loaders = build_run_dataloaders(settings, dataset, backend)
        try:
            model = build_run_model(settings, backend, run_directory)
            optimizer = build_run_optimizer(settings, model, backend)
            state = RunState(settings, run_directory, backend, model, optimizer, dataset, stage_manager, TrainingProgress())
            resume = restore_checkpoint_if_resuming(state)
            progress = state.progress

            with RunLogger.open(
                settings,
                run_directory,
                dataset,
                model,
                stage_manager,
                progress,
                backend,
                setup_started=started_at,
                keep_history=keep_history,
            ) as logger:
                if resume is None or not (run_directory / "run_config.json").exists():
                    record_run_config(settings, run_directory)
                if resume is None:
                    logger.log_fresh_start()
                else:
                    logger.log_resume(resume.checkpoint, progress.step)
                batches = BatchStream(settings, loaders, stage_manager, progress)
                if resume is not None and resume.data_stream is not None:
                    batches.load_state_dict(resume.data_stream)
                logger.status("training")
                stopped = False
                while progress.step < stage_manager.total_steps and not stopped:
                    result = run_one_optimizer_step(
                        settings, backend, model, optimizer, stage_manager, batches, progress
                    )
                    progress.advance()
                    if is_evaluation_step(settings, progress.step, stage_manager):
                        validation_loader = loaders.val_loaders[stage_manager.entering_stage_at(progress.step)]
                        with logger.evaluating():
                            result.validation = evaluate(settings, backend, model, validation_loader)
                    logger.log_step(result, progress)
                    # a request arriving during the last step changes nothing: the run is finished, not stopped
                    stopped = stop_requested(should_stop) and progress.step < stage_manager.total_steps
                    if stopped:
                        logger.status("stopping after this step, saving a checkpoint")
                    if is_checkpoint_step(settings, progress.step, stage_manager) or stopped:
                        save_run_checkpoint(state, logger, batches)
                export_dir = None if stopped else export_if_requested(state, logger)
                return logger.close(progress, export_dir, stopped=stopped)
        finally:
            loaders.close()  # the loader workers stop now, not when the GC finds the iterators


# --- setup -----------------------------------------------------------------------------------------------------------


def create_backend(settings: Settings) -> Backend:
    """
    The run's backend (`settings.backend` at `settings.precision`); its constructor picks the device and sets the
    torch flags. `train()` seeds it right after.
    """

    return get_backend(settings.backend, precision=settings.precision)


def run_directory_of(settings: Settings) -> Path:
    """
    The run directory: `out_dir/<run_name>`, where the checkpoints, logs and wandb files of the run go.
    """

    return Path(settings.out_dir) / settings.run_name


def prepare_run_directory(settings: Settings) -> Path:
    """
    Create the run directory (`run_directory_of`) with its `checkpoints/` folder. Returns the run directory.
    """

    run_directory = run_directory_of(settings)
    checkpoint_dir(run_directory).mkdir(parents=True, exist_ok=True)
    return run_directory


def record_run_config(settings: Settings, run_directory: Path) -> None:
    """
    Write `run_config.json` (the settings as parsed). A fresh run always writes it; a resume only into a run
    directory without one (an explicit `resume_checkpoint_path` into a new directory) and keeps it otherwise.
    """

    with open(run_directory / "run_config.json", "w") as file:
        json.dump(asdict(settings), file, indent=4)


def build_stage_manager(settings: Settings, dataset: ResolvedDataset, world_size: int) -> StageManager:
    """
    The run's `StageManager`: the dataset's stage budgets turned into optimizer-step boundaries.
    """

    return StageManager(
        dataset.stages,
        world_batch_size=settings.world_batch_size,
        block_size=settings.block_size,
        world_size=world_size,
        warmup_steps=settings.warmup_steps,
        cooldown_steps=settings.cooldown_steps,
        micro_batch_size=settings.micro_batch_size,
    )


def check_block_sizes_agree(settings: Settings, model_config: RecurrentConfig) -> None:
    """
    The run config's `block_size` must equal the architecture's (the RoPE table is sized by it). The dataset-side
    check (`block_size` of the dataset config) is the resolver's.
    """

    if model_config.block_size != settings.block_size:
        raise ValueError(
            f"block_size {settings.block_size} of the run config does not match block_size {model_config.block_size} "
            f"of the model architecture config {settings.model_architecture_config} (with model_overwrite applied)"
        )


def build_run_model(settings: Settings, backend: Backend, run_directory: Path) -> Module:
    """
    The run's model: architecture yaml + `model_overwrite`, block-size check, `RecurrentGPT`, `model_config.json`,
    then `backend.setup_model` (device, optional compile).

    Numerics: the parameter init is the first consumer of the global torch RNG after `seed_everything`; nothing that
    draws may run before it.
    """

    model_config = RecurrentConfig.from_yaml(settings.model_architecture_config, **settings.model_overwrite)
    check_block_sizes_agree(settings, model_config)
    model = RecurrentGPT(
        model_config, ignore_index=IGNORE_INDEX, gradient_checkpointing=settings.gradient_checkpointing
    )
    model_config.to_json(run_directory / "model_config.json")
    return backend.setup_model(model, compile_model=settings.compile_model)


def build_run_optimizer(settings: Settings, model: Module, backend: Backend) -> Optimizer:
    """
    The run's optimizer: the three parameter groups of `get_param_groups`, `settings.optimizer` with
    `settings.optim_config`, wrapped by `backend.setup_optimizer`.
    """

    param_groups = get_param_groups(
        model, settings.optim_config.weight_decay, settings.no_weight_decay_for_bias_and_norm_params
    )
    return backend.setup_optimizer(build_optimizer(settings.optimizer, param_groups, settings.optim_config))


def restore_checkpoint_if_resuming(state: RunState) -> ResumePoint | None:
    """
    Restore the run from its checkpoint when it resumes and say where from; None for a fresh run.

    With `settings.resume`: `resume_checkpoint_path` if set, else the most recently written checkpoint of `run_name`
    in the run directory; none found means a fresh start. Restores model and optimizer state, verifies dataset and
    settings against the checkpoint, restores the RNG state and sets `progress.step = progress.resume_step =
    checkpoint step`. The returned data-stream state goes into `BatchStream.load_state_dict` once the stream exists.
    """

    settings = state.settings
    if not settings.resume:
        return None
    if settings.resume_checkpoint_path:
        resume_path: Path | None = Path(settings.resume_checkpoint_path)
    else:
        resume_path = find_latest_checkpoint(state.run_directory, settings.run_name)
    if resume_path is None:
        return None
    metadata = load_training_checkpoint(state.backend, resume_path, state.model, state.optimizer)
    check_dataset_unchanged(metadata, state.dataset, settings.allow_dataset_change)
    model_config = plain_model(state.model).config.to_dict()
    check_settings_unchanged(metadata, settings, model_config, settings.allow_settings_change)
    state.progress.step = state.progress.resume_step = metadata.step
    state.backend.set_rng_state(metadata.rng)
    return ResumePoint(resume_path, metadata.data_stream)


# --- inside the loop -------------------------------------------------------------------------------------------------


def stop_requested(should_stop: StopCheck | None) -> bool:
    """
    Whether the caller asked the run to stop (None: never).
    """

    return should_stop is not None and should_stop()


def save_run_checkpoint(state: RunState, logger: RunLogger, batches: BatchStream) -> None:
    """
    Write the checkpoint of `state.progress.step` completed optimizer steps and tell the logger.

    Named `step-{step:08d}-{run_name}.pth`, plus `-stage-{i}_end` after the last plain step of stage i; `stage` is
    the stage the run is heading for (`StageManager.entering_stage_at`). Numerics: called after evaluation and
    logging, so the stored RNG state includes the evaluation draws.
    """

    settings, progress, stage_manager = state.settings, state.progress, state.stage_manager
    stage_end = stage_manager.stage_ending_at(progress.step - 1)
    path = checkpoint_path(state.run_directory, settings.run_name, progress.step, stage_end)
    metadata = CheckpointMetadata(
        step=progress.step,
        stage=stage_manager.entering_stage_at(progress.step),
        rng=state.backend.rng_state(),
        settings=asdict(settings),
        model_config=plain_model(state.model).config.to_dict(),
        dataset_config_hash=state.dataset.config_hash,
        validation_rows=state.dataset.validation_rows,
        data_stream=batches.state_dict(),
    )
    with logger.saving_checkpoint():
        save_training_checkpoint(state.backend, path, state.model, state.optimizer, metadata)
    logger.log_checkpoint(path)


# --- after the loop --------------------------------------------------------------------------------------------------


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
    trained_model = plain_model(state.model)
    export_to_hf(trained_model, trained_model.config, export_dir, tokenizer_dir=state.dataset.tokenizer_dir)
    logger.log_export(export_dir)
    return export_dir
