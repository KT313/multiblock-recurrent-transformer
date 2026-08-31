# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Multi-stage training loop for the multi-block recurrent transformer.

    python training/train.py --config config/crow_300m_final.yaml [--key value ...]

Steps are optimizer steps (one world batch each). Device/precision/distribution concerns live entirely in
`training.backend`; this file must not touch torch.cuda / torch.distributed directly.
"""

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python training/train.py` from the repo root

from data_preparation.lib.log import LOG_FORMAT, ProgressStreamHandler
from model import RecurrentConfig, RecurrentGPT
from model.hf import export_to_hf
from training.backend import Backend, get_backend
from training.checkpoint import (
    CheckpointMetadata,
    checkpoint_dir,
    checkpoint_path,
    find_latest_checkpoint,
    is_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
    unwrap_compiled,
)
from training.data import IGNORE_INDEX, build_stage_dataloaders
from training.data.dataset_resolver import ResolvedDataset, check_dataset_unchanged, resolve_dataset
from training.evaluation import evaluate, is_evaluation_step
from training.logger import RunLogger, TrainingReport
from training.optim import build_optimizer, get_param_groups
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager
from training.step import TrainingProgress, micro_batch_stream, run_one_optimizer_step

TRAINING_LOGGER_NAME = "training"  # the logger hierarchy of `training/`; `RunLogger` logs on `training.logger`


def build_stage_manager(settings: Settings, dataset: ResolvedDataset, world_size: int) -> StageManager:
    """The run's `StageManager`: the dataset's stage budgets turned into optimizer-step boundaries."""
    return StageManager(
        dataset.training_stages(),
        world_batch_size=settings.world_batch_size,
        block_size=settings.block_size,
        world_size=world_size,
        warmup_steps=settings.warmup_steps,
        cooldown_steps=settings.cooldown_steps,
        micro_batch_size=settings.micro_batch_size,
    )


def prepare_run_directory(settings: Settings) -> Path:
    """Create the run directory (`settings.out_dir`) with its `checkpoints/` folder and write `run_config.json`
    (the settings as parsed, jsonargparse overrides applied). Returns the run directory."""
    run_directory = Path(settings.out_dir)
    checkpoint_dir(run_directory).mkdir(parents=True, exist_ok=True)
    with open(run_directory / "run_config.json", "w") as f:
        json.dump(asdict(settings), f, indent=4)
    return run_directory


def check_block_sizes_agree(settings: Settings, model_config: RecurrentConfig) -> None:
    """The run config's `block_size` must equal the architecture's (the RoPE table is sized by it). The dataset-side
    check (`block_size` of the dataset config) is the resolver's."""
    if model_config.block_size != settings.block_size:
        raise ValueError(
            f"block_size {settings.block_size} of the run config does not match block_size {model_config.block_size} "
            f"of the model architecture config {settings.model_architecture_config} (with model_overwrite applied)"
        )


def build_run_model(settings: Settings, backend: Backend, run_directory: Path) -> torch.nn.Module:
    """The run's model: architecture yaml + `model_overwrite`, block-size check, `RecurrentGPT`, `model_config.json`,
    then `backend.setup_model` (device, optional compile).

    Numerics: the parameter init is the first consumer of the global torch RNG after `seed_everything`, so this must
    run after the dataset is resolved and the loaders are built (nothing that draws may move before it).
    """
    model_config = RecurrentConfig.from_yaml(settings.model_architecture_config, **settings.model_overwrite)
    check_block_sizes_agree(settings, model_config)
    model = RecurrentGPT(
        model_config, ignore_index=IGNORE_INDEX, gradient_checkpointing=settings.gradient_checkpointing
    )
    model_config.to_json(run_directory / "model_config.json")
    return backend.setup_model(model, compile=settings.compile_model)


def build_run_optimizer(settings: Settings, model: torch.nn.Module, backend: Backend) -> torch.optim.Optimizer:
    """The run's optimizer: the three parameter groups of `get_param_groups`, `settings.optimizer` with
    `settings.optim_config`, wrapped by `backend.setup_optimizer`."""
    weight_decay = settings.optim_config.get("weight_decay", 0.0)
    param_groups = get_param_groups(model, weight_decay, settings.no_weight_decay_for_bias_and_norm_params)
    return backend.setup_optimizer(build_optimizer(settings.optimizer, param_groups, **settings.optim_config))


def export_if_requested(settings: Settings, run_directory: Path, model: torch.nn.Module, dataset: ResolvedDataset) -> Path | None:
    """With `export_to_hf`: write the HuggingFace folder (`export_hf_path`, default `run_directory / hf_export`) from
    the unwrapped model and the dataset's tokenizer and return it; None otherwise."""
    if not settings.export_to_hf:
        return None
    export_dir = Path(settings.export_hf_path) if settings.export_hf_path else run_directory / "hf_export"
    plain_model = cast(RecurrentGPT, unwrap_compiled(model))
    export_to_hf(plain_model, plain_model.config, export_dir, tokenizer_dir=dataset.tokenizer_dir)
    return export_dir


def train(settings: Settings) -> TrainingReport:
    setup_started = time.time()
    backend = get_backend(settings.backend, precision=settings.precision)
    backend.seed_everything(settings.seed)
    run_directory = prepare_run_directory(settings)

    resolved = resolve_dataset(settings, backend)  # verifies the dataset config's data, auto-prepares if configured
    stage_manager = build_stage_manager(settings, resolved, backend.world_size)
    loaders = build_stage_dataloaders(settings, resolved, backend)

    model = build_run_model(settings, backend, run_directory)  # after the loaders: first torch RNG draw after seeding
    optimizer = build_run_optimizer(settings, model, backend)

    progress = TrainingProgress()
    resume_path = None
    if settings.resume:
        resume_path = Path(settings.resume_checkpoint_path) if settings.resume_checkpoint_path else find_latest_checkpoint(
            run_directory, settings.run_name
        )
    if resume_path is not None:
        metadata = load_training_checkpoint(backend, resume_path, model, optimizer)
        check_dataset_unchanged(metadata, resolved, settings.allow_dataset_change)
        progress.step = progress.resume_step = metadata.step
        backend.set_rng_state(metadata.rng)

    with RunLogger.open(
        settings, run_directory, resolved, model, stage_manager, progress, backend, setup_started=setup_started
    ) as logger:
        if resume_path is not None:
            logger.log_resume(resume_path, progress.step)
        else:
            logger.log_fresh_start()
        batches = micro_batch_stream(settings, loaders, stage_manager, progress)  # transition RNG seeded with the resume step

        while progress.step < stage_manager.total_steps:
            result = run_one_optimizer_step(settings, backend, model, optimizer, stage_manager, batches, progress)
            progress.advance()
            if is_evaluation_step(settings, progress, stage_manager):
                with logger.evaluating():
                    result.validation = evaluate(settings, backend, model, loaders.val_loaders[result.next_stage.stage_idx])
            logger.log_step(result, progress)

            stage_end = stage_manager.stage_ending_at(result.step)
            if is_checkpoint_step(settings, progress.done, stage_manager.total_steps, stage_end=stage_end is not None):
                path = checkpoint_path(run_directory, settings.run_name, progress.done, stage_end)
                metadata = CheckpointMetadata(
                    step=progress.done,
                    stage=result.next_stage.stage_idx,
                    rng=backend.rng_state(),  # after the evaluation draws of this step
                    settings=asdict(settings),
                    model_config=cast(RecurrentGPT, unwrap_compiled(model)).config.to_dict(),
                    dataset_config_hash=resolved.config_hash,
                    validation_rows=resolved.validation_rows,
                )
                save_training_checkpoint(backend, path, model, optimizer, metadata)
                logger.log_checkpoint(path)

        export_dir = export_if_requested(settings, run_directory, model, resolved)
        if export_dir is not None:
            logger.log_export(export_dir)
        return logger.close(progress, export_dir)


def configure_console_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach one stderr stream handler to the `training` logger so `RunLogger`'s lines reach the terminal; idempotent.

    The CLI's job (library code does not configure logging): the same handler type and line format as
    `data_preparation.lib.log.configure_logging`, which `resolve_dataset` still calls for the `data_preparation`
    hierarchy (task 9 moves that call here). Task 10's dashboard swaps this plain stream handler out for the run and
    restores it afterwards.
    """
    training_logger = logging.getLogger(TRAINING_LOGGER_NAME)
    training_logger.setLevel(level)
    handler = next((h for h in training_logger.handlers if isinstance(h, ProgressStreamHandler)), None)
    if handler is None:
        handler = ProgressStreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        training_logger.addHandler(handler)
    handler.setLevel(level)
    return training_logger


def main() -> None:
    configure_console_logging()
    report = train(parse_settings())
    print(report.summary())


if __name__ == "__main__":
    main()
