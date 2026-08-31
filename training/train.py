# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Multi-stage training loop for the multi-block recurrent transformer.

    python training/train.py --config config/crow_300m_final.yaml [--key value ...]

Steps are optimizer steps (one world batch each). Device/precision/distribution concerns live entirely in
`training.backend`; this file must not touch torch.cuda / torch.distributed directly.
"""

import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python training/train.py` from the repo root

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
from training.logger import Logger, describe_parameters, num_parameters
from training.optim import build_optimizer, get_param_groups
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager
from training.step import TrainingProgress, micro_batch_stream, run_one_optimizer_step


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


def train(settings: Settings) -> None:
    start_time = time.time()
    backend = get_backend(settings.backend, precision=settings.precision)
    backend.seed_everything(settings.seed)
    run_directory = prepare_run_directory(settings)

    resolved = resolve_dataset(settings, backend)  # verifies the dataset config's data, auto-prepares if configured
    stage_manager = build_stage_manager(settings, resolved, backend.world_size)
    max_steps = stage_manager.total_steps
    print(stage_manager.get_stage_summary())
    print(f"Total training steps: {max_steps:,} ({settings.gradient_accumulation_steps} micro-batches each)")
    loaders = build_stage_dataloaders(settings, resolved, backend)

    model = build_run_model(settings, backend, run_directory)  # after the loaders: first torch RNG draw after seeding
    n_params = num_parameters(unwrap_compiled(model))
    print(describe_parameters(model))
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
        print(f"Resumed from {resume_path} at step {progress.step}")
    else:
        print("No checkpoint loaded, starting from scratch.")

    logger = Logger(
        settings.logger_project, settings.run_name, run_directory, offline=settings.wandb_offline, enabled=settings.wandb_enabled
    )
    logger.log_hyperparams(asdict(settings) | {"dataset_config_hash": resolved.config_hash})
    logger.log_summary({"num_parameters": n_params})

    batches = micro_batch_stream(settings, loaders, stage_manager, progress)  # transition RNG seeded with the resume step
    tokens_per_step = settings.world_batch_size * settings.block_size
    sample_counter: Counter[str] = Counter()
    print(f"{time.ctime()[:-5]}: setup took {time.time() - start_time:.1f}s, starting training at step {progress.step}.")

    train_t0 = time.time()
    interval_t0 = time.time()
    while progress.step < max_steps:
        result = run_one_optimizer_step(settings, backend, model, optimizer, stage_manager, batches, progress)
        progress.advance()
        sample_counter.update(result.data_ids)

        if result.next_stage.in_transition and not result.stage.in_transition:
            print(f"Step {progress.done}: starting transition {result.next_stage.prev_stage_idx} -> "
                  f"{result.next_stage.stage_idx} ({result.next_stage.stage_name}), "
                  f"LR {result.next_stage.prev_base_lr:.2e} -> {result.next_stage.base_lr:.2e}")
        elif result.stage.in_transition and not result.next_stage.in_transition:
            print(f"Step {progress.done}: transition complete, now in stage {result.next_stage.stage_idx} "
                  f"({result.next_stage.stage_name})")

        if is_evaluation_step(settings, progress, stage_manager):
            t0 = time.time()
            result.validation = evaluate(settings, backend, model, loaders.val_loaders[result.next_stage.stage_idx])
            result.validation["val_time"] = torch.as_tensor(time.time() - t0)
            print(f"Step {progress.done}: val loss {result.validation['val_loss'].item():.4f} "
                  f"(stage {result.next_stage.stage_idx}, {result.validation['val_time']:.1f}s)")

        if progress.done % settings.log_step_interval == 0:  # task 7 moves this block into RunLogger.log_step
            now = time.time()
            seconds_per_step = (now - interval_t0) / settings.log_step_interval
            interval_t0 = now
            total = sum(sample_counter.values())
            metrics: dict[str, Any] = dict(result.metrics)
            metrics |= result.validation or {}
            metrics |= {
                "loss": result.loss,
                "ppl": result.log_ppl.exp(),
                "lr": result.learning_rate,
                "grad_norm": result.grad_norm,
                "step": progress.done,
                "seconds/step": seconds_per_step,
                "tokens/second": tokens_per_step / seconds_per_step,
                "total_tokens": progress.done * tokens_per_step,
                "total_time": now - train_t0,
                "remaining_time": seconds_per_step * (max_steps - progress.done),
                "stage/current_stage": result.stage.stage_idx,
                "stage/base_lr": result.stage.base_lr,
                "stage/in_transition": int(result.stage.in_transition),
                "stage/transition_progress": result.stage.transition_progress,
                "stage/stage_progress": result.stage.stage_progress,
            }
            metrics |= {f"data_composition/{k}": v / total for k, v in sample_counter.items()}
            sample_counter.clear()
            logger.log(metrics, step=progress.done)
            print(f"step {progress.done}/{max_steps} | loss {result.loss.item():.4f} | lr {result.learning_rate:.2e} | "
                  f"grad_norm {result.grad_norm.item():.3f} | {seconds_per_step:.2f}s/step")

        stage_end = stage_manager.stage_ending_at(result.step)
        if is_checkpoint_step(settings, progress.done, max_steps, stage_end=stage_end is not None):
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
            print(f"Saved checkpoint {path}")

    logger.log_summary({"train_time": time.time() - train_t0})
    logger.finish()
    print(f"Training finished after {progress.done} steps in {time.time() - train_t0:.1f}s.")
    if settings.export_to_hf:
        export_dir = Path(settings.export_hf_path) if settings.export_hf_path else run_directory / "hf_export"
        raw = cast(RecurrentGPT, unwrap_compiled(model))
        export_to_hf(raw, raw.config, export_dir, tokenizer_dir=resolved.tokenizer_dir)
        print(f"Exported HuggingFace model to {export_dir}")


def main() -> None:
    train(parse_settings())


if __name__ == "__main__":
    main()
