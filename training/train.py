# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Multi-stage training loop for the multi-block recurrent transformer.

    python training/train.py --config config/crow_300m_final.yaml [--key value ...]

Steps are optimizer steps (one world batch each). Device/precision/distribution concerns live entirely in
`training.backend`; this file must not touch torch.cuda / torch.distributed directly.
"""

import json
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict, cast

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
from training.data import IGNORE_INDEX, StageDataloaders, build_stage_dataloaders, length_sorted_batches
from training.data.dataset_resolver import ResolvedDataset, check_dataset_unchanged, resolve_dataset
from training.data.loader import Batch, sample_stage_batch
from training.logger import Logger, describe_parameters, num_parameters, track_gradient_metrics
from training.lr_schedule import get_lr_multistage
from training.optim import build_optimizer, get_param_groups, set_lr
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager


class LoopState(TypedDict):
    """Mutable step counter shared between the loop and the micro-batch stream."""

    step: int  # next optimizer step to run
    resume_step: int  # step the run was resumed at, -1 for a fresh run


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


def micro_batch_stream(
    cfg: Settings,
    loaders: StageDataloaders,
    stage_manager: StageManager,
    state: LoopState,
    rng: random.Random,
) -> Iterator[Batch]:
    """Endless stream of micro-batches; every `gradient_accumulation_steps` of them belong to `state["step"]`.

    Inside a stage transition each micro-batch is drawn from the next stage's loader with probability
    `transition_progress`, otherwise from the current stage's loader.
    """

    def raw() -> Iterator[Batch]:
        while True:
            info = stage_manager.get_stage_info(state["step"])
            for _ in range(cfg.gradient_accumulation_steps):
                yield sample_stage_batch(loaders, info.stage_idx, info.prev_stage_idx, info.transition_progress, rng)

    if cfg.sort_batches_by_length:
        return length_sorted_batches(
            raw(),
            cfg.micro_batch_size,
            cfg.gradient_accumulation_steps,
            ignore_index=IGNORE_INDEX,
            padding_multiple=cfg.sequence_padding_multiple,
        )
    return raw()


@torch.no_grad()
def validate(
    cfg: Settings, backend: Backend, model: torch.nn.Module, val_loader: Iterable[Batch]
) -> dict[str, torch.Tensor]:
    """Validation loss at every depth in `partial_depth_eval` and at the model's mean recurrence."""
    model.eval()
    config = cast(RecurrentGPT, unwrap_compiled(model)).config
    mean_recurrence = cast(list[int], config.mean_recurrence)  # broadcast to a list in RecurrentConfig.__post_init__
    depths: list[int | list[int]] = [*cfg.partial_depth_eval, mean_recurrence]
    losses = torch.zeros(cfg.eval_iters, len(depths), device=backend.device)
    for depth_idx, depth in enumerate(depths):
        steps = [(d, 0) for d in depth] if isinstance(depth, list) else [(depth, 0)] * len(mean_recurrence)
        for k, (input_ids, labels, _) in enumerate(val_loader):
            if k >= cfg.eval_iters:
                break
            input_ids, labels = input_ids.to(backend.device), labels.to(backend.device)
            with backend.autocast():
                losses[k, depth_idx] = model(input_ids, labels=labels, num_steps_pair=steps)["loss"]
    losses = backend.all_reduce(losses.mean(dim=0))
    metrics = {"val_loss": losses[-1], "val_ppl": losses[-1].exp()}
    for depth_idx, depth in enumerate(depths):
        metrics[f"val_loss_{depth}"] = losses[depth_idx]
        metrics[f"val_ppl_{depth}"] = losses[depth_idx].exp()
    model.train()
    return metrics


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

    state: LoopState = {"step": 0, "resume_step": -1}
    resume_path = None
    if settings.resume:
        resume_path = Path(settings.resume_checkpoint_path) if settings.resume_checkpoint_path else find_latest_checkpoint(
            run_directory, settings.run_name
        )
    if resume_path is not None:
        metadata = load_training_checkpoint(backend, resume_path, model, optimizer)
        check_dataset_unchanged(metadata, resolved, settings.allow_dataset_change)
        state["step"] = state["resume_step"] = metadata.step
        backend.set_rng_state(metadata.rng)
        print(f"Resumed from {resume_path} at step {state['step']}")
    else:
        print("No checkpoint loaded, starting from scratch.")

    logger = Logger(settings.logger_project, settings.run_name, run_directory, offline=settings.wandb_offline, enabled=settings.wandb_enabled)
    logger.log_hyperparams(asdict(settings) | {"dataset_config_hash": resolved.config_hash})
    logger.log_summary({"num_parameters": n_params})

    rng = random.Random(settings.seed + state["step"])
    batches = micro_batch_stream(settings, loaders, stage_manager, state, rng)
    tokens_per_step = settings.world_batch_size * settings.block_size
    sample_counter: Counter[str] = Counter()
    print(f"{time.ctime()[:-5]}: setup took {time.time() - start_time:.1f}s, starting training at step {state['step']}.")

    train_t0 = time.time()
    interval_t0 = time.time()
    while state["step"] < max_steps:
        step = state["step"]
        cast(RecurrentGPT, unwrap_compiled(model)).step = step
        info = stage_manager.get_stage_info(step)
        lr = get_lr_multistage(
            step,
            max_steps,
            stage_manager,
            min_lr=settings.min_lr,
            warmup_steps=settings.warmup_steps,
            cooldown_steps=settings.cooldown_steps,
            schedule=settings.lr_schedule,
            resume_step=state["resume_step"],
            resume_warmup_steps=settings.resume_warmup_steps,
        )
        set_lr(optimizer, lr)

        loss_sum = torch.zeros((), device=backend.device)
        log_ppl_sum = torch.zeros((), device=backend.device)
        for micro in range(settings.gradient_accumulation_steps):
            input_ids, labels, data_ids = next(batches)
            sample_counter.update(data_ids)
            input_ids = backend.to_device(input_ids)
            labels = backend.to_device(labels)
            with backend.no_sync(model) if micro < settings.gradient_accumulation_steps - 1 else nullcontext():
                with backend.autocast():
                    outputs = model(input_ids, labels=labels)
                backend.backward(outputs["loss"] / settings.gradient_accumulation_steps)
            loss_sum += outputs["loss"].detach()
            log_ppl_sum += outputs["log_ppl"].detach()
        loss = loss_sum / settings.gradient_accumulation_steps
        log_ppl = log_ppl_sum / settings.gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is {loss.item()} at step {step}. Terminating.")

        grad_norm = backend.clip_grad_norm(model, settings.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Gradient norm is non-finite at step {step}. Terminating.")
        if step > 0:  # as in the thesis runs: the very first update is skipped (LR is 0 there anyway with warmup)
            optimizer.step()
        metrics: dict[str, Any] = {}
        if settings.log_gradient_metrics and (step + 1) % settings.log_step_interval == 0:
            metrics |= track_gradient_metrics(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        state["step"] = done = step + 1

        next_info = stage_manager.get_stage_info(done)
        if next_info.in_transition and not info.in_transition:
            print(f"Step {done}: starting transition {next_info.prev_stage_idx} -> {next_info.stage_idx} "
                  f"({next_info.stage_name}), LR {next_info.prev_base_lr:.2e} -> {next_info.base_lr:.2e}")
        elif info.in_transition and not next_info.in_transition:
            print(f"Step {done}: transition complete, now in stage {next_info.stage_idx} ({next_info.stage_name})")

        if done % settings.eval_step_interval == 0 or done >= max_steps:
            t0 = time.time()
            val_metrics = validate(settings, backend, model, loaders.val_loaders[next_info.stage_idx])
            val_metrics["val_time"] = torch.as_tensor(time.time() - t0)
            print(f"Step {done}: val loss {val_metrics['val_loss'].item():.4f} "
                  f"(stage {next_info.stage_idx}, {val_metrics['val_time']:.1f}s)")
            metrics |= val_metrics

        if done % settings.log_step_interval == 0:
            now = time.time()
            steps_in_interval = settings.log_step_interval
            seconds_per_step = (now - interval_t0) / steps_in_interval
            interval_t0 = now
            total = sum(sample_counter.values())
            metrics |= {
                "loss": backend.all_reduce(loss),
                "ppl": log_ppl.exp(),
                "lr": lr,
                "grad_norm": grad_norm,
                "step": done,
                "seconds/step": seconds_per_step,
                "tokens/second": tokens_per_step / seconds_per_step,
                "total_tokens": done * tokens_per_step,
                "total_time": now - train_t0,
                "remaining_time": seconds_per_step * (max_steps - done),
                "stage/current_stage": info.stage_idx,
                "stage/base_lr": info.base_lr,
                "stage/in_transition": int(info.in_transition),
                "stage/transition_progress": info.transition_progress,
                "stage/stage_progress": info.stage_progress,
            }
            metrics |= {f"data_composition/{k}": v / total for k, v in sample_counter.items()}
            sample_counter.clear()
            logger.log(metrics, step=done)
            print(f"step {done}/{max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | grad_norm {grad_norm.item():.3f} | "
                  f"{seconds_per_step:.2f}s/step")

        stage_end = stage_manager.stage_ending_at(step)
        if is_checkpoint_step(settings, done, max_steps, stage_end=stage_end is not None):
            path = checkpoint_path(run_directory, settings.run_name, done, stage_end)
            metadata = CheckpointMetadata(
                step=done,
                stage=next_info.stage_idx,
                rng=backend.rng_state(),
                settings=asdict(settings),
                model_config=cast(RecurrentGPT, unwrap_compiled(model)).config.to_dict(),
                dataset_config_hash=resolved.config_hash,
                validation_rows=resolved.validation_rows,
            )
            save_training_checkpoint(backend, path, model, optimizer, metadata)
            print(f"Saved checkpoint {path}")

    logger.log_summary({"train_time": time.time() - train_t0})
    logger.finish()
    print(f"Training finished after {state['step']} steps in {time.time() - train_t0:.1f}s.")
    if settings.export_to_hf:
        export_dir = Path(settings.export_hf_path) if settings.export_hf_path else run_directory / "hf_export"
        raw = cast(RecurrentGPT, unwrap_compiled(model))
        export_to_hf(raw, raw.config, export_dir, tokenizer_dir=resolved.tokenizer_dir)
        print(f"Exported HuggingFace model to {export_dir}")


def main() -> None:
    train(parse_settings())


if __name__ == "__main__":
    main()
