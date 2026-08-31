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

from model import RecurrentConfig, RecurrentGPT, build_model
from model.hf import export_to_hf
from training.backend import Backend, get_backend
from training.checkpoint import (
    checkpoint_dir,
    checkpoint_name,
    collect_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    should_save_checkpoint,
)
from training.data import StageDataloaders, Tokenizer, build_dataloader, length_sorted_batches
from training.data.dataset_resolver import (
    CHECKPOINT_HASH_KEY,
    CHECKPOINT_VALIDATION_ROWS_KEY,
    ResolvedDataset,
    check_checkpoint_dataset_hash,
    check_checkpoint_validation_rows,
    resolve_dataset,
)
from training.data.loader import Batch, sample_stage_batch
from training.logger import Logger, num_parameters, track_gradient_metrics
from training.lr_schedule import get_lr_multistage
from training.optim import build_optimizer, get_param_groups, set_lr
from training.settings import DataEntry, Settings, parse_settings
from training.stage_manager import StageManager

IGNORE_INDEX = -100


class LoopState(TypedDict):
    """Mutable step counter shared between the loop and the micro-batch stream."""

    step: int  # next optimizer step to run
    resume_step: int  # step the run was resumed at, -1 for a fresh run


def unwrap(model: torch.nn.Module) -> RecurrentGPT:
    """The plain `RecurrentGPT` behind a `torch.compile` wrapper (for `.step`, `.config`, ...)."""
    return cast(RecurrentGPT, getattr(model, "_orig_mod", model))


def build_stage_dataloaders(
    cfg: Settings, resolved: ResolvedDataset, tokenizer: Tokenizer, backend: Backend
) -> StageDataloaders:
    """One train and one validation loader per stage, each mixing its datasets with constant weights."""

    def loader(entries: list[DataEntry], num_workers: int) -> Iterable[Batch]:
        return build_dataloader(
            entries,
            tokenizer,
            block_size=cfg.block_size,
            micro_batch_size=cfg.micro_batch_size,
            num_workers=num_workers,
            seed=cfg.seed + backend.rank,
            shard=(backend.rank, backend.world_size),
            padding_multiple=cfg.sequence_padding_multiple,
            ignore_index=IGNORE_INDEX,
        )

    return StageDataloaders(
        train_loaders=[loader(s.train_data, cfg.dataloader_num_workers) for s in resolved.stages],
        val_loaders=[loader(s.val_data, 0) for s in resolved.stages],
    )


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
    config = unwrap(model).config
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


def train(cfg: Settings) -> None:
    start_time = time.time()
    backend = get_backend(cfg.backend, precision=cfg.precision)
    backend.seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir)
    checkpoint_dir(out_dir).mkdir(parents=True, exist_ok=True)

    resolved = resolve_dataset(cfg, backend)  # verifies the dataset config's data, auto-prepares if configured
    tokenizer = Tokenizer(resolved.tokenizer_dir)
    stage_manager = StageManager(
        resolved.stage_manager_stages(),
        world_batch_size=cfg.world_batch_size,
        block_size=cfg.block_size,
        world_size=backend.world_size,
        warmup_steps=cfg.warmup_steps,
        cooldown_steps=cfg.cooldown_steps,
        micro_batch_size=cfg.micro_batch_size,
    )
    max_steps = stage_manager.total_steps
    print(stage_manager.get_stage_summary())
    print(f"Total training steps: {max_steps:,} ({cfg.gradient_accumulation_steps} micro-batches each)")
    loaders = build_stage_dataloaders(cfg, resolved, tokenizer, backend)

    model_config = RecurrentConfig.from_yaml(cfg.model_architecture_config, **cfg.model_overwrite)
    if model_config.block_size != cfg.block_size:
        raise ValueError(
            f"block_size {cfg.block_size} of the run config does not match block_size {model_config.block_size} of the "
            f"model architecture config {cfg.model_architecture_config} (with model_overwrite applied)"
        )
    raw_model = build_model(model_config, ignore_index=IGNORE_INDEX, gradient_checkpointing=cfg.gradient_checkpointing)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=4)
    raw_model.config.to_json(out_dir / "model_config.json")
    n_params = num_parameters(raw_model)
    core_blocks = cast(Iterable[torch.nn.Module], raw_model.transformer.core_blocks)
    rec_params = sum(p.numel() for block in core_blocks for p in block.parameters())
    mean_recurrence = cast(list[int], raw_model.config.mean_recurrence)  # a list after RecurrentConfig.__post_init__
    mean_rec = sum(mean_recurrence) / len(mean_recurrence)
    print(f"Model: {n_params:,} parameters, {rec_params:,} in recurrent blocks, unfolds to "
          f"{int(n_params - rec_params + rec_params * mean_rec):,} at mean recurrence.")
    model = backend.setup_model(raw_model, compile=cfg.compile_model)

    weight_decay = cfg.optim_config.get("weight_decay", 0.0)
    param_groups = get_param_groups(model, weight_decay, cfg.no_weight_decay_for_bias_and_norm_params)
    optimizer = backend.setup_optimizer(build_optimizer(cfg.optimizer, param_groups, **cfg.optim_config))

    state: LoopState = {"step": 0, "resume_step": -1}
    resume_path = None
    if cfg.resume:
        resume_path = Path(cfg.resume_checkpoint_path) if cfg.resume_checkpoint_path else find_latest_checkpoint(
            out_dir, cfg.run_name
        )
    if resume_path is not None:
        extra = load_checkpoint(backend, resume_path, model, optimizer)
        check_checkpoint_dataset_hash(extra, resolved.config_hash, cfg.allow_dataset_change)
        check_checkpoint_validation_rows(extra, resolved.validation_rows, cfg.allow_dataset_change)
        state["step"] = state["resume_step"] = extra["step"]
        restore_rng_state(extra["rng"])
        print(f"Resumed from {resume_path} at step {state['step']}")
    else:
        print("No checkpoint loaded, starting from scratch.")

    logger = Logger(cfg.logger_project, cfg.run_name, out_dir, offline=cfg.wandb_offline, enabled=cfg.wandb_enabled)
    logger.log_hyperparams(asdict(cfg) | {CHECKPOINT_HASH_KEY: resolved.config_hash})
    logger.log_summary({"num_parameters": n_params})

    rng = random.Random(cfg.seed + state["step"])
    batches = micro_batch_stream(cfg, loaders, stage_manager, state, rng)
    tokens_per_step = cfg.world_batch_size * cfg.block_size
    sample_counter: Counter[str] = Counter()
    print(f"{time.ctime()[:-5]}: setup took {time.time() - start_time:.1f}s, starting training at step {state['step']}.")

    train_t0 = time.time()
    interval_t0 = time.time()
    while state["step"] < max_steps:
        step = state["step"]
        unwrap(model).step = step
        info = stage_manager.get_stage_info(step)
        lr = get_lr_multistage(
            step,
            max_steps,
            stage_manager,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps,
            cooldown_steps=cfg.cooldown_steps,
            schedule=cfg.lr_schedule,
            resume_step=state["resume_step"],
            resume_warmup_steps=cfg.resume_warmup_steps,
        )
        set_lr(optimizer, lr)

        loss_sum = torch.zeros((), device=backend.device)
        log_ppl_sum = torch.zeros((), device=backend.device)
        for micro in range(cfg.gradient_accumulation_steps):
            input_ids, labels, data_ids = next(batches)
            sample_counter.update(data_ids)
            input_ids = input_ids.to(backend.device, non_blocking=True)
            labels = labels.to(backend.device, non_blocking=True)
            with backend.no_sync(model) if micro < cfg.gradient_accumulation_steps - 1 else nullcontext():
                with backend.autocast():
                    outputs = model(input_ids, labels=labels)
                backend.backward(outputs["loss"] / cfg.gradient_accumulation_steps)
            loss_sum += outputs["loss"].detach()
            log_ppl_sum += outputs["log_ppl"].detach()
        loss = loss_sum / cfg.gradient_accumulation_steps
        log_ppl = log_ppl_sum / cfg.gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is {loss.item()} at step {step}. Terminating.")

        grad_norm = backend.clip_grad_norm(model, cfg.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Gradient norm is non-finite at step {step}. Terminating.")
        if step > 0:  # as in the thesis runs: the very first update is skipped (LR is 0 there anyway with warmup)
            optimizer.step()
        metrics: dict[str, Any] = {}
        if cfg.log_gradient_metrics and (step + 1) % cfg.log_step_interval == 0:
            metrics |= track_gradient_metrics(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        state["step"] = done = step + 1

        next_info = stage_manager.get_stage_info(done)
        if next_info.in_transition and not info.in_transition:
            print(f"Step {done}: starting transition {next_info.prev_stage_idx} -> {next_info.stage_idx} "
                  f"({next_info.stage_name}), LR {next_info.prev_base_lr:.2e} -> {next_info.base_lr:.2e}")
        elif info.in_transition and not next_info.in_transition:
            print(f"Step {done}: transition complete, now in stage {next_info.stage_idx} ({next_info.stage_name})")

        if done % cfg.eval_step_interval == 0 or done >= max_steps:
            t0 = time.time()
            val_metrics = validate(cfg, backend, model, loaders.val_loaders[next_info.stage_idx])
            val_metrics["val_time"] = torch.as_tensor(time.time() - t0)
            print(f"Step {done}: val loss {val_metrics['val_loss'].item():.4f} "
                  f"(stage {next_info.stage_idx}, {val_metrics['val_time']:.1f}s)")
            metrics |= val_metrics

        if done % cfg.log_step_interval == 0:
            now = time.time()
            steps_in_interval = cfg.log_step_interval
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

        stage_end, stage_suffix = stage_manager.should_save_stage_checkpoint(step)
        if should_save_checkpoint(
            done,
            max_steps=max_steps,
            save_step_interval=cfg.save_step_interval,
            save_last_step=cfg.save_last_step,
            stage_end=stage_end,
        ):
            stage_idx = int(stage_suffix.split("-")[1].split("_")[0]) if stage_end else None
            path = checkpoint_dir(out_dir) / checkpoint_name(done, cfg.run_name, stage_end=stage_idx)
            extra = {
                "step": done,
                "stage": next_info.stage_idx,
                "rng": collect_rng_state(),
                "config": asdict(cfg),
                CHECKPOINT_HASH_KEY: resolved.config_hash,
                CHECKPOINT_VALIDATION_ROWS_KEY: resolved.validation_rows,
            }
            save_checkpoint(backend, path, model, optimizer, extra)
            print(f"Saved checkpoint {path}")

    logger.log_summary({"train_time": time.time() - train_t0})
    logger.finish()
    print(f"Training finished after {state['step']} steps in {time.time() - train_t0:.1f}s.")
    if cfg.export_to_hf:
        export_dir = Path(cfg.export_hf_path) if cfg.export_hf_path else out_dir / "hf_export"
        raw = unwrap(model)
        export_to_hf(raw, raw.config, export_dir, tokenizer_dir=resolved.tokenizer_dir)
        print(f"Exported HuggingFace model to {export_dir}")


def main() -> None:
    train(parse_settings())


if __name__ == "__main__":
    main()
