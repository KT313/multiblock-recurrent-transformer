# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One optimizer step of the training loop: the micro-batch stream, the scheduled learning rate and
`run_one_optimizer_step` — the only place with autocast / backward / clipping.

Everything here is numerics (see `tasks/training_pipeline_restructure.md`, section 3). The step body is a move of the
thesis loop, bit-identical: `golden_tiny_steps.json` (dataset-independent, `test_step.py`) and `golden_tiny_run.json`
(the 20-step tiny run, `test_run.py` / `golden.py`) pin it. Two things in the stream are NOT thesis behaviour and are
pinned as the new reference instead: the transition draws come from a private `random.Random(seed + resume step)` (the
thesis drew from the checkpointed global `random`, so a thesis resume kept the transition mix of the uninterrupted
run, this one re-seeds it), and the world batch is assembled from unpadded samples and padded once per micro-batch
(the thesis collated and padded per micro-batch, then re-stacked those padded batches; the width of a regrouped
micro-batch therefore no longer depends on how the loader happened to group its rows). Steps are OPTIMIZER steps:
one world batch of `gradient_accumulation_steps` micro-batches, one `optimizer.step()`.
"""

import random
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from model import RecurrentGPT
from training.backend import Backend
from training.checkpoint import unwrap_compiled
from training.data import IGNORE_INDEX, Batch, Sample, StageDataloaders, world_batch_micro_batches
from training.data.loader import sample_stage_batch
from training.logger import track_gradient_metrics
from training.lr_schedule import get_lr_multistage
from training.optim import set_lr
from training.settings import Settings
from training.stage_manager import StageInfo, StageManager


@dataclass
class TrainingProgress:
    """The mutable step counter of a run, shared by the loop and the micro-batch stream.

    `step` is the next optimizer step to run; `train()` calls `advance()` once right after `run_one_optimizer_step`,
    so during a step `step` is the step being run and after `advance()` `done` is the number of completed steps.
    """

    step: int = 0  # next optimizer step to run
    resume_step: int = -1  # step the run was resumed at, -1 for a fresh run

    def advance(self) -> None:
        self.step += 1

    @property
    def done(self) -> int:
        """After `advance()`: the number of completed optimizer steps (evaluation, logging and checkpoint intervals
        count these)."""
        return self.step


@dataclass
class StepResult:
    """What one optimizer step produced."""

    step: int  # the optimizer step that was run
    learning_rate: float  # scheduled LR of that step (before the per-group `base_lr` multiplier)
    loss: Tensor  # mean of the micro-batch losses, all-reduced (identity on one device)
    log_ppl: Tensor  # mean of the micro-batch log-perplexities
    grad_norm: Tensor  # pre-clip gradient norm
    stage: StageInfo  # stage info at `step` (what the step trained on)
    next_stage: StageInfo  # stage info at `step + 1` (transition log lines, the stage evaluation / checkpoints use)
    data_ids: list[str]  # one entry per sample of the world batch (data composition)
    metrics: dict[str, Tensor] = field(default_factory=dict)  # `track_gradient_metrics` at log steps, else {}
    validation: dict[str, Tensor] | None = None  # filled by `train()` when it is an evaluation step


class BatchStream:
    """Endless stream of micro-batches; every `gradient_accumulation_steps` of them belong to one optimizer step.

    One world batch at a time: the stream pulls worker batches of tokenized, unpadded samples until it holds
    `world_batch_size` of them, then `world_batch_micro_batches` sorts them (`sort_batches_by_length`), splits them
    into `gradient_accumulation_steps` micro-batches and pads each one to its own longest sample. A world batch
    therefore always has exactly `world_batch_size` samples, also when a worker batch arrived short (a loader
    reaching the end of its rows) or when a row without a supervised label was dropped: the stream pulls one more
    worker batch and carries the surplus samples over to the next world batch.

    It is also the checkpointable part of the data path (`state_dict` / `load_state_dict`, stored as
    `CheckpointMetadata.data_stream`): it counts the rows it consumed per data entry and keeps the transition RNG.

    Numerics: the stream reads `progress.step` once per world batch, lazily, when the first micro-batch of that step
    is requested (after the previous step advanced the counter). Inside a stage transition each worker batch comes
    from the next stage's loader with probability `transition_progress`, drawn from `random.Random(settings.seed +
    progress.step)` — created here, at construction time, i.e. seeded with the resume step (the stream is created
    once after the resume) unless `load_state_dict` restores the stored state; `rng.random()` is consumed only
    inside a transition, once per pulled worker batch (= once per micro-batch as long as no batch is short). Loader
    iterators are created lazily by `StageDataloaders.next_train_batch`, each `iter(DataLoader)` drawing one base
    seed from the global torch RNG — all pulls of a world batch happen before its first forward (with
    `sort_batches_by_length` they always did, the sorter buffered the whole window; without it they used to be
    interleaved with the forwards).
    """

    def __init__(
        self, settings: Settings, loaders: StageDataloaders, stage_manager: StageManager, progress: TrainingProgress
    ) -> None:
        self.settings = settings
        self.loaders = loaders
        self.stage_manager = stage_manager
        self.progress = progress
        self.rng = random.Random(settings.seed + progress.step)
        self.consumed_rows: dict[str, int] = {}  # data entry prefix (`<stage>-<source>`) -> rows delivered so far
        self._surplus: list[Sample] = []
        self._micro_batches = self._stream()

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        return next(self._micro_batches)

    def state_dict(self) -> dict[str, Any]:
        """What a checkpoint stores: the consumed rows per data entry and the transition RNG state."""
        return {"consumed_rows": dict(self.consumed_rows), "transition_rng": self.rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Continue where the checkpointed run stood: every train dataset skips the rows already consumed from it
        and the transition RNG picks its state back up (instead of re-seeding with `seed + resume step`).

        This is "no repeated rows", not "the same order": the mixture draws live inside the dataloader workers and
        are not replayed, and the worker sharding regroups the remaining rows, so a mid-stage resume trains on the
        rows the interrupted run had not reached yet, in an order of its own. Rows dropped in a worker (no
        supervised label) never reach this counter, so a resume may re-read — and drop again — those few rows.
        """
        self.consumed_rows = {str(prefix): int(rows) for prefix, rows in state["consumed_rows"].items()}
        self.rng.setstate(state["transition_rng"])
        self.loaders.set_resume_offsets(self.consumed_rows)

    def _pull(self, stage: StageInfo) -> list[Sample]:
        """One worker batch (from the transition mix), counted against its data entry."""
        samples = sample_stage_batch(
            self.loaders, stage.stage_idx, stage.prev_stage_idx, stage.transition_progress, self.rng
        )
        for _, _, data_id in samples:
            self.consumed_rows[data_id] = self.consumed_rows.get(data_id, 0) + 1
        return samples

    def _stream(self) -> Iterator[Batch]:
        while True:
            stage = self.stage_manager.get_stage_info(self.progress.step)
            samples, self._surplus = self._surplus, []
            while len(samples) < self.settings.world_batch_size:
                samples += self._pull(stage)
            samples, self._surplus = (
                samples[: self.settings.world_batch_size],
                samples[self.settings.world_batch_size :],
            )
            yield from world_batch_micro_batches(
                samples,
                self.settings.micro_batch_size,
                self.loaders.tokenizer,
                self.settings.block_size,
                sort_by_length=self.settings.sort_batches_by_length,
                padding_multiple=self.settings.sequence_padding_multiple,
                ignore_index=IGNORE_INDEX,
            )


def scheduled_learning_rate(settings: Settings, stage_manager: StageManager, progress: TrainingProgress) -> float:
    """The LR of optimizer step `progress.step`: trapezoid warmup / cooldown over the whole run, the per-stage base
    LR in between (linearly interpolated inside a transition), the resume warmup after `progress.resume_step`."""
    return get_lr_multistage(
        progress.step,
        stage_manager.total_steps,
        stage_manager,
        min_lr=settings.min_lr,
        warmup_steps=settings.warmup_steps,
        cooldown_steps=settings.cooldown_steps,
        schedule=settings.lr_schedule,
        resume_step=progress.resume_step,
        resume_warmup_steps=settings.resume_warmup_steps,
    )


def run_one_optimizer_step(
    settings: Settings,
    backend: Backend,
    model: Module,
    optimizer: Optimizer,
    stage_manager: StageManager,
    batches: Iterator[Batch],
    progress: TrainingProgress,
) -> StepResult:
    """Run optimizer step `progress.step`: one world batch of `gradient_accumulation_steps` micro-batches from
    `batches`, one `optimizer.step()`. Does not advance `progress` (`train()` does, right after).

    Numerics, in order: `model.step` is set on the unwrapped model (seeds the recurrence sampler), the LR is set on
    every group (× `base_lr`); per micro-batch `backend.to_device`, `no_sync` on all but the last, autocast around
    the forward only, `backward(loss / gradient_accumulation_steps)`, `loss_sum += loss.detach()`; then the mean
    loss must be finite, `grad_norm = backend.clip_grad_norm(...)` (pre-clip norm, must be finite), `optimizer.step()`
    only if `progress.step > 0` (the very first update is skipped, as in the thesis), `track_gradient_metrics` at
    log steps before `zero_grad(set_to_none=True)`. The returned loss is `backend.all_reduce`d every step — identity
    on one device (the thesis loop reduced only at log steps: same numbers, one place).
    """
    step = progress.step
    accumulation_steps = settings.gradient_accumulation_steps
    cast(RecurrentGPT, unwrap_compiled(model)).step = step
    stage = stage_manager.get_stage_info(step)
    learning_rate = scheduled_learning_rate(settings, stage_manager, progress)
    set_lr(optimizer, learning_rate)

    loss_sum = torch.zeros((), device=backend.device)
    log_ppl_sum = torch.zeros((), device=backend.device)
    data_ids: list[str] = []
    for micro in range(accumulation_steps):
        input_ids, labels, micro_batch_data_ids = next(batches)
        data_ids.extend(micro_batch_data_ids)
        input_ids = backend.to_device(input_ids)
        labels = backend.to_device(labels)
        with backend.no_sync(model) if micro < accumulation_steps - 1 else nullcontext():
            with backend.autocast():
                outputs = model(input_ids, labels=labels)
            backend.backward(outputs["loss"] / accumulation_steps)
        loss_sum += outputs["loss"].detach()
        log_ppl_sum += outputs["log_ppl"].detach()
    loss = loss_sum / accumulation_steps
    log_ppl = log_ppl_sum / accumulation_steps
    if not torch.isfinite(loss):
        raise RuntimeError(f"Loss is {loss.item()} at step {step}. Terminating.")

    grad_norm = backend.clip_grad_norm(model, settings.grad_clip)
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"Gradient norm is non-finite at step {step}. Terminating.")
    if step > 0:  # as in the thesis runs: the very first update is skipped (LR is 0 there anyway with warmup)
        optimizer.step()
    metrics: dict[str, Tensor] = {}
    if settings.log_gradient_metrics and (step + 1) % settings.log_step_interval == 0:
        metrics = track_gradient_metrics(model, optimizer)
    optimizer.zero_grad(set_to_none=True)

    return StepResult(
        step=step,
        learning_rate=learning_rate,
        loss=backend.all_reduce(loss),
        log_ppl=log_ppl,
        grad_norm=grad_norm,
        stage=stage,
        next_stage=stage_manager.get_stage_info(step + 1),
        data_ids=data_ids,
        metrics=metrics,
    )
