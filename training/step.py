# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One optimizer step of the training loop: the micro-batch stream, the scheduled learning rate and
`run_one_optimizer_step` — the only place with autocast / backward / clipping.

Everything here is numerics (see `tasks/training_pipeline_restructure.md`, section 3). The step body is a move of the
thesis loop, bit-identical: `golden_tiny_steps.json` (dataset-independent, `test_step.py`) and `golden_tiny_run.json`
(the 20-step tiny run, `test_run.py` / `golden.py`) pin it. The stream departs from the thesis recipe and is pinned
as the new reference instead: ONE continuous reader per source for the whole run with per-SAMPLE source draws from a
private `random.Random(seed + resume step)` (the thesis pulled whole worker batches from per-stage mixture loaders,
re-reading a shared source's rows in every stage), and the world batch is assembled from unpadded samples and padded
once per micro-batch (the thesis collated and padded per micro-batch, then re-stacked those padded batches; the
width of a regrouped micro-batch therefore no longer depends on how the loader happened to group its rows). Steps
are OPTIMIZER steps: one world batch of `gradient_accumulation_steps` micro-batches, one `optimizer.step()`.
"""

import random
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from training.backend.base import Backend, plain_model
from training.data.collate import IGNORE_INDEX, Batch, Sample
from training.data.loader import RunDataloaders, world_batch_micro_batches
from training.logger import track_gradient_metrics
from training.lr_schedule import get_lr_multistage
from training.optim import set_lr
from training.settings import Settings
from training.stage_manager import StageInfo, StageManager


@dataclass
class TrainingProgress:
    """The mutable step counter of a run, shared by the loop and the micro-batch stream.

    `step` is the next optimizer step to run; `train()` calls `advance()` once right after `run_one_optimizer_step`.
    """

    step: int = 0  # next optimizer step to run
    resume_step: int = -1  # step the run was resumed at, -1 for a fresh run

    def advance(self) -> None:
        """One optimizer step completed: after this `step` is the number of completed steps (evaluation, logging and
        checkpoint intervals count these) and the index of the next step to run."""
        self.step += 1


@dataclass
class StepResult:
    """What one optimizer step produced."""

    step: int  # the optimizer step that was run
    learning_rate: float  # scheduled LR of that step
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

    ONE continuous reader per source: for every sample of a world batch the stream draws which source it comes
    from — in the main process, from its private `rng` — with the current step's weights
    (`StageManager.data_weights`: the stage's constants, linearly interpolated across a transition window). A
    per-source buffer holds the samples of the last pulled worker batch; a draw whose buffer is empty pulls the
    next worker batch from that source's loader until a sample arrives (a batch can come back empty when every row
    was dropped for lack of a supervised label). The stage structure therefore only changes WEIGHTS: the readers
    run through the whole run and never re-read rows an earlier stage consumed. Once `world_batch_size` samples
    are drawn, `world_batch_micro_batches` sorts them (`sort_batches_by_length`), splits them into
    `gradient_accumulation_steps` micro-batches and pads each one to its own longest sample.

    It is also the checkpointable part of the data path (`state_dict` / `load_state_dict`, stored as
    `CheckpointMetadata.data_stream`): it counts the rows READ per SOURCE — attached at pull time via
    `WorkerBatch.rows_read`, so a row dropped in a worker still counts, the same unit
    `ParquetTextDataset.set_resume_offset` skips — and keeps the draw RNG.

    Numerics: the stream reads `progress.step` once per world batch, lazily, when the first micro-batch of that
    step is requested (after the previous step advanced the counter); all `world_batch_size` draws of the world
    batch use that step's weights. Every sample draw consumes the RNG (`rng.choices` over the sources in
    dataset-config order), inside and outside transitions alike; the RNG is `random.Random(settings.seed +
    progress.step)` — created here, at construction time, i.e. seeded with the resume step (the stream is created
    once after the resume) unless `load_state_dict` restores the stored state. Loader iterators are created lazily
    by `RunDataloaders.next_train_batch` at a source's first pull (and on every restart), each `iter(DataLoader)`
    drawing one base seed from the global torch RNG — the creation order follows the deterministic draw sequence,
    and all pulls of a world batch happen before its first forward.
    """

    def __init__(
        self, settings: Settings, loaders: RunDataloaders, stage_manager: StageManager, progress: TrainingProgress
    ) -> None:
        self.settings = settings
        self.loaders = loaders
        self.stage_manager = stage_manager
        self.progress = progress
        self.rng = random.Random(settings.seed + progress.step)
        self.consumed_rows: dict[str, int] = {}  # source name -> rows read so far (dropped rows included)
        self._buffers: dict[str, deque[Sample]] = {source: deque() for source in loaders.train_sources}
        self._micro_batches = self._stream()

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        return next(self._micro_batches)

    def state_dict(self) -> dict[str, Any]:
        """What a checkpoint stores: the rows read per source (dropped rows included) and the draw RNG state.

        A clean break from the per-stage stream's schema (`transition_rng`, `<stage>-<source>` counters): there is
        no loader for old checkpoints (repo policy).
        """
        return {"consumed_rows": dict(self.consumed_rows), "draw_rng": self.rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Continue where the checkpointed run stood: every train dataset skips the rows already consumed from it
        and the draw RNG picks its state back up (instead of re-seeding with `seed + resume step`).

        This is "no repeated rows": the draw sequence and each source's row order continue exactly, but samples
        that sat in a buffer when the checkpoint was written were already counted as read and are skipped, and the
        fresh loader iterators draw new base seeds from the global torch RNG — so a resume trains on rows the
        interrupted run had not consumed, without reproducing its losses bit for bit. The counters are rows READ
        (`WorkerBatch.rows_read`, dropped rows included) — the unit the offsets skip — so a row dropped for lack
        of a supervised label is not re-read either.
        """
        self.consumed_rows = {str(source): int(rows) for source, rows in state["consumed_rows"].items()}
        self.rng.setstate(state["draw_rng"])
        self.loaders.set_resume_offsets(self.consumed_rows)

    def _next_sample(self, source: str) -> Sample:
        """The next buffered sample of `source`, pulling worker batches until one is there; rows read — dropped
        rows included — are counted against the source at pull time."""
        buffer = self._buffers[source]
        while not buffer:
            batch = self.loaders.next_train_batch(source)
            self.consumed_rows[source] = self.consumed_rows.get(source, 0) + batch.rows_read
            buffer.extend(batch.samples)
        return buffer.popleft()

    def _stream(self) -> Iterator[Batch]:
        sources = self.loaders.train_sources
        while True:
            weights_at_step = self.stage_manager.data_weights(self.progress.step)
            weights = [weights_at_step.get(source, 0.0) for source in sources]
            samples = [
                self._next_sample(self.rng.choices(sources, weights=weights, k=1)[0])
                for _ in range(self.settings.world_batch_size)
            ]
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
    every group; per micro-batch `backend.to_device`, `no_sync` on all but the last, autocast around
    the forward only, `backward(loss / gradient_accumulation_steps)`, `loss_sum += loss.detach()`; then the mean
    loss must be finite, `grad_norm = backend.clip_grad_norm(...)` (pre-clip norm, must be finite), `optimizer.step()`
    only if `progress.step > 0` (the very first update is skipped, as in the thesis), `track_gradient_metrics` at
    log steps before `zero_grad(set_to_none=True)`. The returned loss is `backend.all_reduce`d every step — identity
    on one device (the thesis loop reduced only at log steps: same numbers, one place).
    """
    step = progress.step
    accumulation_steps = settings.gradient_accumulation_steps
    plain_model(model).step = step
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
