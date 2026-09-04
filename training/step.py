# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
One optimizer step of the training loop: the micro-batch stream, the scheduled learning rate and
`run_one_optimizer_step`, the only place with autocast / backward / clipping.

Everything here is numerics, bit-identical to the thesis loop; the golden tests in `test_step.py` and `test_run.py`
fail on any change. The stream is the reference for the data path: ONE continuous reader per source for the whole
run, per-SAMPLE source draws from a private `random.Random(seed + resume step)`, and the world batch assembled from
unpadded samples and padded once per micro-batch. Steps are OPTIMIZER steps: one world batch of
`gradient_accumulation_steps` micro-batches, one `optimizer.step()`.
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
    """
    The mutable step counter of a run, shared by the loop and the micro-batch stream.

    `step` is the next optimizer step to run; `train()` calls `advance()` once right after `run_one_optimizer_step`.
    """

    step: int = 0  # next optimizer step to run
    resume_step: int = -1  # step the run was resumed at, -1 for a fresh run

    def advance(self) -> None:
        """
        One optimizer step completed: after this `step` is the number of completed steps (evaluation, logging and
        checkpoint intervals count these) and the index of the next step to run.
        """

        self.step += 1


@dataclass
class StepResult:
    """
    What one optimizer step produced.
    """

    step: int  # the optimizer step that was run
    learning_rate: float  # scheduled LR of that step
    loss: Tensor  # mean of the micro-batch losses, all-reduced (identity on one device)
    log_ppl: Tensor  # mean of the micro-batch log-perplexities
    grad_norm: Tensor  # pre-clip gradient norm
    stage: StageInfo  # stage info at `step` (what the step trained on)
    data_ids: list[str]  # one entry per sample of the world batch (data composition)
    metrics: dict[str, Tensor] = field(default_factory=dict)  # `track_gradient_metrics` at log steps, else {}
    validation: dict[str, Tensor] | None = None  # filled by `train()` when it is an evaluation step


class BatchStream:
    """
    Endless stream of micro-batches; every `gradient_accumulation_steps` of them form one optimizer step.

    One reader per source for the whole run; stages only change the draw weights, so a source shared by two stages
    is never re-read. Each sample of a world batch comes from a source drawn with the current step's weights; the
    full world batch is then split and padded by `world_batch_micro_batches`.

    Checkpointed (`state_dict`): the rows read per source (dropped rows included), the draw RNG state and the
    samples still buffered per source.

    Reproducibility rules:
    - the draw RNG is `random.Random(seed + resume step)`, restored from a checkpoint when there is one; every
      sample draw consumes it
    - `progress.step` is read once per world batch, when its first micro-batch is requested
    - loader iterators are created at a source's first pull; their base seeds come from the loaders' private
      generator, never from the global torch RNG
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
        """
        What a checkpoint stores: the rows read per source (dropped rows included), the draw RNG state and the
        buffered samples per source. Old checkpoint schemas have no loader (repo policy).
        """

        return {
            "consumed_rows": dict(self.consumed_rows),
            "draw_rng": self.rng.getstate(),
            "buffers": {source: list(buffer) for source, buffer in self._buffers.items() if buffer},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Continue where the checkpointed run stood: every train dataset skips the rows already consumed from it,
        the draw RNG picks its state back up and the buffered samples are trained on first.

        Counters are rows READ (dropped rows included), the unit the offsets skip. Buffered samples of a source
        that no longer exists are dropped.
        """

        self.consumed_rows = {str(source): int(rows) for source, rows in state["consumed_rows"].items()}
        self.rng.setstate(state["draw_rng"])
        self.loaders.set_resume_offsets(self.consumed_rows)
        for source, samples in state["buffers"].items():
            if source in self._buffers:
                self._buffers[source].extend(samples)

    def _next_sample(self, source: str) -> Sample:
        """
        The next buffered sample of `source`, pulling worker batches until one is there; rows read (dropped rows
        included) are counted against the source at pull time.
        """

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
    """
    The LR of optimizer step `progress.step`: trapezoid warmup / cooldown over the whole run, the per-stage base
    LR in between (linearly interpolated inside a transition).
    """

    return get_lr_multistage(
        progress.step,
        stage_manager.total_steps,
        stage_manager,
        min_lr=settings.min_lr,
        warmup_steps=settings.warmup_steps,
        cooldown_steps=settings.cooldown_steps,
        schedule=settings.lr_schedule,
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
    """
    Run optimizer step `progress.step`: one world batch of `gradient_accumulation_steps` micro-batches from
    `batches`, one `optimizer.step()`. Does not advance `progress` (`train()` does, right after).

    Runs the same numerics as the thesis training loop; the golden tests in `test_step.py` and `test_run.py` fail on
    any change. Non-obvious parts: step 0 skips `optimizer.step()`, `grad_norm` is measured before clipping, and the
    loss is all-reduced every step (a no-op on one device).
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
    for micro_batch_index in range(accumulation_steps):
        input_ids, labels, micro_batch_data_ids = next(batches)
        data_ids.extend(micro_batch_data_ids)
        input_ids = backend.to_device(input_ids)
        labels = backend.to_device(labels)
        with backend.no_sync(model) if micro_batch_index < accumulation_steps - 1 else nullcontext():
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
        data_ids=data_ids,
        metrics=metrics,
    )
