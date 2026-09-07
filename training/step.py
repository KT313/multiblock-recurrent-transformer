# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
One optimizer step of the training loop: the micro-batch stream, the scheduled learning rate and
`run_one_optimizer_step`, the only place with autocast / backward / clipping.

Everything here is numerics, bit-identical to the thesis loop; the golden tests in `test_step.py` and `test_run.py`
fail on any change. The stream is the reference for the data path: ONE continuous reader per source for the whole
run, the source of every document chosen deterministically so that the sources' TOKEN shares follow the stage
weights (`BatchStream._pick_source`), and the documents packed into one row per micro-batch
(`training.data.packing`). Steps are OPTIMIZER steps: `gradient_accumulation_steps` packed micro-batches, one
`optimizer.step()`.
"""

import logging
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from model.layers.attention import document_attention_mask
from training.backend.base import Backend, plain_model
from training.data.collate import Sample
from training.data.loader import RunDataloaders
from training.data.packing import PackedBatch, PackPool, pack_samples, shifted_length
from training.data.tokenizer import IGNORE_INDEX
from training.logger import track_gradient_metrics
from training.lr_schedule import get_lr_multistage
from training.optim import set_lr
from training.settings import Settings
from training.stage_manager import StageInfo, StageManager

log = logging.getLogger(__name__)

# How many documents the pool may reject in a row before a refill gives up. A rejected document (longer than the
# pack) moves neither `_loaded` nor `_target`, so `_pick_source` returns the SAME source again: a source whose
# documents never fit would spin the refill loop forever, reading rows and warning per row without ever filling a
# pack. Settings make this unreachable (`tokens_per_micro_batch >= training_max_sequence_length`, and rows are
# truncated to that); if it happens anyway the run must fail loudly rather than hang.
MAX_CONSECUTIVE_REJECTS = 100


class NonFiniteLossError(RuntimeError):
    """
    The step's loss or gradient norm is not finite. Raised before `optimizer.step`, so the model is still the one
    of the completed steps; `train()` checkpoints it and stops.
    """


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
    data_ids: list[str]  # one entry per document of the step's packs (document count per source)
    data_tokens: dict[str, int]  # document slots trained per data id, pack tails excluded (data composition)
    metrics: dict[str, Tensor] = field(default_factory=dict)  # at log steps: `track_gradient_metrics`, `packing/padding_fraction`; else {}
    validation: dict[str, Tensor] | None = None  # filled by `train()` when it is an evaluation step


class BatchStream:
    """
    Endless stream of packed micro-batches; every `gradient_accumulation_steps` of them form one optimizer step.

    One reader per source for the whole run; stages only change the weights, so a source shared by two stages is
    never re-read. Every document goes into a `PackPool`: before every micro-batch the pool is refilled to
    `POOL_TOKEN_FACTOR` pack lengths of tokens, then one pack of `tokens_per_micro_batch` tokens is taken first-fit
    from its front (`_packs`). The weights of a pick are those of the step at which the pool was refilled, up to
    sixteen packs ahead of the step that trains on the document.

    Stage weights are TOKEN shares, realised by filling the pool by deficit instead of drawing sources at random.
    Every train source keeps two numbers, both 0 at the start of a run: `_loaded`, the slots (`shifted_length`) of
    its documents put into the pool so far, and `_target`, the slots it should have had. When a document of `n`
    slots enters the pool, its source's `_loaded` grows by `n` and EVERY source's `_target` grows by its current
    weight times `n`; the next document comes from the source with the largest `_target - _loaded` among those with
    a weight > 0 (`_pick_source`). With constant weights the deficit is `weight x total - loaded`: a source is
    over-served by at most ONE OF ITS OWN documents and under-served by at most the SUM of the other active
    sources' document lengths (it is picked as soon as it leads, but every other source may take one document
    first), so the lag is a constant and every share converges to its weight as O(1/L). When a transition blends
    the weights per step, a source that gains weight earns its share from then on, no catch-up burst for the
    tokens before. Pack tails never enter either number, nor does a document the pool rejects as oversized
    (`MAX_CONSECUTIVE_REJECTS` rejected in a row end the run instead of spinning the refill).

    Checkpointed (`state_dict`): the rows read per source (dropped rows included), the two numbers per source, the
    samples still buffered per source and the pool.

    Reproducibility rules:
    - no RNG anywhere: the pick is a deterministic function of the two numbers, ties go to the alphabetically
      smallest source name (the sources are iterated sorted by name), so reordering the dataset config's
      `sources:` block cannot change the stream
    - `progress.step` is read once per micro-batch, before the pool is refilled
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
        self.consumed_rows: dict[str, int] = {}  # source name -> rows read so far (dropped rows included)
        self._loaded: dict[str, int] = {source: 0 for source in loaders.train_sources}  # slots put into the pool
        self._target: dict[str, float] = {source: 0.0 for source in loaders.train_sources}  # slots it should have
        self._buffers: dict[str, deque[Sample]] = {source: deque() for source in loaders.train_sources}
        self._pool = PackPool(settings.tokens_per_micro_batch)
        self._micro_batches: Iterator[PackedBatch] = self._packs()

    def __iter__(self) -> Iterator[PackedBatch]:
        return self

    def __next__(self) -> PackedBatch:
        return next(self._micro_batches)

    def state_dict(self) -> dict[str, Any]:
        """
        What a checkpoint stores: the rows read per source (dropped rows included), the loaded and target slots
        per source, the buffered samples per source and the packing pool. Old checkpoint schemas (the `draw_rng`
        of the random-draw stream) have no loader (repo policy).
        """

        return {
            "consumed_rows": dict(self.consumed_rows),
            "pool_loaded": dict(self._loaded),
            "pool_target": dict(self._target),
            "buffers": {source: list(buffer) for source, buffer in self._buffers.items() if buffer},
            "pool": self._pool.state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Continue where the checkpointed run stood: every train dataset skips the rows already consumed from it,
        the loaded and target slots per source are what they were (so the next pick is the one the interrupted run
        would have made), the buffered samples are trained on first and the packing pool is what it was.

        Counters are rows READ (dropped rows included), the unit the offsets skip. The row counter, the buffered
        samples and the two numbers of a source that no longer exists are dropped (a re-added source would
        otherwise silently skip that many rows on its first epoch); a source the state does not know starts at 0.
        """

        stored_rows = {str(source): int(rows) for source, rows in state["consumed_rows"].items()}
        self.consumed_rows = {
            source: rows for source, rows in stored_rows.items() if source in self.loaders.train_sources
        }
        dropped = sorted(set(stored_rows) - set(self.consumed_rows))
        if dropped:
            log.info("Dropping the checkpoint's row counters of %s: no longer a train source of this run", dropped)
        self._loaded = {source: int(state["pool_loaded"].get(source, 0)) for source in self.loaders.train_sources}
        self._target = {source: float(state["pool_target"].get(source, 0.0)) for source in self.loaders.train_sources}
        self.loaders.set_resume_offsets(self.consumed_rows)
        for source, samples in state["buffers"].items():
            if source in self._buffers:
                self._buffers[source].extend(samples)
        self._pool.restore(list(state["pool"]))

    def _next_sample(self, source: str) -> Sample:
        """
        The next buffered sample of `source`, pulling worker batches until one is there; rows read (dropped rows
        included) are counted against the source at pull time.

        Single-shard: `rows_read` counts the rows THIS rank's loader yielded, while `set_resume_offset` skips range rows
        over all shards, so with several ranks a resume would rewind every rank by the world size. `train()` refuses
        `world_size != 1` until this counts range rows.
        """

        buffer = self._buffers[source]
        while not buffer:
            batch = self.loaders.next_train_batch(source)
            self.consumed_rows[source] = self.consumed_rows.get(source, 0) + batch.rows_read
            buffer.extend(batch.samples)
        return buffer.popleft()

    def _weights(self) -> dict[str, float]:
        """
        The per-source token-share weights of the current step, in `train_sources` order (0 for a source the stage
        does not use).
        """

        weights_at_step = self.stage_manager.data_weights(self.progress.step)
        return {source: weights_at_step.get(source, 0.0) for source in self.loaders.train_sources}

    def _pick_source(self, weights: dict[str, float]) -> str:
        """
        The source the next document comes from: the largest deficit `_target - _loaded` among the sources with a
        weight > 0; a tie goes to the alphabetically smallest source name (strict `>` while iterating the sources
        sorted by name), so the order the dataset config lists its sources in never changes the stream.
        """

        best: str | None = None
        # the seed is not a threshold: EVERY active deficit is negative for long stretches (a source leaving the
        # mixture freezes its positive deficit, so the sum over the sources still active stays short by that much).
        # `best is None` is what makes the pick correct - it takes the first eligible source whatever the seed.
        best_deficit = 0.0
        for source in sorted(weights):
            if weights[source] <= 0.0:
                continue
            deficit = self._target[source] - self._loaded[source]
            if best is None or deficit > best_deficit:
                best, best_deficit = source, deficit
        if best is None:
            raise RuntimeError(f"no train source has a weight > 0 at step {self.progress.step}: {weights}")
        return best

    def _add_to_pool(self, source: str, sample: Sample, weights: dict[str, float]) -> bool:
        """
        Put `sample` (from `source`) into the pool and account it: `source` has `n` more slots loaded and every
        source's target grows by its weight times `n`. Returns whether the pool took it: a document it rejects as
        oversized (dropped with a warning) touches neither number, it will never be trained on.
        """

        if not self._pool.add(sample):
            return False
        length = shifted_length(sample)
        self._loaded[source] += length
        for other, weight in weights.items():
            if weight > 0.0:
                self._target[other] += weight * length
        return True

    def _packs(self) -> Iterator[PackedBatch]:
        """
        The packed micro-batches: refill the pool to its token target with the current step's weights (picking the
        source of every document by deficit), take one pack first-fit from its front, pack it.
        """

        pool = self._pool
        while True:
            weights = self._weights()
            rejected = 0
            while pool.needs_refill():
                source = self._pick_source(weights)
                sample = self._next_sample(source)
                if self._add_to_pool(source, sample, weights):
                    rejected = 0
                    continue
                # a rejected document leaves the deficits where they were, so the next pick is the same source
                rejected += 1
                if rejected >= MAX_CONSECUTIVE_REJECTS:
                    raise RuntimeError(
                        f"the packing pool rejected {rejected} documents in a row while refilling at step "
                        f"{self.progress.step}: the last came from {source!r} and occupies "
                        f"{shifted_length(sample)} slots, the pack holds {pool.pack_length}. Every document of a "
                        "source must fit into one pack; raise tokens_per_micro_batch or lower "
                        "training_max_sequence_length"
                    )
            samples = pool.take_pack()
            # unreachable: the pool holds many pack lengths of documents, and both `add` and `restore` refuse
            # a document longer than one pack, so the front document always fits an empty pack
            if not samples:
                raise RuntimeError("the packing pool holds documents but none fits into an empty pack")
            yield pack_samples(samples, pool.pack_length, self.loaders.tokenizer, IGNORE_INDEX)


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


def model_inputs(batch: PackedBatch, backend: Backend) -> dict[str, Any]:
    """
    The keyword arguments of the model's forward for one packed micro-batch, on the device: `input_ids`, `labels`,
    the per-document `position_ids` and the document attention mask (`document_attention_mask`, built here, outside
    the model's forward and any compiled region).
    """

    return {
        "input_ids": backend.to_device(batch.input_ids),
        "labels": backend.to_device(batch.labels),
        "position_ids": backend.to_device(batch.position_ids),
        "attention_mask": document_attention_mask(backend.to_device(batch.document_ids)),
    }


def run_one_optimizer_step(
    settings: Settings,
    backend: Backend,
    model: Module,
    optimizer: Optimizer,
    stage_manager: StageManager,
    batches: Iterator[PackedBatch],
    progress: TrainingProgress,
) -> StepResult:
    """
    Run optimizer step `progress.step`: `gradient_accumulation_steps` packed micro-batches from `batches`, one
    `optimizer.step()`. Does not advance `progress` (`train()` does, right after).

    Runs the same numerics as the thesis training loop; the golden tests in `test_step.py` and `test_run.py` fail on
    any change. Non-obvious parts: step 0 skips `optimizer.step()`, `grad_norm` is measured before clipping, and the
    loss is all-reduced every step (a no-op on one device). The loss is the mean over the valid tokens of each pack,
    averaged over the packs: with full packs the valid-token counts are nearly equal (they differ by the pack tails),
    so the step loss is close to token-weighted. Training on padded rows was removed for exactly this reason: the
    loader sorted a world batch by length, so a micro-batch of short rows weighed as much as one of full rows.
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
    data_tokens: dict[str, int] = {}  # document slots per data id, the pack tails left out
    padding_tokens = 0  # the tail positions without a document, over the step's packs
    for micro_batch_index in range(accumulation_steps):
        batch = next(batches)
        data_ids.extend(batch.data_ids)
        for data_id, tokens in zip(batch.data_ids, batch.data_tokens):
            data_tokens[data_id] = data_tokens.get(data_id, 0) + tokens
        padding_tokens += batch.padding_tokens
        inputs = model_inputs(batch, backend)
        with backend.no_sync(model) if micro_batch_index < accumulation_steps - 1 else nullcontext():
            with backend.autocast():
                outputs = model(**inputs)
            backend.backward(outputs["loss"] / accumulation_steps)
        loss_sum += outputs["loss"].detach()
        log_ppl_sum += outputs["log_ppl"].detach()
    loss = loss_sum / accumulation_steps
    log_ppl = log_ppl_sum / accumulation_steps
    if not torch.isfinite(loss):
        raise NonFiniteLossError(f"Loss is {loss.item()} at step {step}")

    grad_norm = backend.clip_grad_norm(model, settings.grad_clip)
    if not torch.isfinite(grad_norm):
        raise NonFiniteLossError(f"Gradient norm is non-finite at step {step}")
    if step > 0:  # as in the thesis runs: the very first update is skipped (LR is 0 there anyway with warmup)
        optimizer.step()
    metrics: dict[str, Tensor] = {}
    if (step + 1) % settings.log_step_interval == 0:
        if settings.log_gradient_metrics:
            metrics = track_gradient_metrics(model, optimizer)
        # packing efficiency: the share of the step's tokens that were pack tails
        metrics["packing/padding_fraction"] = torch.tensor(padding_tokens / settings.tokens_per_optimizer_step)
    optimizer.zero_grad(set_to_none=True)

    return StepResult(
        step=step,
        learning_rate=learning_rate,
        loss=backend.all_reduce(loss),
        log_ppl=log_ppl,
        grad_norm=grad_norm,
        stage=stage,
        data_ids=data_ids,
        data_tokens=data_tokens,
        metrics=metrics,
    )
