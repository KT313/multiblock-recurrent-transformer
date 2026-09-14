# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Deterministic packed microbatch streams and their per-rank views."""

import logging
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import Tensor

from training.backend.base import Backend
from training.data.collate import Sample
from training.data.loader import RunDataloaders
from training.data.packing import PackedBatch, PackPool, pack_samples, shifted_length
from training.data.tokenizer import IGNORE_INDEX
from training.settings import Settings
from training.stage_manager import StageManager
from training.steps.state import TrainingProgress

log = logging.getLogger("training.step")

# How many documents the pool may reject in a row before a refill gives up. A rejected document (longer than the
# pack) moves neither `_loaded` nor `_target`, so `_pick_source` returns the SAME source again: a source whose
# documents never fit would spin the refill loop forever, reading rows and warning per row without ever filling a
# pack. Settings make this unreachable (`tokens_per_micro_batch >= training_max_sequence_length`, and rows are
# truncated to that); if it happens anyway the run must fail loudly rather than hang.
MAX_CONSECUTIVE_REJECTS = 100


class BatchStream:
    """
    Endless stream of packed micro-batches; every `micro_batches_per_step` of them form one optimizer step.

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
        self._loaded: dict[str, int] = dict.fromkeys(loaders.train_sources, 0)  # slots put into the pool
        self._target: dict[str, float] = dict.fromkeys(loaders.train_sources, 0.0)  # slots it should have
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
        # Intentional resume policy: drain saved samples unchanged after explicitly allowed config changes.
        # The pool may retain removed sources; buffers/pool retain old document lengths. This bounded
        # carry-over is accepted for rare mid-run changes, so do not filter/retruncate these samples.
        # Old lengths must still fit model_max_sequence_length: lowering that bound can cause a failure.
        # Pool.restore separately drops documents exceeding the current pack length.
        for source, samples in state["buffers"].items():
            if source in self._buffers:
                self._buffers[source].extend(samples)
        self._pool.restore(list(state["pool"]))

    def _next_sample(self, source: str) -> Sample:
        """
        The next buffered sample of `source`, pulling worker batches until one is there; rows read (dropped rows
        included) are counted against the source at pull time.

        Single reader: the train loaders read every source as ONE shard on the main rank (`RankBatches` scatters the
        packs), so `rows_read` counts range rows and `set_resume_offset` skips exactly them on a resume.
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



# The tensors of a pack a rank receives, in the order they are stacked for the scatter (`RankBatches`); all `(1, L)`
PACK_TENSORS = ("input_ids", "labels", "position_ids", "document_ids")


class RankBatches:
    """
    This rank's packed micro-batches: the `BatchStream` itself at world size 1, otherwise the main rank's stream
    scattered one pack per rank and micro-batch index.

    The main rank pulls `world_size` packs per `__next__`, stacks their four `(1, L)` tensors into a
    `(world_size, 4, L)` int64 tensor (`PACK_TENSORS` order, the int32 document ids widened for the stack), scatters it
    (`Backend.scatter_packs`) and trains on pack 0. The pack it returns carries the data statistics of ALL packs
    (`data_ids` / `data_tokens` concatenated in pack order, `padding_tokens` summed), so `StepResult` and the logger's
    composition and padding metrics describe the world's step without knowing about ranks. The other ranks receive
    their slice and return it with empty statistics (their logger is silent anyway). One collective per micro-batch
    index: the main rank packs micro-batch i + 1 while the devices run i.

    `state_dict` / `load_state_dict` are the stream's on the main rank; the state dict is empty on the other ranks,
    whose checkpoint is never written.
    """

    def __init__(self, backend: Backend, stream: "BatchStream | None", pack_length: int) -> None:
        if backend.is_main and stream is None:
            raise ValueError("the main rank owns the data stream and must pass it")
        if not backend.is_main and stream is not None:
            raise ValueError("only the main rank reads data; a non-main rank passes no stream")
        self.backend = backend
        self.stream = stream
        self.pack_length = pack_length

    def __iter__(self) -> Iterator[PackedBatch]:
        return self

    def __next__(self) -> PackedBatch:
        backend = self.backend
        slice_shape = (len(PACK_TENSORS), self.pack_length)
        if self.stream is not None:
            if backend.world_size == 1:
                return next(self.stream)
            packs = [next(self.stream) for _ in range(backend.world_size)]
            stacked = torch.stack(
                [torch.cat([getattr(pack, name).to(torch.int64) for name in PACK_TENSORS], dim=0) for pack in packs]
            )
            mine = backend.scatter_packs(stacked, slice_shape)
            return self._pack_from(
                mine,
                data_ids=[data_id for pack in packs for data_id in pack.data_ids],
                data_tokens=[tokens for pack in packs for tokens in pack.data_tokens],
                padding_tokens=sum(pack.padding_tokens for pack in packs),
            )
        return self._pack_from(backend.scatter_packs(None, slice_shape), data_ids=[], data_tokens=[], padding_tokens=0)

    @staticmethod
    def _pack_from(rows: Tensor, *, data_ids: list[str], data_tokens: list[int], padding_tokens: int) -> PackedBatch:
        """
        The `PackedBatch` of one received `(4, L)` slice: the rows back to `(1, L)` each, the document ids to int32.
        """

        input_ids, labels, position_ids, document_ids = (rows[index : index + 1] for index in range(len(PACK_TENSORS)))
        return PackedBatch(
            input_ids=input_ids,
            labels=labels,
            data_ids=data_ids,
            position_ids=position_ids,
            document_ids=document_ids.to(torch.int32),
            padding_tokens=padding_tokens,
            data_tokens=data_tokens,
        )

    def state_dict(self) -> dict[str, Any]:
        return self.stream.state_dict() if self.stream is not None else {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.stream is not None:
            self.stream.load_state_dict(state)
