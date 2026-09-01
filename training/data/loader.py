# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Dataloader construction (one loader per stage mixture, `build_stage_dataloaders` for a whole run), the per-stage
batch sampling used by multi-stage training and the assembly of one world batch into padded micro-batches."""

import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from torch.utils.data import DataLoader, IterableDataset

from training.backend import Backend
from training.data.collate import (
    IGNORE_INDEX,
    Batch,
    Sample,
    WorkerBatch,
    collate_fn,
    collate_worker_batch,
    pad_and_shift,
)
from training.data.dataset_resolver import DataEntry, ResolvedDataset
from training.data.datasets import ParquetTextDataset, Row, WeightedMixtureDataset
from training.data.tokenizer import Tokenizer
from training.settings import Settings

SampleBatch = list[Sample]  # the surviving tokenized rows of one worker batch (the `samples` half of a WorkerBatch)


def build_dataloader(
    entries: list[DataEntry],
    tokenizer: Tokenizer,
    block_size: int,
    micro_batch_size: int,
    num_workers: int = 0,
    seed: int = 1337,
    shard: tuple[int, int] = (0, 1),
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    pin_memory: bool = False,
    padded: bool = True,
) -> DataLoader[Row]:
    """Loader over the weighted mixture of ``entries`` (a stage's ``train_data`` / ``val_data`` as resolved by
    `training.data.dataset_resolver`).

    ``padded`` (the validation loaders, and the default) yields ready ``(input_ids, labels, data_ids)`` batches;
    ``padded=False`` (the training loaders) yields a `WorkerBatch` — the unpadded `Sample` list of the batch plus
    the per-entry count of rows read to produce it (dropped rows included) — the workers still do the tokenization,
    the padding happens once per assembled micro-batch in `world_batch_micro_batches`.

    Every entry becomes one `ParquetTextDataset` over its row range ``[skip_rows, skip_rows + max_rows)`` — the
    validation split decided by the resolver — with its ``data_signature`` (None = the text column).
    ``shard=(rank, world)`` is passed to every dataset; with ``world == 1`` it is a no-op. ``pin_memory`` is the
    backend's decision (`Backend.pin_memory`, true on CUDA).
    """
    if len({e.prefix for e in entries}) != len(entries):
        raise ValueError("Dataset prefixes within one loader must be unique.")
    datasets = [
        ParquetTextDataset(
            e.data_dir, e.prefix, e.data_signature, shard=shard, skip_rows=e.skip_rows, max_rows=e.max_rows
        )
        for e in entries
    ]
    dataset: IterableDataset[Row] = (
        datasets[0] if len(datasets) == 1 else WeightedMixtureDataset(datasets, [e.weight for e in entries], seed)
    )
    collate: Callable[[list[Row]], Any]
    if padded:
        collate = partial(
            collate_fn,
            tokenizer=tokenizer,
            block_size=block_size,
            padding_multiple=padding_multiple,
            ignore_index=ignore_index,
        )
    else:
        collate = partial(collate_worker_batch, tokenizer=tokenizer, block_size=block_size)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )


@dataclass
class StageDataloaders:
    """One train and one val loader per training stage, with lazily created and cycled train iterators.

    Train loaders yield unpadded `WorkerBatch`es (surviving samples + rows read per entry), validation loaders
    padded `Batch`es; `tokenizer` is the one every loader was built with and the one `world_batch_micro_batches`
    pads with. What a resume restores is the row offsets of `set_resume_offsets` (`training.step.BatchStream` owns
    the bookkeeping).
    """

    train_loaders: Sequence[Iterable[WorkerBatch]]
    val_loaders: Sequence[Iterable[Batch]]
    tokenizer: Tokenizer
    _train_iterators: list[Iterator[WorkerBatch] | None] = field(init=False)

    def __post_init__(self) -> None:
        self._train_iterators = [None] * len(self.train_loaders)

    def next_train_batch(self, stage_idx: int) -> WorkerBatch:
        """Next worker batch of ``stage_idx``'s train loader; restarts the loader when it is exhausted."""
        iterator = self._train_iterators[stage_idx]
        if iterator is None:
            iterator = self._train_iterators[stage_idx] = iter(self.train_loaders[stage_idx])
        try:
            return next(iterator)
        except StopIteration:
            # the epoch that started at the resume offsets is over; every later one reads the whole range again
            self.clear_resume_offsets(stage_idx)
            iterator = self._train_iterators[stage_idx] = iter(self.train_loaders[stage_idx])
            return next(iterator)

    def train_datasets(self, stage_idx: int) -> list[ParquetTextDataset]:
        """The parquet datasets behind a stage's train loader (empty for a loader that is not a `DataLoader` over
        them, which is what the tests hand in)."""
        dataset = getattr(self.train_loaders[stage_idx], "dataset", None)
        if isinstance(dataset, ParquetTextDataset):
            return [dataset]
        if isinstance(dataset, WeightedMixtureDataset):
            return [member for member in dataset.datasets if isinstance(member, ParquetTextDataset)]
        return []

    def set_resume_offsets(self, consumed_rows: Mapping[str, int]) -> None:
        """Start every train dataset whose prefix appears in `consumed_rows` that many rows into its range, so a
        resumed run does not train on the rows the interrupted run already saw. One-shot (see
        `ParquetTextDataset.set_resume_offset`); prefixes of other runs or of removed sources are ignored."""
        for stage_idx in range(len(self.train_loaders)):
            for dataset in self.train_datasets(stage_idx):
                dataset.set_resume_offset(consumed_rows.get(dataset.prefix, 0))

    def clear_resume_offsets(self, stage_idx: int) -> None:
        """Drop the pending resume offsets of a stage (its loader is about to be re-created for a fresh epoch)."""
        for dataset in self.train_datasets(stage_idx):
            dataset.set_resume_offset(0)


def build_stage_dataloaders(settings: Settings, dataset: ResolvedDataset, backend: Backend) -> StageDataloaders:
    """One train and one validation loader per stage of `dataset`, each mixing its entries with constant weights.

    The tokenizer is loaded once from `dataset.tokenizer_dir` and shared by every loader. Train loaders use
    `settings.dataloader_num_workers` and yield unpadded samples, validation loaders read in-process and yield
    padded batches. Loader seed `settings.seed + rank`, datasets sharded by `(rank, world_size)`.
    """
    tokenizer = Tokenizer(dataset.tokenizer_dir)

    def loader(entries: list[DataEntry], num_workers: int, padded: bool) -> DataLoader[Row]:
        return build_dataloader(
            entries,
            tokenizer,
            block_size=settings.block_size,
            micro_batch_size=settings.micro_batch_size,
            num_workers=num_workers,
            seed=settings.seed + backend.rank,
            shard=(backend.rank, backend.world_size),
            padding_multiple=settings.sequence_padding_multiple,
            ignore_index=IGNORE_INDEX,
            pin_memory=backend.pin_memory,
            padded=padded,
        )

    return StageDataloaders(
        train_loaders=[loader(stage.train_data, settings.dataloader_num_workers, False) for stage in dataset.stages],
        val_loaders=[loader(stage.val_data, 0, True) for stage in dataset.stages],
        tokenizer=tokenizer,
    )


def sample_stage_batch(
    stage_loaders: StageDataloaders,
    stage_idx: int,
    prev_stage_idx: int | None,
    transition_progress: float,
    rng: random.Random,
) -> WorkerBatch:
    """Worker batch for the current step: from ``stage_idx`` with probability ``transition_progress``, else from the
    previous stage. Outside a transition (``prev_stage_idx`` is None) always from ``stage_idx``."""
    if prev_stage_idx is not None and rng.random() >= transition_progress:
        return stage_loaders.next_train_batch(prev_stage_idx)
    return stage_loaders.next_train_batch(stage_idx)


def sample_length(sample: Sample) -> int:
    """Tokens of a sample before padding — what its micro-batch will have to be padded to."""
    return sample[0].shape[0]


def world_batch_micro_batches(
    samples: list[Sample],
    micro_batch_size: int,
    tokenizer: Tokenizer,
    block_size: int,
    sort_by_length: bool,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> list[Batch]:
    """Split one world batch of samples into micro-batches of `micro_batch_size` and pad each of them once.

    With `sort_by_length` the samples are sorted by their token count first (a stable sort: ties keep arrival
    order), so a micro-batch groups rows of similar length and its padding — and with it the compute of the
    forward — shrinks. Without it the arrival order is kept, which reproduces the loader's own batching exactly.
    Every sample keeps all of its supervised positions either way: the width is derived from the full sample
    length, never from the supervised part of it.
    """
    if sort_by_length:
        samples = sorted(samples, key=sample_length)
    return [
        pad_and_shift(samples[start : start + micro_batch_size], tokenizer, block_size, padding_multiple, ignore_index)
        for start in range(0, len(samples), micro_batch_size)
    ]
