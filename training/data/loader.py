# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Dataloader construction (one loader per stage mixture, `build_stage_dataloaders` for a whole run) and the
per-stage batch sampling used by multi-stage training."""

import random
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset

from training.backend import Backend
from training.data.collate import IGNORE_INDEX, collate_fn, find_multiple
from training.data.dataset_resolver import DataEntry, ResolvedDataset
from training.data.datasets import ParquetTextDataset, Row, WeightedMixtureDataset
from training.data.tokenizer import Tokenizer
from training.settings import Settings

Batch = tuple[torch.Tensor, torch.Tensor, list[str]]


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
) -> DataLoader[Row]:
    """Loader over the weighted mixture of ``entries`` (a stage's ``train_data`` / ``val_data`` as resolved by
    `training.data.dataset_resolver`), yielding ``(input_ids, labels, data_ids)`` batches.

    Every entry becomes one `ParquetTextDataset` over its row range ``[skip_rows, skip_rows + max_rows)`` — the
    validation split decided by the resolver — with its ``data_signature`` (None = the text column). Loader state
    is not checkpointed: on resume, loaders are recreated fresh (as in the thesis runs). ``shard=(rank, world)`` is
    passed to every dataset; with ``world == 1`` it is a no-op.
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
    collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        block_size=block_size,
        padding_multiple=padding_multiple,
        ignore_index=ignore_index,
    )
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )


@dataclass
class StageDataloaders:
    """One train and one val loader per training stage, with lazily created and cycled train iterators."""

    train_loaders: Sequence[Iterable[Batch]]
    val_loaders: Sequence[Iterable[Batch]]
    _train_iterators: list[Iterator[Batch] | None] = field(init=False)

    def __post_init__(self) -> None:
        self._train_iterators = [None] * len(self.train_loaders)

    def next_train_batch(self, stage_idx: int) -> Batch:
        """Next batch of ``stage_idx``'s train loader; restarts the loader when it is exhausted."""
        iterator = self._train_iterators[stage_idx]
        if iterator is None:
            iterator = self._train_iterators[stage_idx] = iter(self.train_loaders[stage_idx])
        try:
            return next(iterator)
        except StopIteration:
            iterator = self._train_iterators[stage_idx] = iter(self.train_loaders[stage_idx])
            return next(iterator)


def build_stage_dataloaders(settings: Settings, dataset: ResolvedDataset, backend: Backend) -> StageDataloaders:
    """One train and one validation loader per stage of `dataset`, each mixing its entries with constant weights.

    The tokenizer is loaded once from `dataset.tokenizer_dir` and shared by every loader. Train loaders use
    `settings.dataloader_num_workers`, validation loaders read in-process. Loader seed `settings.seed + rank`,
    datasets sharded by `(rank, world_size)`.
    """
    tokenizer = Tokenizer(dataset.tokenizer_dir)

    def loader(entries: list[DataEntry], num_workers: int) -> Iterable[Batch]:
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
        )

    return StageDataloaders(
        train_loaders=[loader(stage.train_data, settings.dataloader_num_workers) for stage in dataset.stages],
        val_loaders=[loader(stage.val_data, 0) for stage in dataset.stages],
    )


def sample_stage_batch(
    stage_loaders: StageDataloaders,
    stage_idx: int,
    prev_stage_idx: int | None,
    transition_progress: float,
    rng: random.Random,
) -> Batch:
    """Batch for the current step: from ``stage_idx`` with probability ``transition_progress``, else from the
    previous stage. Outside a transition (``prev_stage_idx`` is None) always from ``stage_idx``."""
    if prev_stage_idx is not None and rng.random() >= transition_progress:
        return stage_loaders.next_train_batch(prev_stage_idx)
    return stage_loaders.next_train_batch(stage_idx)


def _fit(row: torch.Tensor, width: int, fill: int) -> torch.Tensor:
    """Slice or right-pad a 1-D row to `width`. Rows come from independently padded micro-batches, so a chunk can
    mix widths; positions past a row's supervised length are never trained on, so the fill value is irrelevant."""
    if row.shape[0] >= width:
        return row[:width]
    return torch.cat([row, row.new_full((width - row.shape[0],), fill)])


def length_sorted_batches(
    batches: Iterable[Batch],
    micro_batch_size: int,
    accumulation_steps: int,
    ignore_index: int = IGNORE_INDEX,
    padding_multiple: int | None = None,
) -> Iterator[Batch]:
    """Regroup every ``accumulation_steps`` micro-batches (one world batch) into micro-batches sorted by length.

    A sample's length is its number of supervised positions (``labels != ignore_index``); ``input_ids`` cannot be
    used because ``collate_fn`` has already replaced padding by EOS. Every re-grouped micro-batch is trimmed to its
    longest sample (rounded up to ``padding_multiple``), which is where the compute saving comes from. Positions
    beyond a sample's length carry only ignore-index labels, so the loss is unchanged.
    """
    buffer: list[Batch] = []

    def flush() -> Iterator[Batch]:
        samples: list[tuple[torch.Tensor, torch.Tensor, str, int]] = []
        for input_ids, labels, data_ids in buffer:
            for i in range(input_ids.shape[0]):
                samples.append((input_ids[i], labels[i], data_ids[i], int((labels[i] != ignore_index).sum())))
        samples.sort(key=lambda s: s[3])
        for start in range(0, len(samples), micro_batch_size):
            chunk = samples[start : start + micro_batch_size]
            width = max(1, max(s[3] for s in chunk))
            if padding_multiple:
                width = find_multiple(width, padding_multiple)
            width = min(width, max(s[0].shape[0] for s in chunk))
            yield (
                torch.stack([_fit(s[0], width, int(s[0][-1])) for s in chunk]),
                torch.stack([_fit(s[1], width, ignore_index) for s in chunk]),
                [s[2] for s in chunk],
            )
        buffer.clear()

    for batch in batches:
        buffer.append(batch)
        if len(buffer) == accumulation_steps:
            yield from flush()
    if buffer:
        yield from flush()
