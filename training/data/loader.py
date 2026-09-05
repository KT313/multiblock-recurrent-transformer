# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Dataloader construction (one train loader per SOURCE for the whole run and one validation loader per stage,
`build_run_dataloaders`) and the assembly of one world batch into padded micro-batches.

The stage structure never touches the train loaders: `training.step.BatchStream` draws the source per sample with
the stage-interpolated weights, so a reader continues across stage boundaries and never re-reads rows.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset

from training.backend.base import Backend
from training.data.collate import (
    IGNORE_INDEX,
    Batch,
    Sample,
    WorkerBatch,
    collate_fn,
    collate_worker_batch,
    pad_and_shift,
)
from training.data.dataset_resolver import TRAIN_LOADER_NUM_WORKERS, DataEntry, ResolvedDataset
from training.data.datasets import ParquetTextDataset, Row, WeightedMixtureDataset
from training.data.tokenizer import Tokenizer
from training.settings import Settings

SampleBatch = list[Sample]  # the surviving tokenized rows of one worker batch (the `samples` half of a WorkerBatch)

# The worker batch of the unpadded train loaders: rows tokenized per worker batch and worker batches kept ready ahead
# (torch's `prefetch_factor`). Their product is how many tokenized rows a source has waiting when `BatchStream` pulls
# a whole world batch at the start of an optimizer step; 64 x 4 = 256 covers `world_batch_size` 256 drawn from one
# source, so the pull waits for no tokenization (with the old 4 x 4 = 16 the rest was tokenized while the GPU idled).
# Neither value touches the sample order or the numerics: the one worker (`TRAIN_LOADER_NUM_WORKERS`) walks its range
# in order and the stream concatenates its batches, `WorkerBatch.rows_read` counts the rows of any batch size and the
# samples pulled ahead travel in the checkpoint (`BatchStream.state_dict`). Fixed here, not settings: a `Settings`
# field is compared on resume, and this one may differ freely.
TRAIN_LOADER_BATCH_ROWS = 64
TRAIN_LOADER_PREFETCH_FACTOR = 4


def entry_dataset(entry: DataEntry, shard: tuple[int, int] = (0, 1)) -> ParquetTextDataset:
    """
    The `ParquetTextDataset` of one data entry: its folder, its row range [skip_rows, skip_rows + max_rows)
    (the validation split decided by the resolver) and its data_signature (None = the text column), dealt over
    shard=(rank, world) (a no-op with world == 1).
    """

    return ParquetTextDataset(
        entry.data_dir,
        entry.prefix,
        entry.data_signature,
        shard=shard,
        skip_rows=entry.skip_rows,
        max_rows=entry.max_rows,
    )


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
    generator: torch.Generator | None = None,
    worker_batch_rows: int | None = None,
) -> DataLoader[Row]:
    """
    Loader over entries: a single per-source train entry, or a stage's val_data mixed by weight
    (`training.data.dataset_resolver` resolves both); every entry becomes one `entry_dataset`, and the loader
    itself is `dataloader_over` the single dataset or the `WeightedMixtureDataset` of several.
    """

    if len({entry.prefix for entry in entries}) != len(entries):
        raise ValueError("Dataset prefixes within one loader must be unique.")
    datasets = [entry_dataset(entry, shard) for entry in entries]
    weights = [entry.weight for entry in entries]
    dataset: IterableDataset[Row] = (
        datasets[0] if len(datasets) == 1 else WeightedMixtureDataset(datasets, weights, seed)
    )
    return dataloader_over(
        dataset,
        tokenizer,
        block_size,
        micro_batch_size,
        num_workers=num_workers,
        padding_multiple=padding_multiple,
        ignore_index=ignore_index,
        pin_memory=pin_memory,
        padded=padded,
        generator=generator,
        worker_batch_rows=worker_batch_rows,
    )


def dataloader_over(
    dataset: IterableDataset[Row],
    tokenizer: Tokenizer,
    block_size: int,
    micro_batch_size: int,
    num_workers: int = 0,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    pin_memory: bool = False,
    padded: bool = True,
    generator: torch.Generator | None = None,
    worker_batch_rows: int | None = None,
) -> DataLoader[Row]:
    """
    The `DataLoader` over one dataset with the run's collate function.

    padded (validation, the default) yields ready (input_ids, labels, data_ids) batches of `micro_batch_size` rows;
    padded=False (training) yields a `WorkerBatch` of `worker_batch_rows` rows (None: `micro_batch_size`), padded
    later per micro-batch in `world_batch_micro_batches`, so a padded loader refuses `worker_batch_rows`. pin_memory
    is the backend's decision. generator is the source of the per-iterator base seed; without one, every `iter()`
    draws it from the global torch RNG. Workers keep `TRAIN_LOADER_PREFETCH_FACTOR` batches ready.
    """

    if worker_batch_rows is not None and worker_batch_rows <= 0:
        raise ValueError(f"worker_batch_rows must be positive, got {worker_batch_rows}")
    collate: Callable[[list[Row]], Any]
    if padded:
        if worker_batch_rows is not None:
            raise ValueError("worker_batch_rows applies to unpadded loaders only: a padded batch is the micro-batch")
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
        batch_size=micro_batch_size if worker_batch_rows is None else worker_batch_rows,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=TRAIN_LOADER_PREFETCH_FACTOR if num_workers > 0 else None,
        generator=generator,
    )


@dataclass
class RunDataloaders:
    """
    One train loader per SOURCE for the whole run and one validation loader per stage, with lazily created and
    cycled train iterators and the resume offsets they start at.

    train_loaders maps source name to loader in dataset-config order (`train_sources`). datasets are the
    parquet datasets behind the train loaders (a test fake passes {}): the offset of `set_resume_offsets` is
    applied to a dataset right before its first iterator after a resume and reset to 0 for every later epoch.
    """

    train_loaders: dict[str, Iterable[WorkerBatch]]
    val_loaders: Sequence[Iterable[Batch]]
    tokenizer: Tokenizer
    datasets: dict[str, ParquetTextDataset]
    pending_offsets: dict[str, int] = field(default_factory=dict, init=False)  # source -> rows its next epoch skips
    _train_iterators: dict[str, Iterator[WorkerBatch] | None] = field(init=False)

    def __post_init__(self) -> None:
        self._train_iterators = dict.fromkeys(self.train_loaders)

    @property
    def train_sources(self) -> list[str]:
        """
        The source names in dataset-config order.
        """

        return list(self.train_loaders)

    def _start_train_iterator(self, source: str) -> Iterator[WorkerBatch]:
        """
        A fresh iterator over source's loader, its dataset set to start at the pending resume offset (0 when
        none is pending).
        """

        offset = self.pending_offsets.pop(source, 0)
        dataset = self.datasets.get(source)
        if dataset is not None:
            dataset.set_resume_offset(offset)
        iterator = self._train_iterators[source] = iter(self.train_loaders[source])
        return iterator

    def next_train_batch(self, source: str) -> WorkerBatch:
        """
        Next worker batch of source's loader; restarts the loader when its epoch is over.

        The restart never spins on an empty range: setup guarantees at least one training row per source
        (`check_entry_rows`). The iterator is created lazily at the first pull and anew on every restart.
        """

        iterator = self._train_iterators[source]
        if iterator is None:
            iterator = self._start_train_iterator(source)
        try:
            return next(iterator)
        except StopIteration:
            return next(self._start_train_iterator(source))  # the epoch is over; the next one reads the whole range

    def set_resume_offsets(self, consumed_rows: Mapping[str, int]) -> None:
        """
        Start every train dataset named in `consumed_rows` that many rows into its range (modulo the range) at
        its next epoch; names of removed sources are ignored.
        """

        self.pending_offsets = {source: rows for source, rows in consumed_rows.items() if source in self.train_loaders}

    def close(self) -> None:
        """
        Shut down the worker processes of every live train iterator (idempotent); `train()` calls it on every way
        out. Without it a run that ends in an exception leaves its iterators in a reference cycle, and each worker's
        shutdown takes 5 s once the GC gets there. torch offers no public shutdown; `_shutdown_workers` is what the
        iterator's own finalizer runs.
        """

        for source, iterator in self._train_iterators.items():
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            self._train_iterators[source] = None


def build_run_dataloaders(settings: Settings, dataset: ResolvedDataset, backend: Backend) -> RunDataloaders:
    """
    The loaders of one run: one train loader per train source and one validation loader per stage (its entries
    mixed with constant weights), all sharing one tokenizer.

    Train loaders: one worker process each (`TRAIN_LOADER_NUM_WORKERS`) that tokenizes `TRAIN_LOADER_BATCH_ROWS`
    rows at a time and yields them unpadded as a `WorkerBatch`; `BatchStream` buffers those per source and pads
    per micro-batch, so the worker batch size is only a grouping and never changes the sample order. They do not
    pin memory: `pad_and_shift` copies the rows into a fresh pageable micro-batch anyway (see there for why it
    stays pageable). Validation loaders read in-process, yield padded batches that go to the device as they are,
    and pin them when the backend wants pinned memory.

    Iterator seeds come from one private generator, so creating an iterator (first pull, epoch restart, each
    evaluation) never touches the global torch RNG and a resume replays the same latent noise.
    """

    tokenizer = Tokenizer(dataset.tokenizer_dir)
    shard = (backend.rank, backend.world_size)
    generator = torch.Generator().manual_seed(settings.seed + backend.rank)  # the worker RNG is unused (no shuffle)
    train_datasets = {entry.prefix: entry_dataset(entry, shard) for entry in dataset.train_sources}
    train_loaders: dict[str, Iterable[WorkerBatch]] = {
        source: dataloader_over(
            parquet_dataset,
            tokenizer,
            block_size=settings.block_size,
            micro_batch_size=settings.micro_batch_size,
            num_workers=TRAIN_LOADER_NUM_WORKERS,
            padding_multiple=settings.sequence_padding_multiple,
            ignore_index=IGNORE_INDEX,
            pin_memory=False,
            padded=False,
            generator=generator,
            worker_batch_rows=TRAIN_LOADER_BATCH_ROWS,
        )
        for source, parquet_dataset in train_datasets.items()
    }
    val_loaders = [
        build_dataloader(
            stage.val_data,
            tokenizer,
            block_size=settings.block_size,
            micro_batch_size=settings.micro_batch_size,
            num_workers=0,
            seed=settings.seed + backend.rank,
            shard=shard,
            padding_multiple=settings.sequence_padding_multiple,
            ignore_index=IGNORE_INDEX,
            pin_memory=backend.pin_memory,
            padded=True,
            generator=generator,
        )
        for stage in dataset.stages
    ]
    return RunDataloaders(train_loaders, val_loaders, tokenizer, train_datasets)


def sample_length(sample: Sample) -> int:
    """
    Tokens of a sample before padding: what its micro-batch will have to be padded to.
    """

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
    """
    Split one world batch of samples into micro-batches of `micro_batch_size` and pad each of them once.

    With `sort_by_length` the samples are sorted by token count first (a stable sort), so a micro-batch groups rows
    of similar length and pads less. Without it the arrival order is kept. The width comes from the full sample
    length either way, never from the supervised part.
    """

    if sort_by_length:
        samples = sorted(samples, key=sample_length)
    return [
        pad_and_shift(samples[start : start + micro_batch_size], tokenizer, block_size, padding_multiple, ignore_index)
        for start in range(0, len(samples), micro_batch_size)
    ]
