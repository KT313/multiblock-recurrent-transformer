# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Dataloader construction (one train loader per SOURCE for the whole run and one validation loader per stage,
`build_run_dataloaders`) and the assembly of one world batch into padded micro-batches.

The stage structure never touches the train loaders: which source a sample comes from is drawn per sample in
`training.step.BatchStream` with the stage-interpolated weights (`StageManager.data_weights`), so a reader simply
continues across stage boundaries and consecutive stages sharing a source never re-read its rows.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

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


def entry_dataset(entry: DataEntry, shard: tuple[int, int] = (0, 1)) -> ParquetTextDataset:
    """The `ParquetTextDataset` of one data entry: its folder, its row range ``[skip_rows, skip_rows + max_rows)``
    (the validation split decided by the resolver) and its ``data_signature`` (None = the text column), dealt over
    ``shard=(rank, world)`` (a no-op with ``world == 1``)."""
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
) -> DataLoader[Row]:
    """Loader over ``entries``: a single per-source train entry, or a stage's ``val_data`` mixed by weight
    (`training.data.dataset_resolver` resolves both); every entry becomes one `entry_dataset`, and the loader
    itself is `dataloader_over` the single dataset or the `WeightedMixtureDataset` of several."""
    if len({e.prefix for e in entries}) != len(entries):
        raise ValueError("Dataset prefixes within one loader must be unique.")
    datasets = [entry_dataset(e, shard) for e in entries]
    dataset: IterableDataset[Row] = (
        datasets[0] if len(datasets) == 1 else WeightedMixtureDataset(datasets, [e.weight for e in entries], seed)
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
) -> DataLoader[Row]:
    """The `DataLoader` over one dataset with the run's collate function.

    ``padded`` (the validation loaders, and the default) yields ready ``(input_ids, labels, data_ids)`` batches;
    ``padded=False`` (the training loaders) yields a `WorkerBatch` — the unpadded `Sample` list of the batch plus
    the count of rows read to produce it (dropped rows included) — the workers still do the tokenization, the
    padding happens once per assembled micro-batch in `world_batch_micro_batches`. ``pin_memory`` is the backend's
    decision (`Backend.pin_memory`, true on CUDA).
    """
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
class RunDataloaders:
    """One train loader per SOURCE for the whole run and one validation loader per stage, with lazily created and
    cycled train iterators and the resume offsets those iterators start at.

    ``train_loaders`` maps source name to loader in dataset-config order — the deterministic order the stream draws
    over (`train_sources`). Train loaders yield unpadded `WorkerBatch`es (surviving samples + rows read), validation
    loaders padded `Batch`es; `tokenizer` is the one every loader was built with and the one
    `world_batch_micro_batches` pads with. ``datasets`` are the parquet datasets behind the train loaders (a test
    fake without one passes ``{}``): the container sets a dataset's resume offset right before it creates an
    iterator over its loader — the pending offset of `set_resume_offsets` for the first epoch after a resume, 0 for
    every later epoch — so no offset outlives the epoch it was meant for (`training.step.BatchStream` owns the
    row bookkeeping).
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
        """The source names in dataset-config order."""
        return list(self.train_loaders)

    def _start_train_iterator(self, source: str) -> Iterator[WorkerBatch]:
        """A fresh iterator over ``source``'s loader, its dataset set to start at the pending resume offset (0 when
        none is pending: every epoch after the first reads the whole range). Numerics: each `iter(DataLoader)`
        draws one base seed from the global torch RNG."""
        offset = self.pending_offsets.pop(source, 0)
        dataset = self.datasets.get(source)
        if dataset is not None:
            dataset.set_resume_offset(offset)
        iterator = self._train_iterators[source] = iter(self.train_loaders[source])
        return iterator

    def next_train_batch(self, source: str) -> WorkerBatch:
        """Next worker batch of ``source``'s loader; restarts the loader when its epoch is over.

        The restart can never spin on an empty range: setup guarantees at least one training row per source
        (`check_entry_rows` in the resolver). Numerics: the iterator is created lazily at the first pull (and
        anew on every restart), and each `iter(DataLoader)` draws one base seed from the global torch RNG.
        """
        iterator = self._train_iterators[source]
        if iterator is None:
            iterator = self._start_train_iterator(source)
        try:
            return next(iterator)
        except StopIteration:
            return next(self._start_train_iterator(source))  # the epoch is over; the next one reads the whole range

    def set_resume_offsets(self, consumed_rows: Mapping[str, int]) -> None:
        """Start every train dataset whose source appears in `consumed_rows` that many rows into its range (modulo
        the range) at its next epoch, so a resumed run does not train on the rows the interrupted run already saw;
        names of removed sources are ignored."""
        self.pending_offsets = {source: rows for source, rows in consumed_rows.items() if source in self.train_loaders}

    def close(self) -> None:
        """Shut down the worker processes of every live train iterator and forget the iterators (idempotent).
        Validation loaders read in-process and no iterator over them is kept here.

        `train()` calls it on every way out of a run. A run that ends in an exception otherwise leaves its
        multi-process iterators alive in a reference cycle (traceback -> frame -> loaders); when the cyclic GC
        later collects one, the worker never receives its stop message and the shutdown takes 5 s per worker.
        torch offers no public shutdown: `_shutdown_workers` is what the iterator's own finalizer runs.
        """
        for source, iterator in self._train_iterators.items():
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            self._train_iterators[source] = None


def build_run_dataloaders(settings: Settings, dataset: ResolvedDataset, backend: Backend) -> RunDataloaders:
    """One train loader per train source of `dataset` (the whole-run readers) and one validation loader per stage
    (mixing its entries with constant weights).

    The tokenizer is loaded once from `dataset.tokenizer_dir` and shared by every loader. Train loaders run
    `TRAIN_LOADER_NUM_WORKERS` (= 1) worker each and yield unpadded samples; validation loaders read in-process
    and yield padded batches. Loader seed `settings.seed + rank`, datasets sharded by `(rank, world_size)`.
    """
    tokenizer = Tokenizer(dataset.tokenizer_dir)
    shard = (backend.rank, backend.world_size)
    train_datasets = {entry.prefix: entry_dataset(entry, shard) for entry in dataset.train_sources}
    train_loaders: dict[str, Iterable[WorkerBatch]] = {
        source: dataloader_over(
            parquet,
            tokenizer,
            block_size=settings.block_size,
            micro_batch_size=settings.micro_batch_size,
            num_workers=TRAIN_LOADER_NUM_WORKERS,
            padding_multiple=settings.sequence_padding_multiple,
            ignore_index=IGNORE_INDEX,
            pin_memory=backend.pin_memory,
            padded=False,
        )
        for source, parquet in train_datasets.items()
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
        )
        for stage in dataset.stages
    ]
    return RunDataloaders(train_loaders, val_loaders, tokenizer, train_datasets)


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
