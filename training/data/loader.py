# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Dataloader construction: one train loader per SOURCE for the whole run (unpadded worker batches the stream packs)
and one validation loader per stage (padded batches), `build_run_dataloaders`.

The stage structure never touches the train loaders: `training.steps.BatchStream` draws the source per sample with
the stage-interpolated weights, so a reader continues across stage boundaries and never re-reads rows.
"""

import logging
import resource
import signal
from functools import partial
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset

from training.backend.base import Backend
from training.data.collate import WorkerBatch, collate_fn, collate_worker_batch
from training.data.dataset_resolver import TRAIN_LOADER_NUM_WORKERS, DataEntry, ResolvedDataset
from training.data.datasets import ParquetTextDataset, Row, WeightedMixtureDataset
from training.data.loader_options import (
    TRAIN_LOADER_BATCH_ROWS, TRAIN_LOADER_PREFETCH_FACTOR,
    UNLIMITED_OPEN_FILES,
)
from training.data.loader_state import EpochCounters, RunDataloaders
from training.data.tokenizer import IGNORE_INDEX, Tokenizer
from training.settings import Settings

__all__ = [
    "EpochCounters", "RunDataloaders", "TRAIN_LOADER_BATCH_ROWS", "TRAIN_LOADER_PREFETCH_FACTOR", "UNLIMITED_OPEN_FILES",
    "build_dataloader", "build_run_dataloaders", "build_train_loaders", "build_validation_loaders", "dataloader_over",
    "entry_dataset", "raise_open_file_limit", "worker_init_fn",
]

log = logging.getLogger(__name__)


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
    training_max_sequence_length: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 1337,
    shard: tuple[int, int] = (0, 1),
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    pin_memory: bool = False,
    padded: bool = True,
    generator: torch.Generator | None = None,
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
        training_max_sequence_length,
        batch_size,
        num_workers=num_workers,
        padding_multiple=padding_multiple,
        ignore_index=ignore_index,
        pin_memory=pin_memory,
        padded=padded,
        generator=generator,
    )


def worker_init_fn(worker_id: int) -> None:
    """
    A DataLoader worker ignores SIGINT. A terminal Ctrl-C signals the whole process group; a worker that took it
    would exit on the KeyboardInterrupt and the parent would fail with "DataLoader worker exited unexpectedly"
    instead of finishing the step and checkpointing (`training.cli.stop_on_interrupt`). Ignoring changes nothing
    about a worker's lifetime: the parent ends it by message (a sentinel once its iterator is dropped) and torch's
    watchdog ends it when the parent dies. SIGTERM keeps its default: torch's shutdown fallback and the interpreter's
    exit both end a leftover worker with it, and a worker that ignored it would hang the parent's exit.
    """

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def dataloader_over(
    dataset: IterableDataset[Row],
    tokenizer: Tokenizer,
    training_max_sequence_length: int,
    batch_size: int,
    num_workers: int = 0,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    pin_memory: bool = False,
    padded: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader[Row]:
    """
    The `DataLoader` over one dataset with the run's collate function, `batch_size` rows per batch.

    padded (validation, the default) yields ready (input_ids, labels, data_ids) batches; padded=False (training)
    yields a `WorkerBatch` of unpadded samples that `BatchStream` packs later, so `padding_multiple` and
    `ignore_index` do not apply to it. pin_memory is the backend's decision. generator is the source of the
    per-iterator base seed; without one, every `iter()` draws it from the global torch RNG. Workers keep
    `TRAIN_LOADER_PREFETCH_FACTOR` batches ready.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    collate: Callable[[list[Row]], Any]
    if padded:
        collate = partial(
            collate_fn,
            tokenizer=tokenizer,
            training_max_sequence_length=training_max_sequence_length,
            padding_multiple=padding_multiple,
            ignore_index=ignore_index,
        )
    else:
        collate = partial(collate_worker_batch, tokenizer=tokenizer, training_max_sequence_length=training_max_sequence_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=TRAIN_LOADER_PREFETCH_FACTOR if num_workers > 0 else None,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def raise_open_file_limit() -> int:
    """
    Lift this process's soft limit on open file descriptors to its hard limit and return the limit in force.

    The default soft limit of 1024 is below what the train loaders' batches in flight can hold (see the note at
    `TRAIN_LOADER_BATCH_ROWS`); the hard limit is the administrator's ceiling and needs no privilege to reach (an
    unlimited hard limit gets `UNLIMITED_OPEN_FILES`). Worker processes inherit the raised limit.
    """

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = max(soft_limit, UNLIMITED_OPEN_FILES) if hard_limit == resource.RLIM_INFINITY else hard_limit
    if soft_limit != target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard_limit))
        log.debug("open file limit raised from %d to %d for the train loader workers", soft_limit, target)
    return target


def build_run_dataloaders(settings: Settings, dataset: ResolvedDataset, backend: Backend) -> RunDataloaders:
    """
    The loaders of one run: one train loader per train source and one validation loader per stage (its entries
    mixed with constant weights), all sharing one tokenizer.

    Train loaders: one worker process each (`TRAIN_LOADER_NUM_WORKERS`) that tokenizes `TRAIN_LOADER_BATCH_ROWS`
    rows at a time and yields them unpadded as a `WorkerBatch`; `BatchStream` buffers those per source and packs
    per micro-batch, so the worker batch size is only a grouping and never changes the sample order. They do not
    pin memory: `pack_samples` copies the rows into a fresh pageable micro-batch anyway (see `pad_and_shift` for
    why it stays pageable). Validation loaders read in-process, yield padded batches of `validation_batch_size`
    rows that go to the device as they are, and pin them when the backend wants pinned memory.

    Iterator seeds come from one private generator, so creating an iterator (first pull, epoch restart, each
    evaluation) never touches the global torch RNG and a resume replays the same latent noise.

    Ranks: the train loaders exist on the main rank only, which reads every source as ONE shard and packs for the
    whole world (`training.steps.RankBatches`); the other ranks get no train loader (and raise no file limit). The
    validation loaders exist on every rank, each reading its shard of the rows.
    """

    # create the shared tokenizer and private iterator seed stream
    tokenizer = Tokenizer(dataset.tokenizer_dir)
    generator = torch.Generator().manual_seed(settings.seed + backend.rank)  # the worker RNG is unused (no shuffle)

    # build main-rank training readers and rank-sharded validation readers
    train_datasets, train_loaders = build_train_loaders(settings, dataset, backend, tokenizer, generator)
    val_loaders = build_validation_loaders(settings, dataset, backend, tokenizer, generator)
    return RunDataloaders(train_loaders, val_loaders, tokenizer, train_datasets, settings.training_max_sequence_length)


def build_train_loaders(
    settings: Settings, dataset: ResolvedDataset, backend: Backend, tokenizer: Tokenizer, generator: torch.Generator,
) -> tuple[dict[str, ParquetTextDataset], dict[str, Iterable[WorkerBatch]]]:
    """Create unsharded source readers only on the rank that packs training microbatches."""

    train_datasets: dict[str, ParquetTextDataset] = {}
    train_loaders: dict[str, Iterable[WorkerBatch]] = {}
    if backend.is_main:
        if TRAIN_LOADER_NUM_WORKERS > 0:
            raise_open_file_limit()
        train_datasets = {entry.prefix: entry_dataset(entry) for entry in dataset.train_sources}  # one shard
        train_loaders = {
            source: dataloader_over(
                parquet_dataset,
                tokenizer,
                training_max_sequence_length=settings.training_max_sequence_length,
                batch_size=TRAIN_LOADER_BATCH_ROWS,
                num_workers=TRAIN_LOADER_NUM_WORKERS,
                pin_memory=False,
                padded=False,
                generator=generator,
            )
            for source, parquet_dataset in train_datasets.items()
        }
    return train_datasets, train_loaders


def build_validation_loaders(
    settings: Settings, dataset: ResolvedDataset, backend: Backend, tokenizer: Tokenizer, generator: torch.Generator,
) -> list[DataLoader[Row]]:
    """Create each stage's finite validation mixture over this rank's row shard."""

    shard = (backend.rank, backend.world_size)
    return [
        build_dataloader(
            stage.val_data,
            tokenizer,
            training_max_sequence_length=settings.training_max_sequence_length,
            batch_size=settings.validation_batch_size,
            num_workers=0,
            seed=settings.seed + backend.rank,
            shard=shard,
            padding_multiple=settings.validation_padding_multiple,
            ignore_index=IGNORE_INDEX,
            pin_memory=backend.pin_memory,
            padded=True,
            generator=generator,
        )
        for stage in dataset.stages
    ]
