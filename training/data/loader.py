# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Dataloader construction: one train loader per SOURCE for the whole run (unpadded worker batches the stream packs)
and one validation loader per stage (padded batches), `build_run_dataloaders`.

The stage structure never touches the train loaders: `training.step.BatchStream` draws the source per sample with
the stage-interpolated weights, so a reader continues across stage boundaries and never re-reads rows.
"""

import logging
import resource
import signal
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset

from training.backend.base import Backend
from training.data.collate import Batch, WorkerBatch, collate_fn, collate_worker_batch
from training.data.dataset_resolver import TRAIN_LOADER_NUM_WORKERS, DataEntry, ResolvedDataset
from training.data.datasets import ParquetTextDataset, Row, WeightedMixtureDataset
from training.data.tokenizer import IGNORE_INDEX, Tokenizer
from training.settings import Settings

# The worker batch of the unpadded train loaders: rows tokenized per worker batch and worker batches kept ready ahead
# (torch's `prefetch_factor`). Their product is how many tokenized rows a source has waiting when `BatchStream`
# refills its packing pool (`POOL_TOKEN_FACTOR` pack lengths of documents, drawn between two micro-batches); 64 x 4 =
# 256 rows covers a refill from one source, so the pull waits for no tokenization (with the old 4 x 4 = 16 the rest was
# tokenized while the GPU idled). Neither value touches the sample order or the numerics: the one worker
# (`TRAIN_LOADER_NUM_WORKERS`) walks its range in order and the stream concatenates its batches, `WorkerBatch.rows_read`
# counts the rows of any batch size and the samples pulled ahead travel in the checkpoint (`BatchStream.state_dict`).
# Fixed here, not settings: a `Settings` field is compared on resume, and this one may differ freely.
# File descriptors: torch shares a worker's tensors with the training process through one descriptor per tensor
# (its `file_descriptor` strategy), held until the batch has been received, so the batches in flight cost up to
# TRAIN_LOADER_BATCH_ROWS x TRAIN_LOADER_PREFETCH_FACTOR x 2 tensors per source of the `ulimit -n` budget. A worker
# that runs out drops the batch with a traceback on stderr and the loader skips it; `build_run_dataloaders` lifts the
# soft limit to the hard one (`raise_open_file_limit`) and `RunDataloaders` refuses an epoch that delivered fewer
# rows than the range holds, so the loss is an error and never silent.
TRAIN_LOADER_BATCH_ROWS = 64
TRAIN_LOADER_PREFETCH_FACTOR = 4
UNLIMITED_OPEN_FILES = 1 << 20  # the soft open-file limit `raise_open_file_limit` sets under an unlimited hard limit

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
    instead of finishing the step and checkpointing (`training.train.stop_on_interrupt`). Ignoring changes nothing
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


@dataclass
class EpochCounters:
    """
    What one source has read since its current iterator was started: rows read, samples that survived the collate
    and the dropped rows between them, plus whether the epoch began at the start of the range.

    full_epoch is False only for the first epoch after a resume, which starts at an offset and may legitimately
    end without a sample; `next_train_batch` refuses a FULL epoch without one. expected_rows is what the dataset
    says the epoch holds for this rank (`ParquetTextDataset.epoch_rows`; None without a dataset, for a test fake),
    and an epoch that ends with fewer rows read lost worker batches. warned is the run-wide flag of the single drop
    warning a source gets and is the one field an epoch reset leaves alone.
    """

    full_epoch: bool = True
    expected_rows: int | None = None
    rows_read: int = 0
    samples: int = 0
    dropped_rows: int = 0
    warned: bool = False

    def start_epoch(self, full_epoch: bool, expected_rows: int | None) -> None:
        """
        Begin counting a new epoch; `warned` survives, so a source warns about dropped rows once per run.
        """

        self.full_epoch = full_epoch
        self.expected_rows = expected_rows
        self.rows_read = self.samples = self.dropped_rows = 0


@dataclass
class RunDataloaders:
    """
    One train loader per SOURCE for the whole run and one validation loader per stage, with lazily created and
    cycled train iterators, the resume offsets they start at and the per-source counters of the running epoch.

    train_loaders maps source name to loader in dataset-config order (`train_sources`). datasets are the
    parquet datasets behind the train loaders (a test fake passes {}): the offset of `set_resume_offsets` is
    applied to a dataset right before its first iterator after a resume and reset to 0 for every later epoch.
    training_max_sequence_length is only quoted in the messages about dropped rows; a fake may leave it out.

    Data wait: `next_train_batch` times every pull from a loader (`clock`, monotonic) and adds the seconds to
    `wait_seconds` per source; `take_wait_seconds` hands them out and resets. A pull blocks only when the worker has
    no batch ready, so this is the time the training process (and every GPU behind it) waits for data. A source's
    first pull and every epoch restart include the worker start-up (process spawn, tokenizer load) and count too.
    """

    train_loaders: dict[str, Iterable[WorkerBatch]]
    val_loaders: Sequence[Iterable[Batch]]
    tokenizer: Tokenizer
    datasets: dict[str, ParquetTextDataset]
    training_max_sequence_length: int | None = None
    clock: Callable[[], float] = time.monotonic  # a test knob
    pending_offsets: dict[str, int] = field(default_factory=dict, init=False)  # source -> rows its next epoch skips
    wait_seconds: dict[str, float] = field(default_factory=dict, init=False)  # source -> seconds blocked since the last take
    _train_iterators: dict[str, Iterator[WorkerBatch] | None] = field(init=False)
    epochs: dict[str, EpochCounters] = field(init=False)  # source -> what its running epoch has read

    def __post_init__(self) -> None:
        self._train_iterators = dict.fromkeys(self.train_loaders)
        self.epochs = {source: EpochCounters() for source in self.train_loaders}

    @property
    def train_sources(self) -> list[str]:
        """
        The source names in dataset-config order.
        """

        return list(self.train_loaders)

    def _drop_cause(self) -> str:
        """
        Why the collate drops a row (`collate_samples`), for the warning and the error below.
        """

        window = "training_max_sequence_length + 1"
        if self.training_max_sequence_length is not None:
            window = f"{window} = {self.training_max_sequence_length + 1}"
        return (
            "the collate drops every row that keeps no supervised label: a row of a single token, or an instruct "
            f"row whose masked prompt alone fills the {window} tokens a row is cut to"
        )

    def _start_train_iterator(self, source: str) -> Iterator[WorkerBatch]:
        """
        A fresh iterator over source's loader, its dataset set to start at the pending resume offset (0 when
        none is pending), with the epoch counters reset to what that offset makes of the epoch.
        """

        offset = self.pending_offsets.pop(source, 0)
        dataset = self.datasets.get(source)
        loader = self.train_loaders[source]
        expected_rows: int | None = None
        if dataset is None:
            offset = 0  # no dataset to skip on (a test fake): the iterator reads its loader from the start
        else:
            dataset.set_resume_offset(offset)
            offset = dataset.resume_offset  # taken modulo the range, so a whole epoch of rows starts at 0 again
            expected_rows = dataset.epoch_rows(getattr(loader, "num_workers", 0))
        self.epochs[source].start_epoch(full_epoch=offset == 0, expected_rows=expected_rows)
        iterator = self._train_iterators[source] = iter(loader)
        return iterator

    def _count_batch(self, source: str, batch: WorkerBatch) -> None:
        """
        Book one worker batch on source's epoch counters, with one WARNING the first time the source drops a row:
        a run that trains on far fewer rows than the source holds is otherwise invisible.
        """

        counters = self.epochs[source]
        dropped = batch.rows_read - len(batch.samples)
        counters.rows_read += batch.rows_read
        counters.samples += len(batch.samples)
        counters.dropped_rows += dropped
        if dropped and not counters.warned:
            counters.warned = True
            log.warning(
                "%s: %d of %d rows dropped without reaching training; %s",
                source,
                dropped,
                batch.rows_read,
                self._drop_cause(),
            )

    def _end_of_epoch(self, source: str) -> None:
        """
        Close the finished epoch of source: refuse an epoch that delivered fewer rows than the dataset holds for
        this rank (the worker lost batches) or a full epoch that yielded no sample, log what it read otherwise.

        A lost batch would otherwise pass silently: torch's loader skips a batch whose worker failed to hand it over
        once the worker reports the end of the range, and the rows are neither trained on nor counted for a resume.
        A restart re-reads the same rows, so a source whose every row the collate drops would restart forever with
        a frozen step counter. `train()` calls `close()` on every way out, so either raise shuts the worker down.
        """

        counters = self.epochs[source]
        if counters.expected_rows is not None and counters.rows_read != counters.expected_rows:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            raise RuntimeError(
                f"train source {source!r}: its loader delivered {counters.rows_read} of the {counters.expected_rows} "
                "rows of this epoch, so the worker lost batches on the way (its tracebacks are above). The usual "
                "cause is 'Too many open files': every tensor a worker hands over holds one file descriptor until "
                f"the training process has received it, up to {TRAIN_LOADER_BATCH_ROWS * TRAIN_LOADER_PREFETCH_FACTOR * 2} "
                f"per source; this process may open {soft_limit} (hard limit {hard_limit}). Raise the hard limit "
                "(`ulimit -Hn`, or LimitNOFILE in the systemd unit), or lower TRAIN_LOADER_PREFETCH_FACTOR"
            )
        if counters.full_epoch and counters.samples == 0:
            raise RuntimeError(
                f"train source {source!r} yielded no usable sample in a full epoch over its {counters.rows_read} "
                f"rows: {self._drop_cause()}. Raise training_max_sequence_length or fix the source; reading the "
                "same rows again is all a restart could do and the run would never take a step"
            )
        finished = "%s: epoch done, %d rows read, %d samples, %d dropped"
        counts = (source, counters.rows_read, counters.samples, counters.dropped_rows)
        if counters.dropped_rows:
            log.info(finished, *counts)
        else:
            log.debug(finished, *counts)  # a small source finishes an epoch every few pulls, with nothing to report

    def next_train_batch(self, source: str) -> WorkerBatch:
        """
        Next worker batch of source's loader; restarts the loader when its epoch is over.

        Setup guarantees rows on disk (`check_entry_rows`), never usable samples: rows the collate drops are read
        but yield nothing, so `_end_of_epoch` raises on a full epoch without a sample instead of restarting. The
        iterator is created lazily at the first pull and anew on every restart.
        """

        iterator = self._train_iterators[source]
        if iterator is None:
            iterator = self._start_train_iterator(source)
        while True:
            started = self.clock()
            batch = next(iterator, None)  # a sentinel, so the error below is not chained onto a StopIteration
            self.wait_seconds[source] = self.wait_seconds.get(source, 0.0) + (self.clock() - started)
            if batch is not None:
                self._count_batch(source, batch)
                return batch
            # the epoch is over and the next one starts at row 0, so the second round either yields or raises
            self._end_of_epoch(source)
            iterator = self._start_train_iterator(source)

    def take_wait_seconds(self) -> dict[str, float]:
        """
        The seconds `next_train_batch` blocked per source since the last call (sources without a pull left out),
        and a reset of the counter; `train()` hands them to `RunLogger.log_step` after every step.
        """

        waited, self.wait_seconds = self.wait_seconds, {}
        return waited

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
    whole world (`training.step.RankBatches`); the other ranks get no train loader (and raise no file limit). The
    validation loaders exist on every rank, each reading its shard of the rows.
    """

    tokenizer = Tokenizer(dataset.tokenizer_dir)
    generator = torch.Generator().manual_seed(settings.seed + backend.rank)  # the worker RNG is unused (no shuffle)
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
    shard = (backend.rank, backend.world_size)
    val_loaders = [
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
    return RunDataloaders(train_loaders, val_loaders, tokenizer, train_datasets, settings.training_max_sequence_length)
