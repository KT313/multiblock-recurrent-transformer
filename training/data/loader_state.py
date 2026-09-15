# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Run-owned loader iterators, resume offsets, and epoch accounting."""

import logging
import resource
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from training.data.collate import Batch, WorkerBatch
from training.data.datasets import ParquetTextDataset
from training.data.loader_options import TRAIN_LOADER_BATCH_ROWS, TRAIN_LOADER_PREFETCH_FACTOR
from training.data.tokenizer import Tokenizer

log = logging.getLogger("training.data.loader")


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
    `wait_seconds` per source; `take_wait_seconds` hands them out and resets. This measures loader retrieval time,
    including waiting for ready batches and retrieval overhead. It cannot distinguish storage, decompression,
    tokenization, collation, worker scheduling or inter-process transfer. The first pull of a fresh iterator
    (a source's first batch, every epoch restart) includes worker start-up (process spawn, tokenizer load) and is
    not counted: it would put nearly every run's first log interval over the warning threshold.
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
        fresh = iterator is None  # a fresh iterator's first pull includes worker start-up
        if iterator is None:
            iterator = self._start_train_iterator(source)
        while True:
            started = self.clock()
            batch = next(iterator, None)  # a sentinel, so the error below is not chained onto a StopIteration
            if not fresh:
                self.wait_seconds[source] = self.wait_seconds.get(source, 0.0) + (self.clock() - started)
            if batch is not None:
                self._count_batch(source, batch)
                return batch
            # the epoch is over and the next one starts at row 0, so the second round either yields or raises
            self._end_of_epoch(source)
            iterator = self._start_train_iterator(source)
            fresh = True

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
