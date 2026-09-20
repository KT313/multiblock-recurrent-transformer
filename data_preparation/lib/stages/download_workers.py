# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Token worker ownership and the ordered shutdown of a download pass."""

from __future__ import annotations

import functools
import queue
import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from types import TracebackType
from typing import Any

from data_preparation.lib.abort import BuildAborted, StopCheck
from data_preparation.lib.download_profile import active_profile, bind_profile, measure, profile_source
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.loaders import Row
from data_preparation.lib.stages.download_state import StoredRow, _Increment
from data_preparation.lib.storage.parquet import ShardWriter

TOKEN_QUEUE_DEPTH = 3  # bound token batches buffered ahead of the worker
PIPELINE_POLL_SECONDS = 0.02  # how often a worker with batches in flight (and room for more) checks the queue and the oldest batch
_Pending = tuple[_Increment, list[StoredRow], "Future[Any] | None"]  # a started batch: its future (None: nothing to wait for)
PASSIVE_SHARD_DIVISOR = 4  # reduce each passive language writer buffer


class _StopGate:
    """
    The stop check of one download pass (handed to every raw folder as its should_stop). A stop request trips it
    once, and from then on it is suspended: the rows the pass already consumed are stored without another abort
    on the way (each shard publish checks the stop again), and the pass ends after that.
    """

    def __init__(self, should_stop: StopCheck | None) -> None:
        self._should_stop = should_stop
        self.tripped = False
        self.suspended = False

    def __call__(self) -> bool:
        if self.suspended or self._should_stop is None:
            return False
        if self._should_stop():
            self.tripped = True
            return True
        return False

    def suspend(self) -> None:
        self.suspended = True


class _DownloadFailures:
    """Keep exception identity and tracebacks; a substantive error takes precedence over cooperative stop."""

    def __init__(self, gate: _StopGate) -> None:
        self._gate = gate
        self._errors: list[tuple[str, BaseException, TracebackType | None]] = []

    @property
    def primary(self) -> BaseException | None:
        if not self._errors:
            return None
        return next((error for _, error, _ in self._errors if not isinstance(error, BuildAborted)), self._errors[0][1])

    def record(self, phase: str, error: BaseException) -> None:
        self._gate.suspend()
        if not any(previous is error for _, previous, _ in self._errors):
            self._errors.append((phase, error, error.__traceback__))

    def attempt(self, phase: str, action: Callable[[], object]) -> None:
        try:
            action()
        except BaseException as error:  # noqa: BLE001  # propagated after subsequent safe, sequential cleanup
            self.record(phase, error)

    def raise_if_failed(self) -> None:
        primary = self.primary
        if primary is None:
            return
        primary_tb = None
        for phase, error, tb in self._errors:
            if error is primary:
                primary_tb = tb
                primary.add_note(f"Download failure during {phase}")
            else:
                detail = "".join(traceback.format_exception(type(error), error, tb))
                primary.add_note(f"Additional download failure during {phase}:\n{detail}")
        raise primary.with_traceback(primary_tb)


class _TokenWorker:
    """
    The tokenizing half of a download pass on its own thread: batches submitted by the fetch thread are started
    (:meth:`_TokenStep.start`), finished (:meth:`_TokenStep.finish`) and stored (:func:`_store`) in submission
    order, so row order, the per-row progress and the shard boundaries are exactly those of the same pass done in
    one thread. The queue holds :data:`TOKEN_QUEUE_DEPTH` batches: submit blocks the fetch thread when the worker
    is that far behind. in_flight batches are started before the oldest is finished: one in-process (start
    computes right away, so this is the plain sequential loop), one per tokenizer process plus one with a pool
    (tokenizer_pool.py), so every process has a batch while the worker stores the finished ones; with batches in
    flight and room for more, the worker polls the queue and the oldest batch every
    :data:`PIPELINE_POLL_SECONDS`.

    A failure on the worker is kept and re-raised on the fetch thread by the next :meth:`submit`, :meth:`drain`
    or :meth:`close` (:attr:`failed` tells earlier). After a tokenizer or write error the worker only settles what
    is queued or in flight without storing it; after :class:`BuildAborted` (the stop check, raised by a shard
    publish) it stores on: the gate is suspended by then (:func:`_store`), and every row the pass consumed belongs
    on disk so the folders' offsets stay the pass's frontier. A later substantive error takes precedence over that
    stop. close is what leaving the with block does: it joins before reporting any stored failure. The fetch owner
    verifies termination before writer cleanup; forced interruption cannot promise that cleanup is complete.
    """

    def __init__(self, name: str, writers: dict[str, ShardWriter], bar: Progress, gate: _StopGate, *, in_flight: int = 1) -> None:
        self._queue: queue.Queue[tuple[_Increment, list[StoredRow]] | None] = queue.Queue(maxsize=TOKEN_QUEUE_DEPTH)
        self._in_flight = max(1, in_flight)
        self._writers = writers
        self._bar = bar
        self._gate = gate
        self._failure: BaseException | None = None
        self._storing = True  # False after an error other than the stop: the queued batches are settled unstored
        self._raised = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=bind_profile(self._run), name=f"tokenize:{name}")
        self._thread.start()

    @property
    def failed(self) -> bool:
        return self._failure is not None

    @property
    def stopped(self) -> bool:
        """The worker released writer ownership and its thread was joined (also after a stored failure)."""

        return self._stopped.is_set() and not self._thread.is_alive()

    def submit(self, increment: _Increment, batch: list[StoredRow]) -> None:
        """
        Queue batch (nothing for an empty one) for increment; raises the worker's failure instead if it has one.
        """

        self._raise_failure()
        if not batch:
            return
        increment.submitted += len(batch)
        with measure("queue_put"):
            self._queue.put((increment, batch))

    def drain(self) -> None:
        """
        Wait until every submitted batch is stored (or dropped), then raise the worker's failure if it has one.
        """

        with measure("queue_drain"):
            self._queue.join()
        self._raise_failure()

    def close(self) -> None:
        """
        End the worker after the queued batches (or after settling them, once failed) and join it; raises the
        worker's failure if it was not raised before.
        """

        with measure("token_worker_shutdown"):
            self._queue.put(None)
            self._thread.join()
        self._raise_failure()

    def __enter__(self) -> _TokenWorker:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _raise_failure(self) -> None:
        if self._failure is not None and not self._raised:
            self._raised = True
            raise self._failure

    def _remember_failure(self, error: BaseException) -> None:
        previous = self._failure
        if previous is None:
            self._failure = error
        elif isinstance(previous, BuildAborted) and not isinstance(error, BuildAborted):
            error.add_note("Token worker previously stopped:\n" + "".join(traceback.format_exception(previous)))
            self._failure = error
            self._raised = False  # a stop already reported must not hide this later storage failure
        elif error is not previous:
            previous.add_note("Additional token worker failure:\n" + "".join(traceback.format_exception(error)))

    def _run(self) -> None:
        try:
            self._process()
        finally:
            self._stopped.set()  # no writer access after this point, even if join itself is interrupted

    def _process(self) -> None:
        pending: deque[_Pending] = deque()  # started batches, oldest first
        closed = False
        while pending or not closed:
            closed = self._fill(pending, closed)
            if pending:
                self._settle(pending.popleft())

    def _fill(self, pending: deque[_Pending], closed: bool) -> bool:
        """
        Start queued batches while the pipeline has room and the oldest started batch is not ready to store;
        returns whether the end marker was taken. With nothing in flight the queue is waited for, otherwise it
        is polled so a batch that finishes is stored promptly.
        """

        while not closed and len(pending) < self._in_flight:
            if pending and _ready(pending[0]):
                break
            try:
                with measure("pipeline_wait" if pending else "queue_get"):
                    item = self._queue.get(timeout=PIPELINE_POLL_SECONDS if pending else None)
            except queue.Empty:
                continue
            if item is None:
                return True
            pending.append(self._start(item))
        return closed

    def _start(self, item: tuple[_Increment, list[StoredRow]]) -> _Pending:
        increment, batch = item
        future: Future[Any] | None = None
        if self._storing:
            try:
                with profile_source(increment.name), measure("tokenize_batch"):
                    future = increment.token_step.start(batch)
            except BaseException as error:  # noqa: BLE001  # whatever it is, the fetch thread re-raises it
                self._remember_failure(error)
                self._storing = False
        return increment, batch, future

    def _settle(self, entry: _Pending) -> None:
        """
        Finish and store one started batch (the oldest), then settle it for the fetch thread.
        """

        increment, batch, future = entry
        try:
            if self._storing:
                try:
                    with profile_source(increment.name):
                        result = None
                        if future is not None:
                            with measure("pool_result_wait"):
                                result = future.result()
                        with measure("finish_batch"):
                            stored = increment.token_step.finish(batch, result)
                        try:
                            with measure("store_batch"):
                                _store(increment, self._writers[increment.name], stored, self._bar, self._gate)
                            profile = active_profile()
                            if profile is not None:
                                profile.record("stored_rows", 0, amount=len(stored))
                                profile.record("stored_tokens", 0, amount=sum(row["tokens"] for row, _ in stored))
                        finally:
                            del stored  # release on failures too; never retain across queue.get
                except BuildAborted as stop:  # the gate is suspended now: keep storing what is queued
                    self._remember_failure(stop)
                except BaseException as error:  # noqa: BLE001  # whatever it is, the fetch thread re-raises it
                    self._remember_failure(error)
                    self._storing = False
            increment.settled += len(batch)  # after the store: `kept` is up to date before the rows leave `in_flight`
        finally:
            self._queue.task_done()


def _ready(entry: _Pending) -> bool:
    future = entry[2]
    return future is None or future.done()


def _flush_partial_shards(writers: dict[str, ShardWriter], failures: _DownloadFailures) -> None:
    """
    After a stop or failure, attempt each buffered shard once in writer insertion order. All failures remain
    visible; callbacks may already have published/recorded a shard, so no failed flush is retried. The worker
    must have terminated before this phase begins. Final writer exits follow after every salvage attempt.
    """

    for name, writer in writers.items():
        failures.attempt(f"partial shard flush for {name}", writer.flush)


def _store(increment: _Increment, writer: ShardWriter, stored: list[StoredRow], bar: Progress, gate: _StopGate) -> None:
    """
    The rows the token step released, appended and counted as kept. A stop raised by a shard publish on the way
    (the shard is published and recorded before the check) suspends the gate, the rest of the batch is stored, and
    the stop is raised at the end.
    """

    stop: BuildAborted | None = None
    for row, row_progress in stored:
        try:
            increment.folder.add(writer, row, row_progress)
        except BuildAborted as error:
            stop = error
            gate.suspend()
        increment.counters.kept += 1
    credit = increment.credit(len(stored))
    if credit:
        bar.update(credit)  # once per batch: the dashboard bar takes a lock per update
    if stop is not None:
        raise stop


def finish_download_pass(
    increments: list[_Increment], rows: Iterator[tuple[str, Row]], writers: dict[str, ShardWriter],
    worker: _TokenWorker | None, failures: _DownloadFailures,
) -> None:
    """Settle, close, join, salvage, and release in order; only then establish exhaustion."""

    # Settle pending batches, close the reader, and establish worker termination.
    if worker is not None:
        active_worker = worker

        def settle_pending(increment: _Increment) -> None:
            active_worker.submit(increment, increment.token_step.take())

        for increment in increments:
            failures.attempt(f"pending settlement for {increment.name}", functools.partial(settle_pending, increment))
    close = getattr(rows, "close", None)
    if close is not None:
        failures.attempt("row iterator close", close)
    if worker is not None:
        failures.attempt("token worker close/join", worker.close)
        if not worker.stopped:
            failures.record(
                "token worker shutdown",
                RuntimeError("Token worker termination could not be established; writer cleanup is incomplete to avoid live worker access"),
            )
            failures.raise_if_failed()
            return

    # Salvage once after failures, then release every writer in insertion order.
    failure = failures.primary
    if failure is not None:
        _flush_partial_shards(writers, failures)
        failure = failures.primary  # salvage may reveal a substantive failure after cooperative cancellation
    exit_args = (type(failure), failure, failure.__traceback__) if failure is not None else (None, None, None)
    for name, writer in writers.items():
        failures.attempt(f"writer exit for {name}", functools.partial(writer.__exit__, *exit_args))  # flush on success; release only after salvage
    failures.raise_if_failed()

    # Establish exhaustion only after successful processing and cleanup.
    for increment in increments:
        if not increment.passive and increment.counters.kept < increment.rows_to_keep:
            increment.counters.exhausted = True  # only a successful pass can establish exhaustion


def open_increment_writer(
    name: str, increments: list[_Increment], increments_by_name: dict[str, _Increment],
    writers: dict[str, ShardWriter], shard_size: int,
) -> _Increment:
    """Find a planned or newly discovered source and open its writer on first use."""

    if name not in increments_by_name:
        increments_by_name.update((increment.name, increment) for increment in increments)
    increment = increments_by_name[name]
    if name not in writers:
        size = max(shard_size // PASSIVE_SHARD_DIVISOR, 1) if increment.passive else shard_size
        writers[name] = ShardWriter(increment.folder.directory, size, start_shard=increment.folder.shard_count, on_shard=increment.folder.record_shard).__enter__()
    return increment
