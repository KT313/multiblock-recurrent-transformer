# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded, ordered, pure key computation. Only the parent admits or publishes rows."""
from __future__ import annotations

import multiprocessing
from collections import deque
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import TracebackType
from typing import Any

from data_preparation.lib.abort import BuildAborted, StopCheck, check_stop
from data_preparation.lib.download_debug import initialize_worker_debug, measured_worker, worker_debug_options
from data_preparation.lib.download_profile import measure, profile_source
from data_preparation.lib.stages.global_dedup import global_key

Row = dict[str, Any]
DEFAULT_GLOBAL_HASH_WORKERS = 1


def check_global_hash_workers(workers: int) -> None:
    if type(workers) is not int or workers < 1:
        raise ValueError("global_hash_workers must be a positive integer")


def _key_fields(kind: str, rows: list[Row]) -> list[Row]:
    fields = {"pretrain": ("text",), "instruct": ("instruction", "input", "output"), "messages": ("messages",)}
    if kind not in fields:
        raise ValueError(f"unknown global key kind {kind!r}")
    # Missing required fields are diagnosed by global_key in the worker, in batch order.
    return [{key: row[key] for key in fields[kind] if key in row} for row in rows]


def _hash_batch(name: str, kind: str, rows: list[Row]) -> list[int]:
    with profile_source(name):
        return _compute_keys(kind, rows)


@measured_worker("global_hash_rows")
def _compute_keys(kind: str, rows: list[Row]) -> list[int]:
    return [global_key(kind, row) for row in rows]


class GlobalHashWorkers:
    """A lazy invocation-owned spawn pool; one worker selects the in-process path.

    The window limits logical batches, not bytes: serialization and normalization
    make extra copies, and document lengths vary. No pending work crosses sources.
    """

    def __init__(self, workers: int = DEFAULT_GLOBAL_HASH_WORKERS) -> None:
        check_global_hash_workers(workers)
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> GlobalHashWorkers:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close(exc)

    def close(self, failure: BaseException | None = None) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        try:
            with measure("global_hash_shutdown"):
                pool.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"Global hash worker cleanup also failed: {error!r}")

    def _submit(self, name: str, kind: str, rows: list[Row]) -> Future[list[int]]:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker_debug, initargs=(worker_debug_options(), "global-hash"),
            )
        return self._pool.submit(_hash_batch, name, kind, _key_fields(kind, rows))

    def batches(
        self, decoded: Iterator[list[Row]], name: str, kind: str, offset: int,
        should_stop: StopCheck | None = None,
    ) -> Generator[tuple[list[Row], list[int] | None], None, None]:
        """Consume in input order; callers close this generator on admission failure."""
        if self.workers == 1:
            for rows in decoded:
                check_stop(should_stop)
                yield rows, None
            return

        pending: deque[tuple[int, list[Row], Future[list[int]]]] = deque()
        exhausted = False
        reader_error: Exception | None = None
        try:
            while True:
                while not exhausted and len(pending) < self.workers:
                    check_stop(should_stop)
                    try:
                        rows = next(decoded)
                    except StopIteration:
                        exhausted = True
                        break
                    except BuildAborted:
                        raise
                    except Exception as error:  # earlier queued batches precede this reader failure
                        reader_error, exhausted = error, True
                        break
                    check_stop(should_stop)
                    try:
                        future = self._submit(name, kind, rows)
                    except BrokenProcessPool as error:
                        raise RuntimeError(f"{name}: global hash worker died at candidate offset {offset}") from error
                    pending.append((offset, rows, future))
                    offset += len(rows)
                if not pending:
                    if reader_error is not None:
                        raise reader_error
                    return
                batch_offset, rows, future = pending[0]
                try:
                    with measure("global_hash_wait"):
                        while True:
                            check_stop(should_stop)
                            try:
                                keys = future.result(timeout=0.1)
                                break
                            except TimeoutError:
                                # A task may itself raise TimeoutError; do not mistake it for a wait timeout.
                                if future.done():
                                    keys = future.result()
                                    break
                except BrokenProcessPool as error:
                    raise RuntimeError(f"{name}: global hash worker died at candidate offset {batch_offset}") from error
                except Exception as error:
                    error.add_note(f"Global hashing source={name}, candidate offset={batch_offset}")
                    raise
                check_stop(should_stop)
                yield rows, keys  # do not refill until the caller has admitted this batch
                pending.popleft()
                del rows, keys, future
        finally:
            for _, _, future in pending:
                future.cancel()
            pending.clear()
