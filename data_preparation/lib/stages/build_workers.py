# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Reading, filtering, hashing and writing the raw shards of a per-raw-shard pretrain build in worker processes.

Build jobs are threads of one process (`lib/build/runner.py`), and everything a pretrain row goes through (Arrow
decode, length and quality filter, the dedup key) holds the GIL: eight builds of a 64-core machine were measured at
1.1 busy cores together. The exact dedup is the only order-dependent step and needs only the hashes, so a
:class:`ShardWorkers` pool moves everything else out of the process and keeps the dedup, the statistics and the
manifest in the build thread; the text never crosses a process boundary:

1. `prepare(index, raw_path)` in a worker: the raw shard through `preprocess_batch` (length filter), the optional
   `check_quality` and `text_hash64` (`row_pipeline.py`, `exact_dedup.py`, the same functions as the in-thread
   path, in the same order); the surviving texts, clamped token counts and hashes stay in the worker, the hashes
   and the filter statistics come back (:class:`PreparedShard`).
2. The build thread runs the Bloom filter over the hashes (first occurrence wins) and asks the same worker to
   `write(index, keep, ...)` the kept rows as processed shards (`build_row_table` + `publish_shard`, the calls of
   `ProcessedOutput.publish`), then records them in the manifest exactly as before.

Raw shard k belongs to worker k % processes (each worker is a single-process executor, so a follow-up task lands
in the process that holds the rows). One prepare per worker is in flight; the next one is queued behind the
write, so a worker holds one shard's rows at a time and works on while the build thread waits for the write and
saves the manifest. Nothing is written until the build thread asks, so a stop or a failure still loses at most
one raw shard, as in the in-thread path; queued prepares are cancelled at shutdown. Spawn, not fork, for the
reason given at `build.Decontaminator`; a killed worker surfaces as a named RuntimeError. The workers report
into the `--debug` overview as `source=<name>/build`.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pyarrow.parquet as pq

from data_preparation.lib.download_debug import WorkerDebugOptions, initialize_worker_debug, measured_worker, worker_debug_options
from data_preparation.lib.download_profile import measure
from data_preparation.lib.stages.exact_dedup import text_hash64
from data_preparation.lib.stages.row_pipeline import check_quality, preprocess_batch
from data_preparation.lib.storage.parquet import build_row_table, publish_shard, shard_name

T = TypeVar("T")
WrittenShard = tuple[str, int, int]  # shard file name, rows, tokens: the arguments of Manifest.add_shard


@dataclass(frozen=True)
class ShardSource:
    """
    The per-source parameters of the row pipeline a worker needs (picklable, sent with every task).
    """

    name: str
    text_field: str
    min_chars: int
    quality_filter: bool
    normalize: bool  # dedup.normalize: hash the normalized text
    max_tokens: int  # dataset_max_sequence_length: stored counts are clamped to it
    batch_size: int  # rows per Arrow batch, as the in-thread path reads them


@dataclass(frozen=True)
class PreparedShard:
    """
    What comes back from a prepared raw shard: the dedup keys of the rows that passed the filters (in row order)
    and the filter statistics of the shard, in the shape of `stats["length_filter"]` and `stats["quality_filter"]`
    (without its `enabled` flag).
    """

    hashes: list[int]
    length_filter: dict[str, int]
    quality_filter: dict[str, object]


class ShardWorkers:
    """
    The worker processes of one per-raw-shard build (a with block); see the module docstring.
    """

    def __init__(self, processes: int, source: ShardSource) -> None:
        if processes < 1:
            raise ValueError(f"shard workers must be >= 1, got {processes}")
        self.processes = processes
        self.source = source
        self._pools: list[ProcessPoolExecutor] = []
        self._prepared: dict[int, Future[PreparedShard]] = {}

    def __enter__(self) -> ShardWorkers:
        context = multiprocessing.get_context("spawn")
        debug = worker_debug_options()
        self._pools = [
            ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_init_build_worker, initargs=(debug,))
            for _ in range(self.processes)
        ]
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        with measure("worker_shutdown"):
            for pool in self._pools:
                pool.shutdown(cancel_futures=True)  # a build that stopped early does not wait for queued shards
        self._pools = []
        self._prepared.clear()

    def prepare(self, index: int, raw_path: Path) -> None:
        """
        Queue raw shard index (its file raw_path) on its worker.
        """

        self._prepared[index] = self._pool(index).submit(_prepare_shard, index, str(raw_path), self.source)

    def prepared(self, index: int) -> PreparedShard:
        """
        Wait for raw shard index, queued by :meth:`prepare` earlier.
        """

        with measure("worker_result_wait"):
            return _result(self._prepared.pop(index))

    def write(
        self, index: int, keep: list[int], directory: Path, first_shard: int, shard_size: int, *, next_raw_path: Path | None,
    ) -> list[WrittenShard]:
        """
        Publish the rows keep (indices into the prepared rows of raw shard index) as processed shards of at most
        shard_size rows named from first_shard on in directory, and return them for the manifest. next_raw_path,
        if given, is raw shard index + processes, queued on the same worker behind the write.
        """

        pool = self._pool(index)
        future = pool.submit(_write_shard, index, keep, str(directory), first_shard, shard_size, self.source.name)
        if next_raw_path is not None:
            self.prepare(index + self.processes, next_raw_path)
        with measure("worker_result_wait"):
            return _result(future)

    def _pool(self, index: int) -> ProcessPoolExecutor:
        if not self._pools:
            raise RuntimeError("the shard workers are not open (use them inside their with block)")
        return self._pools[index % self.processes]


def _result(future: Future[T]) -> T:
    try:
        return future.result()
    except BrokenProcessPool as error:
        raise RuntimeError(f"a build worker died (killed by the OOM killer, or crashed): {error}") from error


# --- in the workers ------------------------------------------------------------------------------------------------------

_HELD: dict[int, tuple[list[str], list[int], list[int]]] = {}  # raw shard index -> (texts, tokens, hashes) of its survivors


def _init_build_worker(debug: WorkerDebugOptions | None) -> None:
    initialize_worker_debug(debug, "build")


@measured_worker("build_prepare_shard")
def _prepare_shard(index: int, raw_path: str, source: ShardSource) -> PreparedShard:
    texts: list[str] = []
    tokens: list[int] = []
    hashes: list[int] = []
    length_filter = {"input_samples": 0, "removed_too_short": 0, "removed_invalid": 0, "output_samples": 0}
    rejection_reasons: dict[str, int] = {}
    parquet = pq.ParquetFile(raw_path)
    for batch in parquet.iter_batches(batch_size=source.batch_size, columns=[source.text_field, "tokens"]):
        kept, batch_stats = preprocess_batch(batch, source.text_field, source.name, source.min_chars)
        for key, value in batch_stats.items():
            length_filter[key] += value
        for row in kept:
            text: str = row["text"]
            if source.quality_filter:
                passes, reason = check_quality(text)
                if not passes:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    continue
            texts.append(text)
            tokens.append(min(int(row["tokens"]), source.max_tokens))
            hashes.append(text_hash64(text, source.normalize))
    _HELD[index] = (texts, tokens, hashes)
    quality_filter = {"filtered_count": sum(rejection_reasons.values()), "rejection_reasons": rejection_reasons}
    return PreparedShard(hashes, length_filter, quality_filter)


@measured_worker("build_write_shard")
def _write_shard(index: int, keep: list[int], directory: str, first_shard: int, shard_size: int, source_name: str) -> list[WrittenShard]:
    texts, tokens, hashes = _HELD.pop(index)
    written: list[WrittenShard] = []
    for number, start in enumerate(range(0, len(keep), shard_size)):
        chunk = keep[start : start + shard_size]
        rows = [{"text": texts[i], "source": source_name, "tokens": tokens[i], "hash": hashes[i]} for i in chunk]
        path = publish_shard(build_row_table(rows), Path(directory) / shard_name(first_shard + number))
        written.append((path.name, len(chunk), sum(tokens[i] for i in chunk)))
    return written
