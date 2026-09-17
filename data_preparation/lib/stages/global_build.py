# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset-scoped final output from reusable, source-local processed candidates.

The source manifest is the atomic commit record: output shards are published first,
then their list, the global frontier and candidate/dependency generations in one
manifest replacement. Unlisted shards never refill the Bloom filter. Every writer
runs under the caller's exclusive dataset lease; readers use completed snapshots.

The admission of one source is a pipeline of three threads with the same commits in the same order as a plain
loop: a reader decodes the candidate batches two ahead (:class:`_Reader`; parquet decode releases the GIL), the
build thread runs `GlobalAdmission.commit_batch` (the Bloom pass, which holds the GIL), and a writer publishes
each batch's shard and manifest (:class:`_Writer`, at most two commits behind the admission). A shard file is
published before the manifest that lists it, and the frontier in the manifest equals the rows on disk, at every
commit; a failed commit poisons the admission at its next batch and leaves the manifest at the last complete
commit, the state :func:`restore_output_generation` and :func:`source_frontier` resume from.
"""
from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterable, Iterator
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Any


from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.download_profile import bind_profile, measure
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.global_dedup import GlobalAdmission, GlobalFrontier, PublishBatch, ordered_sources
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import guarded_path
from data_preparation.lib.storage.parquet import build_row_table, publish_shard, shard_name

from data_preparation.lib.stages.global_output import (
    GLOBAL_BATCH_ROWS, candidate_rows, committed_keys,
    outputs_complete as outputs_complete, source_frontier as source_frontier,  # noqa: PLC0414 - compatibility exports
    build_generation_identity, inspect_owned_output, load_completed_candidates, restore_output_generation,
)


log = get_logger(__name__)
Row = dict[str, Any]
READ_AHEAD_BATCHES = 2  # candidate batches decoded ahead of the admission


def build_global_source(
    config: DatasetConfig, name: str, layout: DatasetLayout, start: GlobalFrontier,
    *, rows_target: int, exhausted: bool, should_stop: StopCheck | None = None,
    batch_rows: int = GLOBAL_BATCH_ROWS, preseed_keys: Iterable[int] = (),
) -> tuple[GlobalFrontier, bool]:
    """Admit one source; return its frontier and whether budget/exhaustion finalized it.

    A changed candidate generation replays this source. Dependency generations force
    downstream replay even if an earlier replacement happened to retain the same keys.
    A matching partial generation resumes behind exactly its committed candidate offset.
    """

    # validate the source priority and guard all output paths
    if batch_rows < 1:
        raise ValueError("global batch_rows must be positive")
    order = ordered_sources(config)
    if start.source_index >= len(order) or order[start.source_index] != name:
        raise ValueError(f"{name}: global source priority does not match starting frontier")
    prior_names = order[:start.source_index]
    directory = guarded_path(layout.root, layout.processed_dir(name))
    for filename in ("MANIFEST.json", "MANIFEST.json.tmp"):
        guarded_path(layout.root, directory / filename)
    candidates_dir = DatasetLayout(layout.root).processed_dir(name)
    candidates = load_completed_candidates(config, name, layout, candidates_dir)
    identity = build_generation_identity(config, name, layout, start, prior_names, candidates)

    # inspect ownership and restore the matching committed generation
    manifest = inspect_owned_output(name, directory, identity, candidates, start)
    manifest, completed_frontier = restore_output_generation(
        config, name, directory, start, candidates, identity, manifest, rows_target=rows_target, exhausted=exhausted,
    )
    if completed_frontier is not None:
        return completed_frontier, True

    # rebuild admission from committed keys before recovering or extending the frontier
    frontier = source_frontier(manifest)
    own_keys = committed_keys(layout, (name,))
    admission = GlobalAdmission(
        order, memory_mb=config.bloom_dedup_memory_mb, frontier=frontier,
        committed_keys=chain(committed_keys(layout, prior_names), own_keys), preseed_keys=preseed_keys,
    )
    admission.check_capacity(start.retained + candidates.rows())
    if frontier.source_index == start.source_index + 1:
        manifest.complete_generation(directory)  # the final frontier committed before a crash in completion
        return frontier, True

    commit = partial(commit_global_batch, layout, directory, manifest, start)
    kind = "messages" if config.sources[name].instruction_format == "messages" else config.sources[name].kind

    # admit candidate batches in their existing order: decoded ahead by the reader, committed behind by the writer
    rows = candidate_rows(candidates_dir, candidates, frontier.source_candidates)
    batches = iter(lambda: list(islice(rows, batch_rows)), [])
    with _Writer(commit) as writer:
        with _Reader(batches) as decoded:
            for batch in decoded:
                check_stop(should_stop)
                admission.commit_batch(name, kind, batch, writer.publish)
        check_stop(should_stop)
        writer.flush()  # the manifest holds every commit before its rows are counted

        # finalize the source budget
        complete = manifest.rows() >= rows_target or exhausted
        if complete:
            admission.finish_source(name, writer.publish)

    # publish filter statistics
    manifest.stats["global_filter"] = admission.statistics()
    manifest.stats["budget_shortfall"] = max(0, rows_target - manifest.rows())
    if complete:
        manifest.complete_generation(directory)
    else:
        manifest.save(directory)
    log.info("%s: global Bloom admission %s; retained %d/%d; shortfall %d", name,
             manifest.stats["global_dedup"], manifest.rows(), rows_target, manifest.stats["budget_shortfall"])
    metrics = manifest.stats["global_filter"]
    log.info("%s: global Bloom filter: %d MiB, %.3fx nominal load, measured FPR %.6f%%", name,
             metrics["memory_mb"], metrics["load"], metrics["measured_false_positive_rate"] * 100)
    check_stop(should_stop)
    return admission.frontier, complete


class _Reader:
    """
    The candidate batches decoded on their own thread, :data:`READ_AHEAD_BATCHES` ahead of the consumer. The with
    block starts the thread and yields the batches; an error of the reader is raised to the consumer in its turn;
    leaving the block (an error, a stop) lets a blocked reader go and joins it.
    """

    def __init__(self, batches: Iterator[list[Row]]) -> None:
        self._batches = batches
        self._queue: queue.Queue[list[Row] | BaseException | None] = queue.Queue(maxsize=READ_AHEAD_BATCHES)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=bind_profile(self._run), name="global-reader", daemon=True)

    def __enter__(self) -> Iterator[list[Row]]:
        self._thread.start()
        return self._items()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stopped.set()
        while self._thread.is_alive():  # a reader blocked on the full queue needs a taker before it sees the stop
            with contextlib.suppress(queue.Empty):
                self._queue.get(timeout=0.05)
        self._thread.join()

    def _items(self) -> Iterator[list[Row]]:
        while True:
            with measure("global_reader_wait"):
                item = self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _run(self) -> None:
        try:
            for batch in self._batches:
                if self._stopped.is_set():
                    return
                self._queue.put(batch)
            self._queue.put(None)
        except BaseException as error:  # noqa: BLE001 - handed to the consumer thread, which raises it
            self._queue.put(error)


class _Writer:
    """
    The commits of one source (`commit_global_batch`: shard, then manifest) on their own thread, in order.
    :meth:`publish` is the admission's PublishBatch: it queues one batch and blocks while an earlier one is still
    queued, so the admission runs at most two commits ahead of the manifest on disk. A failed commit is raised by
    the next publish (poisoning the admission, as a failing commit did in the loop) and by :meth:`flush` and the
    end of the with block; the batches queued after it are discarded, so the manifest stays at the last complete
    commit.
    """

    def __init__(self, commit: PublishBatch) -> None:
        self._commit = commit
        self._queue: queue.Queue[tuple[list[Row], GlobalFrontier] | None] = queue.Queue(maxsize=1)
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=bind_profile(self._run), name="global-writer", daemon=True)

    def __enter__(self) -> _Writer:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._failure is None:
            return
        if exc_type is None:
            raise self._failure
        if exc is not self._failure:
            log.error("the global output writer failed as well: %r", self._failure)

    def publish(self, rows: list[Row], frontier: GlobalFrontier) -> None:
        self._check()
        with measure("global_writer_wait"):
            self._queue.put((rows, frontier))

    def flush(self) -> None:
        """
        Wait until every published batch is on disk.
        """

        with measure("global_writer_wait"):
            self._queue.join()
        self._check()

    def _check(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            try:
                if self._failure is None:
                    self._commit(*item)
            except BaseException as error:  # noqa: BLE001 - raised on the build thread by publish / flush
                self._failure = error
            finally:
                self._queue.task_done()
        self._queue.task_done()


def commit_global_batch(
    layout: DatasetLayout, directory: Path, manifest: Manifest, start: GlobalFrontier,
    rows: list[dict[str, Any]], next_frontier: GlobalFrontier,
) -> None:
    """Publish rows before atomically recording their frontier and counters."""
    if rows:
        path = guarded_path(layout.root, directory / shard_name(len(manifest.shards)))
        guarded_path(layout.root, path.with_name(path.name + ".tmp"))
        publish_shard(build_row_table(rows), path)
        manifest.add_shard(path.name, len(rows), sum(int(row["tokens"]) for row in rows))
    manifest.extra["global_frontier"] = next_frontier.to_dict()
    manifest.stats["global_dedup"] = {
        "candidates": next_frontier.candidates - start.candidates,
        "retained": next_frontier.retained - start.retained,
        "bloom_positive": next_frontier.bloom_positive - start.bloom_positive,
    }
    manifest.save(directory)
