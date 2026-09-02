# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``RawFolder``: the one object that owns the bookkeeping of a ``sources/<source>/raw/`` directory.

A raw manifest records more than its shards: the loader offset reached (``rows_fetched``, where the next download
resumes), how many source rows were rejected on the way (``skipped_malformed`` / ``dropped_too_long``), whether the
loader ran dry (``exhausted``, possibly because a ``check_limit`` was reached: ``check_limit_reached``) and the token
cap the stored rows were cut at (``truncated_at_tokens``). Three code paths change all of that — the per-source
download, the ``github_code`` group pass and the repair step's truncation — and :class:`RawFolder` is the only
writer of these fields, so the offset and the reject counters always move together (a resume never counts a rejected
row twice).

The append side is resumable at shard granularity: every published shard is recorded with the loader offset **and**
the reject totals as of its last stored row (:class:`RowProgress`, carried on the row itself under
:data:`ROW_PROGRESS_KEY` and stored in :class:`~data_preparation.lib.storage.manifest.ShardInfo`), so a stop, a
failure or a truncation to the good prefix (:func:`good_prefix_length`, :meth:`RawFolder.truncate_to`) all leave a
manifest a resume can continue from without counting any rejected row twice. Manifests written before those per-shard fields existed still load: a truncation then resets the
counters to 0 and says so in the log (the folder stays usable and is never treated as stale).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.log import get_logger
from data_preparation.lib.storage.manifest import Manifest, ShardInfo, shard_problem, shard_rows, shard_tokens
from data_preparation.lib.storage.parquet import ShardWriter, list_parquet_files, shard_index

log = get_logger(__name__)

Row = dict[str, Any]

ROW_PROGRESS_KEY = "_progress"  # private row key: the fetch progress right after this row (stripped before it is written)


class RowProgress(NamedTuple):
    """Where an increment stood when a stored row was produced: the loader offset after it and how many rows before
    it were rejected. Persisted with every shard (its last stored row's values), so a resume — which re-reads the
    source from that offset — counts every rejected row exactly once."""

    consumed: int
    skipped_malformed: int
    dropped_too_long: int


def good_prefix_length(directory: Path, manifest: Manifest) -> tuple[int, str | None]:
    """How many leading shards of ``manifest`` verify against ``directory``, and the first problem (None if all do)."""
    for index, shard in enumerate(manifest.shards):
        problem = shard_problem(directory, shard)
        if problem is not None:
            return index, problem
    return len(manifest.shards), None


# --- the folder ---------------------------------------------------------------------------------------------------------


class RawFolder:
    """The bookkeeping of one raw directory around its (already loaded) manifest.

    ``config_cap`` is the config's ``max_seq_length``, used as :attr:`cap` only when the manifest records none yet
    (a fresh folder, or one written before the truncation existed); a folder opened just to inspect or repair needs
    none. ``should_stop`` is polled after every published shard.
    """

    def __init__(self, directory: Path, manifest: Manifest, *, config_cap: int = 0, should_stop: StopCheck | None = None) -> None:
        self.directory = directory
        self.manifest = manifest
        self._config_cap = config_cap
        self._should_stop = should_stop
        self._reset_increment()

    def _reset_increment(self) -> None:
        """(Re)base the append bookkeeping on what the manifest holds right now."""
        self.start_offset = self.manifest.rows_fetched
        self._skipped_before, self._dropped_before = self.manifest.skipped_malformed, self.manifest.dropped_too_long
        self._last = RowProgress(0, 0, 0)  # progress at the row most recently handed to the writer

    # --- state ---------------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.manifest.source

    @property
    def cap(self) -> int:
        """The token cap the rows appended to this folder are cut at: the recorded ``truncated_at_tokens``.

        Appending under a *lower* config cap would otherwise leave a folder whose manifest promises longer rows than
        the appended part holds, and raising the cap back would not be detected
        (:meth:`~data_preparation.lib.storage.manifest.Manifest.is_outdated`). Lowering stays free (the build clamps
        the stored counts). A manifest without the field falls back to the ``config_cap`` given at construction."""
        recorded = self.manifest.truncated_at_tokens
        return recorded if recorded is not None else self._config_cap

    @property
    def rows(self) -> int:
        """Rows on disk (the manifest's shards)."""
        return self.manifest.rows()

    @property
    def rows_fetched(self) -> int:
        """The loader offset reached: source rows consumed, where the next download resumes."""
        return self.manifest.rows_fetched

    @property
    def exhausted(self) -> bool:
        return self.manifest.exhausted

    @property
    def shard_count(self) -> int:
        return len(self.manifest.shards)

    # --- exhaustion ----------------------------------------------------------------------------------------------

    def reopen_if_check_limit_grew(self, check_limit: int | None) -> None:
        """A source marked exhausted because its ``check_limit`` was reached may be read further when the limit grew
        (or was removed): ``check_limit`` is not part of the raw hash, so the manifest is not stale, only its flag.
        Kept in memory — the increment that follows saves it."""
        reached = self.manifest.check_limit_reached
        if not self.exhausted or reached is None:
            return
        if check_limit is None or check_limit > reached:
            log.info("%s: check_limit grew from %s to %s, source no longer exhausted", self.name, reached, check_limit)
            self.manifest.exhausted = False
            self.manifest.check_limit_reached = None

    def mark_exhausted(self, *, check_limit: int | None = None) -> None:
        """Record that there is nothing more to fetch and save; ``check_limit`` names the limit that stopped the
        reads (so a later, larger limit reopens the source)."""
        self._set_exhausted(check_limit)
        self.save()

    def _set_exhausted(self, check_limit: int | None) -> None:
        self.manifest.exhausted = True
        if check_limit is not None:
            self.manifest.check_limit_reached = check_limit

    # --- appending -----------------------------------------------------------------------------------------------

    def add(self, writer: ShardWriter, row: Row) -> None:
        """Hand ``row`` (tagged with :data:`ROW_PROGRESS_KEY` by the caller's token step) to ``writer``."""
        self._last = row.pop(ROW_PROGRESS_KEY)
        writer.add(row)

    def record_shard(self, path: Path) -> None:
        """``ShardWriter`` callback: record the published shard with the offset and the reject totals as of its last
        row, save the manifest and check the stop request."""
        self.manifest.add_shard(
            path.name,
            shard_rows(path),
            shard_tokens(path),
            offset=self.start_offset + self._last.consumed,
            skipped_malformed=self._skipped_before + self._last.skipped_malformed,
            dropped_too_long=self._dropped_before + self._last.dropped_too_long,
        )
        self._store(self._last)
        check_stop(self._should_stop)

    def finish(self, progress: RowProgress, *, exhausted: bool, check_limit: int | None = None) -> None:
        """After the loader ran dry / the target was reached: the final offset, counters and exhaustion flag."""
        if exhausted:
            self._set_exhausted(check_limit)
        self._last = progress
        self._store(progress)

    def _store(self, progress: RowProgress) -> None:
        self.manifest.rows_fetched = self.start_offset + progress.consumed
        self.manifest.skipped_malformed = self._skipped_before + progress.skipped_malformed
        self.manifest.dropped_too_long = self._dropped_before + progress.dropped_too_long
        self.save()

    def save(self) -> None:
        self.manifest.save(self.directory)

    # --- repair --------------------------------------------------------------------------------------------------

    def truncate_to(self, good_shards: int) -> None:
        """Keep the first ``good_shards`` shards (the good prefix :func:`good_prefix_length` found) and drop the rest:
        the manifest keeps the prefix, the offset **and every reject counter** become what the last kept shard
        recorded (so the next download resumes there and counts nothing twice) and the exhaustion flag is cleared.
        The prefix must hold at least one shard with a recorded offset — the repair step plans a deletion otherwise.
        """
        good = good_shards
        if good >= len(self.manifest.shards):
            return  # nothing to drop
        if good < 1 or self.manifest.shards[good - 1].offset is None:
            raise ValueError(f"{self.name}: cannot truncate {self.directory} to {good} shard(s): no resume point")
        dropped = self.manifest.shards[good:]
        log.warning("%s: dropping %d shard(s) from %s (%s and after)", self.name, len(dropped), self.directory, dropped[0].name)
        kept = self.manifest.shards[good - 1]
        self.manifest.shards = self.manifest.shards[:good]
        self.manifest.rows_fetched = int(kept.offset or 0)
        self._restore_reject_counters(kept)
        self.manifest.exhausted = False
        self.manifest.check_limit_reached = None
        for path in list_parquet_files(self.directory):
            index = shard_index(path)
            if index is not None and index >= good:
                path.unlink()
        self.save()
        self._reset_increment()

    def _restore_reject_counters(self, kept: ShardInfo) -> None:
        """The reject totals as of ``kept``'s last row. A shard written before those fields existed carries none:
        the counters restart at 0 (a resume may then undercount rejected rows) and the folder stays usable."""
        if kept.skipped_malformed is None and kept.dropped_too_long is None:
            log.warning(
                "%s: %s was written before the per-shard reject counters existed; resetting skipped_malformed / "
                "dropped_too_long to 0 (they may undercount after the resume)", self.name, self.directory,
            )
        self.manifest.skipped_malformed = int(kept.skipped_malformed or 0)
        self.manifest.dropped_too_long = int(kept.dropped_too_long or 0)
