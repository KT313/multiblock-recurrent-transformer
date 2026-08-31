# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `RawFolder`: the token cap and its fallback, the read-only queries over a raw manifest, the append
bookkeeping (offset and reject counters per shard), exhaustion (also by a `check_limit` that later grows) and the
truncation to a good prefix, which restores every counter from the last kept shard."""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.parquet import ShardWriter
from data_preparation.lib.storage.raw_folder import (
    ROW_PROGRESS_KEY,
    RawFolder,
    RowProgress,
    check_limit_reached,
    is_exhausted,
    rejected_rows,
)


def _manifest(**kwargs: object) -> Manifest:
    return Manifest(source="s", source_hash="h", stage="raw", **kwargs)  # type: ignore[arg-type]  # kwargs are the field types


def _write_shard(directory: Path, index: int, rows: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"data-{index:05d}.parquet"
    pq.write_table(pa.Table.from_pylist([{"text": f"r{i}", "tokens": 1} for i in range(rows)]), path)
    return path


# --- cap and queries ---------------------------------------------------------------------------------------------


def test_cap_is_the_recorded_one_and_falls_back_to_the_config_cap(tmp_path: Path) -> None:
    recorded = RawFolder(tmp_path, _manifest(truncated_at_tokens=64), config_cap=2048)
    assert recorded.cap == 64, "the folder's own cap wins over a lower or higher config cap"
    fresh = RawFolder(tmp_path, _manifest(), config_cap=2048)
    assert fresh.cap == 2048
    assert RawFolder(tmp_path, _manifest()).cap == 0  # opened for inspection only: no cap needed


def test_queries_read_the_bookkeeping_keys(tmp_path: Path) -> None:
    manifest = _manifest()
    assert not is_exhausted(manifest) and check_limit_reached(manifest) is None and rejected_rows(manifest) == (0, 0)
    manifest.extra.update({"exhausted": True, "check_limit": 5, "skipped_malformed": 3, "dropped_too_long": 2})
    assert is_exhausted(manifest) and check_limit_reached(manifest) == 5 and rejected_rows(manifest) == (3, 2)
    folder = RawFolder(tmp_path, manifest)
    assert folder.name == "s" and folder.exhausted and folder.rows == 0 and folder.rows_fetched == 0 and folder.shard_count == 0


# --- exhaustion ---------------------------------------------------------------------------------------------------


def test_mark_exhausted_records_the_limit_and_saves(tmp_path: Path) -> None:
    folder = RawFolder(tmp_path, _manifest())
    folder.mark_exhausted(check_limit=7)
    stored = Manifest.load(tmp_path)
    assert stored is not None and is_exhausted(stored) and check_limit_reached(stored) == 7


def test_reopen_if_check_limit_grew(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    folder = RawFolder(tmp_path, _manifest())
    folder.reopen_if_check_limit_grew(10)  # not exhausted: nothing to do
    assert not is_exhausted(folder.manifest)

    folder.mark_exhausted(check_limit=5)
    folder.reopen_if_check_limit_grew(5)
    assert is_exhausted(folder.manifest), "the same limit still applies"
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        folder.reopen_if_check_limit_grew(10)
    assert not is_exhausted(folder.manifest) and check_limit_reached(folder.manifest) is None
    assert "check_limit grew" in caplog.text

    dry = RawFolder(tmp_path, _manifest())
    dry.mark_exhausted()  # the loader itself ran dry: no limit to grow
    dry.reopen_if_check_limit_grew(None)
    assert is_exhausted(dry.manifest)


# --- appending ------------------------------------------------------------------------------------------------------


def test_appending_records_offset_and_reject_counters_per_shard(tmp_path: Path) -> None:
    """Every published shard carries the loader offset and the reject totals as of its last stored row, on top of
    what the folder already held (`_before`), and the manifest is saved after each one."""
    folder = RawFolder(tmp_path, _manifest(rows_fetched=100, extra={"skipped_malformed": 5, "dropped_too_long": 1}))
    with ShardWriter(tmp_path, 2, start_shard=0, on_shard=folder.record_shard) as writer:
        for consumed, skipped, dropped in ((3, 1, 0), (6, 1, 2), (9, 4, 2), (11, 4, 3)):
            folder.add(writer, {"text": "x", "tokens": 1, ROW_PROGRESS_KEY: RowProgress(consumed, skipped, dropped)})
    folder.finish(RowProgress(12, 4, 3), exhausted=False)

    stored = Manifest.load(tmp_path)
    assert stored is not None
    assert [(s.rows, s.offset, s.skipped_malformed, s.dropped_too_long) for s in stored.shards] == [(2, 106, 6, 3), (2, 111, 9, 4)]
    assert stored.rows_fetched == 112 and rejected_rows(stored) == (9, 4) and not is_exhausted(stored)


def test_record_shard_checks_the_stop_request(tmp_path: Path) -> None:
    folder = RawFolder(tmp_path, _manifest(), should_stop=lambda: True)
    with pytest.raises(BuildAborted), ShardWriter(tmp_path, 1, start_shard=0, on_shard=folder.record_shard) as writer:
        folder.add(writer, {"text": "x", "tokens": 1, ROW_PROGRESS_KEY: RowProgress(1, 0, 0)})
    stored = Manifest.load(tmp_path)
    assert stored is not None and stored.rows() == 1, "the shard is published and recorded before the stop"


def test_finish_marks_exhaustion_with_the_limit_that_caused_it(tmp_path: Path) -> None:
    folder = RawFolder(tmp_path, _manifest())
    folder.finish(RowProgress(9, 0, 0), exhausted=True, check_limit=9)
    assert is_exhausted(folder.manifest) and check_limit_reached(folder.manifest) == 9


# --- truncation -------------------------------------------------------------------------------------------------------


def test_truncate_to_good_prefix_restores_offset_and_every_counter(tmp_path: Path) -> None:
    for index, rows in enumerate((2, 2, 2)):
        _write_shard(tmp_path, index, rows)
    manifest = _manifest(rows_fetched=30, extra={"exhausted": True, "check_limit": 30, "skipped_malformed": 9, "dropped_too_long": 6})
    for index, (offset, skipped, dropped) in enumerate(((10, 3, 2), (20, 6, 4), (30, 9, 6))):
        manifest.add_shard(f"data-{index:05d}.parquet", 2, 2, offset=offset, skipped_malformed=skipped, dropped_too_long=dropped)
    folder = RawFolder(tmp_path, manifest)
    assert folder.truncate_to_good_prefix(), "nothing broken yet"

    (tmp_path / "data-00001.parquet").write_bytes(b"corrupt")
    assert folder.truncate_to_good_prefix()
    assert [s.name for s in manifest.shards] == ["data-00000.parquet"]
    assert manifest.rows_fetched == 10 and rejected_rows(manifest) == (3, 2)
    assert not is_exhausted(manifest) and check_limit_reached(manifest) is None
    assert sorted(p.name for p in tmp_path.glob("data-*.parquet")) == ["data-00000.parquet"]
    assert folder.start_offset == 10, "the next increment appends behind the kept prefix"


def test_truncate_to_good_prefix_of_a_manifest_without_counters_resets_them(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    for index in range(2):
        _write_shard(tmp_path, index, 2)
    manifest = _manifest(rows_fetched=20, extra={"skipped_malformed": 9, "dropped_too_long": 6})
    manifest.add_shard("data-00000.parquet", 2, 2, offset=10)  # written before the per-shard fields existed
    manifest.add_shard("data-00001.parquet", 2, 2, offset=20)
    (tmp_path / "data-00001.parquet").write_bytes(b"corrupt")
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert RawFolder(tmp_path, manifest).truncate_to_good_prefix()
    assert manifest.rows_fetched == 10 and rejected_rows(manifest) == (0, 0)
    assert "written before the per-shard reject counters existed" in caplog.text


def test_truncate_to_good_prefix_without_a_resume_point(tmp_path: Path) -> None:
    _write_shard(tmp_path, 0, 2)
    manifest = _manifest(rows_fetched=20)
    manifest.add_shard("data-00000.parquet", 2, 2, offset=10)
    manifest.add_shard("data-00001.parquet", 2, 2, offset=20)  # file never written
    manifest.shards[0].offset = None  # a legacy manifest without offsets
    assert not RawFolder(tmp_path, manifest).truncate_to_good_prefix()
    (tmp_path / "data-00000.parquet").write_bytes(b"corrupt")
    manifest.shards[0].offset = 10
    assert not RawFolder(tmp_path, manifest).truncate_to_good_prefix(), "the first shard is bad: nothing to keep"
    assert len(manifest.shards) == 2 and manifest.rows_fetched == 20, "nothing changed"
