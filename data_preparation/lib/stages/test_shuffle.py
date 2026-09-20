# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Behavioral contracts for the disposable disk-backed global shuffle."""

from __future__ import annotations

import random
import sqlite3
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.stages import shuffle
from data_preparation.lib.stages.shuffle import shuffled_rows


@pytest.mark.parametrize("count", [0, 1, 200])
def test_shuffle_preserves_rows_seed_and_global_rng(tmp_path: Path, count: int) -> None:
    rows = [{"text": f"日本語\n\x00{index % 7}", "hash": 2**64 - 1, "tokens": 10} for index in range(count)]
    state = random.getstate()
    with shuffled_rows(iter(rows), tmp_path, 42) as output:
        first = list(output)
    assert random.getstate() == state
    assert Counter(tuple(row.items()) for row in first) == Counter(tuple(row.items()) for row in rows)
    with shuffled_rows(iter(rows), tmp_path, 42) as output:
        assert list(output) == first
    if count > 1:
        with shuffled_rows(iter(rows), tmp_path, 43) as output:
            assert list(output) != first
        assert first != rows
    assert not list(tmp_path.iterdir())


def test_key_collisions_do_not_drop_or_reorder_equal_key_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random.Random, "getrandbits", lambda self, bits: 0)
    rows = [{"text": str(index)} for index in range(20)]
    with shuffled_rows(iter(rows), tmp_path, 42) as output:
        assert list(output) == rows


@pytest.mark.parametrize("phase", ["input", "output"])
def test_cancelled_shuffle_closes_database_and_discards_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    monkeypatch.setattr(shuffle, "_CHECK_INTERVAL", 1)
    consumed = 0
    cancelled = False

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal consumed, cancelled
        for index in range(10):
            consumed += 1
            cancelled = phase == "input" and index == 3
            yield {"text": str(index)}

    with pytest.raises(BuildAborted), shuffled_rows(rows(), tmp_path, 42, should_stop=lambda: cancelled) as output:
        assert next(output)
        cancelled = True
        next(output)
    assert consumed == (4 if phase == "input" else 10)
    assert not list(tmp_path.iterdir())


def test_consumer_failure_removes_scratch_and_preserves_exception(tmp_path: Path) -> None:
    error = OSError("output write failed")
    with pytest.raises(OSError) as caught, shuffled_rows(iter([{"text": "x"}]), tmp_path, 42) as output:
        next(output)
        raise error
    assert caught.value is error
    assert not list(tmp_path.iterdir())


def test_sqlite_failure_names_workspace_and_keeps_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    connect = sqlite3.connect

    def limited_connect(path: Path) -> sqlite3.Connection:
        connection = connect(path)
        connection.execute("PRAGMA max_page_count=4")
        return connection

    monkeypatch.setattr(sqlite3, "connect", limited_connect)
    with (
        pytest.raises(RuntimeError, match="Disk-backed shuffle failed.*Check free disk space") as caught,
        shuffled_rows(iter([{"text": "x" * 100_000}]), tmp_path, 42),
    ):
        pytest.fail("the scratch database must hit its page limit while inserting")
    assert isinstance(caught.value.__cause__, sqlite3.Error)
    assert str(tmp_path) in str(caught.value)
    assert not list(tmp_path.iterdir())
