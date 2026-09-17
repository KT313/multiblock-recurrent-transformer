# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the threads of the dataset-wide admission pass (data_preparation.lib.stages.global_build): the reader that
decodes candidate batches ahead and the writer that commits behind, and what happens when either fails or the pass
stops early. The admission itself is tested in test_global_dedup.py, the pass end to end in build/test_global_runner.py.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from data_preparation.lib.stages.global_build import _Reader, _Writer
from data_preparation.lib.stages.global_dedup import GlobalFrontier

Row = dict[str, Any]


def _batches(count: int, fail_after: int | None = None) -> Iterator[list[Row]]:
    for index in range(count):
        if fail_after is not None and index == fail_after:
            raise OSError("candidate shard unreadable")
        yield [{"text": f"row {index}"}]


def _frontier(retained: int) -> GlobalFrontier:
    return GlobalFrontier(("a",), 1, candidates=retained, retained=retained)


# --- reader --------------------------------------------------------------------------------------------------------------


def test_reader_yields_every_batch_in_order_and_raises_its_error_in_turn() -> None:
    with _Reader(_batches(5)) as decoded:
        assert [batch[0]["text"] for batch in decoded] == [f"row {i}" for i in range(5)]

    with _Reader(_batches(5, fail_after=3)) as decoded:
        seen = []
        with pytest.raises(OSError, match="unreadable"):
            for batch in decoded:
                seen.append(batch[0]["text"])
        assert seen == ["row 0", "row 1", "row 2"], "the batches before the error are delivered first"


def test_reader_stops_when_the_consumer_leaves_early() -> None:
    """
    A consumer that raises with the queue full leaves a reader blocked on put; leaving the block lets it go and joins
    it, without reading the rest of the candidates.
    """

    produced = []

    def endless() -> Iterator[list[Row]]:
        index = 0
        while True:
            produced.append(index)
            yield [{"text": f"row {index}"}]
            index += 1

    reader = _Reader(endless())
    with pytest.raises(RuntimeError, match="stop"), reader as decoded:
        next(decoded)
        time.sleep(0.05)  # let the reader fill the queue and block
        raise RuntimeError("stop")
    assert not reader._thread.is_alive()
    assert len(produced) <= 6, "the read-ahead plus what the drain released, not the whole source"


# --- writer --------------------------------------------------------------------------------------------------------------


def test_writer_commits_in_order_and_flush_waits_for_the_disk() -> None:
    committed: list[tuple[list[Row], GlobalFrontier]] = []
    started = threading.Event()

    def slow_commit(rows: list[Row], frontier: GlobalFrontier) -> None:
        started.wait()
        time.sleep(0.01)
        committed.append((rows, frontier))

    with _Writer(slow_commit) as writer:
        writer.publish([{"text": "a"}], _frontier(1))
        writer.publish([{"text": "b"}], _frontier(2))  # queued behind the first, which has not started
        assert committed == []
        started.set()
        writer.flush()
        assert [frontier.retained for _, frontier in committed] == [1, 2]
        writer.publish([], _frontier(2))
    assert [frontier.retained for _, frontier in committed] == [1, 2, 2], "leaving the block commits what is queued"


def test_a_failed_commit_is_raised_by_the_next_publish_and_later_batches_are_dropped() -> None:
    committed: list[int] = []

    def commit(rows: list[Row], frontier: GlobalFrontier) -> None:
        if frontier.retained == 2:
            raise OSError("publication failed")
        committed.append(frontier.retained)

    with pytest.raises(OSError, match="publication failed"), _Writer(commit) as writer:
        writer.publish([{"text": "a"}], _frontier(1))
        writer.publish([{"text": "b"}], _frontier(2))
        writer.flush()
    assert committed == [1]

    with pytest.raises(OSError, match="publication failed"), _Writer(commit) as writer:
        writer.publish([{"text": "a"}], _frontier(1))
        writer.publish([{"text": "b"}], _frontier(2))
        time.sleep(0.05)  # the failure happens on the writer thread
        for retained in (3, 4, 5):
            writer.publish([{"text": "c"}], _frontier(retained))  # the first publish after the failure raises it
        pytest.fail("unreachable")
    assert committed == [1, 1], "nothing after the failed commit reaches the disk"


def test_a_failure_of_the_writer_is_raised_when_the_block_ends() -> None:
    def commit(rows: list[Row], frontier: GlobalFrontier) -> None:
        raise OSError("publication failed")

    with pytest.raises(OSError, match="publication failed"), _Writer(commit) as writer:
        writer.publish([{"text": "a"}], _frontier(1))


def test_a_stop_while_the_writer_is_busy_keeps_the_stop_and_commits_the_queued_batch() -> None:
    committed: list[int] = []

    def commit(rows: list[Row], frontier: GlobalFrontier) -> None:
        time.sleep(0.02)
        committed.append(frontier.retained)

    with pytest.raises(KeyboardInterrupt), _Writer(commit) as writer:
        writer.publish([{"text": "a"}], _frontier(1))
        writer.publish([{"text": "b"}], _frontier(2))
        raise KeyboardInterrupt
    assert committed == [1, 2]
