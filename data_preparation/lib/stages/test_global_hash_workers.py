# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Ordered worker results, bounded speculation, cancellation and real spawn parity."""
from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from data_preparation.conftest import CfgFactory
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.stages.global_build import build_global_source
from data_preparation.lib.stages.global_dedup import GlobalAdmission, global_key
from data_preparation.lib.stages.global_hash_workers import GlobalHashWorkers, _key_fields
from data_preparation.lib.stages.global_session import GlobalAdmissionSession
from data_preparation.lib.stages.test_global_session import candidates, output_rows


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.timeout(60)
def test_real_workers_match_serial_keys_and_output(cfg_factory: CfgFactory, tmp_path: Path, workers: int) -> None:
    cfg, layout, frontier = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, frontier, lambda: ())
    with GlobalHashWorkers(workers) as hashing:
        for name in ("a", "b", "c"):
            frontier, complete = build_global_source(cfg, name, layout, frontier, rows_target=1, exhausted=True,
                                                     session=session, hash_workers=hashing, batch_rows=1)
            assert complete
        cases: dict[str, list[dict[str, Any]]] = {
            "pretrain": [{"text": 'Hello\n世界\t"Q"\x00'}, {"text": "HELLO 世界 \"q\"\x00"}],
            "instruct": [{"instruction": "Do THIS", "output": "yes"},
                         {"instruction": "do  this", "input": None, "output": "YES"}],
            "messages": [{"messages": [{"role": "user", "content": "Hi"},
                                       {"role": "assistant", "content": "Hello"}]}],
        }
        for kind, rows in cases.items():
            result = list(hashing.batches(iter([[row] for row in rows]), "different-source", kind, 0))
            assert [keys[0] if keys is not None else global_key(kind, batch[0]) for batch, keys in result] == [
                global_key(kind, row) for row in rows
            ]
        assert hashing._pool is None if workers == 1 else hashing._pool is not None
    assert hashing._pool is None
    assert session.restorations == 1 and session.reuses == 2
    assert [row["text"] for name in ("a", "b", "c") for row in output_rows(layout, name)] == ["shared", "a", "b", "c"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_worker_counts(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GlobalHashWorkers(value)


@pytest.mark.parametrize("fail_at", [1, 4])
def test_reader_error_keeps_preceding_batches_and_window_bound(monkeypatch: pytest.MonkeyPatch, fail_at: int) -> None:
    submitted = 0
    admitted = 0

    def submit(self: GlobalHashWorkers, name: str, kind: str, rows: list[dict[str, Any]]) -> Future[list[int]]:
        nonlocal submitted
        submitted += 1
        assert submitted - admitted <= self.workers
        future: Future[list[int]] = Future()
        future.set_result([global_key(kind, row) for row in rows])
        return future

    def decoded() -> Iterator[list[dict[str, Any]]]:
        for index in range(fail_at):
            yield [{"text": str(index)}]
        raise OSError("reader failed")

    monkeypatch.setattr(GlobalHashWorkers, "_submit", submit)
    with GlobalHashWorkers(3) as hashing, pytest.raises(OSError, match="reader failed"):
        for rows, _ in hashing.batches(decoded(), "a", "pretrain", 0):
            assert rows[0]["text"] == str(admitted)
            admitted += 1
    assert admitted == fail_at


def test_reversed_completion_keeps_duplicate_winner_and_cancels_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    futures: list[Future[list[int]]] = []

    def submit(self: GlobalHashWorkers, name: str, kind: str, rows: list[dict[str, Any]]) -> Future[list[int]]:
        future: Future[list[int]] = Future()
        futures.append(future)
        if len(futures) == 2:
            future.set_result([global_key(kind, row) for row in rows])
            futures[0].set_result([global_key("pretrain", {"text": "SAME"})])
        return future

    monkeypatch.setattr(GlobalHashWorkers, "_submit", submit)
    admission = GlobalAdmission(("a",), memory_mb=1)
    retained: list[dict[str, Any]] = []
    with GlobalHashWorkers(2) as hashing:
        with contextlib.closing(hashing.batches(iter([[{"text": text}] for text in ("SAME", "same", "last")]),
                                              "a", "pretrain", 0)) as batches:
            for _ in range(2):
                rows, keys = next(batches)
                admission.commit_batch("a", "pretrain", rows, lambda rows, _: retained.extend(rows), keys=keys)
        assert len(futures) == 3 and futures[2].cancelled()
    assert [row["text"] for row in retained] == ["SAME"]


def test_stop_while_waiting_cancels_without_admitting(monkeypatch: pytest.MonkeyPatch) -> None:
    future: Future[list[int]] = Future()
    monkeypatch.setattr(GlobalHashWorkers, "_submit", lambda *args: future)
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 5

    with GlobalHashWorkers(2) as hashing, pytest.raises(BuildAborted):
        list(hashing.batches(iter([[{"text": "a"}]]), "a", "pretrain", 0, stop))
    assert future.cancelled()


@pytest.mark.timeout(60)
def test_worker_exception_and_death_are_visible() -> None:
    with GlobalHashWorkers(2) as hashing:
        with pytest.raises(ValueError, match="string text") as caught:
            list(hashing.batches(iter([[{"text": None}]]), "a", "pretrain", 123))
        assert "123" in str(caught.value.__notes__)
        assert hashing._pool is not None
        killed = hashing._pool.submit(os._exit, 7)
        with pytest.raises(Exception, match="terminated abruptly"):
            killed.result(timeout=10)
        with pytest.raises(RuntimeError, match="a: global hash worker died at candidate offset 124"):
            list(hashing.batches(iter([[{"text": "valid"}]]), "a", "pretrain", 124))


@pytest.mark.parametrize("keys", [[True], [1 << 63], [-(1 << 63) - 1], [], [0, 1]])
def test_invalid_precomputed_keys_do_not_mutate_admission(keys: list[int]) -> None:
    admission = GlobalAdmission(("a",), memory_mb=1)
    before = admission.frontier
    with pytest.raises(ValueError, match="precomputed keys"):
        admission.commit_batch("a", "pretrain", [{"text": "hello"}], lambda rows, frontier: None, keys=keys)
    assert admission.frontier == before and admission.seen.items_in_filter == 0


def test_projection_preserves_missing_null_and_message_fields() -> None:
    rows: list[dict[str, Any]] = [{"instruction": "a", "output": "b", "tokens": 2}, {"instruction": "a", "input": None, "output": "b"}]
    projected = _key_fields("instruct", rows)
    assert "input" not in projected[0] and projected[1]["input"] is None
    assert "tokens" not in projected[0]
    assert [global_key("instruct", row) for row in projected] == [global_key("instruct", row) for row in rows]


def test_task_timeout_is_not_retried_and_cleanup_keeps_primary_error(monkeypatch: pytest.MonkeyPatch) -> None:
    future: Future[list[int]] = Future()
    future.set_exception(TimeoutError("task timeout"))
    monkeypatch.setattr(GlobalHashWorkers, "_submit", lambda *args: future)
    with GlobalHashWorkers(2) as hashing, pytest.raises(TimeoutError, match="task timeout"):
        list(hashing.batches(iter([[{"text": "a"}]]), "a", "pretrain", 0))

    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=1)  # no submit: no child is created
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("shutdown failed")
    monkeypatch.setattr(pool, "shutdown", fail)
    hashing._pool = pool
    primary = ValueError("primary")
    hashing.close(primary)
    assert "shutdown failed" in str(primary.__notes__)


def test_hash_failure_discards_session_and_retry_recovers(
    cfg_factory: CfgFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, layout, start = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    start, _ = build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=session)
    failed: Future[list[int]] = Future()
    failed.set_exception(ValueError("hash failed"))
    with GlobalHashWorkers(2) as hashing, monkeypatch.context() as patch:
        patch.setattr(GlobalHashWorkers, "_submit", lambda *args: failed)
        with pytest.raises(ValueError, match="hash failed"):
            build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True,
                                session=session, hash_workers=hashing)
        assert session._entry is None and hashing._pool is None
    build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True, session=session)
    assert session.restorations == 2
    assert [row["text"] for row in output_rows(layout, "b")] == ["b"]


@pytest.mark.slow
def test_reused_real_bloom_matches_reconstruction_with_false_positives() -> None:
    admission = GlobalAdmission(("a", "b"), memory_mb=1)
    committed: list[int] = []

    def collect(rows: list[dict[str, Any]], _: Any) -> None:
        committed.extend(row["global_hash"] for row in rows)

    for first in range(0, 550_000, 4096):
        keys = list(range(first, min(first + 4096, 550_000)))
        admission.commit_batch("a", "pretrain", [{} for _ in keys], collect, keys=keys)
    admission.finish_source("a", lambda rows, frontier: None)
    assert admission.frontier.bloom_positive > 0, "the fixture must exercise real false positives"
    restored = GlobalAdmission(("a", "b"), memory_mb=1, frontier=admission.frontier, committed_keys=committed)
    for first in range(540_000, 570_000, 4096):
        keys = list(range(first, min(first + 4096, 570_000)))
        results: list[list[int]] = []
        def record(rows: list[dict[str, Any]], _: Any, target: list[list[int]] = results) -> None:
            target.append([row["global_hash"] for row in rows])
        for instance in (admission, restored):
            instance.commit_batch("b", "pretrain", [{} for _ in keys], record, keys=keys)
        assert results[0] == results[1]
    assert admission.frontier == restored.frontier
    assert admission.statistics() == restored.statistics()
