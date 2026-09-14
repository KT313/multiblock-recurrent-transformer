# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Fault injection for the ordered download resource lifetime; no network or large inputs."""

from __future__ import annotations

import importlib
import multiprocessing
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from data_preparation.conftest import REPO, REV, FakeHub
from data_preparation.lib.dataset_config import DatasetConfig, SourceConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.progress import NoProgress
from data_preparation.lib.sources.hub_files import FileIndex
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.raw_folder import RawFolder

module = importlib.import_module("data_preparation.lib.stages.download")


class _Scenario:
    """In-memory resources whose event log records phase and dependency order, including failed actions."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, faults: dict[str, BaseException]) -> None:
        self.events: list[str] = []
        self.faults = faults
        self.gate = module._StopGate(None)
        self.worker: Any = None
        self.increments: list[Any] = []
        for name in ("A", "B"):
            self.increments.append(SimpleNamespace(
                name=name, passive=False, rows_to_keep=1,
                folder=SimpleNamespace(directory=tmp_path / name, shard_count=0, record_shard=lambda *args: None),
                token_step=SimpleNamespace(take=lambda name=name: self.action(f"take {name}")),
                counters=SimpleNamespace(kept=0, exhausted=False),
            ))
        scenario = self

        class Writer:
            def __init__(self, path: Path, *args: Any, **kwargs: Any) -> None:
                self.name = path.name

            def __enter__(self) -> Writer:
                scenario.action(f"enter {self.name}")
                return self

            def flush(self) -> None:
                assert scenario.worker is None or scenario.worker.stopped
                scenario.action(f"flush {self.name}")

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                assert scenario.worker is None or scenario.worker.stopped
                scenario.action(f"exit {self.name} {'success' if exc_type is None else 'error'}")

        class Worker:
            def __init__(self, *args: Any) -> None:
                scenario.action("worker start")
                scenario.worker = self
                self.stopped = False

            def submit(self, increment: Any, batch: Any) -> None:
                scenario.action(f"submit {increment.name}")

            def close(self) -> None:
                self.stopped = "worker live" not in faults
                scenario.action("worker close")

        class Rows:
            def __iter__(self) -> Rows:
                return self

            def __next__(self) -> tuple[str, dict[str, Any]]:
                scenario.action("processing")
                raise StopIteration

            def close(self) -> None:
                scenario.action("rows close")

        monkeypatch.setattr(module, "ShardWriter", Writer)
        monkeypatch.setattr(module, "_TokenWorker", Worker)
        self.rows = Rows()

    def action(self, event: str) -> list[Any]:
        self.events.append(event)
        if event in self.faults:
            raise self.faults[event]
        return []

    def run(self) -> None:
        module._fetch(self.increments, self.rows, NoProgress(), SimpleNamespace(), 10, self.gate)


START = ["enter A", "enter B", "worker start", "processing"]
SETTLE = ["take A", "submit A", "take B", "submit B", "rows close", "worker close"]
FAIL_END = ["flush A", "flush B", "exit A error", "exit B error"]


@pytest.mark.parametrize("phase", ["processing", "take A", "submit A", "rows close", "worker close"])
def test_cleanup_after_each_failure_is_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str,
) -> None:
    error = OSError(f"failure at {phase}")
    scenario = _Scenario(monkeypatch, tmp_path, {phase: error})
    with pytest.raises(OSError) as caught:
        scenario.run()
    assert caught.value is error
    expected = START + SETTLE + FAIL_END
    if phase == "take A":
        expected.remove("submit A")
    assert scenario.events == expected
    assert not any(increment.counters.exhausted for increment in scenario.increments)
    assert scenario.gate.suspended


def test_all_cleanup_errors_keep_primary_identity_and_secondary_tracebacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    primary = ValueError("bad source")
    faults: dict[str, BaseException] = {"processing": primary, "rows close": OSError("index full"), "flush A": OSError("shard full"),
              "exit A error": RuntimeError("release failed")}
    scenario = _Scenario(monkeypatch, tmp_path, faults)
    with pytest.raises(ValueError) as caught:
        scenario.run()
    assert caught.value is primary
    assert scenario.events == START + SETTLE + FAIL_END
    diagnostics = "\n".join(caught.value.__notes__)
    for phrase in ("row iterator close", "index full", "partial shard flush for A", "shard full", "writer exit for A", "release failed", "Traceback"):
        assert phrase in diagnostics
    assert any(frame.name == "__next__" for frame in traceback.extract_tb(caught.value.__traceback__))


def test_successful_exits_continue_flushing_after_one_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scenario = _Scenario(monkeypatch, tmp_path, {"exit A success": OSError("callback failed")})
    with pytest.raises(OSError, match="callback failed"):
        scenario.run()
    assert scenario.events == START + SETTLE + ["exit A success", "exit B success"]
    assert not any(increment.counters.exhausted for increment in scenario.increments)


def test_success_keeps_exit_driven_flush_and_exhaustion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scenario = _Scenario(monkeypatch, tmp_path, {})
    scenario.run()
    assert scenario.events == START + SETTLE + ["exit A success", "exit B success"]
    assert all(increment.counters.exhausted for increment in scenario.increments)
    assert not scenario.gate.suspended


@pytest.mark.parametrize("secondary", [None, OSError("storage failed")])
def test_cooperative_stop_keeps_identity_unless_storage_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, secondary: BaseException | None,
) -> None:
    stop = BuildAborted("cancelled")
    faults: dict[str, BaseException] = {"processing": stop}
    if secondary is not None:
        faults["flush A"] = secondary
    scenario = _Scenario(monkeypatch, tmp_path, faults)
    with pytest.raises(BaseException) as caught:
        scenario.run()
    assert caught.value is (secondary if secondary is not None else stop)
    assert scenario.events == START + SETTLE + FAIL_END
    if secondary is not None:
        assert "cancelled" in "\n".join(caught.value.__notes__)


def test_partial_initialization_closes_only_entered_writers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scenario = _Scenario(monkeypatch, tmp_path, {"enter B": OSError("cannot open B")})
    with pytest.raises(OSError, match="cannot open B"):
        scenario.run()
    assert scenario.events == ["enter A", "enter B", "rows close", "flush A", "exit A error"]


def test_unconfirmed_shutdown_never_touches_worker_owned_writers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    interruption = KeyboardInterrupt()
    scenario = _Scenario(monkeypatch, tmp_path, {"worker close": interruption, "worker live": RuntimeError()})
    with pytest.raises(KeyboardInterrupt) as caught:
        scenario.run()
    assert caught.value is interruption
    assert scenario.events == START + SETTLE
    assert "writer cleanup is incomplete" in "\n".join(caught.value.__notes__)


def test_worker_reports_storage_failure_after_cancellation_and_join(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = BuildAborted("stop first")
    storage = OSError("storage later")
    calls = 0

    def store(*args: Any) -> None:
        nonlocal calls
        calls += 1
        raise stop if calls == 1 else storage

    monkeypatch.setattr(module, "_store", store)
    increment: Any = SimpleNamespace(name="A", submitted=0, settled=0, token_step=SimpleNamespace(tokenize=lambda batch: batch))
    worker = module._TokenWorker("failure-selection", {"A": SimpleNamespace()}, NoProgress(), module._StopGate(None))
    try:
        worker.submit(increment, [({}, None)])
        with pytest.raises(BuildAborted) as caught_stop:
            worker.drain()
        assert caught_stop.value is stop
        worker.submit(increment, [({}, None)])
        with pytest.raises(OSError) as caught:
            worker.close()
        assert caught.value is storage and worker.stopped
        assert increment.settled == 2
        assert "stop first" in "\n".join(storage.__notes__)
    finally:
        if not worker.stopped:
            worker.close()


@pytest.mark.timeout(30)
def test_grouped_index_save_failure_on_early_close_joins_real_worker(
    hub: FakeHub, cfg_factory: Callable[..., DatasetConfig], layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch,
    read_rows: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """Run the real delegated generator/thread path in a bounded child so a regression cannot leak a test worker."""
    hub.add("data/a.parquet", [{"text": f"row {i}", "language": "Python"} for i in range(6)])
    cfg = cfg_factory({"py": SourceConfig(kind="pretrain", loader="github_code", hf_id=REPO, revision=REV,
                                           language="Python", check_limit=1)})
    module.prepare_tokenizer(cfg, layout)
    original_save = FileIndex.save
    save_events: list[str] = []

    def save(index: FileIndex) -> None:
        if index.row_groups:
            save_events.append("index close failed")
            raise OSError("index save full")
        original_save(index)

    monkeypatch.setattr(FileIndex, "save", save)
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def run() -> None:
        try:
            with pytest.raises(OSError, match="index save full"):
                module.download_github_code_group(cfg, ["py"], layout, rows_needed={"py": 5}, shard_size=10)
            assert save_events == ["index close failed"]
            assert not [thread for thread in threading.enumerate() if thread.name.startswith("tokenize:")]
            manifest = Manifest.load(layout.raw_dir("py"))
            assert manifest is not None and manifest.rows() == manifest.rows_fetched == 1
            assert [row["text"] for row in read_rows(layout.raw_dir("py"))] == ["row 0"]
            assert not manifest.exhausted
            send.send(None)
        except BaseException:
            send.send(traceback.format_exc())
        finally:
            send.close()

    child = context.Process(target=run)
    child.start()
    send.close()
    try:
        child.join(timeout=15)
        assert not child.is_alive(), "download cleanup leaked its non-daemon token worker"
        assert child.exitcode == 0 and receive.poll(), "grouped cleanup child failed without a result"
        assert receive.recv() is None
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=3)
            if child.is_alive():
                child.kill()
                child.join(timeout=3)
        receive.close()


@pytest.mark.parametrize("shard_size, expected_rows", [(2, 2), (10, 5)])
def test_published_callback_failure_never_republishes_or_advances_unpersisted_rows(
    cfg_factory: Callable[..., DatasetConfig], layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch,
    read_rows: Callable[[Path], list[dict[str, Any]]], shard_size: int, expected_rows: int,
) -> None:
    cfg = cfg_factory({"s": SourceConfig(kind="pretrain", loader="synthetic")})
    module.prepare_tokenizer(cfg, layout)
    monkeypatch.setattr(module, "TOKEN_BATCH", 2)
    original_record = RawFolder.record_shard
    callbacks: list[str] = []
    failure = OSError("callback after commit")

    def record(folder: RawFolder, path: Path, table: Any) -> None:
        original_record(folder, path, table)
        callbacks.append(path.name)
        raise failure

    monkeypatch.setattr(RawFolder, "record_shard", record)
    with pytest.raises(OSError) as caught:
        module.download(cfg, "s", layout, rows_needed=5, shard_size=shard_size)
    assert caught.value is failure
    manifest = Manifest.load(layout.raw_dir("s"))
    assert manifest is not None and manifest.rows() == manifest.rows_fetched == expected_rows
    assert not manifest.exhausted
    assert callbacks == ["data-00000.parquet"]
    assert len(read_rows(layout.raw_dir("s"))) == expected_rows
    assert len(list(layout.raw_dir("s").glob("*.parquet"))) == 1
