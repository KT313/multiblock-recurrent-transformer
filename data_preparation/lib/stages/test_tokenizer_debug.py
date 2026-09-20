# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tokenizer task accounting across spawn, including lazy loading, failures and idle recovery."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from data_preparation.lib import download_debug as debug
from data_preparation.lib.download_profile import DownloadProfile, active_profile
from data_preparation.lib.stages import tokenizer_pool


def _snapshot() -> tuple[dict[str, float], set[str]]:
    profile = active_profile()
    assert profile is not None
    _, metrics, running = profile.snapshot()
    return {key[2]: value.seconds for key, value in metrics.items()}, {key[2] for key in running}


def _initialize_gated_tokenizer(
    assigned: Synchronized[int], options: debug.WorkerDebugOptions, release: Event, phase: str,
) -> None:
    tokenizer_pool._init_tokenizer_process((1,), assigned, options)
    original = tokenizer_pool._tokenizer

    def load(tokenizer_dir: str) -> object:
        before, active = _snapshot()
        assert phase in active and "pool_idle_or_dispatch" not in active
        assert release.wait(20)
        after, _ = _snapshot()
        assert after["pool_idle_or_dispatch"] == before["pool_idle_or_dispatch"]
        assert after[phase] > before[phase]
        return original(tokenizer_dir)

    tokenizer_pool._tokenizer = load


def _submit(pool: ProcessPoolExecutor, operation: str, path: Path) -> Future[list[int]] | Future[list[tuple[str, int]]]:
    texts = ["tok_1 tok_2"]
    if operation == "count":
        return pool.submit(tokenizer_pool._count_in_process, str(path), texts)
    return pool.submit(tokenizer_pool._truncate_in_process, str(path), texts, 8)


class _TaskReports(logging.Handler):
    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase
        self.condition = threading.Condition()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if f"{self.phase}=" in message and "active=" in message:
            with self.condition:
                self.messages.append(message)
                self.condition.notify_all()

    def wait(self) -> bool:
        with self.condition:
            return self.condition.wait_for(lambda: len(self.messages) >= 2, timeout=15)


@pytest.mark.parametrize("operation", ["count", "truncate"])
@pytest.mark.timeout(40)
def test_tokenizer_reports_busy_then_idle_and_recovers_after_failure(
    operation: str, tiny_tokenizer_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    phase = f"tokenizer_{operation}"
    caplog.set_level(logging.INFO, logger="data_preparation")
    logger = logging.getLogger("data_preparation.lib.download_debug")
    handler = _TaskReports(phase)
    logger.addHandler(handler)
    context = multiprocessing.get_context("spawn")
    assigned, release = context.Value("i", 0), context.Event()
    debug_file = tmp_path / "tokenizers.log"
    try:
        with debug.log_download_debug(DownloadProfile(live=True), 0.03, debug_file=debug_file):
            options = debug.worker_debug_options()
            assert options is not None
            with ProcessPoolExecutor(1, mp_context=context, initializer=_initialize_gated_tokenizer,
                                     initargs=(assigned, options, release, phase)) as pool:
                future = _submit(pool, operation, tiny_tokenizer_dir)
                try:
                    assert handler.wait(), "missing two reports from inside tokenizer task"
                    assert not future.done()
                    # The second interval lies wholly inside the gated task.
                    assert "pool_idle_or_dispatch" not in handler.messages[1]
                finally:
                    release.set()
                expected = [2] if operation == "count" else [("tok_1 tok_2", 2)]
                assert future.result(timeout=10) == expected
                first, active = pool.submit(_snapshot).result(timeout=5)
                assert active == {"pool_idle_or_dispatch"}
                second, _ = pool.submit(_snapshot).result(timeout=5)
                assert second["pool_idle_or_dispatch"] > first["pool_idle_or_dispatch"]
                with pytest.raises(ValueError, match="no tokenizer.json"):
                    _submit(pool, operation, tmp_path / "missing").result(timeout=5)
                _, active = pool.submit(_snapshot).result(timeout=5)
                assert active == {"pool_idle_or_dispatch"}
                assert _submit(pool, operation, tiny_tokenizer_dir).result(timeout=5) == expected
                pid = pool.submit(os.getpid).result(timeout=5)
        assert pid != os.getpid()
        output = debug_file.read_text()
        assert f"pid={pid}" in output and f"{phase}=" in output
        assert all(message in output for message in handler.messages)
    finally:
        release.set()
        logger.removeHandler(handler)
