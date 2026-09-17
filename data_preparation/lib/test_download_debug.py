# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Live interval accounting, ongoing waits, and reports from actual spawn worker initializers."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import pytest

from data_preparation import prepare
from data_preparation.cli.arguments import build_parser
from data_preparation.lib import download_debug as debug
from data_preparation.lib.download_profile import DownloadProfile, measure, profile_downloads, profile_source


@pytest.mark.parametrize(("flags", "expected"), [([], None), (["--debug"], 5), (["--debug", "0.25"], 0.25)])
def test_debug_argument(flags: list[str], expected: float | None) -> None:
    parser = build_parser(description="", log=prepare.log, on_prepare=prepare.run_prepare, on_download=prepare.run_download)
    for command in ("download", "prepare"):
        assert parser.parse_args([command, "--dataset_config", "x.yaml", *flags]).debug == expected


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "hello"])
def test_invalid_interval_is_rejected(interval: str) -> None:
    parser = build_parser(description="", log=prepare.log, on_prepare=prepare.run_prepare, on_download=prepare.run_download)
    with pytest.raises(SystemExit):
        parser.parse_args(["download", "--dataset_config", "x", "--debug", interval])


def test_live_wait_is_split_across_intervals_without_counting_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    profile = DownloadProfile(live=True)
    overview = debug.Overview(profile)
    with profile_source("source"):
        span = profile.begin("queue_put")
        clock[0] = 5
        first = overview.render()
        assert "WAIT:queue_put=5.000s" in first and "oldest=5.00s" in first
        clock[0] = 10
        assert "WAIT:queue_put=5.000s" in overview.render()
        clock[0] = 12
        profile.finish(span)
        clock[0] = 15
        third = overview.render()
        assert "WAIT:queue_put=2.000s" in third and "active=" not in third
        clock[0] = 20
        assert "queue_put" not in overview.render()


class _Observe(logging.Handler):
    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase
        self.observed = threading.Event()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        self.messages.append(message)
        if self.phase in message and "active=" in message:
            self.observed.set()


def test_reports_blocked_main_thread_before_the_wait_finishes(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="data_preparation")
    handler = _Observe("WAIT:wait_jobs")
    logger = logging.getLogger("data_preparation.lib.download_debug")
    logger.addHandler(handler)
    before = set(threading.enumerate())
    try:
        with profile_downloads(debug=0.02) as profile, debug.log_download_debug(profile, 0.02), measure("wait_jobs"):
            assert handler.observed.wait(5), "report must arrive before the measured wait ends"
        assert any(f"pid={os.getpid()}" in line and "MainThread[" in line for line in handler.messages)
        assert set(threading.enumerate()) == before
    finally:
        logger.removeHandler(handler)


def _initialize_blocked_worker(options: debug.WorkerDebugOptions, release: Event, stage: str) -> None:
    # Gate an actual worker operation so the report must cross the process boundary while it is unfinished.
    if stage == "minhash":
        from data_preparation.lib.stages import fuzzy_dedup as fuzzy
        fuzzy._init_worker(8, 1, options)
        original = fuzzy._signature

        def signature(*args: Any, **kwargs: Any) -> Any:
            assert release.wait(30)
            return original(*args, **kwargs)
        fuzzy._signature = signature
    else:
        from data_preparation.lib.stages import build
        from data_preparation.lib.stages.row_pipeline import check_contamination
        build._init_decontamination({}, 1, 0.5, options)

        def check(*args: Any, **kwargs: Any) -> Any:
            assert release.wait(30)
            return check_contamination(*args, **kwargs)
        pytest.MonkeyPatch().setattr(build, "check_contamination", check)


def _run_worker_task(stage: str) -> int:
    if stage == "minhash":
        from data_preparation.lib.stages.fuzzy_dedup import _signatures
        _signatures(["the quick brown fox"])
    else:
        from data_preparation.lib.stages.build import _contaminated_by
        _contaminated_by("the quick brown fox")
    return os.getpid()


@pytest.mark.parametrize("stage", ["minhash", "decontamination"])
@pytest.mark.slow
def test_spawn_worker_reports_during_its_task(stage: str, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="data_preparation")
    handler = _Observe("minhash_signatures" if stage == "minhash" else "decontamination=")
    logger = logging.getLogger("data_preparation.lib.download_debug")
    logger.addHandler(handler)
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    try:
        with profile_downloads(debug=0.05) as profile, debug.log_download_debug(profile, 0.05):
            options = debug.worker_debug_options()
            assert options is not None
            with ProcessPoolExecutor(1, mp_context=context, initializer=_initialize_blocked_worker,
                                     initargs=(options, release, stage)) as pool:
                future = pool.submit(_run_worker_task, stage)
                try:
                    assert handler.observed.wait(25), "missing report from blocked spawn worker"
                    assert not future.done()
                finally:
                    release.set()
                pid = future.result(timeout=10)
        assert pid != os.getpid()
        assert any(f"pid={pid}" in line and "MainThread[" in line for line in handler.messages)
    finally:
        release.set()
        logger.removeHandler(handler)


def test_reporter_failure_preserves_preparation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(message: str) -> None:
        raise OSError("debug sink failed")
    monkeypatch.setattr(debug.DebugSession, "_log", staticmethod(fail))
    failure = RuntimeError("prepare failed")
    with pytest.raises(RuntimeError) as caught, profile_downloads(debug=5) as profile, debug.log_download_debug(profile, 5):
        raise failure
    assert caught.value is failure
    assert "debug sink failed" in str(failure.__notes__)


def test_cli_logs_debug_to_build_log(tmp_path: Path) -> None:
    data = tmp_path / "data"
    prepare.main(["download", "--dataset_config", "config/datasets/tiny.yaml", "--dataset_dir", str(data),
                  "--debug", "0.02"])
    output = (data / "build.log").read_text()
    assert "download debug pid=" in output and "MainThread[" in output and "kernel=" in output
    assert "final last=" in output


def test_debug_dry_run_starts_no_reporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry run started debug reporting")
    monkeypatch.setattr(debug, "DebugSession", forbidden)
    data = tmp_path / "absent"
    prepare.main(["download", "--dataset_config", "config/datasets/tiny.yaml", "--dataset_dir", str(data),
                  "--dry_run", "--debug"])
    assert not data.exists()
