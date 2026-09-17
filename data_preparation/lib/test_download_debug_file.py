# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The dedicated debug log mirrors live overviews without replacing dashboard output."""

import logging
import threading
from pathlib import Path
from typing import TextIO

import pytest

from data_preparation import prepare
from data_preparation.lib.download_debug import DebugSession, log_download_debug
from data_preparation.lib.download_profile import DownloadProfile


def test_debug_file_appends_and_flushes_before_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="data_preparation")
    path = tmp_path / "debug.log"
    path.write_text("previous run\n")
    written = threading.Event()
    original = DebugSession._write

    def write(session: DebugSession, message: str) -> None:
        original(session, message)
        written.set()

    monkeypatch.setattr(DebugSession, "_write", write)
    with log_download_debug(DownloadProfile(live=True), 0.01, debug_file=path):
        assert written.wait(5)
        text = path.read_text()
        assert text.startswith("previous run\n")
        assert "download debug pid=" in text and "final last=" not in text
    assert "final last=" in path.read_text()
    assert "download debug pid=" in caplog.text


@pytest.mark.parametrize("interval", [None, "0.1"])
def test_debug_file_cli_enables_logging_and_creates_parents(
    interval: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    intervals: list[float] = []
    original = DebugSession.__init__

    def initialize(session: DebugSession, profile: DownloadProfile, seconds: float, stream: TextIO | None = None) -> None:
        intervals.append(seconds)
        original(session, profile, seconds, stream)

    monkeypatch.setattr(DebugSession, "__init__", initialize)
    path = tmp_path / "logs" / "nested" / "debug.log"
    data = tmp_path / "data"
    arguments = ["download", "--dataset_config", "config/datasets/tiny.yaml", "--dataset_dir", str(data),
                 "--debug-file", str(path)]
    if interval is not None:
        arguments.extend(["--debug", interval])
    prepare.main(arguments)
    assert intervals == [5.0 if interval is None else float(interval)]
    assert "download debug pid=" in path.read_text()
    assert "final last=" in path.read_text()
    assert "download debug pid=" in (data / "build.log").read_text()


def test_debug_file_dry_run_creates_no_file_or_directories(tmp_path: Path) -> None:
    path = tmp_path / "absent" / "debug.log"
    prepare.main(["download", "--dataset_config", "config/datasets/tiny.yaml", "--dataset_dir", str(tmp_path / "data"),
                  "--debug-file", str(path), "--dry_run"])
    assert not path.parent.exists()


def test_invalid_debug_file_fails_before_starting_reporter(tmp_path: Path) -> None:
    before = set(threading.enumerate())
    with pytest.raises(IsADirectoryError), log_download_debug(DownloadProfile(live=True), 5, debug_file=tmp_path):
        pytest.fail("preparation must not start with an invalid debug file")
    assert set(threading.enumerate()) == before
