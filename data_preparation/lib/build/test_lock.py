# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.build.lock.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
import subprocess
import sys
from pathlib import Path

import pytest

from data_preparation.lib.build.lock import BUILD_LOCK_NAME, TRAIN_LOCK_NAME, RunLocked, build_lock, run_lock


def test_build_lock_is_exclusive_and_names_the_holder(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with build_lock(root):
        record = json.loads((root / BUILD_LOCK_NAME).read_text())
        assert record["pid"] == os.getpid() and record["program"] == "data preparation" and "since" in record
        with pytest.raises(RunLocked, match=f"pid {os.getpid()} on") as exc, build_lock(root):  # a second open conflicts
            pass
        message = str(exc.value)
        assert message.startswith("data preparation expects one run at a time on this system; one is already running (started 20")
        assert message.endswith(f"). Wait for it to finish, or stop it with: kill -INT {os.getpid()}")
        assert exc.value.holder is not None and exc.value.holder.started() == datetime.fromisoformat(record["since"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    assert (root / BUILD_LOCK_NAME).read_text() == ""  # released and cleared
    with build_lock(root):
        pass


def test_build_lock_blocks_another_process(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    holder = subprocess.Popen(
        [sys.executable, "-c", "import sys; from pathlib import Path; from data_preparation.lib.build.lock import build_lock\n"
         f"with build_lock(Path({str(root)!r})):\n    print('locked', flush=True); sys.stdin.readline()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=Path(__file__).resolve().parents[3],
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        with pytest.raises(RunLocked, match=f"pid {holder.pid}"), build_lock(root):
            pass
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.close()
        holder.wait(timeout=10)
    with build_lock(root):
        pass


def test_the_training_lock_names_its_program(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / TRAIN_LOCK_NAME
    with run_lock(path, "training"), pytest.raises(RunLocked, match="^training expects one run at a time"), run_lock(path, "training"):
        pass
    assert path.read_text() == ""


def test_an_unreadable_holder_record_still_reports_the_conflict(tmp_path: Path) -> None:
    path = tmp_path / TRAIN_LOCK_NAME
    with run_lock(path, "training"):
        path.write_text("not json")
        with pytest.raises(RunLocked, match="holder unknown") as exc, run_lock(path, "training"):
            pass
        assert exc.value.holder is None


def test_a_lock_is_released_on_an_exception(tmp_path: Path) -> None:
    path = tmp_path / "out" / TRAIN_LOCK_NAME
    with pytest.raises(ValueError, match="boom"), run_lock(path, "training"):
        raise ValueError("boom")
    with run_lock(path, "training"):  # not left behind
        pass


def test_a_lock_file_left_by_a_dead_holder_is_simply_taken(tmp_path: Path) -> None:
    """
    The OS dropped the `flock` with the process: the next run takes the file and overwrites the dead holder's record.
    """

    path = tmp_path / TRAIN_LOCK_NAME
    path.write_text(json.dumps({"program": "training", "pid": 999999, "host": "dead-host", "since": "2020-01-01T00:00:00+00:00"}))
    with run_lock(path, "training"):
        assert json.loads(path.read_text())["pid"] == os.getpid()
