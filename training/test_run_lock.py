# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `training.run_lock`: the one-run-per-out_dir lock, its message, its release on every way out and the
reclaim of a lock file a dead run left behind."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from training.run_lock import RUN_LOCK_NAME, RunDirectoryLocked, run_directory_lock

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_directory_lock_is_exclusive_and_names_the_holder(tmp_path: Path) -> None:
    """A second acquisition while the first is held fails with the run-directory message and the holder record;
    the lock is released (and its record cleared) when the block ends, so a later run takes it again."""
    run_directory = tmp_path / "out"
    with run_directory_lock(run_directory):
        record = json.loads((run_directory / RUN_LOCK_NAME).read_text())
        assert record["pid"] == os.getpid() and "host" in record and "since" in record
        message = f"another training run is already using {run_directory} \\(pid {os.getpid()} on"
        with pytest.raises(RunDirectoryLocked, match=message), run_directory_lock(run_directory):
            pass
    assert (run_directory / RUN_LOCK_NAME).read_text() == ""  # released and cleared
    with run_directory_lock(run_directory):  # a sequential run (fresh, then a resume) is unaffected
        pass


def test_run_directory_lock_is_released_on_an_exception(tmp_path: Path) -> None:
    run_directory = tmp_path / "out"
    with pytest.raises(ValueError, match="boom"), run_directory_lock(run_directory):
        raise ValueError("boom")
    with run_directory_lock(run_directory):  # not left behind
        pass


def test_run_directory_lock_reclaims_a_stale_lock_file(tmp_path: Path) -> None:
    """A lock file left behind by a run that died is not stale in any way that matters: the OS dropped the `flock`
    with the process, so the next run takes the file and overwrites the dead holder's record."""
    run_directory = tmp_path / "out"
    run_directory.mkdir()
    (run_directory / RUN_LOCK_NAME).write_text(json.dumps({"pid": 999999, "host": "dead-host", "since": "2020-01-01T00:00:00+00:00"}))
    with run_directory_lock(run_directory):
        assert json.loads((run_directory / RUN_LOCK_NAME).read_text())["pid"] == os.getpid()


def test_run_directory_lock_blocks_another_process(tmp_path: Path) -> None:
    run_directory = tmp_path / "out"
    holder = subprocess.Popen(
        [sys.executable, "-c", "import sys; from pathlib import Path; from training.run_lock import run_directory_lock\n"
         f"with run_directory_lock(Path({str(run_directory)!r})):\n    print('locked', flush=True); sys.stdin.readline()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=REPO_ROOT,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        with pytest.raises(RunDirectoryLocked, match=f"pid {holder.pid}"), run_directory_lock(run_directory):
            pass
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.close()
        holder.wait(timeout=30)
    with run_directory_lock(run_directory):
        pass
