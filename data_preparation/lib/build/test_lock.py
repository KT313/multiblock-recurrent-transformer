# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.build.lock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from data_preparation.lib.build.lock import BUILD_LOCK_NAME, BuildLocked, build_lock


def test_build_lock_is_exclusive_and_names_the_holder(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with build_lock(root):
        record = json.loads((root / BUILD_LOCK_NAME).read_text())
        assert record["pid"] == os.getpid() and "since" in record
        with pytest.raises(BuildLocked, match=f"pid {os.getpid()} on"), build_lock(root):  # a second open conflicts
            pass
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
        with pytest.raises(BuildLocked, match=f"pid {holder.pid}"), build_lock(root):
            pass
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.close()
        holder.wait(timeout=10)
    with build_lock(root):
        pass
