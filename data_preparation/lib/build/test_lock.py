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
from contextlib import ExitStack
from pathlib import Path

import pytest

from data_preparation.lib.build.lock import BUILD_LOCK_NAME, TRAIN_LOCK_NAME, DatasetLease, RunLocked, build_lock, dataset_lock, run_lock


def test_shared_readers_exclude_writers_until_last_reader_exits(tmp_path: Path) -> None:
    path = tmp_path / BUILD_LOCK_NAME
    path.write_text("old holder metadata must not be changed or trusted")
    original = path.stat()
    with ExitStack() as remaining:
        with dataset_lock(tmp_path, "training", shared=True):
            remaining.enter_context(dataset_lock(tmp_path, "training", shared=True))
            with pytest.raises(RunLocked, match="conflicting lock") as error, build_lock(tmp_path):
                pass
            assert error.value.holder is None
        with pytest.raises(RunLocked), build_lock(tmp_path):
            pass
    with build_lock(tmp_path), pytest.raises(RunLocked), dataset_lock(tmp_path, shared=True):
        pass
    assert path.read_text() == "old holder metadata must not be changed or trusted"
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    assert path.stat().st_ino == original.st_ino


def test_an_existing_lock_file_is_opened_without_asking_to_create_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Some NFS servers reject an open that carries O_CREAT when the directory is not writable, even though the file
    exists (seen on a read-only dataset directory on a fresh compute node). The lock file is created only when it
    is missing; an existing one is opened plainly. This fake os.open behaves like such a server.
    """

    path = tmp_path / BUILD_LOCK_NAME
    real_open = os.open

    def strict_nfs_open(name: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if Path(os.fsdecode(name)) == path and flags & os.O_CREAT and path.exists():
            raise PermissionError(13, "Permission denied", str(name))
        return real_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", strict_nfs_open)
    with dataset_lock(tmp_path, shared=True):
        assert path.exists(), "a missing lock file is created"
    with dataset_lock(tmp_path, shared=True), pytest.raises(RunLocked), build_lock(tmp_path):
        pass  # an existing lock file is opened without O_CREAT and still locks


def test_read_lease_cannot_be_borrowed_for_writing(tmp_path: Path) -> None:
    with dataset_lock(tmp_path, shared=True) as lease:
        with dataset_lock(tmp_path, shared=True, lease=lease) as borrowed:
            assert borrowed is lease
        with pytest.raises(ValueError, match="cannot authorize preparation"), dataset_lock(tmp_path, lease=lease):
            pass
    with dataset_lock(tmp_path) as lease:
        with dataset_lock(tmp_path, shared=True, lease=lease) as borrowed:
            assert borrowed is lease
        with pytest.raises(RunLocked), dataset_lock(tmp_path, shared=True):
            pass  # borrowing read access did not downgrade the exclusive lock


@pytest.mark.timeout(15)
def test_shared_lock_coexists_across_processes(tmp_path: Path) -> None:
    script = "\n".join([
        "import sys",
        "from pathlib import Path",
        "from data_preparation.lib.build.lock import dataset_lock, build_lock, RunLocked",
        "root = Path(sys.argv[1])",
        "with dataset_lock(root, shared=True):",
        "    try:",
        "        with build_lock(root): raise AssertionError('writer entered')",
        "    except RunLocked: pass",
    ])
    with dataset_lock(tmp_path, shared=True):
        subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=10)
        with pytest.raises(RunLocked), build_lock(tmp_path):
            pass  # child exit did not release the parent's ownership
    with build_lock(tmp_path):
        pass


def test_exclusive_run_lock_names_the_holder(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with run_lock(root / BUILD_LOCK_NAME, "data preparation"):
        record = json.loads((root / BUILD_LOCK_NAME).read_text())
        assert record["pid"] == os.getpid() and record["program"] == "data preparation" and "since" in record
        with pytest.raises(RunLocked, match=f"pid {os.getpid()} on") as exc, run_lock(root / BUILD_LOCK_NAME, "data preparation"):  # a second open conflicts
            pass
        message = str(exc.value)
        assert message.startswith("data preparation expects one run at a time on this system; one is already running (started 20")
        assert str(root) in message
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
        with pytest.raises(RunLocked, match="conflicting lock"), build_lock(root):
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


def test_aliases_borrowing_and_continuous_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with dataset_lock(Path("dataset"), "training") as lease:
        inode = (root / BUILD_LOCK_NAME).stat().st_ino
        with dataset_lock(alias, lease=lease) as borrowed:
            assert borrowed is lease
            with pytest.raises(RunLocked, match=str(root)), build_lock(alias):
                pass
        with pytest.raises(RunLocked), build_lock(root):
            pass
        with dataset_lock(tmp_path / "independent"):
            pass
    with build_lock(root):
        assert (root / BUILD_LOCK_NAME).stat().st_ino == inode


def test_invalid_borrowed_leases_fail(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="cannot be constructed"):
        DatasetLease()
    forged = object.__new__(DatasetLease)
    with pytest.raises(ValueError, match="invalid dataset lease"), dataset_lock(tmp_path, lease=forged):
        pass
    with dataset_lock(tmp_path) as lease, pytest.raises(ValueError, match="invalid dataset lease"), dataset_lock(tmp_path / "other", lease=lease):
        pass
    with pytest.raises(ValueError, match="invalid dataset lease"), dataset_lock(tmp_path, lease=lease):
        pass


@pytest.mark.timeout(15)
def test_forked_child_cannot_borrow_parent_lease(tmp_path: Path) -> None:
    with dataset_lock(tmp_path) as lease:
        pid = os.fork()
        if pid == 0:
            try:
                with dataset_lock(tmp_path, lease=lease):
                    os._exit(2)
            except ValueError:
                os._exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.timeout(15)
def test_dead_standalone_holder_releases_same_inode(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", "import sys; from pathlib import Path; from data_preparation.lib.build.lock import dataset_lock\n"
         f"with dataset_lock(Path({str(tmp_path)!r}), 'training'):\n    print('locked', flush=True); sys.stdin.readline()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        inode = (tmp_path / BUILD_LOCK_NAME).stat().st_ino
        holder.kill()
        holder.wait(timeout=5)
        with build_lock(tmp_path):
            assert (tmp_path / BUILD_LOCK_NAME).stat().st_ino == inode
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)
