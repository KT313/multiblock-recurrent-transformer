# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `write_atomically`: the rename on success, the cleanup on failure, files and directories, the fsyncs
that make both durable.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from data_preparation.lib.storage.atomic import TEMP_SUFFIX, write_atomically


def test_the_temporary_is_a_sibling_and_is_renamed_on_success(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "MANIFEST.json"
    with write_atomically(target) as temporary:
        assert temporary == target.with_name(target.name + TEMP_SUFFIX), "same directory: `os.replace` needs one filesystem"
        assert temporary.parent.is_dir(), "the parent is created"
        temporary.write_text("payload", encoding="utf-8")
        assert not target.exists(), "nothing under the final name until the block ends"
    assert target.read_text(encoding="utf-8") == "payload"
    assert list(target.parent.iterdir()) == [target], "no temporary left behind"


def test_an_existing_file_is_replaced_and_kept_when_the_block_raises(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pth"
    target.write_text("old", encoding="utf-8")
    with write_atomically(target) as temporary:
        temporary.write_text("new", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "new"
    with pytest.raises(RuntimeError, match="disk full"), write_atomically(target) as temporary:
        temporary.write_text("half", encoding="utf-8")
        raise RuntimeError("disk full")
    assert target.read_text(encoding="utf-8") == "new", "the previous file is untouched"
    assert list(tmp_path.iterdir()) == [target], "the half-written temporary is gone"


def test_a_keyboard_interrupt_also_cleans_up(tmp_path: Path) -> None:
    """
    The second Ctrl-C of a run lands here: `BaseException`, not `Exception`.
    """

    target = tmp_path / "checkpoint.pth"
    with pytest.raises(KeyboardInterrupt), write_atomically(target) as temporary:
        temporary.write_text("half", encoding="utf-8")
        raise KeyboardInterrupt
    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_a_leftover_temporary_of_a_crashed_write_is_removed_first(tmp_path: Path) -> None:
    target = tmp_path / "MANIFEST.json"
    leftover = target.with_name(target.name + TEMP_SUFFIX)
    leftover.mkdir()  # a directory this time: whatever the crashed write left, it must not confuse the next one
    (leftover / "stale").write_text("x", encoding="utf-8")
    with write_atomically(target) as temporary:
        assert not temporary.exists()
        temporary.write_text("fresh", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "fresh"


def test_a_directory_can_be_published_the_same_way(tmp_path: Path) -> None:
    """
    The shape the instruct all-at-once build needs: a whole directory renamed into place.
    """

    target = tmp_path / "processed"
    with write_atomically(target) as temporary:
        temporary.mkdir()
        (temporary / "shard-0.parquet").write_bytes(b"rows")
    assert sorted(path.name for path in target.iterdir()) == ["shard-0.parquet"]
    with pytest.raises(OSError), write_atomically(target) as temporary:
        temporary.mkdir()
        (temporary / "shard-0.parquet").write_bytes(b"more")  # replacing a non-empty directory fails on POSIX
    assert (target / "shard-0.parquet").read_bytes() == b"rows"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["processed"], "the temporary directory is removed"


def test_a_failing_rename_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "MANIFEST.json"

    def failing_replace(src: object, dst: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="No space left"), write_atomically(target) as temporary:
        temporary.write_text("payload", encoding="utf-8")
    assert list(tmp_path.iterdir()) == []


def test_the_suffix_can_be_chosen(tmp_path: Path) -> None:
    target = tmp_path / "MANIFEST.json"
    with write_atomically(target, suffix=".partial") as temporary:
        assert temporary.name == "MANIFEST.json.partial"
        temporary.write_text("payload", encoding="utf-8")
    assert target.is_file()


def _fsynced_kinds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    Record "file" / "dir" per `os.fsync` call, in order, in the returned list.
    """

    kinds: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        kinds.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    return kinds


def test_the_bytes_and_then_the_rename_are_made_durable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The rename is a write of the parent directory of its own: fsyncing only the file leaves a power loss free to
    lose the directory entry that names the new bytes.
    """

    kinds = _fsynced_kinds(monkeypatch)
    target = tmp_path / "MANIFEST.json"
    with write_atomically(target) as temporary:
        temporary.write_text("payload", encoding="utf-8")
    assert kinds == ["file", "dir"] and target.read_text(encoding="utf-8") == "payload"


def test_a_filesystem_that_refuses_the_directory_fsync_still_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Some network mounts refuse to open or fsync a directory; the rename is atomic there anyway, so the write must
    not fail over it.
    """

    real_open, real_fsync = os.open, os.fsync

    def refusing_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if Path(str(path)).is_dir():
            raise OSError(13, "Permission denied")
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    def refusing_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(22, "Invalid argument")
        real_fsync(fd)

    target = tmp_path / "MANIFEST.json"
    monkeypatch.setattr(os, "fsync", refusing_fsync)
    with write_atomically(target) as temporary:
        temporary.write_text("one", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "one"
    monkeypatch.setattr(os, "open", refusing_open)
    with write_atomically(target) as temporary:
        temporary.write_text("two", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "two" and list(tmp_path.iterdir()) == [target]
