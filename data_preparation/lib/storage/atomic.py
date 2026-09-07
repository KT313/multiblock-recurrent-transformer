# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The write-a-sibling-then-rename helper every published file goes through: :func:`write_atomically`.

A manifest, a shard directory or a checkpoint must appear under its final name complete or not at all, because the
next run reads whatever is there. os.replace is atomic within one filesystem, so the write goes to a sibling of
the target (same directory, same filesystem) and is renamed over it once complete. A file target and the parent
directory are fsynced (the bytes, then the rename), so a power loss leaves the previous file or the new one, never
an empty one. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TEMP_SUFFIX = ".tmp"


def _fsync_directory(directory: Path) -> None:
    """
    Persist the rename itself: the new file's bytes are on disk after its own fsync, but the directory entry that
    names them is a separate write, and without this a power loss can leave the entry pointing at the old inode.

    Some filesystems refuse to open or fsync a directory (network mounts); the rename is still atomic there, so an
    OSError is ignored.
    """

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _remove(path: Path) -> None:
    """
    Delete path whether it is a file or a directory; a missing path is fine.
    """

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


@contextmanager
def write_atomically(path: Path | str, *, suffix: str = TEMP_SUFFIX) -> Iterator[Path]:
    """
    Yield a temporary path next to path; rename it over path when the block ends without an exception.

    The parent directory is created and a leftover temporary of an earlier crashed write is removed first. If the
    block or the rename raises, the temporary is deleted and the previous path stays untouched.

    The temporary may be a file or a directory. Replacing an existing non-empty directory fails on POSIX, so a
    directory target has to be removed by the caller first::

        with write_atomically(directory / "MANIFEST.json") as tmp:
            tmp.write_text(payload, encoding="utf-8")
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + suffix)
    _remove(temporary)
    try:
        yield temporary
        if temporary.is_file():
            # the rename is atomic against a crash; the fsync makes the bytes durable across a power loss, which
            # could otherwise leave the renamed file empty (a directory's files are the caller's concern)
            fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        _remove(temporary)
        raise
