# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One helper for the write-a-sibling-then-rename pattern: :func:`write_atomically`.

Every file the pipeline publishes must appear under its final name complete or not at all — a manifest, a shard
directory, a training checkpoint. A crash, a full disk or the second Ctrl-C in the middle of a write must never
leave a truncated file behind, because the next run reads whatever is there (``find_latest_checkpoint`` would pick a
half-written checkpoint, a truncated manifest would send a build back to shard 0).

``os.replace`` is atomic within one filesystem, so the write goes to a sibling of the target — same directory, hence
same filesystem — and is renamed over it once it is complete. Stdlib only, no torch, no pyarrow."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TEMP_SUFFIX = ".tmp"


def _remove(path: Path) -> None:
    """Delete ``path`` whether it is a file or a directory; a missing path is fine."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


@contextmanager
def write_atomically(path: Path | str, *, suffix: str = TEMP_SUFFIX) -> Iterator[Path]:
    """Yield a temporary path next to ``path``; rename it over ``path`` when the block ends without an exception.

    The parent directory is created; a leftover temporary of an earlier, crashed write is removed before the block
    (nothing else may be writing the same target: one build per directory, one process per run directory). If the
    block raises — or the rename itself fails — the temporary is deleted and the exception propagates, leaving the
    previous ``path`` (if any) untouched.

    The temporary may be a file or a directory; ``os.replace`` renames either, but replacing an existing *non-empty*
    directory fails on POSIX, so a directory target has to be removed by the caller first::

        with write_atomically(directory / "MANIFEST.json") as tmp:
            tmp.write_text(payload, encoding="utf-8")
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + suffix)
    _remove(temporary)
    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        _remove(temporary)
        raise
