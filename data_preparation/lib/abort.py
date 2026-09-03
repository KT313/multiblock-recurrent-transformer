# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Cooperative cancellation of a build: stages take a ``should_stop`` callable and call :func:`check_stop` between
shards, never inside one. A stop request (another item failed, or Ctrl-C) ends the stage within one shard with
everything already published on disk and its manifest saved."""

from __future__ import annotations

from collections.abc import Callable

StopCheck = Callable[[], bool]


class BuildAborted(RuntimeError):
    """Raised by a stage that stops because the build was cancelled (another item failed, or an interrupt)."""


def check_stop(should_stop: StopCheck | None) -> None:
    """Raise :class:`BuildAborted` when ``should_stop`` says so (None: never)."""
    if should_stop is not None and should_stop():
        raise BuildAborted("build cancelled")
