# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Progress bars for the preparation stages (the only module that imports ``tqdm``).

``progress(...)`` returns a ``tqdm`` bar on stderr, or a no-op with the same interface when progress is disabled:
``DATA_PREP_PROGRESS=0`` in the environment, or stderr is not a terminal. Log lines are written through
:func:`write_line` (``tqdm.write``) so they do not garble an open bar.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Any, Protocol, TextIO, TypeVar

from tqdm import tqdm

ENV_VAR = "DATA_PREP_PROGRESS"

T = TypeVar("T")


class Progress(Protocol):
    """The subset of the ``tqdm`` interface the stages use."""

    def update(self, n: int = 1) -> Any: ...
    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> Any: ...
    def set_description(self, desc: str | None = None, refresh: bool = True) -> Any: ...
    def close(self) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __enter__(self) -> Progress: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> Any: ...


class NoProgress:
    """No-op stand-in for a ``tqdm`` bar (iteration passes the wrapped iterable through)."""

    def __init__(self, iterable: Iterable[Any] | None = None) -> None:
        self._iterable = iterable
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        return None

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        return None

    def close(self) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def __enter__(self) -> NoProgress:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


def progress_enabled(stream: TextIO | None = None) -> bool:
    """False when ``DATA_PREP_PROGRESS=0`` (or ``false``/``no``/``off``) or when ``stream`` (stderr) is not a TTY."""
    if os.environ.get(ENV_VAR, "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    stream = sys.stderr if stream is None else stream
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def progress(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "row",
    leave: bool = True,
) -> Progress:
    """A ``tqdm`` bar over ``iterable`` (or a manual one with ``total``) on stderr, or :class:`NoProgress`."""
    if not progress_enabled():
        return NoProgress(iterable)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
    )


def write_line(text: str, stream: TextIO) -> None:
    """Write ``text`` (plus newline) to ``stream`` without garbling open progress bars."""
    tqdm.write(text, file=stream)
