# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The progress-bar interface of the preparation stages and its no-op.

The stages report progress through :class:`Progress`; the live implementation is the dashboard's task row
(lib/ui/dashboard.py), and :class:`NoProgress` stands in when no dashboard is active. :func:`progress_enabled`
is the rule the dashboard opens under: DATA_PREP_PROGRESS not 0 and stderr a terminal (ui.enabled).
"""

from __future__ import annotations

import sys
from types import TracebackType
from typing import Any, Protocol, TextIO

from ui.enabled import display_enabled

ENV_VAR = "DATA_PREP_PROGRESS"


class Progress(Protocol):
    """
    What the stages do with a bar: count updates, show a postfix, and open it as a with block (close is
    what leaving the block does).
    """

    @property
    def n(self) -> int: ...
    @property
    def total(self) -> int | None: ...
    def update(self, n: int = 1) -> Any: ...
    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> Any: ...
    def close(self) -> None: ...
    def __enter__(self) -> Progress: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> Any: ...


class NoProgress:
    """
    The bar without a display: n still counts the updates, from initial.
    """

    def __init__(self, total: int | None = None, initial: int = 0) -> None:
        self.n = initial
        self.total = total

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> NoProgress:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None


def progress_enabled(stream: TextIO | None = None) -> bool:
    """
    False when DATA_PREP_PROGRESS=0 (or false/no/off) or when stream (stderr) is not a TTY.
    """

    return display_enabled(ENV_VAR, sys.stderr if stream is None else stream)
