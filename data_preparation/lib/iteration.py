# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Iteration helpers shared by the stages.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def chunks(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """
    items grouped into consecutive lists of size (the last one may be shorter); a size below 1 is a ValueError
    (0 used to yield one-element chunks, so a batch size read from a config as 0 quietly ran row by row).
    """

    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
