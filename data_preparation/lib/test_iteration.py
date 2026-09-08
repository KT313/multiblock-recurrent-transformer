# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for lib/iteration.py: chunking, including the rejected sizes.
"""

from __future__ import annotations

import pytest

from data_preparation.lib.iteration import chunks


def test_chunks_groups_consecutive_items_and_keeps_a_short_tail() -> None:
    assert [list(c) for c in chunks(range(7), 3)] == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(chunks([], 3)) == [] and [list(c) for c in chunks(range(2), 5)] == [[0, 1]]


@pytest.mark.parametrize("size", [0, -1])
def test_chunks_rejects_a_size_below_one(size: int) -> None:
    """
    A size of 0 used to yield one-element chunks, which turns a misconfigured batch size into a silent row-by-row
    run instead of an error.
    """

    with pytest.raises(ValueError, match="chunk size must be positive"):
        list(chunks(range(3), size))
