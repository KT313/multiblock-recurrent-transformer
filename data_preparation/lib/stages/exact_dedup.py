# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The exact-dedup "seen" set as a Bloom filter under a fixed memory budget.

A Python ``set[int]`` costs ~100 B per entry, tens of GB at the 200 B-token scale; :class:`SeenDocuments` wraps an
``rbloom.Bloom`` whose bit array is sized from ``memory_mb`` alone (bits = MB x 2^23) and costs O(1) per document.
Its only error is a false positive — a unique document dropped as a duplicate — never a kept duplicate. The filter
is deliberately not persisted: a build refills it from the ``hash`` column of the processed shards already on disk
(:func:`stored_hashes`), which is the same set a full pass would have accumulated.

rbloom's ``hash_func`` must return a Python int in ``[-2^127, 2^127 - 1]`` (``OverflowError`` otherwise); the
filter seeds a 128-bit linear congruential generator with it and takes the ``k`` probe positions from successive
states, so the value should be well mixed across all 128 bits. Python's ``hash(int)`` reduces modulo ``2^61 - 1``
and would collide distinct 64-bit keys, and a 64-bit key with zero upper bits is a poor LCG seed, hence
:func:`mix128`. rbloom sizes the filter as ``m = -n ln p / ln(2)^2`` bits (rounded up to whole bytes) and uses
``k = floor(m / n x ln 2)`` probes; both are mirrored here so the printed false-positive rate is the filter's.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
from rbloom import Bloom

TARGET_FALSE_POSITIVE_RATE = 0.001
"""The false-positive rate the filter is sized for when it holds exactly :func:`expected_items` documents."""

BITS_PER_MB = 1 << 23

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15  # splitmix64's golden-ratio increment


def _splitmix64(x: int) -> int:
    """The splitmix64 finalizer (Steele, Lea & Flood 2014), a bijection on 64-bit integers."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _MASK64
    return x ^ (x >> 31)


def mix128(hash64: int) -> int:
    """Spread a 64-bit row hash (signed or unsigned) over a signed 128-bit integer for rbloom's ``hash_func``.

    Two steps of the splitmix64 generator seeded with the key (add the increment, finalise) give the two halves;
    both steps are bijections, so distinct 64-bit keys give distinct 128-bit values, every output bit depends on
    every input bit, and no key maps to a zero half (the bare finalizer has the fixed point 0). Pure Python and
    ~12 integer operations: negligible next to hashing the document text.
    """
    lo = _splitmix64((hash64 + _GAMMA) & _MASK64)
    hi = _splitmix64((lo + _GAMMA) & _MASK64)
    value = (hi << 64) | lo
    return value - (1 << 128) if hi >> 63 else value


def bits_for_budget(memory_mb: int) -> int:
    """The bit-array size of a filter with a ``memory_mb`` budget."""
    if memory_mb < 1:
        raise ValueError(f"bloom_memory_mb must be at least 1, got {memory_mb}")
    return memory_mb * BITS_PER_MB


def expected_items(memory_mb: int, false_positive_rate: float = TARGET_FALSE_POSITIVE_RATE) -> int:
    """The ``expected_items`` that makes rbloom allocate (as exactly as it allows) ``memory_mb`` for the target rate:
    ``n = m x ln(2)^2 / -ln(p)``, the inverse of rbloom's ``m = -n ln p / ln(2)^2``."""
    return int(bits_for_budget(memory_mb) * math.log(2) ** 2 / -math.log(false_positive_rate))


def probe_count(false_positive_rate: float = TARGET_FALSE_POSITIVE_RATE) -> int:
    """The number of probes ``k`` rbloom derives for the target rate: ``floor(m / n x ln 2) = floor(-log2 p)`` (it
    truncates rather than rounds, verified against the ``k`` in its serialised header; 9 for 0.1 %)."""
    return int(-math.log2(false_positive_rate))


def expected_false_positive_rate(memory_mb: int, rows: int) -> float:
    """The classic Bloom estimate ``(1 - e^(-k rows / m))^k`` for a filter of ``memory_mb`` after ``rows`` inserts,
    with rbloom's ``k`` for the target rate (an upper bound on the share of unique documents dropped)."""
    if rows < 0:
        raise ValueError(f"rows must not be negative, got {rows}")
    m = bits_for_budget(memory_mb)
    k = probe_count()
    return (1.0 - math.exp(-k * rows / m)) ** k


def _human_count(n: int) -> str:
    """``2.6 M``-style row counts for log lines."""
    for unit, scale in (("B", 10**9), ("M", 10**6), ("k", 10**3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return str(n)


class SeenDocuments:
    """The exact-dedup filter of one build: ``add_if_new`` per candidate row, ``add_all`` to refill from disk."""

    def __init__(self, memory_mb: int = 1024) -> None:
        self.memory_mb = memory_mb
        self.expected_items = expected_items(memory_mb)
        # rbloom's stub leaves hash_func unannotated (``hash_func=__builtins__.hash``); the contract is int -> int
        self._bloom = Bloom(self.expected_items, TARGET_FALSE_POSITIVE_RATE, hash_func=mix128)

    @property
    def size_in_bits(self) -> int:
        """The allocated bit-array size (rbloom rounds the budget up to whole bytes)."""
        return self._bloom.size_in_bits

    @property
    def approx_items(self) -> float:
        """rbloom's estimate of the number of distinct hashes inserted (from the fill ratio)."""
        return self._bloom.approx_items

    def __contains__(self, hash64: int) -> bool:
        return hash64 in self._bloom

    def add_if_new(self, hash64: int) -> bool:
        """Insert ``hash64``; True iff it was not seen before (a false positive reports a new hash as seen)."""
        if hash64 in self._bloom:
            return False
        self._bloom.add(hash64)
        return True

    def add_all(self, hashes: Iterable[int]) -> None:
        """Insert every hash (refilling from :func:`stored_hashes` at the start of a build)."""
        self._bloom.update(hashes)

    def expected_false_positive_rate(self, rows: int) -> float:
        """See :func:`expected_false_positive_rate` for this filter's budget."""
        return expected_false_positive_rate(self.memory_mb, rows)

    def describe(self, rows_needed: int) -> str:
        """The one-line log message printed once per source, e.g.
        ``dedup filter: 1024 MB, ~2.6 M rows -> FPR ≈ 3e-08``."""
        fpr = self.expected_false_positive_rate(rows_needed)
        return f"dedup filter: {self.memory_mb} MB, ~{_human_count(rows_needed)} rows -> FPR ≈ {fpr:.0e}"


def stored_hashes(parquet_files: Iterable[Path]) -> Iterator[int]:
    """The ``hash`` column (int64, no nulls) of each processed shard, file by file, to refill the filter."""
    for path in parquet_files:
        column = pq.read_table(path, columns=["hash"]).column("hash")
        yield from cast(list[int], column.to_pylist())  # pyarrow types to_pylist as list[Any]
