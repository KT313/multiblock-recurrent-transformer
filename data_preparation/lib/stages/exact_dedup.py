# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The exact-dedup key of a document and the "seen" set of one build as a Bloom filter under a fixed memory budget.

A Python set[int] costs ~100 B per entry, tens of GB at the 200 B-token scale; :class:`SeenDocuments` wraps an
rbloom.Bloom sized from memory_mb alone (bits = MB x 2^23). Its only error is a false positive (a unique
document dropped as a duplicate), never a kept duplicate. The filter is not persisted: a build refills it from the
hash column of the processed shards on disk (:func:`stored_hashes`).

rbloom's hash_func must return a Python int in [-2^127, 2^127 - 1] that is well mixed across all 128 bits
(the filter seeds a 128-bit LCG with it). Python's hash(int) reduces modulo 2^61 - 1 and would collide
distinct 64-bit keys, hence :func:`mix128`. rbloom sizes the filter as m = -n ln p / ln(2)^2 bits;
:func:`expected_items` inverts that so the memory budget, not a row estimate, decides the size.

How full the filter runs is the build's business, not the config's: the budget is fixed, the rows are whatever the
source turned out to hold. :meth:`SeenDocuments.describe` logs the expected rate for the rows on disk and
:meth:`SeenDocuments.check` refuses a build that would run far past the nominal capacity, where false positives
stop being negligible; both numbers, and the measured load at save time, go into the processed manifest's
`stats["dedup"]` (`lib/stages/build.py`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
from rbloom import Bloom

from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.row_pipeline import normalize_text

log = get_logger(__name__)

TARGET_FALSE_POSITIVE_RATE = 0.001
"""The false-positive rate the filter is sized for when it holds exactly :func:`expected_items` documents."""

BITS_PER_MB = 1 << 23

BLOOM_MAX_LOAD = 2.0
"""Insertions per nominal capacity a build still runs on; above it the filter drops too many unique documents."""

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15  # splitmix64's golden-ratio increment


def text_hash64(text: str, normalize: bool = True) -> int:
    """
    The exact-dedup key of text: the first 64 bits of its MD5 (lone surrogates dropped) as a signed integer, the
    int64 hash column of processed shards. normalize hashes the lower-cased text with whitespace runs collapsed
    (:func:`normalize_text`), so casing and spacing variants of one document share the key.
    """

    if normalize:
        text = normalize_text(text)
    return int.from_bytes(hashlib.md5(text.encode("utf-8", "ignore")).digest()[:8], "big", signed=True)


def _splitmix64(x: int) -> int:
    """
    The splitmix64 finalizer (Steele, Lea & Flood 2014), a bijection on 64-bit integers.
    """

    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _MASK64
    return x ^ (x >> 31)


def mix128(hash64: int) -> int:
    """
    Spread a 64-bit row hash (signed or unsigned) over a signed 128-bit integer for rbloom's hash_func.

    Two steps of the splitmix64 generator seeded with the key give the two halves. Both steps are bijections, so
    distinct 64-bit keys give distinct 128-bit values, every output bit depends on every input bit, and no key maps
    to a zero half (the bare finalizer has the fixed point 0). About 12 integer operations: negligible next to
    hashing the document text.
    """

    low64 = _splitmix64((hash64 + _GAMMA) & _MASK64)
    high64 = _splitmix64((low64 + _GAMMA) & _MASK64)
    value = (high64 << 64) | low64
    return value - (1 << 128) if high64 >> 63 else value


def bits_for_budget(memory_mb: int) -> int:
    """
    The bit-array size of a filter with a memory_mb budget.
    """

    if memory_mb < 1:
        raise ValueError(f"bloom_memory_mb must be at least 1, got {memory_mb}")
    return memory_mb * BITS_PER_MB


def expected_items(memory_mb: int, false_positive_rate: float = TARGET_FALSE_POSITIVE_RATE) -> int:
    """
    The expected_items that makes rbloom allocate (as exactly as it allows) memory_mb for the target rate:
    n = m x ln(2)^2 / -ln(p), the inverse of rbloom's m = -n ln p / ln(2)^2.
    """

    return int(bits_for_budget(memory_mb) * math.log(2) ** 2 / -math.log(false_positive_rate))


def memory_mb_for(rows: int, false_positive_rate: float = TARGET_FALSE_POSITIVE_RATE) -> int:
    """
    The smallest bloom_memory_mb whose nominal capacity covers rows (what an over-full filter is told to ask for).
    """

    return max(1, math.ceil(rows / expected_items(1, false_positive_rate)))


def _compact(rows: int) -> str:
    """
    Row counts in log lines: millions above a million (597 M), grouped digits below.
    """

    return f"{rows / 1_000_000:.0f} M" if rows >= 1_000_000 else f"{rows:,}"


class SeenDocuments:
    """
    The exact-dedup filter of one build: add_if_new per candidate row, add_all to refill from disk.
    """

    def __init__(self, memory_mb: int = 1024) -> None:
        self.memory_mb = memory_mb
        self.expected_items = expected_items(memory_mb)
        # rbloom's stub leaves hash_func unannotated (hash_func=__builtins__.hash); the contract is int -> int
        self._bloom = Bloom(self.expected_items, TARGET_FALSE_POSITIVE_RATE, hash_func=mix128)

    @property
    def nominal_capacity(self) -> int:
        """
        Insertions the filter is sized for: at that many it hits :data:`TARGET_FALSE_POSITIVE_RATE`.
        """

        return self.expected_items

    @property
    def items_in_filter(self) -> float:
        """
        rbloom's estimate of the insertions so far, from the set bits (a refill from disk counts, duplicates do not).
        """

        return self._bloom.approx_items

    def expected_false_positive_rate(self, items: float) -> float:
        """
        The classic Bloom estimate (1 - exp(-k n / m)) ** k for n = items: the share of unique documents this
        filter drops as duplicates once that many were inserted.
        """

        return (1.0 - math.exp(-self._probes * items / self._bloom.size_in_bits)) ** self._probes

    @property
    def _probes(self) -> int:
        """
        The k of the filter, hash functions per key. rbloom derives it from the arguments of the constructor as
        floor(m / n x ln 2) (it truncates, where the classic optimum rounds: k = 9, not 10, at p = 0.001) and does
        not expose it, so it is re-derived here from the same two numbers.
        """

        return max(1, int(self._bloom.size_in_bits / self.expected_items * math.log(2)))

    def add_if_new(self, hash64: int) -> bool:
        """
        Insert hash64; True iff it was not seen before (a false positive reports a new hash as seen).
        """

        if hash64 in self._bloom:
            return False
        self._bloom.add(hash64)
        return True

    def add_all(self, hashes: Iterable[int]) -> None:
        """
        Insert every hash (refilling from :func:`stored_hashes` at the start of a build).
        """

        self._bloom.update(hashes)

    def describe(self, rows_on_disk: int) -> str:
        """
        The one-line log message printed once per source, e.g. `dedup filter: 1024 MB, 2,600,000 rows on disk
        (upper bound of insertions) -> FPR ≈ 0.00 % (0 % of the nominal 597 M rows)`. rows_on_disk is the raw rows
        of the source, the honest upper bound of what this build inserts, not a capacity of the filter.
        """

        rate = self.expected_false_positive_rate(rows_on_disk)
        load = rows_on_disk / self.nominal_capacity
        return (
            f"dedup filter: {self.memory_mb} MB, {rows_on_disk:,} rows on disk (upper bound of insertions) -> "
            f"FPR ≈ {rate * 100:.2f} % ({load * 100:.0f} % of the nominal {_compact(self.nominal_capacity)} rows)"
        )

    def check(self, rows_on_disk: int) -> None:
        """
        Warn about a filter that runs past its nominal capacity, refuse one past :data:`BLOOM_MAX_LOAD` times it:
        beyond that the false positives stop being negligible and the build would silently drop unique documents.
        The budget is a resource knob (unhashed), so raising it costs nothing but memory and invalidates nothing
        on disk.
        """

        load = rows_on_disk / self.nominal_capacity
        if load <= 1.0:
            return
        rate = self.expected_false_positive_rate(rows_on_disk)
        remedy = (
            f"raise dedup.bloom_memory_mb to {memory_mb_for(rows_on_disk)} (it is unhashed: nothing on disk is "
            "invalidated), or give the source fewer rows"
        )
        overload = (
            f"{rows_on_disk:,} rows on disk are {load:.1f}x the nominal capacity of the {self.memory_mb} MB dedup "
            f"filter ({_compact(self.nominal_capacity)} rows), so about {rate * 100:.1f} % of the unique documents "
            "are dropped as duplicates"
        )
        if load > BLOOM_MAX_LOAD:
            raise ValueError(f"{overload}; that is past the {BLOOM_MAX_LOAD:g}x this build accepts: {remedy}")
        log.warning("%s; %s", overload, remedy)


def stored_hashes(parquet_files: Iterable[Path]) -> Iterator[int]:
    """
    The hash column (int64, no nulls) of each processed shard, file by file, to refill the filter.
    """

    for path in parquet_files:
        column = pq.read_table(path, columns=["hash"]).column("hash")
        yield from cast(list[int], column.to_pylist())  # pyarrow types to_pylist as list[Any]
