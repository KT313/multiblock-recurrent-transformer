# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the Bloom-filter exact-dedup set (tiny 1 MB budgets).
"""

from __future__ import annotations

import logging
import math
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.stages.exact_dedup import (
    BITS_PER_MB,
    BLOOM_MAX_LOAD,
    TARGET_FALSE_POSITIVE_RATE,
    SeenDocuments,
    bits_for_budget,
    expected_items,
    memory_mb_for,
    mix128,
    stored_hashes,
    text_hash64,
)

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


@pytest.mark.parametrize(
    ("text", "normalized", "raw"),
    [
        ("hello world", 6824707963431612112, 6824707963431612112),
        ("  Hello\t World\n\nfoo ", 8471811785197293890, -4212963777905507985),
        ("héllo wörld", -1365678327145243118, -1365678327145243118),
        ("a\ud800b", 1765116674205471180, 1765116674205471180),  # a lone surrogate is dropped, not an error
        ("", -3162216497309240828, -3162216497309240828),
    ],
)
def test_text_hash64_pins_the_stored_keys(text: str, normalized: int, raw: int) -> None:
    """
    The `hash` column of every processed shard on disk holds these values: the function must never change them.
    """

    assert text_hash64(text) == normalized and text_hash64(text, normalize=False) == raw


def test_text_hash64_normalizes_case_and_whitespace() -> None:
    assert text_hash64("Hello  World") == text_hash64("hello world") == text_hash64("\nHELLO\tworld\n")
    assert text_hash64("Hello  World", normalize=False) != text_hash64("hello world", normalize=False)
    assert text_hash64("hello world") != text_hash64("hello worlds")
    assert -(2**63) <= text_hash64("x") < 2**63
    pa.array([text_hash64("x")], type=pa.int64())  # fits the parquet column type


def _random_hashes(count: int, seed: int) -> list[int]:
    """
    count distinct signed 64-bit integers, the range :func:`text_hash64` produces.
    """

    rng = random.Random(seed)
    values: set[int] = set()
    while len(values) < count:
        values.add(rng.randint(INT64_MIN, INT64_MAX))
    return sorted(values)


def _write_shard(path: Path, hashes: list[int]) -> Path:
    table = pa.table({"text": [f"doc {h}" for h in hashes], "hash": pa.array(hashes, type=pa.int64())})
    pq.write_table(table, path)
    return path


# --- mixer -----------------------------------------------------------------------------------------------------------


def test_mix128_stays_in_signed_128_bit_range_and_is_injective_on_a_sample() -> None:
    inputs = [*_random_hashes(20_000, seed=1), 0, 1, -1, INT64_MIN, INT64_MAX, 2, 3, 1 << 32]
    outputs = [mix128(h) for h in inputs]
    assert all(-(1 << 127) <= v <= (1 << 127) - 1 for v in outputs)
    assert len(set(outputs)) == len(set(inputs))


def test_mix128_uses_all_128_bits_for_small_inputs() -> None:
    for h in (0, 1, 2, 3, 1000):
        unsigned = mix128(h) % (1 << 128)
        assert unsigned >> 64 != 0, "upper half must not stay zero"
        assert unsigned & ((1 << 64) - 1) != 0


def test_mix128_is_deterministic() -> None:
    assert mix128(123456789) == mix128(123456789)
    assert mix128(-5) == mix128(-5 + (1 << 64)), "signed and unsigned spellings of the same 64-bit key agree"


# --- sizing and the false-positive arithmetic ------------------------------------------------------------------------


def test_bits_for_budget_and_expected_items() -> None:
    assert bits_for_budget(1) == BITS_PER_MB == 8_388_608
    assert bits_for_budget(1024) == 1 << 33
    n = expected_items(1024)
    assert n == int((1 << 33) * math.log(2) ** 2 / -math.log(TARGET_FALSE_POSITIVE_RATE))
    assert 590_000_000 < n < 600_000_000
    with pytest.raises(ValueError, match="at least 1"):
        bits_for_budget(0)


def test_describe_names_the_budget_the_rows_and_the_expected_rate() -> None:
    seen = SeenDocuments(memory_mb=1)
    assert seen.nominal_capacity == 583_450, "the row count the message quotes"
    assert seen.describe(2_000) == (
        "dedup filter: 1 MB, 2,000 rows on disk (upper bound of insertions) -> "
        "FPR ≈ 0.00 % (0 % of the nominal 583,450 rows)"
    )
    assert seen.describe(1_166_900) == (
        "dedup filter: 1 MB, 1,166,900 rows on disk (upper bound of insertions) -> "
        "FPR ≈ 4.83 % (200 % of the nominal 583,450 rows)"
    ), "an over-full filter says so in the same line"
    # a capacity in the millions is compact, as in the README line of the 1024 MB default ("the nominal 597 M rows")
    assert "the nominal 1 M rows" in SeenDocuments(memory_mb=2).describe(2_000)


def test_expected_false_positive_rate_matches_the_target_at_the_nominal_capacity() -> None:
    """
    The filter is sized for `TARGET_FALSE_POSITIVE_RATE` at its nominal capacity; twice as many insertions cost
    roughly 50x that (the rate rises steeply, which is what `check` refuses).
    """

    seen = SeenDocuments(memory_mb=1)
    assert seen.expected_false_positive_rate(0) == 0.0
    at_nominal = seen.expected_false_positive_rate(seen.nominal_capacity)
    assert TARGET_FALSE_POSITIVE_RATE / 2 < at_nominal < 2 * TARGET_FALSE_POSITIVE_RATE
    at_twice = seen.expected_false_positive_rate(2 * seen.nominal_capacity)
    assert 0.02 < at_twice < 0.09, at_twice
    assert seen.expected_false_positive_rate(200) < at_nominal, "monotone in the insertions"


def test_check_is_quiet_up_to_the_nominal_capacity(caplog: pytest.LogCaptureFixture) -> None:
    seen = SeenDocuments(memory_mb=1)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        seen.check(seen.nominal_capacity)
    assert caplog.text == ""


def test_check_warns_over_the_nominal_capacity_and_names_the_budget(caplog: pytest.LogCaptureFixture) -> None:
    seen = SeenDocuments(memory_mb=1)
    rows = int(1.5 * seen.nominal_capacity)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        seen.check(rows)  # a warning, not a refusal: 1.5x is under BLOOM_MAX_LOAD
    assert f"{rows:,} rows on disk are 1.5x the nominal capacity of the 1 MB dedup filter (583,450 rows)" in caplog.text
    assert "1.2 % of the unique documents are dropped as duplicates" in caplog.text
    assert "raise dedup.bloom_memory_mb to 2 (it is unhashed: nothing on disk is invalidated)" in caplog.text


def test_check_refuses_past_the_maximum_load() -> None:
    seen = SeenDocuments(memory_mb=1)
    rows = int(BLOOM_MAX_LOAD * seen.nominal_capacity) + 1
    with pytest.raises(ValueError, match=re.escape("that is past the 2x this build accepts")) as error:
        seen.check(rows)
    assert "4.8 % of the unique documents are dropped as duplicates" in str(error.value)
    assert "raise dedup.bloom_memory_mb to 3" in str(error.value), "the budget that brings it back under the nominal capacity"
    seen.check(int(BLOOM_MAX_LOAD * seen.nominal_capacity))  # exactly at the limit still builds (a warning)


def test_memory_mb_for_covers_the_rows() -> None:
    assert memory_mb_for(0) == 1 and memory_mb_for(1) == 1
    assert memory_mb_for(expected_items(1)) == 1
    assert memory_mb_for(expected_items(1) + 1) == 2
    assert expected_items(memory_mb_for(2_600_000)) > 2_600_000


def test_items_in_filter_estimates_the_insertions() -> None:
    seen = SeenDocuments(memory_mb=1)
    assert seen.items_in_filter == 0
    hashes = _random_hashes(10_000, seed=6)
    seen.add_all(hashes)
    assert 0.98 * len(hashes) < seen.items_in_filter < 1.02 * len(hashes)
    seen.add_all(hashes)  # the same documents again: nothing is inserted
    assert seen.items_in_filter < 1.02 * len(hashes)


def test_size_in_bits_is_the_budget_within_a_byte() -> None:
    seen = SeenDocuments(memory_mb=1)
    assert abs(seen._bloom.size_in_bits - BITS_PER_MB) <= 8
    assert seen._bloom.size_in_bits == BITS_PER_MB  # rbloom rounds up to whole bytes; 1 MB is already whole
    two = SeenDocuments(memory_mb=2)
    assert abs(two._bloom.size_in_bits - 2 * BITS_PER_MB) <= 8


# --- the filter ------------------------------------------------------------------------------------------------------


def test_add_if_new_first_true_then_false() -> None:
    seen = SeenDocuments(memory_mb=1)
    assert seen.add_if_new(42) is True
    assert seen.add_if_new(42) is False
    assert 42 in seen._bloom
    assert seen.add_if_new(-42) is True
    assert seen.add_if_new(INT64_MIN) is True
    assert seen.add_if_new(INT64_MIN) is False
    assert 0.5 < seen._bloom.approx_items < 6


def test_no_false_negatives() -> None:
    seen = SeenDocuments(memory_mb=1)
    hashes = _random_hashes(5_000, seed=2)
    assert all(seen.add_if_new(h) for h in hashes[:2_500])
    seen.add_all(hashes[2_500:])
    assert all(h in seen._bloom for h in hashes)
    assert not any(seen.add_if_new(h) for h in hashes)


def test_default_budget_is_1024_mb() -> None:
    # constructing a 1 GB filter in tests is avoided; the default is checked without allocating
    assert SeenDocuments.__init__.__defaults__ == (1024,)
    assert expected_items(1024) == expected_items(1024, TARGET_FALSE_POSITIVE_RATE)


def test_observed_false_positive_rate_roughly_matches_formula() -> None:
    seen = SeenDocuments(memory_mb=1)
    inserted = 600_000
    rng = random.Random(3)
    seen.add_all(rng.randint(INT64_MIN, INT64_MAX) for _ in range(inserted))
    fresh = [rng.randint(INT64_MIN, INT64_MAX) for _ in range(100_000)]  # collisions with the inserts: ~3e-9 each
    hits = sum(h in seen._bloom for h in fresh)
    observed = hits / len(fresh)
    k = 9  # rbloom's probe count for the 0.1 % target: floor(m / n x ln 2), what `expected_false_positive_rate` re-derives
    expected = (1 - math.exp(-k * inserted / BITS_PER_MB)) ** k  # the classic Bloom estimate: ~1e-3 -> ~100 hits
    assert expected / 3 < observed < expected * 3, (observed, expected)
    assert seen.expected_false_positive_rate(inserted) == pytest.approx(expected), "the method uses rbloom's own k"


def test_planted_duplicate_survives_a_restart_via_stored_hashes(tmp_path: Path) -> None:
    hashes = _random_hashes(3_000, seed=4)
    filter_a = SeenDocuments(memory_mb=1)
    kept = [h for h in hashes if filter_a.add_if_new(h)]
    assert kept == hashes
    shard_files = [
        _write_shard(tmp_path / "data-00000.parquet", kept[:1_000]),
        _write_shard(tmp_path / "data-00001.parquet", kept[1_000:]),
    ]
    # "restart": a new filter refilled from the shards on disk sees A's rows as duplicates
    filter_b = SeenDocuments(memory_mb=1)
    filter_b.add_all(stored_hashes(shard_files))
    planted = hashes[1_234]
    assert filter_b.add_if_new(planted) is False
    assert all(filter_b.add_if_new(h) is False for h in hashes)
    known = set(hashes)
    unseen = [h for h in _random_hashes(1_000, seed=5) if h not in known]
    assert sum(filter_b.add_if_new(h) for h in unseen) == len(unseen)


def test_stored_hashes_yields_each_file_in_order(tmp_path: Path) -> None:
    first = [1, -2, INT64_MAX]
    second = [INT64_MIN, 0]
    files = [_write_shard(tmp_path / "a.parquet", first), _write_shard(tmp_path / "b.parquet", second)]
    values = list(stored_hashes(files))
    assert values == first + second
    assert all(type(v) is int for v in values)
    assert list(stored_hashes([])) == []
