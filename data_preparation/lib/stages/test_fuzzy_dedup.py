# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the streaming MinHash/LSH near-duplicate removal: rows kept are pinned to the pre-rewrite (serial,
in-process) implementation on a synthetic corpus with planted near- and exact duplicates.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from data_preparation.dataset_config import DedupConfig
from data_preparation.lib.stages.fuzzy_dedup import fuzzy_dedup


def _has_datasketch() -> bool:
    try:
        import datasketch  # noqa: F401
    except ImportError:
        return False
    return True


needs_datasketch = pytest.mark.skipif(not _has_datasketch(), reason="datasketch not installed")

VOCAB = [f"w{i}" for i in range(40)]
N_DOCS = 300

# Kept indices recorded from the pre-rewrite serial `_fuzzy_dedup` (datasketch 2.0.0, seed 1, num_perm 128,
# ngram 5) on `make_corpus()`; stored as the removed sets, kept = everything else.
_REMOVED: dict[float, set[int]] = {
    0.8: {47, 66, 71, 89, 90, 95, 104, 106, 120, 125, 126, 130, 131, 132, 140, 144, 146, 152, 154, 159, 160, 161, 172,
          176, 180, 188, 189, 192, 195, 196, 200, 208, 214, 215, 218, 219, 221, 226, 230, 231, 236, 246, 249, 252, 259,
          260, 265, 266, 267, 268, 270, 272, 274, 277, 281, 282, 284, 285, 288, 290, 291, 292, 293, 295, 296},
    0.95: {47, 66, 90, 106, 126, 131, 132, 140, 152, 154, 159, 160, 161, 172, 180, 196, 208, 215, 218, 219, 221, 231,
           236, 246, 259, 260, 265, 267, 268, 272, 277, 281, 284, 285, 288, 290, 291, 292, 295, 296},
}  # fmt: skip
EXPECTED_KEPT: dict[float, list[int]] = {t: sorted(set(range(N_DOCS)) - removed) for t, removed in _REMOVED.items()}


def make_corpus(seed: int = 0, n_base: int = 200) -> list[str]:
    """
    200 random 30-120 word documents + 60 near-duplicates (1-3 words changed) + 40 exact duplicates, shuffled.
    """

    rng = random.Random(seed)
    docs = [" ".join(rng.choice(VOCAB) for _ in range(rng.randint(30, 120))) for _ in range(n_base)]
    out = list(docs)
    for _ in range(60):
        words = rng.choice(docs).split()
        for _ in range(rng.randint(1, 3)):
            words[rng.randrange(len(words))] = rng.choice(VOCAB)
        out.append(" ".join(words))
    for _ in range(40):
        out.append(rng.choice(docs))
    rng.shuffle(out)
    return out


def _rows(docs: list[str]) -> Iterator[dict[str, Any]]:
    for i, text in enumerate(docs):
        yield {"text": text, "i": i}


def test_corpus_is_deterministic() -> None:
    assert make_corpus() == make_corpus()
    assert len(make_corpus()) == N_DOCS


@needs_datasketch
@pytest.mark.parametrize("threshold", [0.8, 0.95])
@pytest.mark.parametrize("pass_workers", [1, 2])
def test_kept_rows_match_reference(threshold: float, pass_workers: int) -> None:
    stats: dict[str, Any] = {}
    dedup = DedupConfig(mode="minhash", threshold=threshold, num_perm=128)
    kept = [r["i"] for r in fuzzy_dedup(_rows(make_corpus()), dedup, stats, pass_workers=pass_workers)]
    assert kept == EXPECTED_KEPT[threshold]
    assert stats["near_duplicates_removed"] == N_DOCS - len(kept)
    assert stats["near_duplicate_rate"] == pytest.approx((N_DOCS - len(kept)) / N_DOCS)
    assert stats["threshold"] == threshold and stats["num_perm"] == 128 and stats["seconds"] >= 0


@needs_datasketch
@pytest.mark.parametrize("pass_workers", [1, 2])
def test_rows_stream_before_input_is_exhausted(pass_workers: int) -> None:
    docs = make_corpus()
    consumed = 0

    def source() -> Iterator[dict[str, Any]]:
        nonlocal consumed
        for i, text in enumerate(docs):
            consumed += 1
            yield {"text": text, "i": i}

    dedup = DedupConfig(mode="minhash", threshold=0.8, num_perm=32)
    out = fuzzy_dedup(source(), dedup, {}, pass_workers=pass_workers, chunk_size=32)
    first = next(out)
    assert first["i"] == 0
    # serial path: exactly one row read; spawn-pool path: at most 2 * pass_workers chunks read ahead
    assert consumed <= (1 if pass_workers <= 1 else 2 * pass_workers * 32) < len(docs)
    list(out)  # drain
    assert consumed == len(docs)


@needs_datasketch
def test_interleaved_in_process_passes_keep_their_own_parameters() -> None:
    """
    Builds run in threads of one process (`lib/build/runner.py`), each with its own dedup settings: two
    in-process passes advanced turn by turn must keep exactly the rows each keeps on its own.
    """

    docs = make_corpus()
    settings = {ngram: DedupConfig(mode="minhash", threshold=0.8, num_perm=32, ngram=ngram) for ngram in (2, 200)}  # 200 > every doc: nothing signed
    alone = {ngram: [r["i"] for r in fuzzy_dedup(_rows(docs), dedup, {})] for ngram, dedup in settings.items()}
    assert alone[2] != alone[200]
    passes = {ngram: fuzzy_dedup(_rows(docs), dedup, {}) for ngram, dedup in settings.items()}
    together: dict[int, list[int]] = {ngram: [] for ngram in passes}
    while passes:
        for ngram, rows in list(passes.items()):
            try:
                together[ngram].append(next(rows)["i"])
            except StopIteration:
                del passes[ngram]
    assert together == alone


@needs_datasketch
def test_rows_too_short_for_an_ngram_are_not_collapsed() -> None:
    rows = [{"text": t} for t in ("int main() {}", "SELECT * FROM users;", "hello world", "another short doc")]
    stats: dict[str, Any] = {}
    kept = list(fuzzy_dedup(iter(rows), DedupConfig(mode="minhash", ngram=5), stats))
    assert kept == rows and stats["too_short_passed"] == 4 and stats["near_duplicates_removed"] == 0


@needs_datasketch
def test_stats_and_empty_input() -> None:
    stats: dict[str, Any] = {}
    assert list(fuzzy_dedup(iter([]), DedupConfig(mode="minhash"), stats)) == []
    assert stats["near_duplicates_removed"] == 0 and "seconds" not in stats


def test_missing_datasketch_is_a_clear_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasketch", None)  # makes `import datasketch` raise ImportError
    with pytest.raises(ImportError, match="datasketch"):
        next(fuzzy_dedup(_rows(["a b c d e f"]), DedupConfig(mode="minhash"), {}))
