# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.stages.build: the per-shard resumable build of pretrain sources (length filter,
dedup, quality, decontamination, token clamp, fuzzy dedup), the all-at-once shuffled build of instruct sources
(inversions, dedup, empty / over-cap removal) and the dedup-filter refill across a restart, on local sources."""

from __future__ import annotations

import logging
import random
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from data_preparation.dataset_config import (
    DatasetConfig,
    DecontaminationConfig,
    DedupConfig,
    ProcessingConfig,
    SourceConfig,
)
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.stages import build as stages_build
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.exact_dedup import text_hash64
from data_preparation.lib.stages.row_pipeline import get_ngram_set, instruct_text
from data_preparation.lib.stages.download import TokenCounter, download, prepare_tokenizer
from data_preparation.lib.storage.manifest import Manifest

Row = dict[str, Any]
CfgFactory = Callable[..., DatasetConfig]
Writer = Callable[[Path, list[Row], str], Path]
Mtimes = Callable[[Path], dict[str, int]]
Reader = Callable[[Path], list[Row]]
Prep = Callable[[DatasetConfig], DatasetConfig]

GOOD = (
    "The quick brown fox jumps over the lazy dog. Then it went home to sleep. It dreamed of chasing rabbits all night."
)


def _words(n: int, start: int = 0) -> str:
    return " ".join(f"tok_{(start + i) % 256}" for i in range(n))


@pytest.fixture
def local_dir(layout: DatasetLayout) -> Path:
    return layout.root.parent / "src"


def _cfg(cfg_factory: CfgFactory, local_dir: Path, processing: ProcessingConfig | None = None, **kwargs: Any) -> DatasetConfig:
    src = SourceConfig(kind="pretrain", loader="local", path=str(local_dir), **kwargs.pop("source", {}))
    return cfg_factory({"s": src}, processing=processing or ProcessingConfig(min_chars=5), **kwargs)


def _prepare(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, texts: Sequence[str | None], with_tokenizer: Prep, **kw: Any
) -> DatasetConfig:
    write = kw.pop("write")
    shard_size = kw.pop("shard_size", 4)
    write(local_dir, [{"text": t} for t in texts], "parquet")
    cfg = with_tokenizer(_cfg(cfg_factory, local_dir, **kw))
    download(cfg, "s", layout, rows_needed=len(texts), shard_size=shard_size)
    return cfg


# --- pretrain: per-shard resumable build -------------------------------------------------------------------------------


def test_build_length_filter_drops_short_and_keeps_stats(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    texts = ["ok " * 5, "tiny", None, _words(6), "y" * 7, "z" * 8]
    cfg = _prepare(cfg_factory, layout, local_dir, texts, with_tokenizer, write=write_local, processing=ProcessingConfig(min_chars=5), shard_size=4)
    m = build_source(cfg, "s", layout)
    rows = read_rows(layout.processed_dir("s"))
    assert [r["text"] for r in rows] == ["ok " * 5, _words(6), "y" * 7, "z" * 8], "short / null dropped, nothing truncated"
    assert [r["tokens"] for r in rows] == [5, 6, 1, 1], "the raw counts are reused as they are"
    assert m.stats["length_filter"] == {"input_samples": 6, "removed_too_short": 1, "removed_invalid": 1, "output_samples": 4}
    assert m.stats["input_rows"] == 6 and m.rows() == 4
    assert m.input_shards == [["data-00000.parquet", 4], ["data-00001.parquet", 2]]
    assert m.columns == ["text", "source", "tokens", "hash"] and m.shuffled is False


def test_build_writes_a_manifest_even_when_every_row_is_dropped(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep
) -> None:
    cfg = _prepare(cfg_factory, layout, local_dir, ["a", "b"], with_tokenizer, write=write_local, processing=ProcessingConfig(min_chars=5))
    m = build_source(cfg, "s", layout)
    assert m.shards == [] and m.rows() == 0 and m.tokens() == 0 and m.stats["length_filter"]["output_samples"] == 0
    assert m.input_shards == [["data-00000.parquet", 2]] and Manifest.load(layout.processed_dir("s")) == m
    assert build_source(cfg, "s", layout) == m


def test_build_of_an_exhausted_raw_dir_with_zero_shards_writes_an_empty_manifest(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep
) -> None:
    """A source whose loader yields nothing (or nothing the converter accepts) is exhausted with zero raw shards; the
    build still records a processed manifest with zero shards, so the planner counts the source as complete."""
    write_local(local_dir, [{"question": "q"}], "parquet")  # the converter rejects the row: nothing is stored
    src = SourceConfig(kind="instruct", loader="local", path=str(local_dir), converter="instruction_input_output")
    cfg = with_tokenizer(cfg_factory({"i": src}))
    raw = download(cfg, "i", layout, rows_needed=5)
    assert raw.shards == [] and raw.exhausted is True
    m = build_source(cfg, "i", layout)
    stored = Manifest.load(layout.processed_dir("i"))
    assert stored == m and m.shards == [] and m.input_shards == [] and m.is_current(cfg.processed_hash("i"))
    # the same for a per-shard (unshuffled pretrain) build
    write_local(local_dir / "empty", [], "jsonl")
    cfg2 = with_tokenizer(cfg_factory({"p": SourceConfig(kind="pretrain", loader="local", path=str(local_dir / "empty"))}))
    raw2 = download(cfg2, "p", layout, rows_needed=5)
    assert raw2.shards == [] and raw2.exhausted is True
    m2 = build_source(cfg2, "p", layout)
    assert Manifest.load(layout.processed_dir("p")) == m2 and m2.shards == [] and m2.input_shards == []


def test_build_exact_dedup_tokens_and_idempotence(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep,
    read_rows: Reader, mtimes: Mtimes,
) -> None:  # fmt: skip
    texts = [_words(5), _words(3, 100), "  " + _words(5).upper() + "\n", _words(5), _words(64)]  # the last one exactly at the cap
    cfg = _prepare(cfg_factory, layout, local_dir, texts, with_tokenizer, write=write_local, max_seq_length=64)
    m = build_source(cfg, "s", layout, shard_size=2)
    processed = layout.processed_dir("s")
    assert m.stage == "processed" and m.token_count == "tokenizer" and m.tokenizer == "synthetic"
    rows = read_rows(processed)
    assert [r["text"] for r in rows] == [_words(5), _words(3, 100), _words(64)], "normalized duplicates dropped, text untouched"
    assert [r["tokens"] for r in rows] == [5, 3, 64], "the raw counts (true counts of the stored text) reused"
    assert {r["source"] for r in rows} == {"s"} and [set(r) for r in rows] == [{"text", "source", "tokens", "hash"}] * 3
    assert [r["hash"] for r in rows] == [text_hash64(r["text"]) for r in rows]
    assert [(s.rows, s.tokens) for s in m.shards] == [(2, 8), (1, 64)] and m.tokens() == 72
    assert m.stats["dedup"] == {"mode": "exact", "duplicates_removed": 2}
    assert m.input_shards == [["data-00000.parquet", 4], ["data-00001.parquet", 1]]
    before = mtimes(processed)
    assert build_source(cfg, "s", layout, shard_size=2) == m and mtimes(processed) == before


def test_build_clamps_stored_counts_to_a_lowered_cap(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    """Lowering `max_seq_length` never re-downloads (it is not part of the raw hash); the build clamps the stored
    counts to the new cap instead."""
    cfg = _prepare(cfg_factory, layout, local_dir, [_words(30), _words(3)], with_tokenizer, write=write_local, max_seq_length=64)
    assert [r["tokens"] for r in read_rows(layout.raw_dir("s"))] == [30, 3]
    lowered = replace(cfg, max_seq_length=8)
    assert lowered.raw_hash("s") == cfg.raw_hash("s") and lowered.processed_hash("s") != cfg.processed_hash("s")
    m = build_source(lowered, "s", layout)
    assert [r["tokens"] for r in read_rows(layout.processed_dir("s"))] == [8, 3] and m.tokens() == 11
    assert [r["tokens"] for r in read_rows(layout.raw_dir("s"))] == [30, 3], "raw untouched"


def test_build_estimate_mode_caps_too(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    cfg = _prepare(cfg_factory, layout, local_dir, ["a" * 40, "b" * 400], with_tokenizer, write=write_local, token_count="estimate", max_seq_length=50)
    m = build_source(cfg, "s", layout)
    assert m.token_count == "estimate" and m.tokenizer is None
    assert [r["tokens"] for r in read_rows(layout.processed_dir("s"))] == [10, 50]
    assert [len(r["text"]) for r in read_rows(layout.processed_dir("s"))] == [40, 200], "the download cut the long text at 4 chars/token"


def test_build_appends_only_the_new_shards_and_refills_the_dedup_filter(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep,
    read_rows: Reader, mtimes: Mtimes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:  # fmt: skip
    """A top-up processes only the raw shards not yet covered, leaves the old processed shards untouched, removes
    duplicates across old and new shards (the fresh Bloom filter of the second call is refilled from the `hash`
    column on disk) and ends with exactly the rows of a fresh full pass."""
    first = [_words(6, i) for i in range(6)] + [_words(6, 0)]  # one duplicate inside
    cfg = _prepare(cfg_factory, layout, local_dir, first, with_tokenizer, write=write_local, shard_size=3)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        m1 = build_source(cfg, "s", layout, shard_size=4)
    assert "s: dedup filter: 1 MB, ~7 rows" in caplog.text, "the test config's 1 MB budget, sized for the raw rows"
    processed = layout.processed_dir("s")
    old_rows = read_rows(processed)
    assert len(old_rows) == 6 and m1.stats["input_rows"] == 7 and m1.columns == ["text", "source", "tokens", "hash"]
    before = mtimes(processed)

    # append: duplicates of old rows plus new ones (the duplicates sit in a NEW raw shard, their originals in OLD processed shards)
    write_local(local_dir, [{"text": _words(6, 2)}, {"text": _words(6, 50)}, {"text": _words(6, 3)}, {"text": _words(6, 51)}], "parquet")
    download(cfg, "s", layout, rows_needed=11, shard_size=3)
    counted: list[int] = []
    original = TokenCounter.count_many

    def spy(self: Any, texts: list[str]) -> list[int]:
        counted.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(TokenCounter, "count_many", spy)
    m2 = build_source(cfg, "s", layout, shard_size=4)
    assert sum(counted) == 0, "raw token counts reused, nothing tokenized"
    assert m2.stats["dedup"]["duplicates_removed"] == 3 and m2.stats["input_rows"] == 11
    assert m2.input_shards == [[f"data-{i:05d}.parquet", n] for i, n in enumerate([3, 3, 1, 3, 1])]
    after = mtimes(processed)
    assert {k: after[k] for k in before} == before, "old processed shards were not rewritten"
    new_rows = read_rows(processed)
    assert new_rows[: len(old_rows)] == old_rows, "old rows byte-identical and in place"
    assert [r["text"] for r in new_rows[len(old_rows) :]] == [_words(6, 50), _words(6, 51)]
    # one processed shard per raw shard with survivors (raw shard 2 is a single duplicate -> no shard; raw shard 3 =
    # two duplicates + one new row -> 1 row)
    assert [(sh.rows, sh.tokens) for sh in m2.shards] == [(3, 18), (3, 18), (1, 6), (1, 6)]

    # golden: a fresh full pass over the same raw data keeps exactly the same rows
    fresh = DatasetLayout(tmp_path / "fresh")
    prepare_tokenizer(cfg, fresh)
    download(cfg, "s", fresh, rows_needed=11, shard_size=3)
    m_fresh = build_source(cfg, "s", fresh, shard_size=4)
    assert read_rows(fresh.processed_dir("s")) == new_rows
    assert m_fresh.tokens() == m2.tokens() and m_fresh.stats == m2.stats
    assert [sh.rows for sh in m_fresh.shards] == [3, 3, 1, 1], "same layout: one processed shard per raw shard"


def test_incremental_build_with_quality_filter_equals_a_full_pass(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep,
    read_rows: Reader, tmp_path: Path,
) -> None:  # fmt: skip
    """A row the quality filter drops must not claim its dedup hash: a later (differently cased) duplicate that
    passes the filter is kept, in the incremental and in the full pass alike."""
    bad_caps = GOOD.upper()  # dropped: too many ALL-CAPS words; same normalized hash as GOOD
    proc = ProcessingConfig(min_chars=5, quality_filter=True)
    cfg = _prepare(cfg_factory, layout, local_dir, [bad_caps, "Another good text. It has sentences. Three of them here."],
                   with_tokenizer, write=write_local, processing=proc, max_seq_length=500, shard_size=2)  # fmt: skip
    m1 = build_source(cfg, "s", layout)
    assert m1.rows() == 1 and m1.stats["quality_filter"]["filtered_count"] == 1
    write_local(local_dir, [{"text": GOOD}], "parquet")
    download(cfg, "s", layout, rows_needed=3, shard_size=2)
    m2 = build_source(cfg, "s", layout)
    incremental = read_rows(layout.processed_dir("s"))
    assert [r["text"] for r in incremental][-1] == GOOD and m2.stats["dedup"]["duplicates_removed"] == 0

    fresh = DatasetLayout(tmp_path / "fresh")
    prepare_tokenizer(cfg, fresh)
    download(cfg, "s", fresh, rows_needed=3, shard_size=2)
    m_fresh = build_source(cfg, "s", fresh)
    assert read_rows(fresh.processed_dir("s")) == incremental and m_fresh.stats == m2.stats


def test_build_rebuilds_when_shards_predate_the_columns_or_raw_changed(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep,
    read_rows: Reader, caplog: pytest.LogCaptureFixture,
) -> None:  # fmt: skip
    cfg = _prepare(cfg_factory, layout, local_dir, [_words(4, i) for i in range(3)], with_tokenizer, write=write_local)
    m = build_source(cfg, "s", layout)
    processed = layout.processed_dir("s")
    rows = read_rows(processed)

    # a processed directory from before the current columns: rebuilt from the raw shards (no download)
    legacy = Manifest.load(processed)
    assert legacy is not None
    legacy.columns = []
    legacy.save(processed)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        rebuilt = build_source(cfg, "s", layout)
    assert "predate the current columns" in caplog.text and rebuilt.columns == m.columns
    assert read_rows(processed) == rows

    # raw shards that are no longer a prefix of what was covered: everything is rebuilt
    caplog.clear()
    changed = Manifest.load(processed)
    assert changed is not None
    changed.input_shards[0][1] = 99
    changed.save(processed)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        rebuilt = build_source(cfg, "s", layout)
    assert "raw shards changed" in caplog.text and rebuilt.input_shards == m.input_shards
    assert read_rows(processed) == rows


def test_stale_rebuild_removes_the_shards_of_the_previous_build(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    """A rebuild from a fresh manifest (stale hash) publishes fewer shards than before: none of the old ones may
    survive unlisted next to the new manifest (round-2 finding)."""
    texts = [_words(6, i) for i in range(6)]
    cfg = _prepare(cfg_factory, layout, local_dir, texts, with_tokenizer, write=write_local, processing=ProcessingConfig(min_chars=5), shard_size=6)
    processed = layout.processed_dir("s")
    build_source(cfg, "s", layout, shard_size=1)
    assert len(list(processed.glob("data-*.parquet"))) == 6
    stricter = with_tokenizer(_cfg(cfg_factory, local_dir, ProcessingConfig(min_chars=1000)))  # a new processed hash, every row dropped
    m = build_source(stricter, "s", layout, shard_size=1)
    assert m.shards == [] and list(processed.glob("data-*.parquet")) == [] and read_rows(processed) == []
    assert Manifest.load(processed) == m


def test_build_no_dedup_mode_keeps_duplicates(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    proc = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="none"))
    cfg = _prepare(cfg_factory, layout, local_dir, ["dup", "dup", "DUP"], with_tokenizer, write=write_local, processing=proc)
    build_source(cfg, "s", layout)
    assert [r["text"] for r in read_rows(layout.processed_dir("s"))] == ["dup", "dup", "DUP"]
    exact_raw = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="exact", normalize=False))
    cfg2 = with_tokenizer(_cfg(cfg_factory, local_dir, exact_raw))
    download(cfg2, "s", layout, rows_needed=3)
    build_source(cfg2, "s", layout)
    assert [r["text"] for r in read_rows(layout.processed_dir("s"))] == ["dup", "DUP"]


def test_build_quality_filter_only_when_enabled(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    texts = [GOOD, "This is one sentence. Another one here."]
    cfg = _prepare(cfg_factory, layout, local_dir, texts, with_tokenizer, write=write_local, max_seq_length=500)
    build_source(cfg, "s", layout)
    assert len(read_rows(layout.processed_dir("s"))) == 2
    on = with_tokenizer(_cfg(cfg_factory, local_dir, ProcessingConfig(min_chars=5, quality_filter=True), max_seq_length=500))
    download(on, "s", layout, rows_needed=2)
    m = build_source(on, "s", layout)
    assert [r["text"] for r in read_rows(layout.processed_dir("s"))] == [GOOD]
    assert m.stats["quality_filter"] == {"enabled": True, "filtered_count": 1, "rejection_reasons": {"too_few_sentences": 1}}


@pytest.mark.parametrize("pass_workers", [1, 2])
def test_build_decontamination_only_when_enabled(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep,
    read_rows: Reader, monkeypatch: pytest.MonkeyPatch, pass_workers: int,
) -> None:  # fmt: skip
    """`pass_workers=2` runs the real spawn pool end to end: the n-grams are loaded once in the parent (where the
    patched loader lives — a spawn child would not see the monkeypatch) and reach the workers as pickled init args."""
    planted = " ".join(f"w{i}" for i in range(20))
    calls: list[tuple[list[str], int, str]] = []

    def fake_load(names: list[str], n: int = 13, cache_dir: str | None = None) -> dict[str, set[str]]:
        calls.append((names, n, str(cache_dir)))
        return {"gsm8k_test": get_ngram_set(planted, n), "mmlu_test": set()}

    monkeypatch.setattr(stages_build, "load_benchmark_ngrams", fake_load)
    texts = [planted, GOOD, planted + " tail"]
    cfg = _prepare(cfg_factory, layout, local_dir, texts, with_tokenizer, write=write_local, max_seq_length=500)
    build_source(cfg, "s", layout, pass_workers=pass_workers)
    assert len(read_rows(layout.processed_dir("s"))) == 3 and calls == []
    decon = DecontaminationConfig(enabled=True, benchmarks=["gsm8k_test", "mmlu_test"])
    on = with_tokenizer(_cfg(cfg_factory, local_dir, ProcessingConfig(min_chars=5, decontamination=decon), max_seq_length=500))
    download(on, "s", layout, rows_needed=3)
    m = build_source(on, "s", layout, pass_workers=pass_workers)
    assert [r["text"] for r in read_rows(layout.processed_dir("s"))] == [GOOD]
    assert m.stats["decontamination"] == {"enabled": True, "contaminated_count": 2, "contaminated_by_benchmark": {"gsm8k_test": 2}}
    assert calls == [(["gsm8k_test", "mmlu_test"], 13, str(layout.benchmark_cache_dir()))]


def test_build_benchmark_load_failure_is_an_error(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    def failing(names: list[str], n: int = 13, cache_dir: str | None = None) -> dict[str, set[str]]:
        raise OSError("hub unreachable")

    monkeypatch.setattr(stages_build, "load_benchmark_ngrams", failing)
    proc = ProcessingConfig(min_chars=1, decontamination=DecontaminationConfig(enabled=True))
    cfg = _prepare(cfg_factory, layout, local_dir, [GOOD], with_tokenizer, write=write_local, processing=proc, max_seq_length=500)
    with pytest.raises(OSError, match="hub unreachable"):
        build_source(cfg, "s", layout)
    assert not (layout.processed_dir("s") / "MANIFEST.json").exists()


def test_build_minhash_removes_near_duplicates_all_at_once(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader, mtimes: Mtimes
) -> None:
    pytest.importorskip("datasketch")
    base = " ".join(f"word{i}" for i in range(200))
    near = base.replace("word100", "changed")
    other = " ".join(f"other{i}" for i in range(200))
    partial = " ".join(f"word{i}" for i in range(100)) + " " + " ".join(f"new{i}" for i in range(100))
    short_a, short_b, short_a_variant = "hello world", "SELECT * FROM users;", "Hello   World"  # < 5 words: no n-grams
    proc = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="minhash", threshold=0.8, num_perm=64))
    cfg = _prepare(cfg_factory, layout, local_dir, [base, near, other, base, partial, short_a, short_b, short_a_variant], with_tokenizer,
                   write=write_local, processing=proc, max_seq_length=500)  # fmt: skip
    m = build_source(cfg, "s", layout)
    processed = layout.processed_dir("s")
    # minhash mode = the normalized exact pass first (second `base`, the case variant of `short_a`), then the fuzzy
    # one (`near`); rows too short for an n-gram are not all collapsed onto the first of them
    assert [r["text"] for r in read_rows(processed)] == [base, other, partial, short_a, short_b]
    assert not processed.with_name("s.tmp").exists()
    dedup_stats = dict(m.stats["dedup"])
    assert dedup_stats.pop("seconds") >= 0
    assert dedup_stats == {
        "mode": "minhash", "duplicates_removed": 2, "threshold": 0.8, "num_perm": 64, "near_duplicates_removed": 1,
        "near_duplicate_rate": 0.25, "too_short_passed": 2,  # rate over the rows that went through the LSH
    }  # fmt: skip
    before = mtimes(processed)
    assert build_source(cfg, "s", layout) == m and mtimes(processed) == before, "complete: a no-op"
    # a top-up rebuilds the folder whole (the LSH index needs every signature)
    write_local(local_dir, [{"text": other.replace("other5", "x")}], "parquet")
    download(cfg, "s", layout, rows_needed=9)
    m2 = build_source(cfg, "s", layout)
    assert m2.stats["input_rows"] == 9 and m2.stats["dedup"]["near_duplicates_removed"] == 2
    assert [r["text"] for r in read_rows(processed)] == [base, other, partial, short_a, short_b]


def test_build_minhash_without_datasketch_raises(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    monkeypatch.setitem(sys.modules, "datasketch", None)  # makes `import datasketch` raise ImportError
    proc = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="minhash"))
    cfg = _prepare(cfg_factory, layout, local_dir, ["a b c d e f"], with_tokenizer, write=write_local, processing=proc)
    with pytest.raises(ImportError, match="datasketch"):
        build_source(cfg, "s", layout)


def test_build_requires_a_raw_manifest(cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path) -> None:
    cfg = _cfg(cfg_factory, local_dir)
    with pytest.raises(FileNotFoundError, match="no current raw manifest"):
        build_source(cfg, "s", layout)


def test_build_publishes_per_raw_shard_and_resumes_after_a_stop(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    texts = [_words(6, i) for i in range(9)] + [_words(6, 1)]  # 10 rows, one duplicate in the last raw shard
    cfg = _prepare(cfg_factory, layout, local_dir, texts, with_tokenizer, write=write_local, shard_size=3)
    processed = layout.processed_dir("s")
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    with pytest.raises(BuildAborted):
        build_source(cfg, "s", layout, should_stop=stop_after_two)
    partial = Manifest.load(processed)
    assert partial is not None and partial.input_shards == [["data-00000.parquet", 3], ["data-00001.parquet", 3]]
    assert [s.rows for s in partial.shards] == [3, 3] and partial.stats["input_rows"] == 6
    assert len(read_rows(processed)) == 6

    m = build_source(cfg, "s", layout)  # resumes behind the covered raw shards
    assert m.input_shards == [[f"data-{i:05d}.parquet", n] for i, n in enumerate([3, 3, 3, 1])]
    assert [s.rows for s in m.shards] == [3, 3, 3] and m.stats["input_rows"] == 10
    assert m.stats["dedup"]["duplicates_removed"] == 1
    assert [r["text"] for r in read_rows(processed)] == texts[:9]


def test_pretrain_source_with_shuffle_is_built_all_at_once_in_seeded_order(
    cfg_factory: CfgFactory, layout: DatasetLayout, local_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    texts = [_words(6, i) for i in range(12)]
    cfg = _prepare(cfg_factory, layout, local_dir, texts, with_tokenizer, write=write_local, shard_size=4, source={"shuffle": True, "seed": 5})
    m = build_source(cfg, "s", layout, shard_size=5)
    out = [r["text"] for r in read_rows(layout.processed_dir("s"))]
    expected = list(texts)
    random.Random(5).shuffle(expected)
    assert out == expected != texts and m.shuffled is True and [s.rows for s in m.shards] == [5, 5, 2]
    assert m.input_shards == [[f"data-{i:05d}.parquet", 4] for i in range(3)]
    # a top-up rebuilds the whole folder in the seeded order of the larger list
    write_local(local_dir, [{"text": _words(6, 100)}], "parquet")
    download(cfg, "s", layout, rows_needed=13, shard_size=4)
    m2 = build_source(cfg, "s", layout, shard_size=5)
    expected = texts + [_words(6, 100)]
    random.Random(5).shuffle(expected)
    assert [r["text"] for r in read_rows(layout.processed_dir("s"))] == expected and m2.stats["input_rows"] == 13


# --- instruct: all-at-once shuffled build --------------------------------------------------------------------------------


def _instruct_row(i: int, n_out: int = 4) -> Row:
    return {"instruction": f"tok_{i} tok_{i + 1}", "input": "", "output": " ".join(f"tok_{(i * 7 + j) % 256}" for j in range(n_out))}


def _instruct_cfg(cfg_factory: CfgFactory, with_tokenizer: Prep, local_dir: Path, max_seq_length: int = 64, **source_kwargs: Any) -> DatasetConfig:
    src = SourceConfig(kind="instruct", loader="local", path=str(local_dir), converter="instruction_input_output", **source_kwargs)
    return with_tokenizer(cfg_factory({"i": src}, max_seq_length=max_seq_length))


def test_instruct_build_columns_dedup_empty_removal_and_seeded_shuffle(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, write_local: Writer, read_rows: Reader, mtimes: Mtimes, tmp_path: Path
) -> None:
    src_dir = layout.root.parent / "c"
    rows = [_instruct_row(i) for i in range(20)] + [_instruct_row(0), {"instruction": "tok_0  TOK_1", "input": "", "output": _instruct_row(0)["output"]}]
    rows += [{"instruction": "", "input": "", "output": "tok_1"}, {"instruction": "tok_1", "input": "", "output": "  "}]
    write_local(src_dir, rows, "jsonl")
    cfg = _instruct_cfg(cfg_factory, with_tokenizer, src_dir, seed=3)
    download(cfg, "i", layout, rows_needed=len(rows), shard_size=8)
    processed = layout.processed_dir("i")
    m = build_source(cfg, "i", layout, shard_size=8)
    out = read_rows(processed)
    assert [set(r) for r in out] == [{"instruction", "input", "output", "tokens", "hash"}] * 20
    assert m.columns == ["instruction", "input", "output", "tokens", "hash"] and m.shuffled is True and m.shuffle_seed == 3
    assert m.stats == {"input_rows": 24, "dedup": {"mode": "exact", "duplicates_removed": 2}, "inverted": 0, "removed_empty": 2, "removed_too_long": 0}
    assert m.input_shards == [["data-00000.parquet", 8], ["data-00001.parquet", 8], ["data-00002.parquet", 8]]
    assert [s.rows for s in m.shards] == [8, 8, 4] and m.tokens() == sum(r["tokens"] for r in out) == 20 * 6
    assert all(r["hash"] == text_hash64(instruct_text(r)) for r in out)
    expected = [{**_instruct_row(i), "tokens": 6} for i in range(20)]
    random.Random(3).shuffle(expected)
    assert [{k: r[k] for k in ("instruction", "input", "output", "tokens")} for r in out] == expected, "seeded shuffle of the survivors, in raw order before the shuffle"
    assert not processed.with_name("i.tmp").exists()
    before = mtimes(processed)
    assert build_source(cfg, "i", layout, shard_size=8) == m and mtimes(processed) == before, "complete: a no-op"

    # same seed -> same order elsewhere; another seed -> another order
    other = DatasetLayout(tmp_path / "other")
    prepare_tokenizer(cfg, other)
    download(cfg, "i", other, rows_needed=len(rows), shard_size=8)
    build_source(cfg, "i", other, shard_size=8)
    assert read_rows(other.processed_dir("i")) == out
    reseeded = _instruct_cfg(cfg_factory, with_tokenizer, src_dir, seed=4)
    reseeded_layout = DatasetLayout(tmp_path / "reseeded")  # a new seed is a new raw hash; the download never deletes raw
    prepare_tokenizer(reseeded, reseeded_layout)
    download(reseeded, "i", reseeded_layout, rows_needed=len(rows), shard_size=8)
    build_source(reseeded, "i", reseeded_layout, shard_size=8)
    reseeded_rows = read_rows(reseeded_layout.processed_dir("i"))
    assert reseeded_rows != out and sorted(r["instruction"] for r in reseeded_rows) == sorted(r["instruction"] for r in out)


def test_instruct_inversions_are_seeded_per_row_and_survive_a_resume(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, write_local: Writer, read_rows: Reader, tmp_path: Path
) -> None:
    """`shuffle: false` puts an instruct source on the per-shard resumable path; the inversion of a row depends only
    on (seed, global raw row index), so a build stopped after two raw shards and resumed equals one uninterrupted
    build, and a shuffled build of the same source inverts the same rows."""
    src_dir = layout.root.parent / "inv"
    rows = [_instruct_row(i) for i in range(40)]
    write_local(src_dir, rows, "jsonl")
    cfg = _instruct_cfg(cfg_factory, with_tokenizer, src_dir, seed=7, input_inversions=0.3, shuffle=False)
    download(cfg, "i", layout, rows_needed=40, shard_size=10)
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    with pytest.raises(BuildAborted):
        build_source(cfg, "i", layout, should_stop=stop_after_two)
    partial = Manifest.load(layout.processed_dir("i"))
    assert partial is not None and len(partial.input_shards) == 2
    m = build_source(cfg, "i", layout)
    resumed = read_rows(layout.processed_dir("i"))
    inverted = [r for r in resumed if r["instruction"].startswith("Given this output")]
    assert 4 <= len(inverted) <= 20 and m.stats["inverted"] == len(inverted) and m.shuffled is False
    counter = TokenCounter(cfg, layout)
    assert all(r["tokens"] == counter.count(instruct_text(r)) != 6 for r in inverted), "tokens recounted"
    assert [r["instruction"] for r in resumed if not r["instruction"].startswith("Given")] == [
        r["instruction"] for i, r in enumerate(rows) if not resumed[i]["instruction"].startswith("Given")
    ], "raw order kept without shuffle"

    fresh = DatasetLayout(tmp_path / "fresh")
    prepare_tokenizer(cfg, fresh)
    download(cfg, "i", fresh, rows_needed=40, shard_size=10)
    build_source(cfg, "i", fresh)
    assert read_rows(fresh.processed_dir("i")) == resumed, "resume == uninterrupted build"
    other_shards = DatasetLayout(tmp_path / "shards")
    prepare_tokenizer(cfg, other_shards)
    download(cfg, "i", other_shards, rows_needed=40, shard_size=7)  # different raw shard boundaries
    build_source(cfg, "i", other_shards)
    assert read_rows(other_shards.processed_dir("i")) == resumed, "inversions independent of shard boundaries"

    shuffled = replace(cfg, sources={"i": replace(cfg.sources["i"], shuffle=None)})
    build_source(shuffled, "i", layout)
    assert sorted(r["instruction"] for r in read_rows(layout.processed_dir("i"))) == sorted(r["instruction"] for r in resumed)


def test_instruct_rows_over_the_cap_are_dropped_and_counted(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "long"
    rows = [_instruct_row(i, n_out=4) for i in range(5)] + [_instruct_row(i, n_out=8) for i in range(5)]  # 6 and 10 tokens
    write_local(src_dir, rows, "jsonl")
    cfg = _instruct_cfg(cfg_factory, with_tokenizer, src_dir, max_seq_length=8)
    raw = download(cfg, "i", layout, rows_needed=10)
    assert [r["tokens"] for r in read_rows(layout.raw_dir("i"))] == [6] * 5, "rows over the cap are dropped at download, never truncated"
    assert raw.dropped_too_long == 5 and raw.exhausted is True
    m = build_source(cfg, "i", layout)
    out = read_rows(layout.processed_dir("i"))
    assert len(out) == 5 and all(r["tokens"] == 6 for r in out) and m.stats["removed_too_long"] == 0  # the build's safety net has nothing left to do


def test_instruct_build_starts_over_when_its_tmp_folder_is_left_behind(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, write_local: Writer, read_rows: Reader, caplog: pytest.LogCaptureFixture
) -> None:
    src_dir = layout.root.parent / "tmp"
    write_local(src_dir, [_instruct_row(i) for i in range(4)], "jsonl")
    cfg = _instruct_cfg(cfg_factory, with_tokenizer, src_dir)
    download(cfg, "i", layout, rows_needed=4)
    leftover = layout.processed_dir("i").with_name("i.tmp")
    leftover.mkdir(parents=True)
    (leftover / "data-00000.parquet").write_bytes(b"junk")
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        m = build_source(cfg, "i", layout)
    assert "leftover" in caplog.text and not leftover.exists() and m.rows() == 4 and len(read_rows(layout.processed_dir("i"))) == 4
    assert not layout.processed_dir("i").with_name("i.old").exists()


def test_swap_into_place_is_rename_aside(tmp_path: Path) -> None:
    """Old aside, new in place, only then a deletion — and without an old folder the aside step is skipped."""
    processed = tmp_path / "i"
    processed.mkdir()
    (processed / "data-00000.parquet").write_bytes(b"old")
    temporary = tmp_path / "i.tmp"
    temporary.mkdir()
    (temporary / "data-00000.parquet").write_bytes(b"new")
    stages_build._swap_into_place(temporary, processed)
    assert (processed / "data-00000.parquet").read_bytes() == b"new"
    assert not temporary.exists() and not (tmp_path / "i.old").exists()
    # first build: no old folder to step aside
    fresh = tmp_path / "j.tmp"
    fresh.mkdir()
    (fresh / "data-00000.parquet").write_bytes(b"only")
    stages_build._swap_into_place(fresh, tmp_path / "j")
    assert (tmp_path / "j" / "data-00000.parquet").read_bytes() == b"only" and not fresh.exists()
    # a stale .old of an earlier crashed swap is cleared before the renames
    stale_old = tmp_path / "i.old"
    stale_old.mkdir()
    (stale_old / "data-00000.parquet").write_bytes(b"stale")
    again = tmp_path / "i.tmp"
    again.mkdir()
    (again / "data-00000.parquet").write_bytes(b"newer")
    stages_build._swap_into_place(again, processed)
    assert (processed / "data-00000.parquet").read_bytes() == b"newer" and not stale_old.exists()


def test_swap_crash_while_deleting_the_old_folder_keeps_the_new_data_in_place(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, write_local: Writer, read_rows: Reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename-aside window: a crash on the final delete leaves the new folder in place and the replaced one as
    ``.old`` — never a moment without the data (the repair step removes the ``.old``)."""
    src_dir = layout.root.parent / "swap"
    write_local(src_dir, [_instruct_row(i) for i in range(4)], "jsonl")
    cfg = _instruct_cfg(cfg_factory, with_tokenizer, src_dir)
    download(cfg, "i", layout, rows_needed=4)
    build_source(cfg, "i", layout)
    write_local(src_dir, [_instruct_row(i) for i in range(4, 8)], "jsonl")
    download(cfg, "i", layout, rows_needed=8)  # a top-up: the next build rebuilds the folder whole and swaps again

    real_rmtree = shutil.rmtree

    def crash_on_old(path: str | Path, *args: Any, **kwargs: Any) -> None:
        if str(path).endswith(".old"):
            raise RuntimeError("crash while deleting the old folder")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", crash_on_old)  # build.py calls it as `shutil.rmtree`, so patching the module reaches it
    with pytest.raises(RuntimeError, match="crash while deleting the old folder"):
        build_source(cfg, "i", layout)
    processed = layout.processed_dir("i")
    old = processed.with_name("i.old")
    assert len(read_rows(processed)) == 8, "the new folder is in place"
    assert len(read_rows(old)) == 4, "the replaced folder survived the crash aside"
    assert not processed.with_name("i.tmp").exists()
