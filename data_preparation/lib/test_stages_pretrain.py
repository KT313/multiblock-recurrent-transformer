# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.stages_pretrain: incremental length filter, streaming process (dedup, quality,
decontamination, token counting, fuzzy dedup, repeat_to_budget) on local parquet sources."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from data_preparation.lib import stages_pretrain
from data_preparation.lib.dataset_config import (
    DatasetConfig,
    DecontaminationConfig,
    DedupConfig,
    ProcessingConfig,
    SourceConfig,
)
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.manifest import Manifest
from data_preparation.lib.row_pipeline import get_ngram_set
from data_preparation.lib.stages_pretrain import length_filter, process
from data_preparation.lib.stages_shared import download

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
def source_dir(layout: DatasetLayout) -> Path:
    return layout.root.parent / "src"


def _cfg(
    cfg_factory: CfgFactory, source_dir: Path, processing: ProcessingConfig | None = None, **kwargs: Any
) -> DatasetConfig:
    src = SourceConfig(kind="pretrain", loader="local", path=str(source_dir), **kwargs.pop("source", {}))
    return cfg_factory({"s": src}, processing=processing or ProcessingConfig(min_chars=5), **kwargs)


# --- length_filter -----------------------------------------------------------------------------------------------------


def test_length_filter_is_1_to_1_incremental_and_idempotent(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, mtimes: Mtimes, read_rows: Reader
) -> None:
    write_local(source_dir, [{"text": t} for t in ["ok " * 5, "tiny", None, "x" * 30, "y" * 7, "z" * 8]], "parquet")
    cfg = _cfg(cfg_factory, source_dir, ProcessingConfig(min_chars=5, max_chars=20))
    download(cfg, "s", layout, rows_needed=6, shard_size=4)  # raw shards: 4 + 2 rows
    m = length_filter(cfg, "s", layout)
    filtered = layout.source_dir("s", "filtered")
    assert m.stage == "filtered" and [(s.name, s.rows) for s in m.shards] == [("data-00000.parquet", 2), ("data-00001.parquet", 2)]
    assert m.extra["input_shards"] == [["data-00000.parquet", 4], ["data-00001.parquet", 2]]
    assert m.extra["shard_stats"]["data-00000.parquet"] == {
        "input_samples": 4, "removed_too_short": 1, "removed_invalid": 1, "truncated": 1, "output_samples": 2,
    }  # fmt: skip
    rows = read_rows(filtered)
    assert [r["text"] for r in rows] == ["ok " * 5, "x" * 20, "y" * 7, "z" * 8]
    assert [r["original_length"] for r in rows] == [15, 30, 7, 8] and {r["source"] for r in rows} == {"s"}
    before = mtimes(filtered)
    assert length_filter(cfg, "s", layout) == m and mtimes(filtered) == before

    # append raw rows: only the new raw shards are filtered, old filtered shards untouched
    write_local(source_dir, [{"text": "new " * 3}, {"text": "no"}], "parquet")
    download(cfg, "s", layout, rows_needed=8, shard_size=4)
    m2 = length_filter(cfg, "s", layout)
    assert [s.name for s in m2.shards] == [f"data-{i:05d}.parquet" for i in range(3)]
    after = mtimes(filtered)
    assert {k: after[k] for k in before} == before
    assert [r["text"] for r in read_rows(filtered)][-1] == "new new new "


def test_length_filter_writes_empty_shard_when_everything_is_dropped(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer
) -> None:
    write_local(source_dir, [{"text": "a"}, {"text": "b"}, {"text": "long enough text"}], "parquet")
    cfg = _cfg(cfg_factory, source_dir, ProcessingConfig(min_chars=5))
    download(cfg, "s", layout, rows_needed=3, shard_size=2)
    m = length_filter(cfg, "s", layout)
    assert [(s.name, s.rows) for s in m.shards] == [("data-00000.parquet", 0), ("data-00001.parquet", 1)]
    empty = pq.read_table(layout.source_dir("s", "filtered") / "data-00000.parquet")
    assert empty.num_rows == 0 and empty.column_names == ["text", "source", "original_length"]


def test_length_filter_requires_raw_and_pretrain_kind(cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path) -> None:
    cfg = _cfg(cfg_factory, source_dir)
    with pytest.raises(FileNotFoundError, match="no current raw manifest"):
        length_filter(cfg, "s", layout)
    hold = cfg_factory({"h": SourceConfig(kind="holdout", loader="synthetic", rows=1)})
    with pytest.raises(ValueError, match="pretrain sources only"):
        length_filter(hold, "h", layout)


def test_length_filter_refilters_when_raw_shards_changed(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    write_local(source_dir, [{"text": "hello world"}] * 3, "parquet")
    cfg = _cfg(cfg_factory, source_dir)
    download(cfg, "s", layout, rows_needed=3, shard_size=2)
    length_filter(cfg, "s", layout)
    raw = Manifest.load(layout.source_dir("s", "raw"))
    assert raw is not None
    raw.shards[0].rows = 99  # pretend the raw shard was rewritten with a different row count
    raw.save(layout.source_dir("s", "raw"))
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        m = length_filter(cfg, "s", layout)
    assert "raw shards changed" in caplog.text and m.extra["input_shards"][0] == ["data-00000.parquet", 99]


# --- process -----------------------------------------------------------------------------------------------------------


def _prepare(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, texts: Sequence[str | None], with_tokenizer: Prep, **kw: Any
) -> DatasetConfig:
    return _prepare_rows(cfg_factory, layout, source_dir, [{"text": t} for t in texts], with_tokenizer, **kw)


def _prepare_rows(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, rows: list[Row], with_tokenizer: Prep, **kw: Any
) -> DatasetConfig:
    write = kw.pop("write")
    shard_size = kw.pop("shard_size", 4)
    write(source_dir, rows, "parquet")
    cfg = with_tokenizer(_cfg(cfg_factory, source_dir, **kw))
    download(cfg, "s", layout, rows_needed=len(rows), shard_size=shard_size)
    length_filter(cfg, "s", layout)
    return cfg


def test_process_exact_dedup_tokens_and_idempotence(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep,
    read_rows: Reader, mtimes: Mtimes,
) -> None:  # fmt: skip
    texts = [_words(5), _words(3, 100), "  " + _words(5).upper() + "\n", _words(5), _words(70)]
    cfg = _prepare(cfg_factory, layout, source_dir, texts, with_tokenizer, write=write_local, max_seq_length=64)
    m = process(cfg, "s", layout, shard_size=2)
    processed = layout.source_dir("s", "processed")
    assert m.stage == "processed" and m.token_count == "tokenizer" and m.tokenizer == "synthetic"
    rows = read_rows(processed)
    assert [r["text"] for r in rows] == [_words(5), _words(3, 100), _words(70)], "normalized duplicates dropped, text untouched"
    assert [r["tokens"] for r in rows] == [5, 3, 64], "token count capped at max_seq_length"
    assert {r["source"] for r in rows} == {"s"} and [set(r) for r in rows] == [{"text", "source", "tokens"}] * 3
    assert [(s.rows, s.tokens) for s in m.shards] == [(2, 8), (1, 64)] and m.tokens() == 72
    assert m.extra["stats"]["dedup"] == {"mode": "exact", "duplicates_removed": 2}
    assert m.extra["input_shards"] == [["data-00000.parquet", 4], ["data-00001.parquet", 1]]
    before = mtimes(processed)
    assert process(cfg, "s", layout, shard_size=2) == m and mtimes(processed) == before


def test_process_estimate_mode_caps_too(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    cfg = _prepare(cfg_factory, layout, source_dir, ["a" * 40, "b" * 400], with_tokenizer, write=write_local,
                   token_count="estimate", max_seq_length=50)  # fmt: skip
    m = process(cfg, "s", layout)
    assert m.token_count == "estimate" and m.tokenizer is None
    assert [r["tokens"] for r in read_rows(layout.source_dir("s", "processed"))] == [10, 50]
    assert [len(r["text"]) for r in read_rows(layout.source_dir("s", "processed"))] == [40, 400]


def test_process_keeps_old_rows_as_prefix_when_shards_are_appended(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    first = [_words(6, i) for i in range(6)] + [_words(6, 0)]  # one duplicate inside
    cfg = _prepare(cfg_factory, layout, source_dir, first, with_tokenizer, write=write_local, shard_size=3)
    m1 = process(cfg, "s", layout, shard_size=4)
    old_rows = read_rows(layout.source_dir("s", "processed"))
    assert len(old_rows) == 6
    # append: duplicates of old rows plus new ones
    write_local(source_dir, [{"text": _words(6, 2)}, {"text": _words(6, 50)}, {"text": _words(6, 3)}, {"text": _words(6, 51)}], "parquet")
    download(cfg, "s", layout, rows_needed=11, shard_size=3)
    length_filter(cfg, "s", layout)
    m2 = process(cfg, "s", layout, shard_size=4)
    assert m2 != m1 and m2.extra["stats"]["dedup"]["duplicates_removed"] == 3
    new_rows = read_rows(layout.source_dir("s", "processed"))
    assert new_rows[: len(old_rows)] == old_rows, "old rows byte-identical and in place"
    assert [r["text"] for r in new_rows[len(old_rows) :]] == [_words(6, 50), _words(6, 51)]


def test_process_no_dedup_mode_keeps_duplicates(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    proc = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="none"))
    cfg = _prepare(cfg_factory, layout, source_dir, ["dup", "dup", "DUP"], with_tokenizer, write=write_local, processing=proc)
    process(cfg, "s", layout)
    assert [r["text"] for r in read_rows(layout.source_dir("s", "processed"))] == ["dup", "dup", "DUP"]
    exact_raw = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="exact", normalize=False))
    cfg2 = with_tokenizer(_cfg(cfg_factory, source_dir, exact_raw))
    download(cfg2, "s", layout, rows_needed=3)
    length_filter(cfg2, "s", layout)
    process(cfg2, "s", layout)
    assert [r["text"] for r in read_rows(layout.source_dir("s", "processed"))] == ["dup", "DUP"]


def test_process_quality_filter_only_when_enabled(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    texts = [GOOD, "This is one sentence. Another one here."]
    cfg = _prepare(cfg_factory, layout, source_dir, texts, with_tokenizer, write=write_local, max_seq_length=500)
    process(cfg, "s", layout)
    assert len(read_rows(layout.source_dir("s", "processed"))) == 2
    on = with_tokenizer(_cfg(cfg_factory, source_dir, ProcessingConfig(min_chars=5, quality_filter=True), max_seq_length=500))
    download(on, "s", layout, rows_needed=2)
    length_filter(on, "s", layout)
    m = process(on, "s", layout)
    assert [r["text"] for r in read_rows(layout.source_dir("s", "processed"))] == [GOOD]
    assert m.extra["stats"]["quality_filter"] == {
        "enabled": True, "filtered_count": 1, "rejection_reasons": {"too_few_sentences": 1},
    }  # fmt: skip


@pytest.mark.parametrize("num_workers", [1, 2])
def test_process_decontamination_only_when_enabled(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep,
    read_rows: Reader, monkeypatch: pytest.MonkeyPatch, num_workers: int,
) -> None:  # fmt: skip
    planted = " ".join(f"w{i}" for i in range(20))
    calls: list[tuple[list[str], int, str]] = []

    def fake_load(names: list[str], n: int = 13, cache_dir: str | None = None) -> dict[str, set[str]]:
        calls.append((names, n, str(cache_dir)))
        return {"gsm8k_test": get_ngram_set(planted, n), "mmlu_test": set()}

    monkeypatch.setattr(stages_pretrain, "load_benchmark_ngrams", fake_load)
    texts = [planted, GOOD, planted + " tail"]
    cfg = _prepare(cfg_factory, layout, source_dir, texts, with_tokenizer, write=write_local, max_seq_length=500)
    process(cfg, "s", layout, num_workers=num_workers)
    assert len(read_rows(layout.source_dir("s", "processed"))) == 3 and calls == []
    decon = DecontaminationConfig(enabled=True, benchmarks=["gsm8k_test", "mmlu_test"])
    on = with_tokenizer(_cfg(cfg_factory, source_dir, ProcessingConfig(min_chars=5, decontamination=decon), max_seq_length=500))
    download(on, "s", layout, rows_needed=3)
    length_filter(on, "s", layout)
    m = process(on, "s", layout, num_workers=num_workers)
    assert [r["text"] for r in read_rows(layout.source_dir("s", "processed"))] == [GOOD]
    assert m.extra["stats"]["decontamination"] == {
        "enabled": True, "contaminated_count": 2, "contaminated_by_benchmark": {"gsm8k_test": 2},
    }  # fmt: skip
    if num_workers == 1:
        assert calls == [(["gsm8k_test", "mmlu_test"], 13, str(layout.benchmark_cache_dir()))]


def test_process_benchmark_load_failure_is_an_error(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    def failing(names: list[str], n: int = 13, cache_dir: str | None = None) -> dict[str, set[str]]:
        raise OSError("hub unreachable")

    monkeypatch.setattr(stages_pretrain, "load_benchmark_ngrams", failing)
    proc = ProcessingConfig(min_chars=1, decontamination=DecontaminationConfig(enabled=True))
    cfg = _prepare(cfg_factory, layout, source_dir, [GOOD], with_tokenizer, write=write_local, processing=proc, max_seq_length=500)
    with pytest.raises(OSError, match="hub unreachable"):
        process(cfg, "s", layout)
    assert not (layout.source_dir("s", "processed") / "MANIFEST.json").exists()


def test_process_minhash_removes_near_duplicates(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    pytest.importorskip("datasketch")
    base = " ".join(f"word{i}" for i in range(200))
    near = base.replace("word100", "changed")
    other = " ".join(f"other{i}" for i in range(200))
    partial = " ".join(f"word{i}" for i in range(100)) + " " + " ".join(f"new{i}" for i in range(100))
    proc = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="minhash", threshold=0.8, num_perm=64))
    cfg = _prepare(cfg_factory, layout, source_dir, [base, near, other, base, partial], with_tokenizer,
                   write=write_local, processing=proc, max_seq_length=500)  # fmt: skip
    m = process(cfg, "s", layout)
    assert [r["text"] for r in read_rows(layout.source_dir("s", "processed"))] == [base, other, partial]
    assert m.extra["stats"]["dedup"] == {
        "mode": "minhash", "duplicates_removed": 0, "threshold": 0.8, "num_perm": 64, "near_duplicates_removed": 2,
    }  # fmt: skip


def test_process_minhash_without_datasketch_raises(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    monkeypatch.setitem(sys.modules, "datasketch", None)  # makes `import datasketch` raise ImportError
    proc = ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="minhash"))
    cfg = _prepare(cfg_factory, layout, source_dir, ["a b c d e f"], with_tokenizer, write=write_local, processing=proc)
    with pytest.raises(ImportError, match="datasketch"):
        process(cfg, "s", layout)


def test_process_repeat_to_budget_reaches_target(
    cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path, write_local: Writer, with_tokenizer: Prep, read_rows: Reader
) -> None:
    texts = [_words(3, 0), _words(4, 10), _words(5, 20)]  # 12 tokens per cycle
    cfg = _prepare(cfg_factory, layout, source_dir, texts, with_tokenizer, write=write_local, source={"repeat_to_budget": True})
    m = process(cfg, "s", layout, shard_size=5, target_tokens=30)
    rows = read_rows(layout.source_dir("s", "processed"))
    assert [r["tokens"] for r in rows] == [3, 4, 5, 3, 4, 5, 3, 4], "2 full copies (24) + rows until the remainder 6 is covered"
    assert m.tokens() == 31 and m.extra["target_tokens"] == 30
    assert m.extra["stats"]["repeat_to_budget"] == {"unique_rows": 3, "unique_tokens": 12, "target_tokens": 30, "repeated_rows": 8}
    assert process(cfg, "s", layout, shard_size=5, target_tokens=30) == m
    # a different target rebuilds; None or a target below the unique tokens does not repeat
    m2 = process(cfg, "s", layout, shard_size=5, target_tokens=13)
    assert [r["tokens"] for r in read_rows(layout.source_dir("s", "processed"))] == [3, 4, 5, 3]
    assert m2.tokens() == 15
    assert process(cfg, "s", layout, target_tokens=None).tokens() == 12
    assert process(cfg, "s", layout, target_tokens=12).tokens() == 12


def test_process_requires_filtered_manifest(cfg_factory: CfgFactory, layout: DatasetLayout, source_dir: Path) -> None:
    cfg = _cfg(cfg_factory, source_dir)
    with pytest.raises(FileNotFoundError, match="no current filtered manifest"):
        process(cfg, "s", layout)
