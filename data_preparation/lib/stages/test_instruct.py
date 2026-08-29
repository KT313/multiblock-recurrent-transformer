# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.stages.instruct.build_instruct_mixture on local instruct sources."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.schema.dataset_config import DatasetConfig, InstructMixtureConfig, SourceConfig
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest
import pyarrow.parquet as pq

from data_preparation.lib.stages.instruct import build_instruct_mixture
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.lib.stages.shared import TokenCounter, download

Row = dict[str, Any]
CfgFactory = Callable[..., DatasetConfig]
Writer = Callable[[Path, list[Row], str], Path]
Mtimes = Callable[[Path], dict[str, int]]
Reader = Callable[[Path], list[Row]]
Prep = Callable[[DatasetConfig], DatasetConfig]


def _row(i: int, n_out: int = 4, prefix: str = "a") -> Row:
    return {"instruction": f"tok_{i} tok_{i + 1}", "input": "", "output": " ".join(f"tok_{prefix}{j}" for j in range(n_out))}


@pytest.fixture
def two_sources(layout: DatasetLayout, write_local: Writer) -> dict[str, SourceConfig]:
    a_dir, b_dir = layout.root.parent / "a", layout.root.parent / "b"
    write_local(a_dir, [_row(i, 4, "a") for i in range(40)], "jsonl")  # 6 tokens per row
    write_local(b_dir, [_row(i, 8, "b") for i in range(40)], "jsonl")  # 10 tokens per row
    return {
        "a": SourceConfig(kind="instruct", loader="local", path=str(a_dir), converter="instruction_input_output"),
        "b": SourceConfig(kind="instruct", loader="local", path=str(b_dir), converter="instruction_input_output"),
    }


def _build(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, sources: dict[str, SourceConfig],
    mixture: InstructMixtureConfig, rows_needed: int = 40, **kw: Any,
) -> DatasetConfig:  # fmt: skip
    cfg = with_tokenizer(cfg_factory(sources, instruct_mixtures={"m": mixture}, **kw))
    for name in sources:
        download(cfg, name, layout, rows_needed=rows_needed, shard_size=16)
    return cfg


def test_build_instruct_mixture_counts_split_and_columns(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, two_sources: dict[str, SourceConfig],
    read_rows: Reader, mtimes: Mtimes,
) -> None:  # fmt: skip
    mixture = InstructMixtureConfig(sources={"a": 0.6, "b": 0.4}, max_tokens=64, input_inversions=0.0, val_split=0.25, seed=1)
    cfg = _build(cfg_factory, layout, with_tokenizer, two_sources, mixture)
    result = build_instruct_mixture(cfg, "m", layout, budget_tokens=200, shard_size=8)
    train, val = result["train"], result["validation"]
    assert train.stage == val.stage == "instruct_mixture" and train.source_hash == cfg.instruct_mixture_hash("m")
    # a: 120 tokens / 6 per row -> 20 rows; b: 80 / 10 -> 8 rows
    assert train.extra["counts"]["a"]["needed_rows"] == ceil(120 / 6) == 20 and train.extra["counts"]["b"]["needed_rows"] == 8
    assert train.extra["tokens_per_row"] == {"a": 6.0, "b": 10.0} and train.extra["short_sources"] == {}
    assert train.rows() == 21 and val.rows() == 7 and train.extra["metadata"]["total_examples"] == 28
    rows = read_rows(layout.instruct_mixture_dir("t", "m", "train")) + read_rows(layout.instruct_mixture_dir("t", "m", "validation"))
    assert [set(r) for r in rows] == [{"instruction", "input", "output", "tokens"}] * 28
    assert sum(r["output"].startswith("tok_a") for r in rows) == 20 and sum(r["output"].startswith("tok_b") for r in rows) == 8
    assert all(r["tokens"] in (6, 10) for r in rows)
    assert train.tokens() == sum(r["tokens"] for r in read_rows(layout.instruct_mixture_dir("t", "m", "train")))
    assert [s.rows for s in train.shards] == [8, 8, 5]
    # deterministic shuffle: not the source order
    assert [r["instruction"] for r in rows] != sorted(r["instruction"] for r in rows)
    before = {split: mtimes(layout.instruct_mixture_dir("t", "m", split)) for split in result}
    again = build_instruct_mixture(cfg, "m", layout, budget_tokens=200, shard_size=8)
    assert again == result
    assert {split: mtimes(layout.instruct_mixture_dir("t", "m", split)) for split in result} == before
    assert Manifest.load(layout.instruct_mixture_dir("t", "m", "validation")) == val


def test_build_instruct_mixture_reads_only_the_rows_it_needs(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, two_sources: dict[str, SourceConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows carry their token count from the download, so a source is read shard by shard only until its share
    of the budget is reached: with 16-row shards a 60-token share (10 rows of `a`) opens one shard of three."""
    mixture = InstructMixtureConfig(sources={"a": 0.6, "b": 0.4}, max_tokens=64, input_inversions=0.0, val_split=0.0, seed=1)
    cfg = _build(cfg_factory, layout, with_tokenizer, two_sources, mixture)  # 40 rows per source in 16-row shards
    opened: list[str] = []
    original = pq.ParquetFile

    def spy(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(path).parent.parent.name + "/" + Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", spy)  # `instruct.py` looks it up on the module at call time
    train = build_instruct_mixture(cfg, "m", layout, budget_tokens=100, shard_size=8)["train"]
    assert train.extra["counts"]["a"] == {
        "available_rows": 40, "needed_rows": 10, "taken_rows": 10, "dropped_too_long": 0, "kept_rows": 10,
        "tokens_per_row": 6.0, "target_tokens": 60.0,
    }  # fmt: skip
    assert train.extra["counts"]["b"]["taken_rows"] == 4 and train.extra["short_sources"] == {}
    assert opened == ["a/data-00000.parquet", "b/data-00000.parquet"]


def test_build_instruct_mixture_short_sources_and_length_check(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, two_sources: dict[str, SourceConfig], read_rows: Reader
) -> None:
    mixture = InstructMixtureConfig(sources={"a": 0.5, "b": 0.5}, max_tokens=6, val_split=0.0, seed=0)
    cfg = _build(cfg_factory, layout, with_tokenizer, two_sources, mixture, rows_needed=5)
    result = build_instruct_mixture(cfg, "m", layout, budget_tokens=600)
    train = result["train"]
    assert train.extra["short_sources"] == {
        "a": {"available_rows": 5, "needed_rows": 50},
        "b": {"available_rows": 5, "needed_rows": 30},
    }
    assert train.extra["counts"]["b"]["dropped_too_long"] == 5 and train.extra["counts"]["b"]["kept_rows"] == 0
    assert train.rows() == 5 and result["validation"].rows() == 0
    assert all(r["tokens"] == 6 for r in read_rows(layout.instruct_mixture_dir("t", "m", "train")))


def test_build_instruct_mixture_inversions_dedup_and_empty_removal(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "c"
    rows = [_row(i) for i in range(20)] + [_row(0), {"instruction": "tok_1  TOK_2", "input": "", "output": _row(0)["output"]}]
    rows += [{"instruction": "", "input": "", "output": "tok_1"}, {"instruction": "tok_1", "input": "", "output": "  "}]
    write_local(src_dir, rows, "jsonl")
    src = SourceConfig(kind="instruct", loader="local", path=str(src_dir), converter="instruction_input_output")
    mixture = InstructMixtureConfig(sources={"c": 1.0}, max_tokens=64, input_inversions=0.5, val_split=0.0, seed=3)
    cfg = _build(cfg_factory, layout, with_tokenizer, {"c": src}, mixture)
    train = build_instruct_mixture(cfg, "m", layout, budget_tokens=10_000)["train"]
    meta = train.extra["metadata"]
    out = read_rows(layout.instruct_mixture_dir("t", "m", "train"))
    inverted = [r for r in out if r["instruction"].startswith("Given this output")]
    # guarded rows (empty field) are not counted; two inverted duplicates collapse in the dedup afterwards
    assert len(inverted) <= meta["inverted"] <= int(24 * 0.5)
    counter = TokenCounter(cfg, layout)
    assert all(r["tokens"] == counter.count(instruct_text(r)) != 6 for r in inverted), "tokens recounted"
    assert meta["duplicates_removed"] >= 1 and meta["empty_removed"] >= 1
    assert all(r["instruction"].strip() and r["output"].strip() for r in out)
    assert len(out) == meta["total_examples"] == len({(r["instruction"], r["input"], r["output"]) for r in out})


def test_build_instruct_mixture_rebuilds_on_new_budget_hash_or_input_shards(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Prep, two_sources: dict[str, SourceConfig], mtimes: Mtimes
) -> None:
    mixture = InstructMixtureConfig(sources={"a": 0.5, "b": 0.5}, max_tokens=64, val_split=0.2, seed=0)
    cfg = _build(cfg_factory, layout, with_tokenizer, two_sources, mixture, rows_needed=10)
    train_dir = layout.instruct_mixture_dir("t", "m", "train")
    first = build_instruct_mixture(cfg, "m", layout, budget_tokens=100)["train"]
    t0 = mtimes(train_dir)
    second = build_instruct_mixture(cfg, "m", layout, budget_tokens=120)["train"]
    assert second.extra["budget_tokens"] == 120 and second != first and mtimes(train_dir) != t0
    t1 = mtimes(train_dir)
    download(cfg, "a", layout, rows_needed=20, shard_size=16)  # more raw rows -> input shards changed
    third = build_instruct_mixture(cfg, "m", layout, budget_tokens=120)["train"]
    assert third.extra["input_shards"]["a"] == [["data-00000.parquet", 10], ["data-00001.parquet", 10]]
    assert mtimes(train_dir) != t1
    reseeded = _build(cfg_factory, layout, with_tokenizer, two_sources,
                      InstructMixtureConfig(sources={"a": 0.5, "b": 0.5}, max_tokens=64, val_split=0.2, seed=7), rows_needed=20)  # fmt: skip
    fourth = build_instruct_mixture(reseeded, "m", layout, budget_tokens=120)["train"]
    assert fourth.source_hash == reseeded.instruct_mixture_hash("m") != third.source_hash


def test_build_instruct_mixture_requires_raw_manifests(cfg_factory: CfgFactory, layout: DatasetLayout, two_sources: dict[str, SourceConfig]) -> None:
    mixture = InstructMixtureConfig(sources={"a": 1.0})
    cfg = cfg_factory({"a": two_sources["a"]}, instruct_mixtures={"m": mixture})
    with pytest.raises(FileNotFoundError, match="no current raw manifest"):
        build_instruct_mixture(cfg, "m", layout, budget_tokens=10)
