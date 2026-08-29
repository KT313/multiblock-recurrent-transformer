# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.stages.shared: tokenizer stage, token counter, incremental download, validation."""

from __future__ import annotations

import json
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.schema.dataset_config import DatasetConfig, SourceConfig, TokenizerConfig, load_dataset_config
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.sources import synthetic_row
from data_preparation.lib.stages.shared import (
    TokenCounter,
    current_manifest,
    download,
    validation,
    prepare_tokenizer,
    require_manifest,
)

Row = dict[str, Any]
CfgFactory = Callable[..., DatasetConfig]
Writer = Callable[[Path, list[Row], str], Path]
Mtimes = Callable[[Path], dict[str, int]]
Reader = Callable[[Path], list[Row]]


def _synthetic(kind: str = "pretrain", seed: int = 0, **kwargs: Any) -> SourceConfig:
    return SourceConfig(kind=kind, loader="synthetic", seed=seed, **kwargs)  # type: ignore[arg-type]  # Literal kind


def _local(path: Path, kind: str = "pretrain", **kwargs: Any) -> SourceConfig:
    return SourceConfig(kind=kind, loader="local", path=str(path), **kwargs)  # type: ignore[arg-type]  # Literal kind


# --- tokenizer -----------------------------------------------------------------------------------------------------


def test_prepare_tokenizer_synthetic_is_idempotent(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()})
    manifest = prepare_tokenizer(cfg, layout)
    out = layout.tokenizer_dir("synthetic")
    assert (out / "tokenizer.json").is_file() and (out / "tokenizer_config.json").is_file()
    assert manifest.stage == "tokenizer" and manifest.source_hash == cfg.tokenizer_hash() and manifest.shards == []
    before = (out / "tokenizer.json").stat().st_mtime_ns
    assert prepare_tokenizer(cfg, layout) == manifest
    assert (out / "tokenizer.json").stat().st_mtime_ns == before


def test_prepare_tokenizer_rebuilds_on_stale_hash(
    cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = cfg_factory({"p": _synthetic()})
    manifest = prepare_tokenizer(cfg, layout)
    manifest.source_hash = "stale"
    manifest.save(layout.tokenizer_dir("synthetic"))
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        rebuilt = prepare_tokenizer(cfg, layout)
    assert rebuilt.source_hash == cfg.tokenizer_hash()
    assert "rebuilding from scratch" in caplog.text


def test_prepare_tokenizer_hf_uses_from_pretrained_with_revision(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, tiny_tokenizer_dir: Path
) -> None:
    import transformers

    calls: list[tuple[Any, ...]] = []
    real = transformers.AutoTokenizer.from_pretrained

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((name, kwargs.get("revision")))
        return real(str(tiny_tokenizer_dir), *args)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    tok = TokenizerConfig(name="llama", kind="hf", hf_id="org/tok", revision="abc")
    cfg = cfg_factory({"p": _synthetic()}, tokenizer=tok)
    manifest = prepare_tokenizer(cfg, layout)
    assert calls == [("org/tok", "abc")]
    assert (layout.tokenizer_dir("llama") / "tokenizer_config.json").is_file()
    assert manifest.extra == {"kind": "hf", "hf_id": "org/tok", "revision": "abc"}
    assert prepare_tokenizer(cfg, layout) == manifest and len(calls) == 1


# --- token counter ---------------------------------------------------------------------------------------------------


def test_token_counter_tokenizer_mode_caps_at_max_seq_length(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Callable[[DatasetConfig], DatasetConfig]
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}, max_seq_length=5))
    counter = TokenCounter(cfg, layout)
    assert counter.count("tok_1 tok_2 tok_3") == 3
    assert counter.count(" ".join(["tok_1"] * 9)) == 5
    assert counter.count_many(["tok_1", " ".join(["tok_2"] * 7), ""]) == [1, 5, 0]


def test_token_counter_estimate_mode_caps_and_needs_no_tokenizer(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()}, token_count="estimate", max_seq_length=10)
    counter = TokenCounter(cfg, layout)  # tokenizer dir does not exist
    assert counter.count("a" * 8) == 2 and counter.count("a" * 400) == 10
    assert counter.count_many(["a" * 8, "a" * 400]) == [2, 10]


def test_token_counter_missing_tokenizer_raises(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()})
    with pytest.raises(FileNotFoundError, match="run the tokenizer stage first"):
        TokenCounter(cfg, layout)


# --- download: pretrain ------------------------------------------------------------------------------------------------


def test_download_synthetic_appends_incrementally(
    cfg_factory: CfgFactory, layout: DatasetLayout, mtimes: Mtimes, read_rows: Reader
) -> None:
    cfg = cfg_factory({"p": _synthetic(seed=3)})
    raw = layout.source_dir("p", "raw")
    m1 = download(cfg, "p", layout, rows_needed=25, shard_size=10)
    assert [s.rows for s in m1.shards] == [10, 10, 5] and m1.rows_fetched == 25 and m1.stage == "raw"
    assert m1.source_hash == cfg.source_hash("p") and not m1.extra.get("exhausted")
    assert Manifest.load(raw) == m1
    first = mtimes(raw)

    assert download(cfg, "p", layout, rows_needed=25, shard_size=10) == m1  # no-op
    assert download(cfg, "p", layout, rows_needed=10, shard_size=10) == m1  # fewer rows needed: still no-op
    assert mtimes(raw) == first

    m2 = download(cfg, "p", layout, rows_needed=40, shard_size=10)
    assert [s.name for s in m2.shards] == [f"data-{i:05d}.parquet" for i in range(5)]
    assert [s.rows for s in m2.shards] == [10, 10, 5, 10, 5] and m2.rows_fetched == 40
    after = mtimes(raw)
    assert {k: after[k] for k in first} == first, "old shards untouched"
    rows = read_rows(raw)
    assert [r["text"] for r in rows] == [synthetic_row("pretrain", 3, i)["text"] for i in range(40)]


def test_download_passes_index_dir_and_on_file_to_the_loader(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hub-file loaders get `index_dir=<dataset>/hub_index` and an `on_file` callback from the download stage."""
    from data_preparation.lib.sources import loaders as loaders_mod

    seen: dict[str, Any] = {}

    def fake_loader(source: SourceConfig, offset: int, count: int, **kwargs: Any) -> Any:
        seen.update(kwargs)
        on_file = kwargs["on_file"]
        on_file("data/x.parquet")
        return iter([{"text": "a"}, {"text": "b"}][:count])

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", fake_loader)
    cfg = cfg_factory({"p": _synthetic()})
    manifest = download(cfg, "p", layout, rows_needed=2, hf_token="tok")
    assert manifest.rows() == 2
    assert seen["index_dir"] == layout.hub_index_dir() and seen["token"] == "tok" and callable(seen["on_file"])


def test_download_local_applies_converter_and_flags_exhaustion(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "gsm"
    write_local(src_dir, [{"question": f"q{i}", "answer": f"a{i}", "extra": i} for i in range(7)], "parquet")
    cfg = cfg_factory({"g": _local(src_dir, converter="gsm8k_question_answer")})
    m = download(cfg, "g", layout, rows_needed=10, shard_size=4)
    assert m.rows() == 7 and m.rows_fetched == 7 and m.extra["exhausted"] is True
    rows = read_rows(layout.source_dir("g", "raw"))
    assert rows[0] == {"text": "Question: q0\n\nAnswer: a0"}
    # exhausted: a larger request is a no-op
    assert download(cfg, "g", layout, rows_needed=100, shard_size=4) == m


def test_download_keeps_extra_columns_and_requires_text_field(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "code"
    write_local(src_dir, [{"code": "print(1)" * 10, "lang": "py"}], "jsonl")
    cfg = cfg_factory({"c": _local(src_dir, text_field="code")})
    download(cfg, "c", layout, rows_needed=1)
    assert read_rows(layout.source_dir("c", "raw")) == [{"code": "print(1)" * 10, "lang": "py"}]
    bad = cfg_factory({"c": _local(src_dir, text_field="text")})
    with pytest.raises(ValueError, match="no 'text' column"):
        download(bad, "c", DatasetLayout(layout.root / "other"), rows_needed=1)



def test_download_rebuilds_on_stale_hash(
    cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture, read_rows: Reader
) -> None:
    cfg = cfg_factory({"p": _synthetic(seed=0)})
    download(cfg, "p", layout, rows_needed=12, shard_size=5)
    changed = cfg_factory({"p": _synthetic(seed=9)})  # a different seed -> different source hash
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        m = download(changed, "p", layout, rows_needed=7, shard_size=5)
    assert "rebuilding from scratch" in caplog.text
    assert m.rows_fetched == 7 and [s.rows for s in m.shards] == [5, 2]
    raw = layout.source_dir("p", "raw")
    assert sorted(p.name for p in raw.glob("*.parquet")) == ["data-00000.parquet", "data-00001.parquet"]
    assert [r["text"] for r in read_rows(raw)] == [synthetic_row("pretrain", 9, i)["text"] for i in range(7)]


def test_download_keeps_every_row_a_loader_yields_beyond_rows_needed(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    """A remote parquet loader finishes its row group: all rows land on disk, `rows_fetched` is the boundary and a
    later call below that boundary never touches the loader."""
    from data_preparation.lib.sources import loaders as loaders_mod

    calls: list[tuple[int, int, list[str] | None, bool]] = []

    def group_loader(source: SourceConfig, offset: int, count: int, **kwargs: Any) -> Any:
        calls.append((offset, count, kwargs["columns"], kwargs["align_to_row_group"]))
        return iter([{"text": f"row {i}"} for i in range(offset, offset + max(count, 20))])  # a 20-row "row group"

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", group_loader)
    cfg = cfg_factory({"p": _synthetic()})
    m = download(cfg, "p", layout, rows_needed=11, shard_size=8)
    assert m.rows() == 20 and m.rows_fetched == 20 and [s.rows for s in m.shards] == [8, 8, 4]
    assert calls == [(0, 11, ["text"], True)]
    assert [r["text"] for r in read_rows(layout.source_dir("p", "raw"))] == [f"row {i}" for i in range(20)]
    assert download(cfg, "p", layout, rows_needed=15, shard_size=8) == m and len(calls) == 1  # below the boundary: no-op
    m2 = download(cfg, "p", layout, rows_needed=21, shard_size=8)
    assert calls[-1] == (20, 1, ["text"], True) and m2.rows_fetched == 40 and m2.rows() == 40


def test_download_columns_follow_the_converter_and_check_limit_bounds_over_reads(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data_preparation.lib.sources import loaders as loaders_mod

    seen: list[list[str] | None] = []
    closed: list[bool] = []

    def loader(source: SourceConfig, offset: int, count: int, **kwargs: Any) -> Any:
        seen.append(kwargs["columns"])
        try:
            for i in range(offset, offset + 20):
                yield {"question": f"q{i}", "answer": f"a{i}", "text": f"t{i}"}
        finally:
            closed.append(True)

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", loader)
    cfg = cfg_factory({"conv": _synthetic(converter="gsm8k_question_answer"), "lim": _synthetic(check_limit=5)})
    download(cfg, "conv", layout, rows_needed=1)
    m = download(cfg, "lim", layout, rows_needed=3)
    assert seen == [None, ["text"]]
    assert m.rows() == 5 and m.rows_fetched == 5 and closed == [True, True]  # consumption stopped at check_limit
    assert download(cfg, "lim", layout, rows_needed=8).extra["exhausted"] is True and len(seen) == 2


def test_validation_asks_for_exact_rows(cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_preparation.lib.sources import loaders as loaders_mod

    seen: dict[str, Any] = {}

    def loader(source: SourceConfig, offset: int, count: int, **kwargs: Any) -> Any:
        seen.update(kwargs, count=count)
        return iter([{"text": f"t{i}"} for i in range(count)])

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", loader)
    cfg = cfg_factory({"v": _synthetic(kind="validation", rows=4)}, token_count="estimate")
    m = validation(cfg, "v", layout)
    assert m.rows() == 4 and seen["count"] == 4 and seen["align_to_row_group"] is False and seen["columns"] == ["text"]


# --- download: instruct ------------------------------------------------------------------------------------------------


def _sharegpt(human: str, gpt: str) -> Row:
    return {"conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}]}


def test_download_instruct_converts_filters_and_counts_malformed(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "inst"
    rows: list[Row] = [
        {"q": "what", "a": "that", "junk": 1},
        {"q": "only question"},  # malformed: converter raises
        {"q": "how", "a": "so", "ctx": "background"},
        {"a": "no question"},  # malformed
        {"q": "why", "a": "because"},
    ]
    write_local(src_dir, rows, "jsonl")
    src = _local(src_dir, kind="instruct", fields={"instruction": "q", "output": "a", "input": "ctx"})
    cfg = cfg_factory({"i": src})
    m = download(cfg, "i", layout, rows_needed=3, shard_size=10)
    assert m.rows() == 3 and m.rows_fetched == 5 and m.extra["skipped_malformed"] == 2
    assert read_rows(layout.source_dir("i", "raw")) == [
        {"instruction": "what", "input": "", "output": "that"},
        {"instruction": "how", "input": "background", "output": "so"},
        {"instruction": "why", "input": "", "output": "because"},
    ]


def test_download_instruct_filter_and_multiple_loader_calls(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    good, bad = _sharegpt("h" * 60, "g" * 60), _sharegpt("short", "g" * 60)
    src_dir = layout.root.parent / "sg"
    write_local(src_dir, [bad, bad, good, bad, good, good, bad, good], "jsonl")
    src = _local(src_dir, kind="instruct", converter="sharegpt_conversations", filter="sharegpt_quality")
    cfg = cfg_factory({"s": src})
    m = download(cfg, "s", layout, rows_needed=3, shard_size=10)
    # 3 kept rows need 6 source rows: first call (3) yields 1 kept, second (2) yields 1, third (1) yields 1
    assert m.rows() == 3 and m.rows_fetched == 6 and not m.extra.get("exhausted")
    assert all(r == {"instruction": "h" * 60, "input": "", "output": "g" * 60} for r in read_rows(layout.source_dir("s", "raw")))
    m2 = download(cfg, "s", layout, rows_needed=10, shard_size=10)
    assert m2.rows() == 4 and m2.rows_fetched == 8 and m2.extra["exhausted"] is True


def test_download_instruct_check_limit_bounds_inspected_rows(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer
) -> None:
    src_dir = layout.root.parent / "lim"
    write_local(src_dir, [{"instruction": f"i{i}", "output": f"o{i}"} for i in range(10)], "jsonl")
    src = _local(src_dir, kind="instruct", converter="instruction_input_output", check_limit=4)
    cfg = cfg_factory({"l": src})
    m = download(cfg, "l", layout, rows_needed=3, shard_size=10)
    assert m.rows() == 3 and m.rows_fetched == 3 and not m.extra.get("exhausted")
    m = download(cfg, "l", layout, rows_needed=8, shard_size=10)
    assert m.rows() == 4 and m.rows_fetched == 4 and m.extra["exhausted"] is True
    assert download(cfg, "l", layout, rows_needed=8, shard_size=10) == m


def test_download_synthetic_instruct_rows(cfg_factory: CfgFactory, layout: DatasetLayout, read_rows: Reader) -> None:
    cfg = cfg_factory({"i": _synthetic(kind="instruct", seed=2)})
    m = download(cfg, "i", layout, rows_needed=3)
    assert m.rows() == 3
    rows = read_rows(layout.source_dir("i", "raw"))
    assert rows == [synthetic_row("instruct", 2, i) for i in range(3)]


# --- validation -----------------------------------------------------------------------------------------------------------


def test_validation_synthetic_disjoint_shuffled_and_idempotent(
    cfg_factory: CfgFactory,
    layout: DatasetLayout,
    with_tokenizer: Callable[[DatasetConfig], DatasetConfig],
    read_rows: Reader,
    mtimes: Mtimes,
) -> None:
    cfg = with_tokenizer(
        cfg_factory({"train": _synthetic(seed=0), "val": _synthetic(kind="validation", seed=1, rows=20)}, max_seq_length=100)
    )
    download(cfg, "train", layout, rows_needed=40)
    m = validation(cfg, "val", layout, shard_size=8)
    out = layout.validation_dir("val")
    assert m.stage == "validation" and [s.rows for s in m.shards] == [8, 8, 4] and m.rows_fetched == 20
    rows = read_rows(out)
    assert [set(r) for r in rows] == [{"text", "source", "tokens"}] * 20
    assert {r["source"] for r in rows} == {"val"}
    train_texts = {r["text"] for r in read_rows(layout.source_dir("train", "raw"))}
    assert not train_texts & {r["text"] for r in rows}, "validation rows must not appear in the training source"
    generated = [synthetic_row("validation", 1, i)["text"] for i in range(20)]
    assert sorted(r["text"] for r in rows) == sorted(generated) and [r["text"] for r in rows] != generated
    assert all(r["tokens"] == min(len(r["text"].split()), 100) for r in rows)
    assert m.tokens() == sum(r["tokens"] for r in rows)
    before = mtimes(out)
    assert validation(cfg, "val", layout, shard_size=8) == m and mtimes(out) == before
    # deterministic: a second root gets the same order
    other = DatasetLayout(layout.root.parent / "other")
    prepare_tokenizer(cfg, other)
    validation(cfg, "val", other, shard_size=8)
    assert read_rows(other.validation_dir("val")) == rows


def test_validation_local_takes_the_last_rows(
    cfg_factory: CfgFactory,
    layout: DatasetLayout,
    with_tokenizer: Callable[[DatasetConfig], DatasetConfig],
    write_local: Writer,
    read_rows: Reader,
) -> None:
    src_dir = layout.root.parent / "loc"
    write_local(src_dir, [{"text": f"doc {i}"} for i in range(6)], "parquet")
    write_local(src_dir, [{"text": f"doc {i}"} for i in range(6, 10)], "jsonl")
    cfg = with_tokenizer(cfg_factory({"v": _local(src_dir, kind="validation", rows=3)}, token_count="estimate"))
    m = validation(cfg, "v", layout)
    assert m.extra["offset"] == 7 and m.rows_fetched == 10
    assert sorted(r["text"] for r in read_rows(layout.validation_dir("v"))) == ["doc 7", "doc 8", "doc 9"]


def test_validation_hf_stream_takes_the_first_rows_and_warns_when_short(
    cfg_factory: CfgFactory,
    layout: DatasetLayout,
    with_tokenizer: Callable[[DatasetConfig], DatasetConfig],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    read_rows: Reader,
) -> None:
    class Stream:
        def __init__(self, rows: list[Row]) -> None:
            self.rows = rows

        def skip(self, n: int) -> Stream:
            return Stream(self.rows[n:])

        def __iter__(self) -> Any:
            return iter(self.rows)

    module = types.ModuleType("datasets")
    module.load_dataset = lambda **kw: Stream([{"text": f"s{i}"} for i in range(4)])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    src = SourceConfig(kind="validation", loader="hf_stream", hf_id="org/x", rows=6)
    cfg = with_tokenizer(cfg_factory({"v": src}, token_count="estimate"))
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        m = validation(cfg, "v", layout)
    assert "only 4 of 6" in caplog.text and m.rows() == 4 and m.extra["offset"] == 0
    assert sorted(r["text"] for r in read_rows(layout.validation_dir("v"))) == ["s0", "s1", "s2", "s3"]


def test_validation_rejects_non_validation_source(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()})
    with pytest.raises(ValueError, match="kind validation"):
        validation(cfg, "p", layout)


# --- manifest helpers --------------------------------------------------------------------------------------------------


def test_current_manifest_stage_mismatch_and_require(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    Manifest(source="s", source_hash="h", stage="raw").save(tmp_path)
    assert current_manifest(tmp_path, "h", "raw") is not None
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert current_manifest(tmp_path, "h", "filtered") is None
    assert "manifest stage 'raw' != 'filtered'" in caplog.text
    assert current_manifest(tmp_path, "other", "raw") is None
    with pytest.raises(FileNotFoundError, match="no current raw manifest"):
        require_manifest(tmp_path / "missing", "h", "raw", "s")
    payload = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert payload["stage"] == "raw"


def test_fetch_source_forces_range_requests_by_default() -> None:
    from dataclasses import replace

    from data_preparation.lib.schema.dataset_config import SourceConfig
    from data_preparation.lib.stages.shared import fetch_source

    cfg = load_dataset_config(Path("config/datasets/crow_300m_mini.yaml"))
    assert cfg.always_range_requests
    src = cfg.sources["fineweb_edu"]
    assert fetch_source(cfg, src).load_kwargs["max_cached_file_mb"] == 0
    assert "max_cached_file_mb" not in src.load_kwargs  # original untouched
    github = cfg.sources["github_code_clean_python"]
    assert fetch_source(cfg, github).load_kwargs["max_cached_file_mb"] == 0
    assert fetch_source(cfg, cfg.sources["gsm8k"]) is cfg.sources["gsm8k"]  # hf_split: not a hub_files source
    off = replace(cfg, always_range_requests=False)
    assert fetch_source(off, src) is src
    custom = SourceConfig(kind="pretrain", loader="hf_files", hf_id="a/b", load_kwargs={"data_files": "*.parquet", "max_cached_file_mb": 7})
    assert fetch_source(off, custom).load_kwargs["max_cached_file_mb"] == 7
    assert fetch_source(cfg, custom).load_kwargs["max_cached_file_mb"] == 0
