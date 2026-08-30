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

from data_preparation.dataset_config import DatasetConfig, SourceConfig, TokenizerConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.sources import synthetic_row
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.conftest import REPO, REV, FakeHub
from data_preparation.lib.stages.shared import (
    TokenCounter,
    current_manifest,
    download,
    download_github_code_group,
    ensure_raw_tokens,
    prepare_tokenizer,
    require_manifest,
    validation,
)

Row = dict[str, Any]
CfgFactory = Callable[..., DatasetConfig]
Writer = Callable[[Path, list[Row], str], Path]
Mtimes = Callable[[Path], dict[str, int]]
Reader = Callable[[Path], list[Row]]
Prep = Callable[[DatasetConfig], DatasetConfig]


def _synthetic(kind: str = "pretrain", seed: int = 0, **kwargs: Any) -> SourceConfig:
    return SourceConfig(kind=kind, loader="synthetic", seed=seed, **kwargs)  # type: ignore[arg-type]  # Literal kind


def _local(path: Path, kind: str = "pretrain", **kwargs: Any) -> SourceConfig:
    return SourceConfig(kind=kind, loader="local", path=str(path), **kwargs)  # type: ignore[arg-type]  # Literal kind


def _github(language: str, **kwargs: Any) -> SourceConfig:
    return SourceConfig(kind="pretrain", loader="github_code", hf_id=REPO, revision=REV, language=language, **kwargs)


def _code_rows(prefix: str, n: int) -> list[Row]:
    languages = ("Python", "Java", "Go")
    return [{"id": f"{prefix}{i}", "text": f"{prefix} code {i}", "language": languages[i % 3]} for i in range(n)]


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
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(), "i": _synthetic(kind="instruct")}, max_seq_length=5))
    counter = TokenCounter.for_source(cfg, layout, "p")
    assert counter.count("tok_1 tok_2 tok_3") == 3
    assert counter.count(" ".join(["tok_1"] * 9)) == 5
    assert counter.count_many(["tok_1", " ".join(["tok_2"] * 7), ""]) == [1, 5, 0]
    uncapped = TokenCounter.for_source(cfg, layout, "i")  # instruct examples: the full length
    assert uncapped.cap is None and uncapped.count(" ".join(["tok_1"] * 9)) == 9
    assert uncapped.count_many([" ".join(["tok_2"] * 7)]) == [7] and TokenCounter(cfg, layout).cap is None


def test_token_counter_estimate_mode_caps_and_needs_no_tokenizer(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()}, token_count="estimate", max_seq_length=10)
    counter = TokenCounter.for_source(cfg, layout, "p")  # tokenizer dir does not exist
    assert counter.count("a" * 8) == 2 and counter.count("a" * 400) == 10
    assert counter.count_many(["a" * 8, "a" * 400]) == [2, 10]


def test_token_counter_missing_tokenizer_raises(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()})
    with pytest.raises(FileNotFoundError, match="run the tokenizer stage first"):
        TokenCounter(cfg, layout)


# --- download: pretrain ------------------------------------------------------------------------------------------------


def test_download_synthetic_appends_incrementally(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, mtimes: Mtimes, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}))
    raw = layout.source_dir("p", "raw")
    m1 = download(cfg, "p", layout, rows_needed=25, shard_size=10)
    assert [s.rows for s in m1.shards] == [10, 10, 5] and m1.rows_fetched == 25 and m1.stage == "raw"
    assert m1.source_hash == cfg.raw_hash("p") and not m1.extra.get("exhausted")
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
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
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
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    manifest = download(cfg, "p", layout, rows_needed=2, hf_token="tok")
    assert manifest.rows() == 2
    assert seen["index_dir"] == layout.hub_index_dir() and seen["token"] == "tok" and callable(seen["on_file"])


def test_download_local_applies_converter_and_flags_exhaustion(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "gsm"
    write_local(src_dir, [{"question": f"q{i}", "answer": f"a{i}", "extra": i} for i in range(7)], "parquet")
    cfg = with_tokenizer(cfg_factory({"g": _local(src_dir, converter="gsm8k_question_answer")}))
    m = download(cfg, "g", layout, rows_needed=10, shard_size=4)
    assert m.rows() == 7 and m.rows_fetched == 7 and m.extra["exhausted"] is True
    rows = read_rows(layout.source_dir("g", "raw"))
    assert rows[0] == {"text": "Question: q0\n\nAnswer: a0", "tokens": 6}
    assert m.token_count == "tokenizer" and m.tokenizer == "synthetic" and m.tokens() == sum(r["tokens"] for r in rows)
    # exhausted: a larger request is a no-op
    assert download(cfg, "g", layout, rows_needed=100, shard_size=4) == m


def test_download_keeps_extra_columns_and_requires_text_field(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "code"
    write_local(src_dir, [{"code": "print(1)" * 10, "lang": "py"}], "jsonl")
    cfg = with_tokenizer(cfg_factory({"c": _local(src_dir, text_field="code")}))
    download(cfg, "c", layout, rows_needed=1)
    assert read_rows(layout.source_dir("c", "raw")) == [{"code": "print(1)" * 10, "lang": "py", "tokens": 40}]  # tokens of `code`
    bad = cfg_factory({"c": _local(src_dir, text_field="text")})
    other = DatasetLayout(layout.root / "other")
    prepare_tokenizer(bad, other)
    with pytest.raises(ValueError, match="no 'text' column"):
        download(bad, "c", other, rows_needed=1)



def test_download_rebuilds_on_stale_hash(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, caplog: pytest.LogCaptureFixture, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}))
    download(cfg, "p", layout, rows_needed=12, shard_size=5)
    changed = cfg_factory({"p": _synthetic(seed=9)})  # a different seed -> different source hash
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        m = download(changed, "p", layout, rows_needed=7, shard_size=5)
    assert "rebuilding from scratch" in caplog.text
    assert m.rows_fetched == 7 and [s.rows for s in m.shards] == [5, 2]
    raw = layout.source_dir("p", "raw")
    assert sorted(p.name for p in raw.glob("*.parquet")) == ["data-00000.parquet", "data-00001.parquet"]
    assert [r["text"] for r in read_rows(raw)] == [synthetic_row("pretrain", 9, i)["text"] for i in range(7)]


def test_download_refuses_to_restart_over_shards_without_a_manifest(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}))
    download(cfg, "p", layout, rows_needed=3)
    raw = layout.source_dir("p", "raw")
    (raw / "MANIFEST.json").unlink()
    with pytest.raises(RuntimeError, match="holds shards but no manifest"):
        download(cfg, "p", layout, rows_needed=3)
    assert (raw / "data-00000.parquet").is_file()


def test_download_keeps_every_row_a_loader_yields_beyond_rows_needed(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    """A remote parquet loader finishes its row group: all rows land on disk, `rows_fetched` is the boundary and a
    later call below that boundary never touches the loader."""
    from data_preparation.lib.sources import loaders as loaders_mod

    calls: list[tuple[int, int, list[str] | None, bool]] = []

    def group_loader(source: SourceConfig, offset: int, count: int, **kwargs: Any) -> Any:
        calls.append((offset, count, kwargs["columns"], kwargs["align_to_row_group"]))
        return iter([{"text": f"row {i}"} for i in range(offset, offset + max(count, 20))])  # a 20-row "row group"

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", group_loader)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    m = download(cfg, "p", layout, rows_needed=11, shard_size=8)
    assert m.rows() == 20 and m.rows_fetched == 20 and [s.rows for s in m.shards] == [8, 8, 4]
    assert calls == [(0, 11, ["text"], True)]
    assert [r["text"] for r in read_rows(layout.source_dir("p", "raw"))] == [f"row {i}" for i in range(20)]
    assert download(cfg, "p", layout, rows_needed=15, shard_size=8) == m and len(calls) == 1  # below the boundary: no-op
    m2 = download(cfg, "p", layout, rows_needed=21, shard_size=8)
    assert calls[-1] == (20, 1, ["text"], True) and m2.rows_fetched == 40 and m2.rows() == 40


def test_download_columns_follow_the_converter_and_check_limit_bounds_over_reads(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
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
    cfg = with_tokenizer(cfg_factory({"conv": _synthetic(converter="gsm8k_question_answer"), "lim": _synthetic(check_limit=5)}))
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
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
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
    cfg = with_tokenizer(cfg_factory({"i": src}))
    m = download(cfg, "i", layout, rows_needed=3, shard_size=10)
    assert m.rows() == 3 and m.rows_fetched == 5 and m.extra["skipped_malformed"] == 2
    assert read_rows(layout.source_dir("i", "raw")) == [
        {"instruction": "what", "input": "", "output": "that", "tokens": 2},
        {"instruction": "how", "input": "background", "output": "so", "tokens": 3},
        {"instruction": "why", "input": "", "output": "because", "tokens": 2},
    ]  # tokens: instruction + input + output


def test_download_instruct_filter_reads_the_source_once(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    good, bad = _sharegpt("h" * 60, "g" * 60), _sharegpt("short", "g" * 60)
    src_dir = layout.root.parent / "sg"
    write_local(src_dir, [bad, bad, good, bad, good, good, bad, good], "jsonl")
    src = _local(src_dir, kind="instruct", converter="sharegpt_conversations", filter="sharegpt_quality")
    cfg = with_tokenizer(cfg_factory({"s": src}))
    m = download(cfg, "s", layout, rows_needed=3, shard_size=10)
    # 3 kept rows need 6 source rows, read through one loader call that stops at the third kept row
    assert m.rows() == 3 and m.rows_fetched == 6 and not m.extra.get("exhausted")
    assert all(r == {"instruction": "h" * 60, "input": "", "output": "g" * 60, "tokens": 2} for r in read_rows(layout.source_dir("s", "raw")))
    m2 = download(cfg, "s", layout, rows_needed=10, shard_size=10)
    assert m2.rows() == 4 and m2.rows_fetched == 8 and m2.extra["exhausted"] is True


def test_download_instruct_check_limit_bounds_inspected_rows(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer
) -> None:
    src_dir = layout.root.parent / "lim"
    write_local(src_dir, [{"instruction": f"i{i}", "output": f"o{i}"} for i in range(10)], "jsonl")
    src = _local(src_dir, kind="instruct", converter="instruction_input_output", check_limit=4)
    cfg = with_tokenizer(cfg_factory({"l": src}))
    m = download(cfg, "l", layout, rows_needed=3, shard_size=10)
    assert m.rows() == 3 and m.rows_fetched == 3 and not m.extra.get("exhausted")
    m = download(cfg, "l", layout, rows_needed=8, shard_size=10)
    assert m.rows() == 4 and m.rows_fetched == 4 and m.extra["exhausted"] is True and m.extra["check_limit"] == 4
    assert download(cfg, "l", layout, rows_needed=8, shard_size=10) == m

    # check_limit is not part of the raw hash: a grown limit reads further instead of re-downloading
    cfg.sources["l"] = _local(src_dir, kind="instruct", converter="instruction_input_output", check_limit=6)
    m = download(cfg, "l", layout, rows_needed=8, shard_size=10)
    assert m.rows() == 6 and m.rows_fetched == 6 and m.extra["exhausted"] is True and m.extra["check_limit"] == 6
    assert [s.rows for s in m.shards] == [3, 1, 2]  # appended, nothing rewritten
    cfg.sources["l"] = _local(src_dir, kind="instruct", converter="instruction_input_output")
    m = download(cfg, "l", layout, rows_needed=8, shard_size=10)
    assert m.rows() == 8 and not m.extra.get("exhausted") and "check_limit" not in m.extra


def test_download_synthetic_instruct_rows(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader) -> None:
    cfg = with_tokenizer(cfg_factory({"i": _synthetic(kind="instruct", seed=2)}))
    m = download(cfg, "i", layout, rows_needed=3)
    assert m.rows() == 3
    rows = read_rows(layout.source_dir("i", "raw"))
    assert rows == [{**synthetic_row("instruct", 2, i), "tokens": r["tokens"]} for i, r in enumerate(rows)]
    counter = TokenCounter(cfg, layout)
    assert all(r["tokens"] == counter.count(instruct_text(r)) for r in rows)


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
        assert current_manifest(tmp_path, "h", "processed") is None
    assert "manifest stage 'raw' != 'processed'" in caplog.text
    assert current_manifest(tmp_path, "other", "raw") is None
    with pytest.raises(FileNotFoundError, match="no current raw manifest"):
        require_manifest(tmp_path / "missing", "h", "raw", "s")
    payload = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert payload["stage"] == "raw"


def test_fetch_source_forces_range_requests_by_default() -> None:
    from dataclasses import replace

    from data_preparation.dataset_config import SourceConfig
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


# --- github_code group download ----------------------------------------------------------------------------------------


def _raw_state(layout: DatasetLayout, names: list[str], read_rows: Reader) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in names:
        manifest = Manifest.load(layout.source_dir(name, "raw"))
        assert manifest is not None
        state[name] = {
            "rows_fetched": manifest.rows_fetched,
            "exhausted": manifest.extra.get("exhausted", False),
            "shards": [(s.name, s.rows) for s in manifest.shards],
            "rows": read_rows(layout.source_dir(name, "raw")),
        }
    return state


def test_download_github_code_group_equals_separate_downloads(
    hub: FakeHub, cfg_factory: CfgFactory, with_tokenizer: Prep, tmp_path: Path, read_rows: Reader
) -> None:
    """Golden: one group pass leaves exactly the raw shards / offsets / exhausted flags of three separate downloads,
    while opening every repo file once."""
    hub.add("data/a.parquet", _code_rows("a", 8))  # row groups of 2; Python a0 a3 a6, Java a1 a4 a7, Go a2 a5
    hub.add("data/b.parquet", _code_rows("b", 8))
    sources = {"py": _github("Python"), "java": _github("Java"), "rust": _github("Rust")}
    cfg = cfg_factory(sources)
    rows_needed = {"py": 4, "java": 2, "rust": 3}

    separate = DatasetLayout(tmp_path / "separate")
    prepare_tokenizer(cfg, separate)
    for name in sources:
        download(cfg, name, separate, rows_needed=rows_needed[name], shard_size=3)
    expected = _raw_state(separate, list(sources), read_rows)
    assert [r["text"] for r in expected["py"]["rows"]] == ["a code 0", "a code 3", "a code 6", "b code 0"]
    assert expected["py"]["rows_fetched"] == 4 and set(expected["py"]["rows"][0]) == {"text", "language", "tokens"}
    assert expected["rust"]["rows"] == [] and expected["rust"]["exhausted"]

    hub.streams.clear()
    grouped = DatasetLayout(tmp_path / "grouped")
    prepare_tokenizer(cfg, grouped)
    manifests = download_github_code_group(cfg, list(sources), grouped, rows_needed=rows_needed, shard_size=3)
    assert set(manifests) == set(sources)
    assert _raw_state(grouped, list(sources), read_rows) == expected
    assert hub.streams == ["data/a.parquet", "data/b.parquet"]  # each file opened once for all three languages

    # a second call with the same needs is a no-op; a larger need for one language tops up only that one
    hub.streams.clear()
    again = download_github_code_group(cfg, list(sources), grouped, rows_needed=rows_needed, shard_size=3)
    assert hub.streams == [] and {n: m.rows() for n, m in again.items()} == {"py": 4, "java": 2, "rust": 0}
    topped = download_github_code_group(cfg, list(sources), grouped, rows_needed={**rows_needed, "java": 4}, shard_size=3)
    assert [r["text"] for r in read_rows(grouped.source_dir("java", "raw"))] == ["a code 1", "a code 4", "a code 7", "b code 1"]
    assert topped["java"].rows_fetched == 4 and topped["py"].rows() == 4
    download(cfg, "java", separate, rows_needed=4, shard_size=3)
    assert _raw_state(grouped, ["java"], read_rows) == _raw_state(separate, ["java"], read_rows)


def test_download_github_code_group_rejects_other_sources(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"py": _github("Python"), "s": _synthetic(), "lim": _github("Java", check_limit=5)}))
    with pytest.raises(ValueError, match="github_code sources without check_limit"):
        download_github_code_group(cfg, ["py", "s"], layout, rows_needed={"py": 1, "s": 1})
    with pytest.raises(ValueError, match="github_code sources without check_limit"):
        download_github_code_group(cfg, ["py", "lim"], layout, rows_needed={"py": 1, "lim": 1})


# --- raw tokens column --------------------------------------------------------------------------------------------------


def _strip_tokens(raw_dir: Path) -> None:
    """Turn a raw directory into one from before the tokens column (same rows, names and offsets)."""
    import pyarrow.parquet as pq

    manifest = Manifest.load(raw_dir)
    assert manifest is not None
    for shard in manifest.shards:
        table = pq.read_table(raw_dir / shard.name).drop_columns(["tokens"])
        pq.write_table(table, raw_dir / shard.name)
        shard.tokens = None
    manifest.token_count = None
    manifest.tokenizer = None
    manifest.save(raw_dir)


def test_ensure_raw_tokens_upgrades_in_place_without_downloading(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}))
    raw_dir = layout.source_dir("p", "raw")
    fresh = download(cfg, "p", layout, rows_needed=25, shard_size=10)
    rows_with_tokens = read_rows(raw_dir)
    _strip_tokens(raw_dir)
    legacy = Manifest.load(raw_dir)
    assert legacy is not None and legacy.tokens() is None and "tokens" not in read_rows(raw_dir)[0]

    from data_preparation.lib.sources import loaders as loaders_mod

    def no_download(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the upgrade must not touch the loader")

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", no_download)
    upgraded = ensure_raw_tokens(cfg, "p", layout)
    assert upgraded is not None and upgraded.tokens() == fresh.tokens() and upgraded.rows_fetched == 25
    assert [(s.name, s.rows, s.tokens) for s in upgraded.shards] == [(s.name, s.rows, s.tokens) for s in fresh.shards]
    assert upgraded.token_count == "tokenizer" and upgraded.tokenizer == "synthetic"
    assert read_rows(raw_dir) == rows_with_tokens
    assert Manifest.load(raw_dir) == upgraded and ensure_raw_tokens(cfg, "p", layout) == upgraded
    assert download(cfg, "p", layout, rows_needed=25, shard_size=10) == upgraded  # still nothing to fetch
    assert ensure_raw_tokens(cfg, "p", DatasetLayout(layout.root / "elsewhere")) is None  # no raw manifest there


def test_download_upgrades_a_legacy_raw_dir_before_topping_up(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}))
    raw_dir = layout.source_dir("p", "raw")
    download(cfg, "p", layout, rows_needed=10, shard_size=10)
    _strip_tokens(raw_dir)
    m = download(cfg, "p", layout, rows_needed=15, shard_size=10)
    assert [(s.name, s.rows) for s in m.shards] == [("data-00000.parquet", 10), ("data-00001.parquet", 5)] and m.rows_fetched == 15
    rows = read_rows(raw_dir)
    assert all("tokens" in r for r in rows) and m.tokens() == sum(r["tokens"] for r in rows)


# --- per-shard publishing, crash safety, cancellation ---------------------------------------------------------------


def _failing_loader(monkeypatch: pytest.MonkeyPatch, fail_at: int | None, total: int = 40) -> list[int]:
    """Stub the synthetic loader with one that yields ``total`` rows from ``offset`` and raises after the
    ``fail_at``-th row of the whole source (None: never); returns the list of offsets it was called with."""
    from data_preparation.lib.sources import loaders as loaders_mod

    offsets: list[int] = []

    def loader(source: SourceConfig, offset: int, count: int, **kwargs: Any) -> Any:
        offsets.append(offset)
        for i in range(offset, min(offset + count, total)):
            if fail_at is not None and i == fail_at:
                raise OSError("connection reset")
            yield {"text": f"row {i}"}

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", loader)
    return offsets


def test_download_publishes_shards_as_they_fill_and_resumes_after_a_failure(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader, tmp_path: Path
) -> None:
    from data_preparation.lib.stages import shared

    monkeypatch.setattr(shared, "TOKEN_BATCH", 5)  # rows reach the writer in small batches (256 in production)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    offsets = _failing_loader(monkeypatch, fail_at=27)
    with pytest.raises(OSError, match="connection reset"):
        download(cfg, "p", layout, rows_needed=40, shard_size=10)
    raw = layout.source_dir("p", "raw")
    m = Manifest.load(raw)
    assert m is not None and [(s.rows, s.offset) for s in m.shards] == [(10, 10), (10, 20)] and m.rows_fetched == 20
    assert m.rows() == 20 and not m.extra.get("exhausted") and not list(raw.glob("*.tmp")) and not (raw.parent / "raw.tmp").exists()

    # the next call resumes at the last complete shard; the loader is asked from offset 20 and nothing is lost
    _failing_loader(monkeypatch, fail_at=None)
    m2 = download(cfg, "p", layout, rows_needed=40, shard_size=10)
    assert offsets == [0] and m2.rows() == 40 and m2.rows_fetched == 40
    assert [(s.name, s.rows, s.offset) for s in m2.shards] == [(f"data-{i:05d}.parquet", 10, 10 * (i + 1)) for i in range(4)]
    assert [r["text"] for r in read_rows(raw)] == [f"row {i}" for i in range(40)]

    # golden: identical to one uninterrupted download
    other = DatasetLayout(tmp_path / "other")
    from data_preparation.lib.stages.shared import prepare_tokenizer

    prepare_tokenizer(cfg, other)
    reference = download(cfg, "p", other, rows_needed=40, shard_size=10)
    assert [(s.name, s.rows, s.tokens, s.offset) for s in reference.shards] == [(s.name, s.rows, s.tokens, s.offset) for s in m2.shards]
    assert read_rows(other.source_dir("p", "raw")) == read_rows(raw)


def test_download_instruct_shard_offsets_count_consumed_source_rows(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer
) -> None:
    """Instruct rows are filtered while downloading: a shard's offset is the number of *source* rows consumed up to
    its last kept row (what a resume must skip), not the number of rows kept."""
    src_dir = layout.root.parent / "ins"
    rows = [{"instruction": f"i{i}", "output": f"o{i}"} if i % 2 else {"instruction": f"i{i}"} for i in range(12)]  # even: malformed
    write_local(src_dir, rows, "jsonl")
    cfg = with_tokenizer(cfg_factory({"l": _local(src_dir, kind="instruct", converter="instruction_input_output")}))
    m = download(cfg, "l", layout, rows_needed=6, shard_size=2)
    assert [(s.rows, s.offset) for s in m.shards] == [(2, 4), (2, 8), (2, 12)] and m.rows_fetched == 12
    assert m.extra["skipped_malformed"] == 6


def test_download_stops_within_one_shard_when_asked(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data_preparation.lib.abort import BuildAborted

    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    _failing_loader(monkeypatch, fail_at=None, total=100)
    stop = {"now": False}
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        stop["now"] = calls["n"] >= 2  # requested after the second shard was published
        return stop["now"]

    with pytest.raises(BuildAborted):
        download(cfg, "p", layout, rows_needed=100, shard_size=10, should_stop=should_stop)
    m = Manifest.load(layout.source_dir("p", "raw"))
    assert m is not None and m.rows() == 20 and m.rows_fetched == 20 and calls["n"] == 2
    assert download(cfg, "p", layout, rows_needed=100, shard_size=10).rows() == 100  # resumes to completion


def test_truncate_raw_to_good_prefix(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    from data_preparation.lib.stages.shared import truncate_raw_to_good_prefix

    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    _failing_loader(monkeypatch, fail_at=None)
    download(cfg, "p", layout, rows_needed=40, shard_size=10)
    raw = layout.source_dir("p", "raw")
    m = Manifest.load(raw)
    assert m is not None and truncate_raw_to_good_prefix(raw, m) and len(m.shards) == 4  # nothing wrong: untouched

    (raw / "data-00002.parquet").write_bytes(b"corrupt")
    assert truncate_raw_to_good_prefix(raw, m)
    assert [s.name for s in m.shards] == ["data-00000.parquet", "data-00001.parquet"] and m.rows_fetched == 20
    assert sorted(p.name for p in raw.glob("*.parquet")) == ["data-00000.parquet", "data-00001.parquet"]
    stored = Manifest.load(raw)
    assert stored is not None and stored.rows_fetched == 20
    m2 = download(cfg, "p", layout, rows_needed=40, shard_size=10)  # resumes from the kept prefix
    assert m2.rows() == 40 and [r["text"] for r in read_rows(raw)] == [f"row {i}" for i in range(40)]

    (raw / "data-00000.parquet").unlink()  # nothing to keep
    m3 = Manifest.load(raw)
    assert m3 is not None and not truncate_raw_to_good_prefix(raw, m3)
    m3.shards[0].offset = None  # a legacy manifest without offsets: no safe resume point
    (raw / "data-00000.parquet").write_bytes(b"x")
    assert not truncate_raw_to_good_prefix(raw, m3)


def test_raw_tokens_count_the_max_chars_prefix_and_a_changed_max_chars_recounts_in_place(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data_preparation.dataset_config import ProcessingConfig
    from data_preparation.lib.stages import shared

    src_dir = layout.root.parent / "long"
    write_local(src_dir, [{"text": " ".join(["tok_1"] * 50)}], "parquet")  # 299 chars, 50 tokens
    cfg = with_tokenizer(cfg_factory({"p": _local(src_dir)}, processing=ProcessingConfig(min_chars=1, max_chars=59), max_seq_length=4096))
    m = download(cfg, "p", layout, rows_needed=1)
    assert [r["tokens"] for r in read_rows(layout.source_dir("p", "raw"))] == [10], "only the first max_chars are counted"
    assert m.extra["counted_chars"] == 59 and m.tokens() == 10

    def no_fetch(*args: object, **kwargs: object) -> object:
        raise AssertionError("a max_chars change must not download")

    monkeypatch.setattr(shared, "_fetch_rows", no_fetch)
    cfg.processing = ProcessingConfig(min_chars=1, max_chars=119)
    m2 = download(cfg, "p", layout, rows_needed=1)  # the raw hash is unchanged: recounted in place
    assert m2.rows_fetched == 1 and m2.extra["counted_chars"] == 119 and m2.tokens() == 20
    assert [r["tokens"] for r in read_rows(layout.source_dir("p", "raw"))] == [20]


def test_download_instruct_filter_calls_the_loader_once_and_closes_it(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filtered instruct download must not re-open the source per iteration (a remote JSON file would be
    re-streamed from byte 0 every time): one loader call, closed as soon as enough rows are kept."""
    from data_preparation.lib.sources import loaders as loaders_mod

    calls: list[tuple[int, int]] = []
    closed: list[bool] = []

    def loader(source: SourceConfig, offset: int, count: int, **kwargs: Any) -> Any:
        calls.append((offset, count))
        try:
            for i in range(offset, offset + count):
                yield {"instruction": f"i{i}", "output": f"o{i}" if i % 4 == 0 else "x"}  # the filter keeps 1 in 4
        finally:
            closed.append(True)

    def keep_every_fourth(row: dict[str, Any]) -> bool:
        return bool(row["output"] != "x")

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", loader)
    from data_preparation.lib.sources import converters as converters_mod

    monkeypatch.setitem(converters_mod.FILTERS, "every_fourth", keep_every_fourth)
    src = _synthetic(kind="instruct", converter="instruction_input_output", filter="every_fourth")
    cfg = with_tokenizer(cfg_factory({"s": src}))
    m = download(cfg, "s", layout, rows_needed=20, shard_size=10)
    assert m.rows() == 20 and m.rows_fetched == 77 and not m.extra.get("exhausted")
    assert calls == [(0, 2**62)] and closed == [True], "one call, closed after the 20th kept row"
    m2 = download(cfg, "s", layout, rows_needed=25, shard_size=10)
    assert m2.rows() == 25 and calls[1] == (77, 2**62)
