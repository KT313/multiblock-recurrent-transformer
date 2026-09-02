# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.stages.download: tokenizer step, token counter, incremental download (truncation at
the token cap, dropped long instruct rows, resumable counters), raw manifest states."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import os

import pytest

from data_preparation.dataset_config import DatasetConfig, SourceConfig, TokenizerConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.raw_folder import RawFolder
from data_preparation.lib.sources.synthetic import synthetic_row
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.conftest import REPO, REV, FakeHub
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.stages.download import (
    _auto_tokenizer,
    RawFolderError,
    TokenCounter,
    current_manifest,
    current_raw_manifest,
    download,
    download_github_code_group,
    prepare_tokenizer,
    raw_manifest_problem,
    raw_manifest_state,
)
from data_preparation.lib.storage.parquet import estimate_tokens

download_module = importlib.import_module("data_preparation.lib.stages.download")  # the package attribute `download` is the function

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


def test_token_counter_tokenizer_mode_counts_uncapped_and_truncates(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Callable[[DatasetConfig], DatasetConfig]
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(), "i": _synthetic(kind="instruct")}, max_seq_length=5))
    counter = TokenCounter(cfg, layout)
    assert counter.count("tok_1 tok_2 tok_3") == 3
    assert counter.count(" ".join(["tok_1"] * 9)) == 9, "no cap: the download truncates text / drops rows instead"
    assert counter.count_many(["tok_1", " ".join(["tok_2"] * 7), ""]) == [1, 7, 0]
    assert counter.truncate_many([" ".join(["tok_2"] * 7), "tok_1"], 5) == [("tok_2 tok_2 tok_2 tok_2 tok_2 ", 5), ("tok_1", 1)]
    assert counter.mode == "tokenizer" and counter.tokenizer_name == "synthetic"


def test_token_counter_estimate_mode_needs_no_tokenizer(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()}, token_count="estimate", max_seq_length=10)
    counter = TokenCounter(cfg, layout)  # tokenizer dir does not exist
    assert counter.count("a" * 8) == 2 and counter.count("a" * 400) == 100
    assert counter.count_many(["a" * 8, "a" * 400]) == [2, 100]
    assert counter.truncate_many(["a" * 400], 10) == [("a" * 40, 10)]


def test_token_counter_missing_tokenizer_raises(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()})
    with pytest.raises(FileNotFoundError, match="run the tokenizer stage first"):
        TokenCounter(cfg, layout)


# --- download: pretrain ------------------------------------------------------------------------------------------------


def test_download_synthetic_appends_incrementally(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, mtimes: Mtimes, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, max_seq_length=512))  # above every synthetic row: nothing truncated
    raw = layout.raw_dir("p")
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
    rows = read_rows(layout.raw_dir("g"))
    assert rows[0] == {"text": "Question: q0\n\nAnswer: a0", "tokens": 6}
    assert m.token_count == "tokenizer" and m.tokenizer == "synthetic" and m.tokens() == sum(r["tokens"] for r in rows)
    # exhausted: a larger request is a no-op
    assert download(cfg, "g", layout, rows_needed=100, shard_size=4) == m


def test_download_projects_to_the_text_field_and_requires_it(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    """A pretrain source read as-is stores only its `text_field` (plus `tokens`): the projection applies to every
    file format, local `.jsonl` included, so the surplus `lang` column never reaches the raw shards."""
    src_dir = layout.root.parent / "code"
    write_local(src_dir, [{"code": "print(1)" * 10, "lang": "py"}], "jsonl")
    cfg = with_tokenizer(cfg_factory({"c": _local(src_dir, text_field="code")}))
    download(cfg, "c", layout, rows_needed=1)
    assert read_rows(layout.raw_dir("c")) == [{"code": "print(1)" * 10, "tokens": 40}]  # tokens of `code`; no `lang`
    bad = cfg_factory({"c": _local(src_dir, text_field="text")})
    other = DatasetLayout(layout.root / "other")
    prepare_tokenizer(bad, other)
    with pytest.raises(ValueError, match="no 'text' column"):
        download(bad, "c", other, rows_needed=1)



def test_download_raises_instead_of_deleting_a_stale_or_outdated_raw_folder(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, mtimes: Mtimes, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}))
    m = download(cfg, "p", layout, rows_needed=12, shard_size=5)
    raw = layout.raw_dir("p")
    before, rows_before = mtimes(raw), read_rows(raw)

    stale = cfg_factory({"p": _synthetic(seed=9)})  # a different seed -> different raw hash
    with pytest.raises(RawFolderError, match=r"p: raw folder .* is stale: identity/tokenizer changed") as info:
        download(stale, "p", layout, rows_needed=7, shard_size=5)
    assert isinstance(info.value, RuntimeError) and info.value.directory == raw
    outdated = replace(cfg, max_seq_length=cfg.max_seq_length * 2)
    with pytest.raises(RawFolderError, match=r"p: raw folder .* is outdated: max_seq_length 64 -> 128") as info:
        download(outdated, "p", layout, rows_needed=7, shard_size=5)
    assert "download never deletes raw" in str(info.value)
    assert mtimes(raw) == before and read_rows(raw) == rows_before and Manifest.load(raw) == m, "nothing deleted or rewritten"
    # a lowered cap is fine: the rows are at most 64 tokens long, which is more than the config now needs
    assert download(replace(cfg, max_seq_length=32), "p", layout, rows_needed=12, shard_size=5) == m


def test_download_refuses_to_restart_over_shards_without_a_manifest(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}))
    download(cfg, "p", layout, rows_needed=3)
    raw = layout.raw_dir("p")
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
    assert [r["text"] for r in read_rows(layout.raw_dir("p"))] == [f"row {i}" for i in range(20)]
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
    assert read_rows(layout.raw_dir("i")) == [
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
    assert all(r == {"instruction": "h" * 60, "input": "", "output": "g" * 60, "tokens": 2} for r in read_rows(layout.raw_dir("s")))
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
    cfg = with_tokenizer(cfg_factory({"i": _synthetic(kind="instruct", seed=2)}, max_seq_length=128))  # above every synthetic row
    m = download(cfg, "i", layout, rows_needed=3)
    assert m.rows() == 3
    rows = read_rows(layout.raw_dir("i"))
    assert rows == [{**synthetic_row("instruct", 2, i), "tokens": r["tokens"]} for i, r in enumerate(rows)]
    counter = TokenCounter(cfg, layout)
    assert all(r["tokens"] == counter.count(instruct_text(r)) for r in rows)


# --- manifest helpers --------------------------------------------------------------------------------------------------


def test_current_manifest_stage_mismatch_and_require(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    Manifest(source="s", source_hash="h", stage="raw").save(tmp_path)
    assert current_manifest(tmp_path, "h", "raw") is not None
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert current_manifest(tmp_path, "h", "processed") is None
    assert "manifest stage 'raw' != 'processed'" in caplog.text
    assert current_manifest(tmp_path, "other", "raw") is None
    payload = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert payload["stage"] == "raw"


def test_fetch_source_forces_range_requests_by_default() -> None:
    from dataclasses import replace

    from data_preparation.dataset_config import SourceConfig
    from data_preparation.lib.stages.download import fetch_source

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
        manifest = Manifest.load(layout.raw_dir(name))
        assert manifest is not None
        state[name] = {
            "rows_fetched": manifest.rows_fetched,
            "exhausted": manifest.extra.get("exhausted", False),
            "shards": [(s.name, s.rows) for s in manifest.shards],
            "rows": read_rows(layout.raw_dir(name)),
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
    assert [r["text"] for r in read_rows(grouped.raw_dir("java"))] == ["a code 1", "a code 4", "a code 7", "b code 1"]
    assert topped["java"].rows_fetched == 4 and topped["py"].rows() == 4
    download(cfg, "java", separate, rows_needed=4, shard_size=3)
    assert _raw_state(grouped, ["java"], read_rows) == _raw_state(separate, ["java"], read_rows)


def test_download_github_code_group_rejects_other_sources(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"py": _github("Python"), "s": _synthetic(), "lim": _github("Java", check_limit=5)}))
    with pytest.raises(ValueError, match="github_code sources without check_limit"):
        download_github_code_group(cfg, ["py", "s"], layout, rows_needed={"py": 1, "s": 1})
    with pytest.raises(ValueError, match="github_code sources without check_limit"):
        download_github_code_group(cfg, ["py", "lim"], layout, rows_needed={"py": 1, "lim": 1})


# --- truncation at the token cap, dropped instruct rows, raw manifest state ---------------------------------------------


def test_loading_the_tokenizer_disables_tokenizers_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rust tokenizer threads plus a later fork is the known `tokenizers` deadlock; loading must set the guard."""
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    _auto_tokenizer()
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")  # an explicit user choice is respected
    _auto_tokenizer()
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"



def test_download_truncates_pretrain_text_at_the_token_cap(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader
) -> None:
    """Stored text is a prefix of the source text that re-tokenizes to <= the cap; `tokens` is the stored text's own
    count; rows under the cap are stored unchanged; the manifest records the cap."""
    cap = 100
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, max_seq_length=cap))
    m = download(cfg, "p", layout, rows_needed=40, shard_size=10)
    counter = TokenCounter(cfg, layout)
    rows = read_rows(layout.raw_dir("p"))
    assert len(rows) == 40 and m.truncated_at_tokens == cap and m.token_count == "tokenizer" and m.tokenizer == "synthetic"
    truncated = unchanged = 0
    for index, row in enumerate(rows):
        original = synthetic_row("pretrain", 3, index)["text"]
        assert original.startswith(row["text"]) and row["tokens"] == counter.count(row["text"]) <= cap
        if counter.count(original) <= cap:
            assert row["text"] == original
            unchanged += 1
        else:
            assert row["tokens"] == cap and len(row["text"]) < len(original)
            truncated += 1
    assert truncated > 0 and unchanged > 0, "the synthetic rows (64..384 words) fall on both sides of the cap"
    assert m.tokens() == sum(r["tokens"] for r in rows) and all(s.tokens is not None for s in m.shards)


def test_download_appends_at_the_folder_cap_when_the_config_cap_is_lower(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader
) -> None:
    """Lowering `max_seq_length` is free, so an append under the lower cap must not shorten the folder's rows: the
    manifest keeps promising `truncated_at_tokens`, and raising the cap back must still be `current`, not a lie."""
    high = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, max_seq_length=100))
    download(high, "p", layout, rows_needed=20, shard_size=10)
    low = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, max_seq_length=40))
    assert raw_manifest_state(low, "p", layout) == "current", "a lower cap never outdates the folder"
    m = download(low, "p", layout, rows_needed=40, shard_size=10)
    rows = read_rows(layout.raw_dir("p"))
    assert len(rows) == 40 and m.truncated_at_tokens == 100
    counter = TokenCounter(high, layout)
    appended = rows[20:]
    assert all(counter.count(row["text"]) <= 100 for row in appended)
    assert any(counter.count(row["text"]) > 40 for row in appended), "appended rows are cut at the folder's cap, not the config's"
    assert raw_manifest_state(high, "p", layout) == "current" and read_rows(layout.raw_dir("p"))[:20] == rows[:20]


def test_download_truncates_in_estimate_mode_at_four_chars_per_token(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "est"
    write_local(src_dir, [{"text": "x" * 1000}, {"text": "short"}], "parquet")
    cfg = cfg_factory({"e": _local(src_dir)}, token_count="estimate", max_seq_length=10)
    m = download(cfg, "e", layout, rows_needed=2)  # no tokenizer stage needed
    rows = read_rows(layout.raw_dir("e"))
    assert rows == [{"text": "x" * 40, "tokens": 10}, {"text": "short", "tokens": estimate_tokens("short")}]
    assert m.truncated_at_tokens == 10 and m.token_count == "estimate" and m.tokenizer is None


def test_download_github_code_group_truncates_like_separate_downloads(
    hub: FakeHub, cfg_factory: CfgFactory, tmp_path: Path, read_rows: Reader
) -> None:
    """The group path (`_fetch_group`) runs the same token step as `download`: identical truncated texts and counts."""
    long_rows = [
        {"id": f"r{i}", "text": " ".join(f"tok_{(i + j) % 256}" for j in range(20)), "language": ("Python", "Java")[i % 2]} for i in range(6)
    ]
    hub.add("data/a.parquet", long_rows)
    cap = 7
    cfg = cfg_factory({"py": _github("Python"), "java": _github("Java")}, max_seq_length=cap)
    separate, grouped = DatasetLayout(tmp_path / "separate"), DatasetLayout(tmp_path / "grouped")
    for layout in (separate, grouped):
        prepare_tokenizer(cfg, layout)
    for name in ("py", "java"):
        download(cfg, name, separate, rows_needed=3, shard_size=2)
    manifests = download_github_code_group(cfg, ["py", "java"], grouped, rows_needed={"py": 3, "java": 3}, shard_size=2)
    counter = TokenCounter(cfg, grouped)
    for name in ("py", "java"):
        rows = read_rows(grouped.raw_dir(name))
        assert rows == read_rows(separate.raw_dir(name)) and len(rows) == 3
        assert all(r["tokens"] == cap == counter.count(r["text"]) and r["text"].count(" ") == cap for r in rows)  # cut where token 7 starts
        assert all(any(source["text"].startswith(r["text"]) for source in long_rows) for r in rows)
        assert manifests[name].truncated_at_tokens == cap and manifests[name].tokens() == 3 * cap


def _instruct_rows_with_long_and_malformed(n: int) -> list[Row]:
    """Index i: malformed (no output) when i % 3 == 0, too long (10 words) when i % 3 == 1, kept (2 words) otherwise."""
    rows: list[Row] = []
    for i in range(n):
        if i % 3 == 0:
            rows.append({"instruction": f"i{i}"})
        else:
            rows.append({"instruction": f"i{i}", "output": " ".join(f"tok_{i}" for _ in range(9 if i % 3 == 1 else 1))})
    return rows


@pytest.mark.parametrize("token_batch", [256, 3])
def test_download_instruct_drops_long_rows_and_counts_them_once_across_a_resume(
    cfg_factory: CfgFactory,
    with_tokenizer: Prep,
    layout: DatasetLayout,
    write_local: Writer,
    read_rows: Reader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    token_batch: int,
) -> None:
    """Rows over `max_seq_length` are not stored, never truncated; `dropped_too_long` / `skipped_malformed` are
    recorded per shard up to its last stored row, so a stop after the first shard and a resume count every rejected
    row exactly once (round-2 bug: the totals were saved from the running counters)."""
    monkeypatch.setattr(download_module, "TOKEN_BATCH", token_batch)
    src_dir = layout.root.parent / "drop"
    write_local(src_dir, _instruct_rows_with_long_and_malformed(30), "jsonl")
    cfg = with_tokenizer(cfg_factory({"d": _local(src_dir, kind="instruct", converter="instruction_input_output")}, max_seq_length=5))

    with pytest.raises(BuildAborted):
        download(cfg, "d", layout, rows_needed=10, shard_size=4, should_stop=lambda: True)  # checked after each shard
    partial = Manifest.load(layout.raw_dir("d"))
    assert partial is not None and [(s.rows, s.offset) for s in partial.shards] == [(4, 12)]  # kept rows i = 2, 5, 8, 11
    assert partial.rows_fetched == 12 and partial.extra == {"skipped_malformed": 4, "dropped_too_long": 4}

    m = download(cfg, "d", layout, rows_needed=10, shard_size=4)
    assert m.rows() == 10 and m.rows_fetched == 30 and m.extra["skipped_malformed"] == 10 and m.extra["dropped_too_long"] == 10
    assert m.truncated_at_tokens == 5
    rows = read_rows(layout.raw_dir("d"))
    assert [r["instruction"] for r in rows] == [f"i{i}" for i in range(30) if i % 3 == 2]
    assert all(r["tokens"] == 2 and len(r["output"].split()) == 1 for r in rows), "no long row stored, none truncated"

    other = DatasetLayout(tmp_path / "other")
    prepare_tokenizer(cfg, other)
    reference = download(cfg, "d", other, rows_needed=10, shard_size=4)
    assert reference.extra == m.extra and reference.rows_fetched == m.rows_fetched
    assert [(s.name, s.rows, s.tokens, s.offset) for s in reference.shards] == [(s.name, s.rows, s.tokens, s.offset) for s in m.shards]
    assert read_rows(other.raw_dir("d")) == rows


def test_download_instruct_estimate_mode_drops_by_estimated_count(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "est_i"
    write_local(src_dir, [{"instruction": "a" * 20, "output": "b" * 20}, {"instruction": "q", "output": "a"}], "jsonl")
    src = _local(src_dir, kind="instruct", converter="instruction_input_output")
    cfg = cfg_factory({"e": src}, token_count="estimate", max_seq_length=8)
    m = download(cfg, "e", layout, rows_needed=2)
    stored = read_rows(layout.raw_dir("e"))
    assert stored == [{"instruction": "q", "input": "", "output": "a", "tokens": estimate_tokens(instruct_text(stored[0]))}]
    assert stored[0]["tokens"] <= 8 < estimate_tokens(instruct_text({"instruction": "a" * 20, "input": "", "output": "b" * 20}))
    assert m.extra["dropped_too_long"] == 1 and m.extra["exhausted"] is True


def test_raw_manifest_state_and_current_raw_manifest(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}, max_seq_length=64))
    assert raw_manifest_state(cfg, "p", layout) == "missing" and current_raw_manifest(cfg, "p", layout) is None
    assert raw_manifest_problem(cfg, "p", layout) is None
    m = download(cfg, "p", layout, rows_needed=5)
    assert raw_manifest_state(cfg, "p", layout) == "current" and current_raw_manifest(cfg, "p", layout) == m
    assert raw_manifest_problem(cfg, "p", layout) is None

    raised = replace(cfg, max_seq_length=4096)
    assert raw_manifest_state(raised, "p", layout) == "outdated" and current_raw_manifest(raised, "p", layout) is None
    assert raw_manifest_problem(raised, "p", layout) == "outdated: max_seq_length 64 -> 4096"
    lowered = replace(cfg, max_seq_length=16)
    assert raw_manifest_state(lowered, "p", layout) == "current" and current_raw_manifest(lowered, "p", layout) == m

    other_tokenizer = cfg_factory({"p": _synthetic(seed=0)}, tokenizer=TokenizerConfig(name="other", kind="synthetic"))
    assert raw_manifest_state(other_tokenizer, "p", layout) == "stale" and current_raw_manifest(other_tokenizer, "p", layout) is None
    assert raw_manifest_problem(other_tokenizer, "p", layout) == "stale: identity/tokenizer changed"
    assert raw_manifest_state(cfg_factory({"p": _synthetic(seed=1)}), "p", layout) == "stale"
    # stale wins over outdated (the folder holds other rows altogether)
    assert raw_manifest_state(replace(other_tokenizer, max_seq_length=4096), "p", layout) == "stale"
    # a manifest of another stage in the raw folder is stale too
    Manifest(source="p", source_hash=cfg.raw_hash("p"), stage="processed").save(layout.raw_dir("p"))
    assert raw_manifest_state(cfg, "p", layout) == "stale"


def test_download_github_code_group_raises_for_an_outdated_member(
    hub: FakeHub, cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, mtimes: Mtimes
) -> None:
    hub.add("data/a.parquet", _code_rows("a", 6))
    cfg = with_tokenizer(cfg_factory({"py": _github("Python"), "java": _github("Java")}))
    download_github_code_group(cfg, ["py", "java"], layout, rows_needed={"py": 2, "java": 2})
    before = {name: mtimes(layout.raw_dir(name)) for name in ("py", "java")}
    hub.streams.clear()
    with pytest.raises(RawFolderError, match="py: raw folder .* is outdated: max_seq_length 64 -> 65"):
        download_github_code_group(replace(cfg, max_seq_length=65), ["py", "java"], layout, rows_needed={"py": 4, "java": 4})
    assert hub.streams == [] and {name: mtimes(layout.raw_dir(name)) for name in ("py", "java")} == before


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
    monkeypatch.setattr(download_module, "TOKEN_BATCH", 5)  # rows reach the writer in small batches (256 in production)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    offsets = _failing_loader(monkeypatch, fail_at=27)
    with pytest.raises(OSError, match="connection reset"):
        download(cfg, "p", layout, rows_needed=40, shard_size=10)
    raw = layout.raw_dir("p")
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
    prepare_tokenizer(cfg, other)
    reference = download(cfg, "p", other, rows_needed=40, shard_size=10)
    assert [(s.name, s.rows, s.tokens, s.offset) for s in reference.shards] == [(s.name, s.rows, s.tokens, s.offset) for s in m2.shards]
    assert read_rows(other.raw_dir("p")) == read_rows(raw)


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
    m = Manifest.load(layout.raw_dir("p"))
    assert m is not None and m.rows() == 20 and m.rows_fetched == 20 and calls["n"] == 2
    assert download(cfg, "p", layout, rows_needed=100, shard_size=10).rows() == 100  # resumes to completion


def test_truncate_raw_to_good_prefix(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    _failing_loader(monkeypatch, fail_at=None)
    download(cfg, "p", layout, rows_needed=40, shard_size=10)
    raw = layout.raw_dir("p")
    m = Manifest.load(raw)
    assert m is not None and RawFolder(raw, m).truncate_to_good_prefix() and len(m.shards) == 4  # nothing wrong: untouched

    (raw / "data-00002.parquet").write_bytes(b"corrupt")
    assert RawFolder(raw, m).truncate_to_good_prefix()
    assert [s.name for s in m.shards] == ["data-00000.parquet", "data-00001.parquet"] and m.rows_fetched == 20
    assert sorted(p.name for p in raw.glob("*.parquet")) == ["data-00000.parquet", "data-00001.parquet"]
    stored = Manifest.load(raw)
    assert stored is not None and stored.rows_fetched == 20
    m2 = download(cfg, "p", layout, rows_needed=40, shard_size=10)  # resumes from the kept prefix
    assert m2.rows() == 40 and [r["text"] for r in read_rows(raw)] == [f"row {i}" for i in range(40)]

    (raw / "data-00000.parquet").unlink()  # nothing to keep
    m3 = Manifest.load(raw)
    assert m3 is not None and not RawFolder(raw, m3).truncate_to_good_prefix()
    m3.shards[0].offset = None  # a legacy manifest without offsets: no safe resume point
    (raw / "data-00000.parquet").write_bytes(b"x")
    assert not RawFolder(raw, m3).truncate_to_good_prefix()


def test_download_instruct_counts_rejected_rows_once_across_a_truncate_and_resume(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    """A repair that drops a broken shard restores **every** counter from the last kept shard, so the resume behind
    it counts each rejected source row exactly once (round-2 finding D-M4: only `rows_fetched` and the exhaustion
    flag were reset, so the malformed / too-long rows of the dropped shards were counted twice)."""
    src_dir = layout.root.parent / "retruncate"
    write_local(src_dir, _instruct_rows_with_long_and_malformed(30), "jsonl")
    cfg = with_tokenizer(cfg_factory({"d": _local(src_dir, kind="instruct", converter="instruction_input_output")}, max_seq_length=5))
    full = download(cfg, "d", layout, rows_needed=10, shard_size=4)
    assert full.extra == {"skipped_malformed": 10, "dropped_too_long": 10} and full.rows_fetched == 30
    shards = [(s.rows, s.offset, s.skipped_malformed, s.dropped_too_long) for s in full.shards]
    assert shards == [(4, 12, 4, 4), (4, 24, 8, 8), (2, 30, 10, 10)]  # per shard: the totals up to its last stored row
    rows = read_rows(layout.raw_dir("d"))

    raw = layout.raw_dir("d")
    (raw / "data-00001.parquet").write_bytes(b"corrupt")
    broken = Manifest.load(raw)
    assert broken is not None and RawFolder(raw, broken).truncate_to_good_prefix()
    assert broken.rows_fetched == 12 and broken.extra == {"skipped_malformed": 4, "dropped_too_long": 4}

    resumed = download(cfg, "d", layout, rows_needed=10, shard_size=4)
    assert resumed.extra == full.extra and resumed.rows_fetched == 30
    assert [(s.rows, s.offset, s.skipped_malformed, s.dropped_too_long) for s in resumed.shards] == shards
    assert read_rows(raw) == rows


def test_truncating_a_manifest_without_shard_counters_resets_them(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A raw folder written before the per-shard counters existed still loads and truncates: the counters restart at
    0 with a log line instead of crashing or making the folder stale."""
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    _failing_loader(monkeypatch, fail_at=None)
    download(cfg, "p", layout, rows_needed=40, shard_size=10)
    raw = layout.raw_dir("p")
    legacy = Manifest.load(raw)
    assert legacy is not None
    for shard in legacy.shards:  # a manifest from before the fields existed
        shard.skipped_malformed = shard.dropped_too_long = None
    legacy.extra["skipped_malformed"] = legacy.extra["dropped_too_long"] = 7
    legacy.save(raw)

    (raw / "data-00002.parquet").write_bytes(b"corrupt")
    reloaded = Manifest.load(raw)
    assert reloaded is not None and all(s.skipped_malformed is None for s in reloaded.shards)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert RawFolder(raw, reloaded).truncate_to_good_prefix()
    assert reloaded.rows_fetched == 20 and reloaded.extra["skipped_malformed"] == 0 and reloaded.extra["dropped_too_long"] == 0
    assert "written before the per-shard reject counters existed" in caplog.text
    assert raw_manifest_state(cfg, "p", layout) == "current"  # never stale because of the missing fields


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
