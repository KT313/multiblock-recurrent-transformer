# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.stages.download: tokenizer step, token counter, incremental download (truncation at
the token cap, dropped long instruct rows, resumable counters), raw manifest states.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from data_preparation.dataset_config import DatasetConfig, SourceConfig, TokenizerConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.raw_folder import RawFolder
from data_preparation.lib.sources.loaders import SharedLoaderParameters
from data_preparation.lib.sources.synthetic import synthetic_row
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.lib.ui.dashboard import DataDashboard
from data_preparation.conftest import REPO, REV, FakeHub, truncate_to_good_prefix
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.stages.download import (
    _load_tokenizer,
    RawFolderError,
    TokenCounter,
    current_manifest,
    download,
    download_github_code_group,
    inspect_raw,
    prepare_tokenizer,
    reopen_raw,
)
from data_preparation.lib.stages.truncation import CHARS_PER_TOKEN_ESTIMATE, NUMBER_OF_SPECIAL_TOKENS, estimate_tokens

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
        calls.append((name, kwargs.get("revision"), kwargs.get("token")))
        return real(str(tiny_tokenizer_dir), *args)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    tok = TokenizerConfig(name="llama", kind="hf", hf_id="org/tok", revision="abc")
    cfg = cfg_factory({"p": _synthetic()}, tokenizer=tok)
    manifest = prepare_tokenizer(cfg, layout, hf_token="hf_secret")
    assert calls == [("org/tok", "abc", "hf_secret")]
    assert (layout.tokenizer_dir("llama") / "tokenizer_config.json").is_file()
    assert manifest.extra == {"kind": "hf", "hf_id": "org/tok", "revision": "abc"}
    assert prepare_tokenizer(cfg, layout) == manifest and len(calls) == 1


# --- token counter ---------------------------------------------------------------------------------------------------


def test_token_counter_tokenizer_mode_counts_uncapped_and_truncates(
    cfg_factory: CfgFactory, layout: DatasetLayout, with_tokenizer: Callable[[DatasetConfig], DatasetConfig]
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(), "i": _synthetic(kind="instruct")}, dataset_max_sequence_length=5))
    counter = TokenCounter(cfg, layout)
    assert counter.count("tok_1 tok_2 tok_3") == 3
    assert counter.count(" ".join(["tok_1"] * 9)) == 9, "no cap: the download truncates text / drops rows instead"
    assert counter.count_many(["tok_1", " ".join(["tok_2"] * 7), ""]) == [1, 7, 0]
    assert counter.truncate_many([" ".join(["tok_2"] * 7), "tok_1"], 5) == [("tok_2 tok_2 tok_2 tok_2 tok_2 ", 5), ("tok_1", 1)]
    assert counter.mode == "tokenizer" and counter.tokenizer_name == "synthetic"


def test_token_counter_estimate_mode_needs_no_tokenizer(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = cfg_factory({"p": _synthetic()}, token_count="estimate", dataset_max_sequence_length=10)
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
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, dataset_max_sequence_length=512))  # above every synthetic row: nothing truncated
    raw = layout.raw_dir("p")
    m1 = download(cfg, "p", layout, rows_needed=25, shard_size=10)
    assert [s.rows for s in m1.shards] == [10, 10, 5] and m1.rows_fetched == 25 and m1.stage == "raw"
    assert m1.source_hash == cfg.raw_hash("p") and not m1.exhausted
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
    """
    Hub-file loaders get `index_dir=<dataset>/hub_index` and an `on_file` callback from the download stage.
    """

    from data_preparation.lib.sources import loaders as loaders_mod

    seen: list[SharedLoaderParameters] = []

    def fake_loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
        seen.append(shared_parameters)
        assert shared_parameters.on_file is not None
        shared_parameters.on_file("data/x.parquet")
        return iter([{"text": "a"}, {"text": "b"}][:count])

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", fake_loader)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    manifest = download(cfg, "p", layout, rows_needed=2, hf_token="tok")
    assert manifest.rows() == 2
    assert [p.index_dir for p in seen] == [layout.hub_index_dir()] and seen[0].token == "tok"


def test_the_download_row_reports_the_bytes_the_loader_fetched(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The dashboard row of a download reads the loader's `FetchStats` live and keeps the final count once closed.
    """

    from data_preparation.lib.sources import loaders as loaders_mod

    def fake_loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
        assert shared_parameters.stats is not None
        shared_parameters.stats.bytes_fetched += 12_345
        return iter([{"text": "a"}, {"text": "b"}][:count])

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", fake_loader)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    with DataDashboard(enabled=True, console=console) as board:
        assert download(cfg, "p", layout, rows_needed=2).rows() == 2
        (task,) = board._panels["downloads"].done
        assert task.bytes_fetched == 12_345 and "12" not in task.postfix.values()


def test_download_local_applies_converter_and_flags_exhaustion(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "gsm"
    write_local(src_dir, [{"question": f"q{i}", "answer": f"a{i}", "extra": i} for i in range(7)], "parquet")
    cfg = with_tokenizer(cfg_factory({"g": _local(src_dir, converter="gsm8k_question_answer")}))
    m = download(cfg, "g", layout, rows_needed=10, shard_size=4)
    assert m.rows() == 7 and m.rows_fetched == 7 and m.exhausted is True
    rows = read_rows(layout.raw_dir("g"))
    assert rows[0] == {"text": "Question: q0\n\nAnswer: a0", "tokens": 6 + NUMBER_OF_SPECIAL_TOKENS}
    assert m.token_count == "tokenizer" and m.tokenizer == "synthetic" and m.tokens() == sum(r["tokens"] for r in rows)
    # exhausted: a larger request is a no-op
    assert download(cfg, "g", layout, rows_needed=100, shard_size=4) == m


def test_download_projects_to_the_text_field_and_requires_it(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    """
    A pretrain source read as-is stores only its `text_field` (plus `tokens`): the projection applies to every
    file format, local `.jsonl` included, so the surplus `lang` column never reaches the raw shards.
    """

    src_dir = layout.root.parent / "code"
    write_local(src_dir, [{"code": "print(1)" * 10, "lang": "py"}], "jsonl")
    cfg = with_tokenizer(cfg_factory({"c": _local(src_dir, text_field="code")}))
    download(cfg, "c", layout, rows_needed=1)
    assert read_rows(layout.raw_dir("c")) == [{"code": "print(1)" * 10, "tokens": 40 + NUMBER_OF_SPECIAL_TOKENS}]  # tokens of `code`; no `lang`
    bad = cfg_factory({"c": _local(src_dir, text_field="text")})
    other = DatasetLayout(layout.root / "other")
    prepare_tokenizer(bad, other)
    with pytest.raises(ValueError, match="no 'text' column"):
        download(bad, "c", other, rows_needed=1)



def test_pretrain_rows_are_projected_to_a_string_text_field_whatever_the_loader_yields(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    """
    Loaders without a column projection (hf_split, hf_stream) deliver every source column: a surplus column whose
    type varies between rows would fail the Arrow conversion, and a non-string text value would be stored as-is.
    """

    from data_preparation.lib.sources import loaders as loaders_mod

    def loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
        yield {"text": "tok_1 tok_2", "meta": {"nested": 1}}
        yield {"text": 42, "meta": "a string this time"}
        yield {"text": None, "meta": None}

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", loader)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    download(cfg, "p", layout, rows_needed=3)
    (shard,) = sorted(layout.raw_dir("p").glob("data-*.parquet"))
    schema = pq.read_schema(shard)
    assert schema.names == ["text", "tokens"]
    assert pa.types.is_string(schema.field("text").type) and pa.types.is_integer(schema.field("tokens").type)
    assert [r["text"] for r in read_rows(layout.raw_dir("p"))] == ["tok_1 tok_2", "42", ""]


def test_a_written_shard_holds_only_the_row_columns(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """
    The fetch progress travels next to the row, never in it: no bookkeeping column reaches the shard.
    """

    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}))
    download(cfg, "p", layout, rows_needed=3, shard_size=5)
    (shard,) = sorted(layout.raw_dir("p").glob("data-*.parquet"))
    assert pq.read_schema(shard).names == ["text", "tokens"]


def test_download_raises_instead_of_deleting_a_stale_or_outdated_raw_folder(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, mtimes: Mtimes, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}))
    m = download(cfg, "p", layout, rows_needed=12, shard_size=5)
    raw = layout.raw_dir("p")
    before, rows_before = mtimes(raw), read_rows(raw)

    stale = cfg_factory({"p": _synthetic(seed=9)})  # a different seed -> different raw hash
    with pytest.raises(RawFolderError, match=r"p: raw folder .* is stale: source.seed: 0 -> 9; it must be deleted and downloaded again") as info:
        download(stale, "p", layout, rows_needed=7, shard_size=5)
    assert isinstance(info.value, RuntimeError) and info.value.directory == raw
    outdated = replace(cfg, dataset_max_sequence_length=cfg.dataset_max_sequence_length * 2)
    with pytest.raises(RawFolderError, match=r"p: raw folder .* is outdated: dataset_max_sequence_length 64 -> 128") as info:
        download(outdated, "p", layout, rows_needed=7, shard_size=5)
    assert "download never deletes raw" in str(info.value)
    assert mtimes(raw) == before and read_rows(raw) == rows_before and Manifest.load(raw) == m, "nothing deleted or rewritten"
    # a lowered cap is fine: the rows are at most 64 tokens long, which is more than the config now needs
    assert download(replace(cfg, dataset_max_sequence_length=32, training_target_sequence_length=32), "p", layout, rows_needed=12, shard_size=5) == m


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
    """
    A remote parquet loader finishes its row group: all rows land on disk, `rows_fetched` is the boundary and a
    later call below that boundary never touches the loader.
    """

    from data_preparation.lib.sources import loaders as loaders_mod

    calls: list[tuple[int, int, list[str] | None, bool]] = []

    def group_loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
        calls.append((offset, count, shared_parameters.columns, shared_parameters.align_to_row_group))
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

    def loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
        seen.append(shared_parameters.columns)
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
    assert download(cfg, "lim", layout, rows_needed=8).exhausted is True and len(seen) == 2


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
    assert m.rows() == 3 and m.rows_fetched == 5 and m.skipped_malformed == 2
    assert read_rows(layout.raw_dir("i")) == [
        {"instruction": "what", "input": "", "output": "that", "tokens": 2 + NUMBER_OF_SPECIAL_TOKENS},
        {"instruction": "how", "input": "background", "output": "so", "tokens": 3 + NUMBER_OF_SPECIAL_TOKENS},
        {"instruction": "why", "input": "", "output": "because", "tokens": 2 + NUMBER_OF_SPECIAL_TOKENS},
    ]  # tokens: instruction + input + output, plus the trainer's BOS and EOS


def test_download_instruct_skips_converter_results_without_instruction_or_output(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A converter that returns a row without `instruction` / `output` is as malformed as one that raises; the row
    is skipped and counted, the download goes on.
    """

    from data_preparation.lib.sources import converters as converters_mod

    def half_converter(row: Row) -> Row:
        return {key: row[key] for key in ("instruction", "output") if key in row}

    monkeypatch.setitem(converters_mod.CONVERTERS, "half", half_converter)
    src_dir = layout.root.parent / "half"
    write_local(src_dir, [{"instruction": "a"}, {"output": "b"}, {"instruction": "c", "output": "d"}], "jsonl")
    cfg = with_tokenizer(cfg_factory({"h": _local(src_dir, kind="instruct", converter="half")}))
    m = download(cfg, "h", layout, rows_needed=3)
    assert m.rows() == 1 and m.rows_fetched == 3 and m.skipped_malformed == 2 and m.exhausted is True
    assert read_rows(layout.raw_dir("h")) == [{"instruction": "c", "input": "", "output": "d", "tokens": 2 + NUMBER_OF_SPECIAL_TOKENS}]


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
    assert m.rows() == 3 and m.rows_fetched == 6 and not m.exhausted
    assert all(r == {"instruction": "h" * 60, "input": "", "output": "g" * 60, "tokens": 2 + NUMBER_OF_SPECIAL_TOKENS} for r in read_rows(layout.raw_dir("s")))
    m2 = download(cfg, "s", layout, rows_needed=10, shard_size=10)
    assert m2.rows() == 4 and m2.rows_fetched == 8 and m2.exhausted is True


def test_download_instruct_check_limit_bounds_inspected_rows(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer
) -> None:
    src_dir = layout.root.parent / "lim"
    write_local(src_dir, [{"instruction": f"i{i}", "output": f"o{i}"} for i in range(10)], "jsonl")
    src = _local(src_dir, kind="instruct", converter="instruction_input_output", check_limit=4)
    cfg = with_tokenizer(cfg_factory({"l": src}))
    m = download(cfg, "l", layout, rows_needed=3, shard_size=10)
    assert m.rows() == 3 and m.rows_fetched == 3 and not m.exhausted
    m = download(cfg, "l", layout, rows_needed=8, shard_size=10)
    assert m.rows() == 4 and m.rows_fetched == 4 and m.exhausted is True and m.check_limit_reached == 4
    assert download(cfg, "l", layout, rows_needed=8, shard_size=10) == m

    # check_limit is not part of the raw hash: a grown limit reads further instead of re-downloading
    cfg.sources["l"] = _local(src_dir, kind="instruct", converter="instruction_input_output", check_limit=6)
    m = download(cfg, "l", layout, rows_needed=8, shard_size=10)
    assert m.rows() == 6 and m.rows_fetched == 6 and m.exhausted is True and m.check_limit_reached == 6
    assert [s.rows for s in m.shards] == [3, 1, 2]  # appended, nothing rewritten
    cfg.sources["l"] = _local(src_dir, kind="instruct", converter="instruction_input_output")
    m = download(cfg, "l", layout, rows_needed=8, shard_size=10)
    assert m.rows() == 8 and not m.exhausted and m.check_limit_reached is None


def test_download_synthetic_instruct_rows(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader) -> None:
    cfg = with_tokenizer(cfg_factory({"i": _synthetic(kind="instruct", seed=2)}, dataset_max_sequence_length=128))  # above every synthetic row
    m = download(cfg, "i", layout, rows_needed=3)
    assert m.rows() == 3
    rows = read_rows(layout.raw_dir("i"))
    assert rows == [{**synthetic_row("instruct", 2, i), "tokens": r["tokens"]} for i, r in enumerate(rows)]
    counter = TokenCounter(cfg, layout)
    assert all(r["tokens"] == counter.count(instruct_text(r)) + NUMBER_OF_SPECIAL_TOKENS for r in rows)


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


def test_fetch_source_forces_range_requests_by_default(cfg_factory: CfgFactory) -> None:
    from data_preparation.lib.stages.download import fetch_source

    files = SourceConfig(kind="pretrain", loader="hf_files", hf_id="o/files", load_kwargs={"data_files": "data/*.parquet"})
    split = SourceConfig(kind="pretrain", loader="hf_split", hf_id="o/split", load_kwargs={"name": "main"})
    cfg = cfg_factory({"files": files, "py": _github("Python"), "split": split})
    assert cfg.always_range_requests
    src = cfg.sources["files"]
    assert fetch_source(cfg, src).load_kwargs["max_cached_file_mb"] == 0
    assert "max_cached_file_mb" not in src.load_kwargs  # original untouched
    assert fetch_source(cfg, cfg.sources["py"]).load_kwargs["max_cached_file_mb"] == 0
    assert fetch_source(cfg, cfg.sources["split"]) is cfg.sources["split"]  # hf_split: not a hub_files source
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
            "exhausted": manifest.exhausted,
            "shards": [(s.name, s.rows) for s in manifest.shards],
            "rows": read_rows(layout.raw_dir(name)),
        }
    return state


def test_download_github_code_group_equals_separate_downloads(
    hub: FakeHub, cfg_factory: CfgFactory, with_tokenizer: Prep, tmp_path: Path, read_rows: Reader
) -> None:
    """
    Golden: one group pass gives every member the rows, offsets and exhausted flag of its separate download as a
    prefix, while opening every repo file once; what the pass reads on for Rust (never satisfied: the whole repo)
    is kept as well: Python's and Java's rows past their targets, and Go's rows in a folder of their own.
    """

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
    assert expected["py"]["rows_fetched"] == 4 and set(expected["py"]["rows"][0]) == {"text", "tokens"}  # `language` only routes the row
    assert expected["rust"]["rows"] == [] and expected["rust"]["exhausted"]

    hub.streams.clear()
    grouped = DatasetLayout(tmp_path / "grouped")
    prepare_tokenizer(cfg, grouped)
    manifests = download_github_code_group(cfg, list(sources), grouped, rows_needed=rows_needed, shard_size=3)
    assert set(manifests) == {*sources, "name_go"}  # REPO is org/name: Go's folder is sources/name_go
    state = _raw_state(grouped, [*sources, "name_go"], read_rows)
    assert state["rust"] == expected["rust"]
    for name in ("py", "java"):
        assert state[name]["rows"][: rows_needed[name]] == expected[name]["rows"] and not state[name]["exhausted"]
    assert [r["text"] for r in state["py"]["rows"]] == ["a code 0", "a code 3", "a code 6", "b code 0", "b code 3", "b code 6"]
    assert [r["text"] for r in state["java"]["rows"]] == ["a code 1", "a code 4", "a code 7", "b code 1", "b code 4", "b code 7"]
    assert [r["text"] for r in state["name_go"]["rows"]] == ["a code 2", "a code 5", "b code 2", "b code 5"]
    assert {name: state[name]["rows_fetched"] for name in state} == {"py": 6, "java": 6, "rust": 0, "name_go": 4}
    assert set(state["name_go"]["rows"][0]) == {"text", "tokens"} and all(r["tokens"] == 5 for r in state["name_go"]["rows"])
    go = Manifest.load(grouped.raw_dir("name_go"))
    assert go is not None and go.extra == {"github_code_group": ["py", "java", "rust"], "surplus": True} and not go.exhausted
    assert go.source_hash == cfg.raw_hash_of(replace(sources["py"], language="Go"))  # a config entry `name_go: {..., language: Go}` adopts it
    assert hub.streams == ["data/a.parquet", "data/b.parquet"]  # each file opened once for all languages

    # a second call with the same needs reads nothing (no member has rows to fetch); a member asked for more than
    # the repo holds is exhausted without a read: its rows in both files are known
    hub.streams.clear()
    again = download_github_code_group(cfg, list(sources), grouped, rows_needed=rows_needed, shard_size=3)
    assert hub.streams == [] and {n: m.rows() for n, m in again.items()} == {"py": 6, "java": 6, "rust": 0}
    topped = download_github_code_group(cfg, list(sources), grouped, rows_needed={**rows_needed, "py": 8}, shard_size=3)
    assert hub.streams == [] and topped["py"].rows() == 6 and topped["py"].exhausted


def test_download_github_code_group_rejects_other_sources(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"py": _github("Python"), "s": _synthetic()}))
    with pytest.raises(ValueError, match="needs github_code sources"):
        download_github_code_group(cfg, ["py", "s"], layout, rows_needed={"py": 1, "s": 1})


# --- truncation at the token cap, dropped instruct rows, raw manifest state ---------------------------------------------


def test_loading_the_tokenizer_disables_tokenizers_parallelism(monkeypatch: pytest.MonkeyPatch, tiny_tokenizer_dir: Path) -> None:
    """
    Rust tokenizer threads plus a later fork is the known `tokenizers` deadlock; loading must set the guard.
    """

    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    _load_tokenizer(tiny_tokenizer_dir, "synthetic")
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")  # an explicit user choice is respected
    _load_tokenizer(tiny_tokenizer_dir, "synthetic")
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"



def test_download_truncates_pretrain_text_at_the_token_cap(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader
) -> None:
    """
    Stored text is a prefix of the source text; `tokens` is the stored text's own count plus the trainer's BOS and
    EOS and is <= the cap; rows that fit with the specials are stored unchanged; the manifest records the cap.
    """

    cap = 100
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, dataset_max_sequence_length=cap))
    m = download(cfg, "p", layout, rows_needed=40, shard_size=10)
    counter = TokenCounter(cfg, layout)
    rows = read_rows(layout.raw_dir("p"))
    assert len(rows) == 40 and m.truncated_at_tokens == cap and m.token_count == "tokenizer" and m.tokenizer == "synthetic"
    truncated = unchanged = 0
    for index, row in enumerate(rows):
        original = synthetic_row("pretrain", 3, index)["text"]
        assert original.startswith(row["text"]) and row["tokens"] == counter.count(row["text"]) + NUMBER_OF_SPECIAL_TOKENS <= cap
        if counter.count(original) + NUMBER_OF_SPECIAL_TOKENS <= cap:
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
    """
    Lowering `dataset_max_sequence_length` is free, so an append under the lower cap must not shorten the folder's rows: the
    manifest keeps promising `truncated_at_tokens`, and raising the cap back must still be `current`, not a lie.
    """

    high = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, dataset_max_sequence_length=100))
    download(high, "p", layout, rows_needed=20, shard_size=10)
    low = with_tokenizer(cfg_factory({"p": _synthetic(seed=3)}, dataset_max_sequence_length=40))
    assert inspect_raw(low, "p", layout).state == "current", "a lower cap never outdates the folder"
    m = download(low, "p", layout, rows_needed=40, shard_size=10)
    rows = read_rows(layout.raw_dir("p"))
    assert len(rows) == 40 and m.truncated_at_tokens == 100
    counter = TokenCounter(high, layout)
    appended = rows[20:]
    assert all(counter.count(row["text"]) <= 100 for row in appended)
    assert any(counter.count(row["text"]) > 40 for row in appended), "appended rows are cut at the folder's cap, not the config's"
    assert inspect_raw(high, "p", layout).state == "current" and read_rows(layout.raw_dir("p"))[:20] == rows[:20]


def test_download_truncates_in_estimate_mode_at_four_chars_per_token(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "est"
    write_local(src_dir, [{"text": "x" * 1000}, {"text": "short"}], "parquet")
    cfg = cfg_factory({"e": _local(src_dir)}, token_count="estimate", dataset_max_sequence_length=10)
    m = download(cfg, "e", layout, rows_needed=2)  # no tokenizer stage needed
    rows = read_rows(layout.raw_dir("e"))
    text_cap = (10 - NUMBER_OF_SPECIAL_TOKENS) * CHARS_PER_TOKEN_ESTIMATE  # the specials take 2 of the 10 tokens
    assert rows == [{"text": "x" * text_cap, "tokens": 10}, {"text": "short", "tokens": estimate_tokens("short") + NUMBER_OF_SPECIAL_TOKENS}]
    assert m.truncated_at_tokens == 10 and m.token_count == "estimate" and m.tokenizer is None


def test_download_github_code_group_truncates_like_separate_downloads(
    hub: FakeHub, cfg_factory: CfgFactory, tmp_path: Path, read_rows: Reader
) -> None:
    """
    The group path (`_fetch_group`) runs the same token step as `download`: identical truncated texts and counts.
    """

    long_rows = [
        {"id": f"r{i}", "text": " ".join(f"tok_{(i + j) % 256}" for j in range(20)), "language": ("Python", "Java")[i % 2]} for i in range(6)
    ]
    hub.add("data/a.parquet", long_rows)
    cap = 7
    cfg = cfg_factory({"py": _github("Python"), "java": _github("Java")}, dataset_max_sequence_length=cap)
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
        words = cap - NUMBER_OF_SPECIAL_TOKENS
        assert all(r["tokens"] == cap == counter.count(r["text"]) + NUMBER_OF_SPECIAL_TOKENS and r["text"].count(" ") == words for r in rows)  # cut where token 5 starts
        assert all(any(source["text"].startswith(r["text"]) for source in long_rows) for r in rows)
        assert manifests[name].truncated_at_tokens == cap and manifests[name].tokens() == 3 * cap


def _instruct_rows_with_long_and_malformed(n: int) -> list[Row]:
    """
    Index i: malformed (no output) when i % 3 == 0, too long (10 words) when i % 3 == 1, kept (2 words) otherwise.
    """

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
    """
    Rows over `dataset_max_sequence_length` are not stored, never truncated; `dropped_too_long` / `skipped_malformed` are
    recorded per shard up to its last stored row, so a stop after the first shard and a resume count every rejected
    row exactly once (round-2 bug: the totals were saved from the running counters).
    """

    monkeypatch.setattr(download_module, "TOKEN_BATCH", token_batch)
    src_dir = layout.root.parent / "drop"
    write_local(src_dir, _instruct_rows_with_long_and_malformed(30), "jsonl")
    cfg = with_tokenizer(cfg_factory({"d": _local(src_dir, kind="instruct", converter="instruction_input_output")}, dataset_max_sequence_length=5))

    with pytest.raises(BuildAborted):
        download(cfg, "d", layout, rows_needed=10, shard_size=4, should_stop=lambda: True)  # checked after each shard
    partial = Manifest.load(layout.raw_dir("d"))
    # kept rows i = 2, 5, 8, 11 in the shard that tripped the stop, then the row consumed before the stop was seen
    # (i = 14, published as a short shard): everything consumed is stored, the offset is where the fetch stood
    assert partial is not None and [(s.rows, s.offset) for s in partial.shards] == [(4, 12), (1, 15)]
    assert partial.rows_fetched == 15 and (partial.skipped_malformed, partial.dropped_too_long) == (5, 5)

    m = download(cfg, "d", layout, rows_needed=10, shard_size=4)
    assert m.rows() == 10 and m.rows_fetched == 30 and m.skipped_malformed == 10 and m.dropped_too_long == 10
    assert m.truncated_at_tokens == 5
    rows = read_rows(layout.raw_dir("d"))
    assert [r["instruction"] for r in rows] == [f"i{i}" for i in range(30) if i % 3 == 2]
    assert all(r["tokens"] == 2 + NUMBER_OF_SPECIAL_TOKENS and len(r["output"].split()) == 1 for r in rows), "no long row stored, none truncated"

    other = DatasetLayout(tmp_path / "other")
    prepare_tokenizer(cfg, other)
    reference = download(cfg, "d", other, rows_needed=10, shard_size=4)
    assert (reference.skipped_malformed, reference.dropped_too_long) == (m.skipped_malformed, m.dropped_too_long) and reference.rows_fetched == m.rows_fetched
    assert (reference.rows(), reference.tokens()) == (m.rows(), m.tokens()) and read_rows(other.raw_dir("d")) == rows  # only the shard boundaries differ


def test_download_instruct_estimate_mode_drops_by_estimated_count(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    src_dir = layout.root.parent / "est_i"
    write_local(src_dir, [{"instruction": "a" * 20, "output": "b" * 20}, {"instruction": "q", "output": "a"}], "jsonl")
    src = _local(src_dir, kind="instruct", converter="instruction_input_output")
    cfg = cfg_factory({"e": src}, token_count="estimate", dataset_max_sequence_length=8)
    m = download(cfg, "e", layout, rows_needed=2)
    stored = read_rows(layout.raw_dir("e"))
    assert stored == [{"instruction": "q", "input": "", "output": "a", "tokens": estimate_tokens(instruct_text(stored[0])) + NUMBER_OF_SPECIAL_TOKENS}]
    assert stored[0]["tokens"] <= 8 < estimate_tokens(instruct_text({"instruction": "a" * 20, "input": "", "output": "b" * 20})) + NUMBER_OF_SPECIAL_TOKENS
    assert m.dropped_too_long == 1 and m.exhausted is True


def test_inspect_raw_states(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}, dataset_max_sequence_length=64))
    assert inspect_raw(cfg, "p", layout).state == "missing" and inspect_raw(cfg, "p", layout).current_manifest is None
    assert inspect_raw(cfg, "p", layout).reason == "missing"
    m = download(cfg, "p", layout, rows_needed=5)
    assert inspect_raw(cfg, "p", layout).state == "current" and inspect_raw(cfg, "p", layout).current_manifest == m
    assert inspect_raw(cfg, "p", layout).reason == "current"

    raised = replace(cfg, dataset_max_sequence_length=4096)
    assert inspect_raw(raised, "p", layout).state == "outdated" and inspect_raw(raised, "p", layout).current_manifest is None
    assert inspect_raw(raised, "p", layout).reason == "outdated: dataset_max_sequence_length 64 -> 4096"
    lowered = replace(cfg, dataset_max_sequence_length=16, training_target_sequence_length=16)
    assert inspect_raw(lowered, "p", layout).state == "current" and inspect_raw(lowered, "p", layout).current_manifest == m

    other_seed = cfg_factory({"p": _synthetic(seed=1)}, dataset_max_sequence_length=64)
    assert inspect_raw(other_seed, "p", layout).state == "stale" and inspect_raw(other_seed, "p", layout).current_manifest is None
    assert inspect_raw(other_seed, "p", layout).reason == "stale: source.seed: 0 -> 1", "the changed fields, from the recorded payload"
    # stale wins over outdated (the folder holds other rows altogether)
    assert inspect_raw(replace(other_seed, dataset_max_sequence_length=4096), "p", layout).state == "stale"

    # the tokenizer is not raw identity: the same rows, counted differently, are a choice for the repair step
    other_tokenizer = cfg_factory({"p": _synthetic(seed=0)}, dataset_max_sequence_length=64, tokenizer=TokenizerConfig(name="other", kind="synthetic"))
    inspection = inspect_raw(other_tokenizer, "p", layout)
    assert (inspection.state, inspection.current_manifest, inspection.manifest) == ("tokenizer_changed", None, m)
    assert inspection.reason == (
        "tokenizer changed: synthetic (synthetic) -> other (synthetic); 5 rows were counted and truncated under the old one, "
        "their token counts and truncation will not match the new tokenizer"
    )
    estimate = cfg_factory({"p": _synthetic(seed=0)}, dataset_max_sequence_length=64, token_count="estimate")
    assert inspect_raw(estimate, "p", layout).state == "tokenizer_changed"
    assert inspect_raw(estimate, "p", layout).reason == "token_count changed: tokenizer -> estimate; 5 rows were counted the old way, their stored token counts will not match the new one"
    # outdated wins over tokenizer_changed (the folder is re-downloaded with the new tokenizer anyway)
    assert inspect_raw(replace(other_tokenizer, dataset_max_sequence_length=4096), "p", layout).state == "outdated"
    # a same-named tokenizer whose definition changed: the tokenizer step already replaced tokenizers/<name>, so only the hash names the old one
    _edit_raw_manifest(layout, "p", tokenizer_hash="0123456789abcdef")
    assert inspect_raw(cfg, "p", layout).reason.startswith("tokenizer changed: synthetic (definition 0123456789abcdef, no longer under tokenizers/) -> synthetic (synthetic); 5 rows")
    # a manifest from before the tokenizer hash was recorded: unknown is not a change
    _edit_raw_manifest(layout, "p", tokenizer_hash=None)
    assert inspect_raw(other_tokenizer, "p", layout).state == "current" and inspect_raw(estimate, "p", layout).state == "tokenizer_changed"
    # a manifest from before the payload was recorded says so instead of listing fields
    _edit_raw_manifest(layout, "p", hash_payload=None)
    assert inspect_raw(other_seed, "p", layout).reason == "stale: (no field detail recorded)"
    # a manifest of another stage in the raw folder is stale too
    Manifest(source="p", source_hash=cfg.raw_hash("p"), stage="processed").save(layout.raw_dir("p"))
    assert inspect_raw(cfg, "p", layout).state == "stale" and inspect_raw(cfg, "p", layout).reason == "stale: a processed manifest where a raw one belongs"


def _edit_raw_manifest(layout: DatasetLayout, name: str, **changes: Any) -> None:
    manifest = Manifest.load(layout.raw_dir(name))
    assert manifest is not None
    for key, value in changes.items():
        setattr(manifest, key, value)
    manifest.save(layout.raw_dir(name))


def test_download_refuses_a_folder_counted_with_another_tokenizer(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, mtimes: Mtimes
) -> None:
    """
    Appending rows counted with the new tokenizer to rows counted with the old one is the repair step's adopt
    decision, not the download's: it refuses like for a stale folder, names the remedy, and rewrites nothing.
    The raw manifest records how the counts were made so the repair step can re-label it.
    """

    cfg = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}))
    m = download(cfg, "p", layout, rows_needed=5)
    assert (m.tokenizer, m.tokenizer_hash, m.token_count, m.hash_payload) == ("synthetic", cfg.tokenizer_hash(), "tokenizer", cfg.raw_hash_payload("p"))
    before = mtimes(layout.raw_dir("p"))
    other = with_tokenizer(cfg_factory({"p": _synthetic(seed=0)}, tokenizer=TokenizerConfig(name="other", kind="synthetic")))
    with pytest.raises(RawFolderError, match=r"is tokenizer changed: synthetic \(synthetic\) -> other \(synthetic\); 5 rows .*; the download never deletes raw data; the repair step asks whether to keep the folder and go on with the new tokenizer$"):
        download(other, "p", layout, rows_needed=10)
    assert mtimes(layout.raw_dir("p")) == before and Manifest.load(layout.raw_dir("p")) == m
    estimate = cfg_factory({"p": _synthetic(seed=0)}, token_count="estimate")
    with pytest.raises(RawFolderError, match="is token_count changed: tokenizer -> estimate"):
        download(estimate, "p", layout, rows_needed=10)
    estimate_only = cfg_factory({"e": _synthetic(seed=0)}, token_count="estimate")
    e = download(estimate_only, "e", layout, rows_needed=5)
    assert (e.tokenizer, e.tokenizer_hash, e.token_count) == (None, None, "estimate"), "an estimate needs no tokenizer"


def test_an_unreadable_raw_manifest_is_a_state_the_download_refuses(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """
    `Manifest.load` raises for a manifest it cannot parse next to shards; `inspect_raw` reports that as a state
    (status and the planner describe it) and the download refuses the folder like a stale one, deleting nothing.
    """

    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    download(cfg, "p", layout, rows_needed=5)
    (layout.raw_dir("p") / "MANIFEST.json").write_text("{ not json")
    inspection = inspect_raw(cfg, "p", layout)
    assert (inspection.state, inspection.manifest, inspection.current_manifest) == ("unreadable", None, None)
    assert inspection.reason == "unreadable manifest next to shards; fix or delete the directory by hand"
    with pytest.raises(RawFolderError, match=r"p: raw folder .* is unreadable manifest next to shards; fix or delete the directory by hand; the download never deletes raw data$"):
        download(cfg, "p", layout, rows_needed=10)
    assert (layout.raw_dir("p") / "data-00000.parquet").is_file()
    with pytest.raises(RuntimeError, match="unreadable manifest"):
        Manifest.load(layout.raw_dir("p"))  # other callers still get the error


def test_reopen_raw_clears_the_exhausted_latch(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer) -> None:
    """
    A loader that yielded fewer rows than asked leaves the manifest exhausted for good; `reopen_raw` (prepare
    --reopen) clears the flag and the limit it recorded, so the next download reads on from the offset.
    """

    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": f"tok_{i}"} for i in range(3)], "parquet")
    cfg = with_tokenizer(cfg_factory({"g": _local(src_dir, check_limit=3)}))
    assert not reopen_raw(cfg, "g", layout)  # no folder yet
    m = download(cfg, "g", layout, rows_needed=5)
    assert m.exhausted and m.check_limit_reached == 3
    assert download(cfg, "g", layout, rows_needed=5) == m  # exhausted: a no-op
    assert reopen_raw(cfg, "g", layout) and not reopen_raw(cfg, "g", layout)  # cleared once, nothing to clear twice
    reopened = Manifest.load(layout.raw_dir("g"))
    assert reopened is not None and not reopened.exhausted and reopened.check_limit_reached is None and reopened.rows_fetched == 3
    write_local(src_dir, [{"text": f"tok_{i}"} for i in range(3, 8)], "parquet")
    cfg.sources["g"].check_limit = None
    m2 = download(cfg, "g", layout, rows_needed=5)
    assert m2.rows() == 5 and m2.rows_fetched == 5 and not m2.exhausted and [s.rows for s in m2.shards] == [3, 2]


def test_download_github_code_group_raises_for_an_outdated_member(
    hub: FakeHub, cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, mtimes: Mtimes
) -> None:
    hub.add("data/a.parquet", _code_rows("a", 6))
    cfg = with_tokenizer(cfg_factory({"py": _github("Python"), "java": _github("Java")}))
    download_github_code_group(cfg, ["py", "java"], layout, rows_needed={"py": 2, "java": 2})
    before = {name: mtimes(layout.raw_dir(name)) for name in ("py", "java")}
    hub.streams.clear()
    with pytest.raises(RawFolderError, match="py: raw folder .* is outdated: dataset_max_sequence_length 64 -> 65"):
        download_github_code_group(replace(cfg, dataset_max_sequence_length=65), ["py", "java"], layout, rows_needed={"py": 4, "java": 4})
    assert hub.streams == [] and {name: mtimes(layout.raw_dir(name)) for name in ("py", "java")} == before


# --- per-shard publishing, crash safety, cancellation ---------------------------------------------------------------


def _failing_loader(monkeypatch: pytest.MonkeyPatch, fail_at: int | None, total: int = 40) -> list[int]:
    """
    Stub the synthetic loader with one that yields total rows from offset and raises after the
    fail_at-th row of the whole source (None: never); returns the list of offsets it was called with.
    """

    from data_preparation.lib.sources import loaders as loaders_mod

    offsets: list[int] = []

    def loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
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
    # the two full shards, plus every row consumed when the loader failed (20..26: tokenized and buffered, or still
    # waiting for the tokenizer), published as a short shard so the offset is where the download really stood
    assert m is not None and [(s.rows, s.offset) for s in m.shards] == [(10, 10), (10, 20), (7, 27)] and m.rows_fetched == 27
    assert m.rows() == 27 and not m.exhausted and not list(raw.glob("*.tmp")) and not (raw.parent / "raw.tmp").exists()

    # the next call resumes behind the last published shard; the loader is asked from offset 27 and nothing is lost
    _failing_loader(monkeypatch, fail_at=None)
    m2 = download(cfg, "p", layout, rows_needed=40, shard_size=10)
    assert offsets == [0] and m2.rows() == 40 and m2.rows_fetched == 40
    assert [(s.name, s.rows, s.offset) for s in m2.shards] == [
        ("data-00000.parquet", 10, 10), ("data-00001.parquet", 10, 20), ("data-00002.parquet", 7, 27), ("data-00003.parquet", 10, 37), ("data-00004.parquet", 3, 40),
    ]
    assert [r["text"] for r in read_rows(raw)] == [f"row {i}" for i in range(40)]

    # golden: the same rows and counts as one uninterrupted download (only the shard boundaries differ)
    other = DatasetLayout(tmp_path / "other")
    prepare_tokenizer(cfg, other)
    reference = download(cfg, "p", other, rows_needed=40, shard_size=10)
    assert (reference.rows(), reference.tokens(), reference.rows_fetched) == (m2.rows(), m2.tokens(), m2.rows_fetched)
    assert read_rows(other.raw_dir("p")) == read_rows(raw)


def test_download_instruct_shard_offsets_count_consumed_source_rows(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer
) -> None:
    """
    Instruct rows are filtered while downloading: a shard's offset is the number of *source* rows consumed up to
    its last kept row (what a resume must skip), not the number of rows kept.
    """

    src_dir = layout.root.parent / "ins"
    rows = [{"instruction": f"i{i}", "output": f"o{i}"} if i % 2 else {"instruction": f"i{i}"} for i in range(12)]  # even: malformed
    write_local(src_dir, rows, "jsonl")
    cfg = with_tokenizer(cfg_factory({"l": _local(src_dir, kind="instruct", converter="instruction_input_output")}))
    m = download(cfg, "l", layout, rows_needed=6, shard_size=2)
    assert [(s.rows, s.offset) for s in m.shards] == [(2, 4), (2, 8), (2, 12)] and m.rows_fetched == 12
    assert m.skipped_malformed == 6


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
    # the stop is seen at the second shard and asked no more (the check is suspended from then on); the rows the
    # fetch thread had consumed meanwhile (an instant loader runs ahead by a few batches) are stored, not dropped
    assert m is not None and 20 <= m.rows() == m.rows_fetched <= 100 and calls["n"] == 2
    assert [shard.rows for shard in m.shards[:2]] == [10, 10]
    assert download(cfg, "p", layout, rows_needed=100, shard_size=10).rows() == 100  # resumes to completion


def test_truncate_raw_to_good_prefix(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    _failing_loader(monkeypatch, fail_at=None)
    download(cfg, "p", layout, rows_needed=40, shard_size=10)
    raw = layout.raw_dir("p")
    m = Manifest.load(raw)
    assert m is not None and truncate_to_good_prefix(RawFolder(raw, m)) and len(m.shards) == 4  # nothing wrong: untouched

    (raw / "data-00002.parquet").write_bytes(b"corrupt")
    assert truncate_to_good_prefix(RawFolder(raw, m))
    assert [s.name for s in m.shards] == ["data-00000.parquet", "data-00001.parquet"] and m.rows_fetched == 20
    assert sorted(p.name for p in raw.glob("*.parquet")) == ["data-00000.parquet", "data-00001.parquet"]
    stored = Manifest.load(raw)
    assert stored is not None and stored.rows_fetched == 20
    m2 = download(cfg, "p", layout, rows_needed=40, shard_size=10)  # resumes from the kept prefix
    assert m2.rows() == 40 and [r["text"] for r in read_rows(raw)] == [f"row {i}" for i in range(40)]

    (raw / "data-00000.parquet").unlink()  # nothing to keep
    m3 = Manifest.load(raw)
    assert m3 is not None and not truncate_to_good_prefix(RawFolder(raw, m3))
    m3.shards[0].offset = None  # a legacy manifest without offsets: no safe resume point
    (raw / "data-00000.parquet").write_bytes(b"x")
    assert not truncate_to_good_prefix(RawFolder(raw, m3))


def test_download_instruct_counts_rejected_rows_once_across_a_truncate_and_resume(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, write_local: Writer, read_rows: Reader
) -> None:
    """
    A repair that drops a broken shard restores every counter from the last kept shard, so the resume behind
    it counts each rejected source row exactly once (round-2 finding D-M4: only `rows_fetched` and the exhaustion
    flag were reset, so the malformed / too-long rows of the dropped shards were counted twice).
    """

    src_dir = layout.root.parent / "retruncate"
    write_local(src_dir, _instruct_rows_with_long_and_malformed(30), "jsonl")
    cfg = with_tokenizer(cfg_factory({"d": _local(src_dir, kind="instruct", converter="instruction_input_output")}, dataset_max_sequence_length=5))
    full = download(cfg, "d", layout, rows_needed=10, shard_size=4)
    assert (full.skipped_malformed, full.dropped_too_long) == (10, 10) and full.rows_fetched == 30
    shards = [(s.rows, s.offset, s.skipped_malformed, s.dropped_too_long) for s in full.shards]
    assert shards == [(4, 12, 4, 4), (4, 24, 8, 8), (2, 30, 10, 10)]  # per shard: the totals up to its last stored row
    rows = read_rows(layout.raw_dir("d"))

    raw = layout.raw_dir("d")
    (raw / "data-00001.parquet").write_bytes(b"corrupt")
    broken = Manifest.load(raw)
    assert broken is not None and truncate_to_good_prefix(RawFolder(raw, broken))
    assert broken.rows_fetched == 12 and (broken.skipped_malformed, broken.dropped_too_long) == (4, 4)

    resumed = download(cfg, "d", layout, rows_needed=10, shard_size=4)
    assert (resumed.skipped_malformed, resumed.dropped_too_long) == (10, 10) and resumed.rows_fetched == 30
    assert [(s.rows, s.offset, s.skipped_malformed, s.dropped_too_long) for s in resumed.shards] == shards
    assert read_rows(raw) == rows


def test_truncating_a_manifest_without_shard_counters_resets_them(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A raw folder written before the per-shard counters existed still loads and truncates: the counters restart at
    0 with a log line instead of crashing or making the folder stale.
    """

    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    _failing_loader(monkeypatch, fail_at=None)
    download(cfg, "p", layout, rows_needed=40, shard_size=10)
    raw = layout.raw_dir("p")
    legacy = Manifest.load(raw)
    assert legacy is not None
    for shard in legacy.shards:  # a manifest from before the fields existed
        shard.skipped_malformed = shard.dropped_too_long = None
    legacy.skipped_malformed = legacy.dropped_too_long = 7
    legacy.save(raw)

    (raw / "data-00002.parquet").write_bytes(b"corrupt")
    reloaded = Manifest.load(raw)
    assert reloaded is not None and all(s.skipped_malformed is None for s in reloaded.shards)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert truncate_to_good_prefix(RawFolder(raw, reloaded))
    assert reloaded.rows_fetched == 20 and reloaded.skipped_malformed == 0 and reloaded.dropped_too_long == 0
    assert "written before the per-shard reject counters existed" in caplog.text
    assert inspect_raw(cfg, "p", layout).state == "current"  # never stale because of the missing fields


def test_download_instruct_filter_calls_the_loader_once_and_closes_it(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A filtered instruct download must not re-open the source per iteration (a remote JSON file would be
    re-streamed from byte 0 every time): one loader call, closed as soon as enough rows are kept.
    """

    from data_preparation.lib.sources import loaders as loaders_mod

    calls: list[tuple[int, int]] = []
    closed: list[bool] = []

    def loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
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
    assert m.rows() == 20 and m.rows_fetched == 77 and not m.exhausted
    assert calls == [(0, 2**62)] and closed == [True], "one call, closed after the 20th kept row"
    m2 = download(cfg, "s", layout, rows_needed=25, shard_size=10)
    assert m2.rows() == 25 and calls[1] == (77, 2**62)


# --- the token worker ----------------------------------------------------------------------------------------------


class _FakeCounter:
    """
    A `TokenCounter` stand-in (estimate counts) whose `truncate_many` calls go through `on_batch(call number)` first.
    """

    on_batch: Callable[[int], None] = staticmethod(lambda call: None)
    calls = 0

    def __init__(self, config: DatasetConfig, layout: DatasetLayout) -> None:
        pass

    def truncate_many(self, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
        type(self).calls += 1
        type(self).on_batch(type(self).calls)
        return [(text, estimate_tokens(text)) for text in texts]

    def count_many(self, texts: list[str]) -> list[int]:
        return [estimate_tokens(text) for text in texts]


def _counting_loader(monkeypatch: pytest.MonkeyPatch, on_row: Callable[[int], None], total: int = 100) -> list[bool]:
    """
    Stub the synthetic loader with one that yields `total` rows and calls on_row(rows yielded so far) before
    each; returns the list that records the loader's generator being closed.
    """

    from data_preparation.lib.sources import loaders as loaders_mod

    closed: list[bool] = []

    def loader(source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters) -> Any:
        try:
            for i in range(offset, min(offset + count, total)):
                on_row(i - offset + 1)
                yield {"text": f"row {i}"}
        finally:
            closed.append(True)

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", loader)
    return closed


def test_the_fetch_thread_runs_ahead_of_the_tokenizer(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    """
    The first tokenizer call blocks until the loader has yielded two more batches: only a fetch thread that runs
    ahead of the token worker gets there (in one thread the tokenizer would wait for rows that are never pulled).
    """

    monkeypatch.setattr(download_module, "TOKEN_BATCH", 5)
    tokenizer_may_go = threading.Event()

    def on_batch(call: int) -> None:
        if call == 1:
            assert tokenizer_may_go.wait(timeout=10), "the fetch thread did not run ahead of the tokenizer"

    def on_row(yielded: int) -> None:
        if yielded == 3 * download_module.TOKEN_BATCH:
            tokenizer_may_go.set()

    monkeypatch.setattr(_FakeCounter, "on_batch", staticmethod(on_batch))
    monkeypatch.setattr(_FakeCounter, "calls", 0)
    monkeypatch.setattr(download_module, "TokenCounter", _FakeCounter)
    closed = _counting_loader(monkeypatch, on_row, total=40)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    m = download(cfg, "p", layout, rows_needed=40, shard_size=10)
    assert tokenizer_may_go.is_set() and closed == [True]
    assert m.rows() == 40 and m.rows_fetched == 40 and [(s.rows, s.offset) for s in m.shards] == [(10, 10 * (i + 1)) for i in range(4)]
    assert [r["text"] for r in read_rows(layout.raw_dir("p"))] == [f"row {i}" for i in range(40)]  # in order, whatever the threads did


def test_a_failure_on_the_token_worker_ends_the_download_like_a_loader_failure(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader
) -> None:
    """
    The tokenizer raises on the fifth batch (rows 20-24, after two shards were published): the download raises
    that error on its own thread, the loader's stream is closed, the published shards stay and the partial one is
    discarded (the same outcome as the loader failing there), and the next call resumes at shard 2.
    """

    monkeypatch.setattr(download_module, "TOKEN_BATCH", 5)

    def on_batch(call: int) -> None:
        if call == 5:
            raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr(_FakeCounter, "on_batch", staticmethod(on_batch))
    monkeypatch.setattr(_FakeCounter, "calls", 0)
    monkeypatch.setattr(download_module, "TokenCounter", _FakeCounter)
    closed = _counting_loader(monkeypatch, lambda yielded: None, total=100)
    cfg = with_tokenizer(cfg_factory({"p": _synthetic()}))
    threads_before = threading.active_count()
    with pytest.raises(RuntimeError, match="tokenizer exploded"):
        download(cfg, "p", layout, rows_needed=100, shard_size=10)
    raw = layout.raw_dir("p")
    m = Manifest.load(raw)
    assert closed == [True] and threading.active_count() == threads_before, "the stream is closed and the worker joined"
    assert m is not None and [(s.rows, s.offset) for s in m.shards] == [(10, 10), (10, 20)] and m.rows_fetched == 20
    assert not list(raw.glob("*.tmp")) and [r["text"] for r in read_rows(raw)] == [f"row {i}" for i in range(20)]

    monkeypatch.setattr(_FakeCounter, "on_batch", staticmethod(lambda call: None))
    m2 = download(cfg, "p", layout, rows_needed=100, shard_size=10)
    assert m2.rows() == 100 and [r["text"] for r in read_rows(raw)] == [f"row {i}" for i in range(100)]


# --- the group pass keeps every language it decodes ----------------------------------------------------------------------


def test_download_github_code_group_resumes_every_folder_aligned_after_a_stop(
    hub: FakeHub, cfg_factory: CfgFactory, tmp_path: Path, read_rows: Reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A stop publishes the buffered rows of every folder (members and extras) as short shards, so each folder's
    offset is where it really stood; the next pass realigns every one of them and the end state equals an
    uninterrupted pass. Rows reach the writers in small batches so the stop lands mid-pass.
    """

    monkeypatch.setattr(download_module, "TOKEN_BATCH", 2)
    for prefix in "abcd":
        hub.add(f"data/{prefix}.parquet", _code_rows(prefix, 12))  # row groups of 2; Python, Java, Go in turns
    sources = {"py": _github("Python"), "rust": _github("Rust")}
    cfg = cfg_factory(sources)
    rows_needed = {"py": 2, "rust": 3}  # Rust never appears: the whole repo is read; Python turns passive early

    reference = DatasetLayout(tmp_path / "reference")
    prepare_tokenizer(cfg, reference)
    download_github_code_group(cfg, list(sources), reference, rows_needed=rows_needed, shard_size=4)
    names = ["py", "rust", "name_java", "name_go"]
    expected = _raw_state(reference, names, read_rows)
    assert [len(expected[name]["rows"]) for name in names] == [16, 0, 16, 16]

    stopped = DatasetLayout(tmp_path / "stopped")
    prepare_tokenizer(cfg, stopped)
    published = {"n": 0}

    def stop_after_first_shard() -> bool:
        published["n"] += 1
        return published["n"] >= 1

    with pytest.raises(BuildAborted):
        download_github_code_group(cfg, list(sources), stopped, rows_needed=rows_needed, shard_size=4, should_stop=stop_after_first_shard)
    partial = {name: Manifest.load(stopped.raw_dir(name)) for name in names}
    assert partial["py"] is not None and any(partial[name] is not None for name in ("name_java", "name_go"))
    for name in ("py", "name_java", "name_go"):
        manifest = partial[name]
        if manifest is None:  # no row of that language reached its writer before the stop: no manifest yet, next pass from 0
            assert not list(stopped.raw_dir(name).glob("*.parquet"))
            continue
        assert manifest.rows() == manifest.rows_fetched < 16  # the flushed short shard moved the offset to the last stored row
        assert [r["text"] for r in read_rows(stopped.raw_dir(name))] == [r["text"] for r in expected[name]["rows"][: manifest.rows()]]
    resumed = download_github_code_group(cfg, list(sources), stopped, rows_needed=rows_needed, shard_size=4)
    assert resumed["rust"].exhausted
    state = _raw_state(stopped, names, read_rows)
    for name in names:
        assert state[name]["rows"] == expected[name]["rows"] and state[name]["rows_fetched"] == expected[name]["rows_fetched"], name
    assert state["rust"]["exhausted"] and not state["py"]["exhausted"]


def test_download_github_code_group_skips_extras_it_cannot_store(
    hub: FakeHub, cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Reader, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A language whose derived name is a configured source outside the group, or whose folder is stale, is left
    alone (warned about, nothing deleted); the others are stored.
    """

    hub.add("data/a.parquet", _code_rows("a", 12))
    cfg = with_tokenizer(cfg_factory({"py": _github("Python"), "rust": _github("Rust"), "name_go": _synthetic()}))
    stale_dir = layout.raw_dir("name_java")
    stale_dir.mkdir(parents=True)
    Manifest(source="name_java", source_hash="0000000000000000", stage="raw").save(stale_dir)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        manifests = download_github_code_group(cfg, ["py", "rust"], layout, rows_needed={"py": 1, "rust": 1}, shard_size=4)
    assert set(manifests) == {"py", "rust"}
    assert "name_go is a configured source outside this github_code group" in caplog.text
    assert "not storing rows of Java" in caplog.text and "stale" in caplog.text
    stale = Manifest.load(stale_dir)
    assert stale is not None and stale.source_hash == "0000000000000000" and not list(stale_dir.glob("*.parquet"))
    assert not layout.raw_dir("name_go").exists()
    assert [r["text"] for r in read_rows(layout.raw_dir("py"))] == [f"a code {i}" for i in range(0, 12, 3)]
