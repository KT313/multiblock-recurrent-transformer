# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.sources: loaders honour offset/count/order against stubbed `datasets`, local
files, synthetic determinism, every converter/filter on hand-written rows, and that every name used by the shipped
dataset configs is registered. Offline, CPU, fast."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib import sources
from data_preparation.dataset_config import SourceConfig, SourceKind, load_dataset_config
from data_preparation.lib.sources import (
    CONVERTERS,
    FILTERS,
    LOADERS,
    Row,
    fields_converter,
    first_two_turns,
    get_converter,
    get_filter,
    get_loader,
    gsm8k_question_answer,
    instruction_input_output,
    sharegpt_conversations,
    sharegpt_quality,
    synthetic_row,
    write_synthetic_tokenizer,
)

REPO = Path(__file__).resolve().parents[3]
CONFIGS = [REPO / "config" / "datasets" / "crow_300m_final.yaml", REPO / "config" / "datasets" / "tiny.yaml"]


# --- stub `datasets` module -------------------------------------------------------------------------------------------


class FakeStream:
    """Minimal stand-in for an IterableDataset: iteration + `.skip()`; records how far it was consumed."""

    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows
        self.consumed = 0

    def skip(self, n: int) -> FakeStream:
        child = FakeStream(self.rows[n:])
        return child

    def __iter__(self) -> Iterator[Row]:
        for row in self.rows:
            self.consumed += 1
            yield row


class FakeDatasets:
    """Fake `datasets.load_dataset`: serves `rows` in order, applies `split` slicing, records the call kwargs."""

    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []
        self.last_stream: FakeStream | None = None

    def load_dataset(self, path: str, **kwargs: Any) -> Any:
        self.calls.append({"path": path, **kwargs})
        split = kwargs.get("split", "train")
        if kwargs.get("streaming"):
            assert "[" not in split
            self.last_stream = FakeStream(self.rows)
            return self.last_stream
        if "[" in split:
            bounds = split[split.index("[") + 1 : -1]
            a, b = (int(x) for x in bounds.split(":"))
            return self.rows[a:b]
        return list(self.rows)


@pytest.fixture
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> FakeDatasets:
    rows = [{"id": i, "text": f"doc {i}", "language": "Python" if i % 3 == 0 else "Java"} for i in range(20)]
    fake = FakeDatasets(rows)
    module = types.ModuleType("datasets")
    module.load_dataset = fake.load_dataset  # type: ignore[attr-defined]  # fake module attribute
    monkeypatch.setitem(sys.modules, "datasets", module)
    return fake


def _src(**kwargs: Any) -> SourceConfig:
    defaults: dict[str, Any] = {"kind": "pretrain", "loader": "hf_split", "hf_id": "org/name", "revision": "abc"}
    defaults.update(kwargs)
    return SourceConfig(**defaults)


# --- hf_split ---------------------------------------------------------------------------------------------------------


def test_hf_split_slices_and_passes_kwargs(fake_datasets: FakeDatasets) -> None:
    source = _src(load_kwargs={"name": "cfg"}, split="validation")
    rows = list(LOADERS["hf_split"](source, 5, 3, token="tok"))
    assert [r["id"] for r in rows] == [5, 6, 7]
    call = fake_datasets.calls[0]
    assert call == {
        "path": "org/name",
        "split": "validation[5:8]",
        "revision": "abc",
        "token": "tok",
        "name": "cfg",
    }


def test_hf_split_count_zero_does_not_load(fake_datasets: FakeDatasets) -> None:
    assert list(LOADERS["hf_split"](_src(), 3, 0)) == []
    assert fake_datasets.calls == []


def test_hf_split_short_source(fake_datasets: FakeDatasets) -> None:
    assert [r["id"] for r in LOADERS["hf_split"](_src(), 18, 10)] == [18, 19]


def test_negative_offset_rejected(fake_datasets: FakeDatasets) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        list(LOADERS["hf_split"](_src(), -1, 2))


# --- hf_stream --------------------------------------------------------------------------------------------------------


def test_hf_stream_skips_and_takes(fake_datasets: FakeDatasets) -> None:
    source = _src(loader="hf_stream", load_kwargs={"data_files": "x.json"})
    rows = list(LOADERS["hf_stream"](source, 4, 3))
    assert [r["id"] for r in rows] == [4, 5, 6]
    call = fake_datasets.calls[0]
    assert call["streaming"] is True and call["split"] == "train" and call["data_files"] == "x.json"
    assert call["revision"] == "abc" and call["token"] is None


def test_hf_stream_stops_pulling_after_count(fake_datasets: FakeDatasets) -> None:
    rows = list(LOADERS["hf_stream"](_src(loader="hf_stream"), 0, 2))
    assert [r["id"] for r in rows] == [0, 1]
    assert fake_datasets.last_stream is not None and fake_datasets.last_stream.consumed == 2


def test_hf_stream_count_zero_does_not_load(fake_datasets: FakeDatasets) -> None:
    assert list(LOADERS["hf_stream"](_src(loader="hf_stream"), 0, 0)) == []
    assert fake_datasets.calls == []


# --- github_code (the loader itself is tested in test_hf_files.py) ------------------------------------------------


# --- local ------------------------------------------------------------------------------------------------------------


def test_local_reads_parquet_and_jsonl_in_sorted_order(tmp_path: Path) -> None:
    pq.write_table(pa.table({"text": ["b0", "b1"]}), tmp_path / "b.parquet")
    (tmp_path / "a.jsonl").write_text(json.dumps({"text": "a0"}) + "\n\n" + json.dumps({"text": "a1"}) + "\n")
    (tmp_path / "c.txt").write_text("ignored")
    source = _src(loader="local", path=str(tmp_path), hf_id=None, revision=None)
    assert [r["text"] for r in LOADERS["local"](source, 0, 10)] == ["a0", "a1", "b0", "b1"]
    assert [r["text"] for r in LOADERS["local"](source, 1, 2)] == ["a1", "b0"]
    assert list(LOADERS["local"](source, 4, 2)) == []
    assert list(LOADERS["local"](source, 0, 0)) == []


def test_local_projects_jsonl_and_parquet_to_the_requested_columns(tmp_path: Path) -> None:
    """`local` goes through the shared reading contract: `columns` projects both formats (a `.jsonl` row's surplus
    column is dropped, a requested column a row lacks stays absent), None keeps every column."""
    pq.write_table(pa.table({"text": ["b0"], "extra": [1]}), tmp_path / "b.parquet")
    (tmp_path / "a.jsonl").write_text(json.dumps({"text": "a0", "extra": 0}) + "\n")
    source = _src(loader="local", path=str(tmp_path), hf_id=None, revision=None)
    assert list(LOADERS["local"](source, 0, 10, columns=["text"])) == [{"text": "a0"}, {"text": "b0"}]
    assert list(LOADERS["local"](source, 0, 1, columns=["text", "missing"])) == [{"text": "a0"}]  # jsonl: absent stays absent
    assert list(LOADERS["local"](source, 0, 10)) == [{"text": "a0", "extra": 0}, {"text": "b0", "extra": 1}]


def test_local_missing_directory(tmp_path: Path) -> None:
    source = _src(loader="local", path=str(tmp_path / "nope"), hf_id=None, revision=None)
    with pytest.raises(FileNotFoundError):
        list(LOADERS["local"](source, 0, 1))


# --- synthetic --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["pretrain", "instruct"])
def test_synthetic_offset_property_and_determinism(kind: SourceKind) -> None:
    source = SourceConfig(kind=kind, loader="synthetic", seed=7)
    load = LOADERS["synthetic"]
    full = list(load(source, 0, 8))
    assert list(load(source, 5, 3)) == full[5:8]
    assert list(load(source, 0, 8)) == full
    assert len(full) == 8 and len({json.dumps(r) for r in full}) == 8
    other_seed = SourceConfig(kind=kind, loader="synthetic", seed=8)
    assert list(load(other_seed, 0, 8)) != full


def test_synthetic_row_shapes() -> None:
    doc = synthetic_row("pretrain", 0, 3)
    assert set(doc) == {"text"}
    words = doc["text"].split()
    assert 64 <= len(words) <= 384 and all(w.startswith("tok_") and int(w[4:]) < 256 for w in words)
    ins = synthetic_row("instruct", 0, 3)
    assert set(ins) == {"instruction", "input", "output"} and ins["input"] == ""
    assert ins["instruction"].startswith("tok_") and ins["output"].startswith("tok_")


def test_write_synthetic_tokenizer(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    write_synthetic_tokenizer(tmp_path / "tok")
    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path / "tok"))
    assert tokenizer.convert_tokens_to_ids(["<pad>", "<bos>", "<eos>", "tok_0", "tok_255"]) == [0, 1, 2, 3, 258]
    assert tokenizer.encode("tok_1 tok_2", add_special_tokens=False) == [4, 5]
    assert len(tokenizer) == sources.VOCAB_SIZE == 259


# --- registries -------------------------------------------------------------------------------------------------------


def test_registry_names_and_unknown() -> None:
    assert set(LOADERS) == {"hf_files", "hf_split", "hf_stream", "github_code", "local", "synthetic"}
    assert get_loader("local") is sources.load_local
    with pytest.raises(ValueError, match="hf_split"):
        get_loader("nope")
    with pytest.raises(ValueError, match="sharegpt_quality"):
        get_filter("nope")
    with pytest.raises(ValueError, match="gsm8k_question_answer"):
        get_converter(_src(kind="instruct", converter="nope"))


@pytest.mark.parametrize("path", CONFIGS)
def test_shipped_configs_reference_registered_names(path: Path) -> None:
    config = load_dataset_config(path)
    for name, source in config.sources.items():
        assert source.loader in LOADERS, name
        if source.converter is not None:
            assert source.converter in CONVERTERS, name
        if source.filter is not None:
            assert source.filter in FILTERS, name
        converter = get_converter(source)
        if source.kind == "instruct" and source.loader != "synthetic":
            assert converter is not None, name


# --- converters -------------------------------------------------------------------------------------------------------


def test_gsm8k_question_answer() -> None:
    row = gsm8k_question_answer({"question": "1+1?", "answer": "2", "extra": 1})
    assert row == {"text": "Question: 1+1?\n\nAnswer: 2"}
    with pytest.raises(ValueError, match=r"\['answer'\].*'question'"):
        gsm8k_question_answer({"question": "x"})


def test_sharegpt_conversations() -> None:
    row = {
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": 42},
        ]
    }
    assert sharegpt_conversations(row) == {"instruction": "hi", "input": "sys", "output": "42"}
    assert sharegpt_conversations({"conversations": [{"from": "human", "value": "q"}]}) == {
        "instruction": "q",
        "input": "",
        "output": "",
    }
    with pytest.raises(ValueError, match="conversations"):
        sharegpt_conversations({"text": "x"})
    with pytest.raises(ValueError, match="from"):
        sharegpt_conversations({"conversations": [{"value": "no role"}]})


def test_first_two_turns() -> None:
    row = {"conversations": [{"value": "ask"}, {"value": "answer"}, {"value": "ignored"}]}
    assert first_two_turns(row) == {"instruction": "ask", "input": "", "output": "answer"}
    with pytest.raises(ValueError, match="two turns"):
        first_two_turns({"conversations": [{"value": "only"}]})
    with pytest.raises(ValueError, match="conversations"):
        first_two_turns({"conv": []})


def test_instruction_input_output() -> None:
    assert instruction_input_output({"instruction": "a", "output": 1}) == {"instruction": "a", "input": "", "output": "1"}
    assert instruction_input_output({"instruction": "a", "input": None, "output": "b"})["input"] == ""
    assert instruction_input_output({"instruction": "a", "input": "i", "output": "b", "z": 0})["input"] == "i"
    with pytest.raises(ValueError, match=r"\['output'\]"):
        instruction_input_output({"instruction": "a"})


def test_fields_converter() -> None:
    convert = fields_converter({"instruction": "inputs", "output": "targets"})
    assert convert({"inputs": "q", "targets": 3}) == {"instruction": "q", "input": "", "output": "3"}
    with_input = fields_converter({"instruction": "q", "input": "ctx", "output": "a"})
    assert with_input({"q": "x", "ctx": "c", "a": "y"}) == {"instruction": "x", "input": "c", "output": "y"}
    assert with_input({"q": "x", "a": "y"})["input"] == ""
    with pytest.raises(ValueError, match=r"\['targets'\]"):
        convert({"inputs": "q"})
    with pytest.raises(ValueError, match="at least"):
        fields_converter({"instruction": "a"})
    with pytest.raises(ValueError, match="unknown keys"):
        fields_converter({"instruction": "a", "output": "b", "bogus": "c"})


def test_get_converter_resolution() -> None:
    assert get_converter(_src()) is None
    assert get_converter(_src(converter="gsm8k_question_answer")) is gsm8k_question_answer
    by_fields = get_converter(_src(kind="instruct", fields={"instruction": "i", "output": "o"}, converter="first_two_turns"))
    assert by_fields is not None
    assert by_fields({"i": "a", "o": "b"}) == {"instruction": "a", "input": "", "output": "b"}


# --- filters ----------------------------------------------------------------------------------------------------------


def _sharegpt(human: str, gpt: str, first: str = "human", second: str = "gpt") -> Row:
    return {"conversations": [{"from": first, "value": human}, {"from": second, "value": gpt}]}


def test_converters_turn_null_values_into_empty_fields_not_the_string_none() -> None:
    """A null turn / column must become an empty field (dropped at build), never the text "None" trained on."""
    conversation = {"conversations": [{"from": "human", "value": None}, {"from": "gpt", "value": "x"}]}
    assert sharegpt_conversations(conversation) == {"instruction": "", "input": "", "output": "x"}
    assert first_two_turns({"conversations": [{"value": None}, {"value": "y"}]}) == {"instruction": "", "input": "", "output": "y"}
    assert gsm8k_question_answer({"question": None, "answer": "a"}) == {"text": "Question: \n\nAnswer: a"}


def test_sharegpt_quality() -> None:
    ok_h, ok_g = "h" * 100, "g" * 100
    assert sharegpt_quality(_sharegpt(ok_h, ok_g))
    assert FILTERS["sharegpt_quality"] is get_filter("sharegpt_quality")
    assert not sharegpt_quality(_sharegpt("short", ok_g))
    assert not sharegpt_quality(_sharegpt(ok_h, "g" * 2001))
    assert not sharegpt_quality(_sharegpt(ok_h, ok_g, first="gpt", second="human"))
    assert not sharegpt_quality(_sharegpt(ok_h, ok_g + "```python\nprint(1)"))
    assert not sharegpt_quality({"conversations": [{"from": "human", "value": ok_h}]})
    assert not sharegpt_quality({"conversations": None})
    assert not sharegpt_quality({})


# --- misc -------------------------------------------------------------------------------------------------------------



def test_hub_load_kwargs_routes_script_repos_through_generic_builder() -> None:
    from data_preparation.lib.sources.loaders import hub_load_kwargs

    plain = _src(load_kwargs={"name": "cfg"})
    assert hub_load_kwargs(plain, "tok", split="train") == {
        "token": "tok", "split": "train", "path": "org/name", "revision": "abc", "name": "cfg",
    }
    scripted = _src(load_kwargs={"builder": "json", "data_files": "sub/*.jsonl.zst", "encoding": "utf-8"})
    assert hub_load_kwargs(scripted, None, streaming=True) == {
        "token": None, "streaming": True, "path": "json",
        "data_files": "hf://datasets/org/name@abc/sub/*.jsonl.zst", "encoding": "utf-8",
    }
    unpinned = _src(revision=None, load_kwargs={"builder": "parquet", "data_files": "data/*.parquet"})
    assert hub_load_kwargs(unpinned, None)["data_files"] == "hf://datasets/org/name/data/*.parquet"
    with pytest.raises(ValueError, match="requires load_kwargs.data_files"):
        hub_load_kwargs(_src(load_kwargs={"builder": "json"}), None)


def test_fields_converter_turns_null_values_into_empty_strings() -> None:
    from data_preparation.lib.sources.converters import fields_converter

    convert = fields_converter({"instruction": "q", "input": "ctx", "output": "a"})
    assert convert({"q": None, "ctx": None, "a": "x"}) == {"instruction": "", "input": "", "output": "x"}
