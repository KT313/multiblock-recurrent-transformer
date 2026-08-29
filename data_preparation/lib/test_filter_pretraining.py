# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.filter_pretraining: text-field detection, batch filtering, and the CLI end to end."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typing import cast

from data_preparation.lib import filter_pretraining as fp
from data_preparation.lib.common import list_parquet_files


@pytest.mark.parametrize("field", ["text", "TEXT", "content", "code"])
def test_detect_text_field(field: str) -> None:
    fields: list[tuple[str, pa.DataType]] = [("id", pa.int64()), (field, pa.string())]
    schema = pa.schema(fields)
    assert fp.detect_text_field(schema, "src") == field


def test_detect_text_field_prefers_first_candidate() -> None:
    schema = pa.schema([("code", pa.string()), ("text", pa.string())])
    assert fp.detect_text_field(schema, "src") == "text"


def test_detect_text_field_missing_raises() -> None:
    with pytest.raises(ValueError, match="No text field found in src"):
        fp.detect_text_field(pa.schema([("body", pa.string())]), "src")


def test_preprocess_batch_filters_truncates_and_counts() -> None:
    batch = pa.RecordBatch.from_pydict(
        {"content": ["short", None, "x" * 50, "y" * 120, "z" * 51], "meta": [1, 2, 3, 4, 5]}
    )
    out, stats = fp.preprocess_batch(batch, "content", "mysrc", min_chars=50, max_chars=100)
    assert out.schema == fp.OUTPUT_SCHEMA
    assert stats == {
        "input_samples": 5,
        "removed_too_short": 1,
        "removed_invalid": 1,
        "truncated": 1,
        "output_samples": 3,
    }
    rows = out.to_pylist()
    assert [len(r["text"]) for r in rows] == [50, 100, 51]
    assert [r["original_length"] for r in rows] == [50, 120, 51]
    assert {r["source"] for r in rows} == {"mysrc"}


def test_preprocess_batch_length_boundaries() -> None:
    # min_chars is inclusive (>=), max_chars is inclusive too: a text of exactly max_chars is not truncated
    batch = pa.RecordBatch.from_pydict({"text": ["a" * 49, "b" * 50, "c" * 100, "d" * 101]})
    out, stats = fp.preprocess_batch(batch, "text", "s", min_chars=50, max_chars=100)
    assert stats == {"input_samples": 4, "removed_too_short": 1, "removed_invalid": 0, "truncated": 1,
                     "output_samples": 3}  # fmt: skip
    rows = out.to_pylist()
    assert [(len(r["text"]), r["original_length"]) for r in rows] == [(50, 50), (100, 100), (100, 101)]
    assert rows[2]["text"] == "d" * 100


def test_preprocess_batch_all_invalid_or_short_returns_empty_with_schema() -> None:
    batch = pa.RecordBatch.from_pydict({"text": [None, None]})
    out, stats = fp.preprocess_batch(batch, "text", "s", 50, 100)
    assert len(out) == 0 and out.schema == fp.OUTPUT_SCHEMA
    assert stats["removed_invalid"] == 2 and stats["output_samples"] == 0

    batch = pa.RecordBatch.from_pydict({"text": ["tiny", "tiny2"]})
    out, stats = fp.preprocess_batch(batch, "text", "s", 50, 100)
    assert len(out) == 0 and stats["removed_too_short"] == 2 and stats["output_samples"] == 0


def test_preprocess_batch_uses_unicode_length() -> None:
    batch = pa.RecordBatch.from_pydict({"text": ["é" * 10]})
    out, stats = fp.preprocess_batch(batch, "text", "s", min_chars=10, max_chars=5)
    assert stats["output_samples"] == 1 and stats["truncated"] == 1
    assert out.to_pylist()[0]["text"] == "é" * 5 and out.to_pylist()[0]["original_length"] == 10


# --- CLI end to end -------------------------------------------------------------------------------------------------


def _write_raw(raw_dir: Path, name: str, field: str, texts: list[str | None], shards: int = 1) -> None:
    d = raw_dir / name
    d.mkdir(parents=True)
    per = -(-len(texts) // shards)
    for i in range(shards):
        chunk = texts[i * per : (i + 1) * per]
        pq.write_table(pa.table({field: chunk, "id": list(range(len(chunk)))}), d / f"shard-{i:05d}.parquet")


@pytest.fixture
def raw_tree(tmp_path: Path) -> Path:
    raw = tmp_path / "pretraining" / "raw"
    texts_a: list[str | None] = [*(f"doc {i} " + "a" * 60 for i in range(7)), "short", None, "b" * 300]
    _write_raw(raw, "alpha", "text", texts_a, shards=2)
    _write_raw(raw, "beta_code", "code", ["c" * 80, "d" * 80])
    (raw / "empty_src").mkdir()  # no shards -> counted as failed
    _write_raw(raw, "gamma_nofield", "body", ["e" * 80])  # no text field -> error, counted as failed
    return tmp_path


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["filter_pretraining", *argv])
    fp.main()


def test_main_end_to_end(raw_tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _run_main(
        monkeypatch, ["--dataset_dir", str(raw_tree), "--min_chars", "50", "--max_chars", "100", "--shard_size", "4"]
    )
    out_root = raw_tree / "pretraining" / "filtered"
    stats = json.loads((out_root / "preprocessing_stats.json").read_text())
    assert set(stats) == {"alpha", "beta_code"}

    alpha = stats["alpha"]
    assert alpha["input_samples"] == 10 and alpha["output_samples"] == 8
    assert alpha["removed_too_short"] == 1 and alpha["removed_invalid"] == 1 and alpha["truncated"] == 1
    assert alpha["pass_rate"] == pytest.approx(0.8)
    assert alpha["avg_original_length"] == int((7 * 66 + 300) / 8)
    assert alpha["avg_output_length"] == int((7 * 66 + 100) / 8)
    assert set(alpha) == {
        "input_samples", "output_samples", "removed_too_short", "removed_invalid", "truncated",
        "pass_rate", "avg_original_length", "avg_output_length",
    }  # fmt: skip

    files = list_parquet_files(out_root / "alpha", "data")
    assert [f.name for f in files] == ["data-00000.parquet", "data-00001.parquet"]
    table = pa.concat_tables([pq.read_table(f) for f in files])
    assert table.schema == fp.OUTPUT_SCHEMA
    assert table.num_rows == 8 and set(table["source"].to_pylist()) == {"alpha"}
    assert max(len(t) for t in cast(list[str], table["text"].to_pylist())) == 100

    beta = pq.read_table(list_parquet_files(out_root / "beta_code", "data")[0])
    assert beta.column_names == ["text", "source", "original_length"] and beta.num_rows == 2

    assert not (out_root / "empty_src").exists()
    assert not list_parquet_files(out_root / "gamma_nofield", "data")
    printed = capsys.readouterr().out
    assert "Failed: ['empty_src', 'gamma_nofield']" in printed
    assert "Successful: 2 / 4" in printed


def test_main_dataset_selection_by_glob(raw_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_main(monkeypatch, ["--dataset_dir", str(raw_tree), "--datasets", "beta*"])
    out_root = raw_tree / "pretraining" / "filtered"
    stats = json.loads((out_root / "preprocessing_stats.json").read_text())
    assert list(stats) == ["beta_code"]
    assert not (out_root / "alpha").exists()


def test_main_no_sources_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="No dataset directories found"):
        _run_main(monkeypatch, ["--dataset_dir", str(tmp_path)])


def test_process_dataset_direct(raw_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = raw_tree / "pretraining" / "raw"
    out = raw_tree / "out"
    assert fp.process_dataset("empty_src", raw, out, 50, 100, 10, 10) is None
    assert "No parquet files found" in capsys.readouterr().out
    with pytest.raises(ValueError, match="No text field found"):
        fp.process_dataset("gamma_nofield", raw, out, 50, 100, 10, 10)
    stats = fp.process_dataset("beta_code", raw, out, 50, 100, 10, 10)
    assert stats is not None
    assert stats["input_samples"] == 2 and stats["output_samples"] == 2 and stats["pass_rate"] == 1.0
    assert stats["avg_original_length"] == 80 and stats["avg_output_length"] == 80
    assert len(list_parquet_files(out / "beta_code", "data")) == 1
    # batch_size smaller than a shard still yields one merged output shard
    stats = fp.process_dataset("alpha", raw, raw_tree / "out2", 50, 100, 1, 100)
    assert stats is not None and stats["output_samples"] == 8
    assert len(list_parquet_files(raw_tree / "out2" / "alpha", "data")) == 1


def test_parser_defaults() -> None:
    args = fp.build_parser().parse_args([])
    assert (args.min_chars, args.max_chars, args.batch_size, args.shard_size, args.datasets) == (
        50, 20000, 100000, 100000, None,
    )  # fmt: skip
