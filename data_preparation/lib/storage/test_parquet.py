# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.storage.parquet: HF cache setup, the dedup key, token estimate, parquet shard I/O."""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.storage.parquet import (
    ShardWriter,
    configure_hf_cache,
    estimate_tokens,
    list_parquet_files,
    shard_index,
    text_hash64,
)


def _read_all(out_dir: Path) -> list[pa.Table]:
    return [pq.read_table(f) for f in list_parquet_files(out_dir)]


# --- CLI / environment ----------------------------------------------------------------------------------------------



def test_configure_hf_cache_none_leaves_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        monkeypatch.delenv(var, raising=False)
    configure_hf_cache(None)
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        assert var not in os.environ


def test_configure_hf_cache_sets_all_vars_and_creates_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv *before* the call records the original value (or its absence) so monkeypatch restores it; a
    # delenv afterwards would "restore" the value the test itself set and leak the temp cache path into the
    # rest of the session, and delenv(raising=False) on an absent variable records nothing at all
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        monkeypatch.setenv(var, "placeholder")
    cache = tmp_path / "hf_cache"
    configure_hf_cache(cache)
    assert cache.is_dir()
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        assert os.environ[var] == str(cache.resolve())



# --- pure helpers ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"), [("", 0), ("abc", 0), ("abcd", 1), ("a" * 4000, 1000), ("a" * 4003, 1000)]
)
def test_estimate_tokens_is_chars_div_4(text: str, expected: object) -> None:
    assert estimate_tokens(text) == expected


def test_list_parquet_files_sorted_any_name(tmp_path: Path) -> None:
    assert list_parquet_files(tmp_path / "missing") == []
    for name in ("data-00002.parquet", "data-00000.parquet", "other.parquet", "data-x.txt", "MANIFEST.json"):
        (tmp_path / name).write_bytes(b"")
    names = [p.name for p in list_parquet_files(tmp_path)]
    assert names == ["data-00000.parquet", "data-00002.parquet", "other.parquet"]


def test_shard_index() -> None:
    assert shard_index(Path("data-00007.parquet")) == 7
    assert shard_index(Path("/x/data-123456.parquet")) == 123456
    assert shard_index(Path("other.parquet")) is None
    assert shard_index(Path("data-7.parquet")) is None


@pytest.mark.parametrize(
    ("text", "normalized", "raw"),
    [
        ("hello world", 6824707963431612112, 6824707963431612112),
        ("  Hello\t World\n\nfoo ", 8471811785197293890, -4212963777905507985),
        ("héllo wörld", -1365678327145243118, -1365678327145243118),
        ("a\ud800b", 1765116674205471180, 1765116674205471180),  # a lone surrogate is dropped, not an error
        ("", -3162216497309240828, -3162216497309240828),
    ],
)
def test_text_hash64_pins_the_stored_keys(text: str, normalized: int, raw: int) -> None:
    """The `hash` column of every processed shard on disk holds these values: the function must never change them."""
    assert text_hash64(text) == normalized and text_hash64(text, normalize=False) == raw


def test_text_hash64_normalizes_case_and_whitespace() -> None:
    assert text_hash64("Hello  World") == text_hash64("hello world") == text_hash64("\nHELLO\tworld\n")
    assert text_hash64("Hello  World", normalize=False) != text_hash64("hello world", normalize=False)
    assert text_hash64("hello world") != text_hash64("hello worlds")
    assert -(2**63) <= text_hash64("x") < 2**63
    pa.array([text_hash64("x")], type=pa.int64())  # fits the parquet column type


# --- parquet shard writer -------------------------------------------------------------------------------------------


def _write_rows(rows: Iterable[dict[str, Any]], out_dir: Path, shard_size: int, *, start_shard: int = 0) -> list[str]:
    """Write dict rows as shards through `ShardWriter`; the names of the published shards, in order."""
    published: list[Path] = []
    with ShardWriter(out_dir, shard_size, start_shard=start_shard, on_shard=published.append) as writer:
        for row in rows:
            writer.add(row)
    return [path.name for path in published]


def test_shard_writer_writes_shards_of_shard_size_and_appends(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _write_rows(({"text": f"t{i}", "n": i} for i in range(12)), out, shard_size=5) == [f"data-{i:05d}.parquet" for i in range(3)]
    tables = _read_all(out)
    assert [t.num_rows for t in tables] == [5, 5, 2]
    merged = pa.concat_tables(tables)
    assert merged.column_names == ["text", "n"] and merged["n"].to_pylist() == list(range(12))
    assert _write_rows([{"text": "x", "n": 99}], out, 5, start_shard=3) == ["data-00003.parquet"]
    assert [p.name for p in list_parquet_files(out)] == [f"data-{i:05d}.parquet" for i in range(4)]
    assert _write_rows([], out, 5, start_shard=1) == []  # nothing written: shards >= 1 are still cleared (append mode replaces them)
    assert [p.name for p in list_parquet_files(out)] == ["data-00000.parquet"]


def test_shard_writer_empty_input_writes_nothing_but_creates_dir(tmp_path: Path) -> None:
    assert _write_rows([], tmp_path / "out", shard_size=5) == [] and (tmp_path / "out").is_dir()
    assert list_parquet_files(tmp_path / "out") == []


def test_shard_writer_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_size"):
        ShardWriter(tmp_path / "out", shard_size=0, on_shard=lambda path: None)
    with pytest.raises(ValueError, match="start_shard"):
        ShardWriter(tmp_path / "out", shard_size=1, start_shard=-1, on_shard=lambda path: None)


def test_shard_writer_publishes_each_shard_and_keeps_them_on_failure(tmp_path: Path) -> None:
    published: list[str] = []
    out = tmp_path / "out"
    _write_rows(({"n": i} for i in range(7)), out, shard_size=2)  # 4 shards; the writer appends at 2
    (out / "data-00003.parquet.tmp").write_bytes(b"leftover")

    def on_shard(path: Path) -> None:
        published.append(path.name)
        assert path.is_file() and 1 <= pq.read_table(path).num_rows <= 2

    with pytest.raises(RuntimeError, match="boom"), ShardWriter(out, shard_size=2, start_shard=2, on_shard=on_shard) as writer:
        for i in range(5):
            writer.add({"m": i})
            if i == 3:
                assert published == ["data-00002.parquet", "data-00003.parquet"], "each full shard published at once"
        raise RuntimeError("boom")  # the buffered 5th row is discarded, the published shards stay
    assert [p.name for p in list_parquet_files(out)] == [f"data-{i:05d}.parquet" for i in range(4)]
    assert [t.to_pylist() for t in _read_all(out)][2:] == [[{"m": 0}, {"m": 1}], [{"m": 2}, {"m": 3}]]
    assert not list(out.glob("*.tmp"))

    with ShardWriter(out, shard_size=2, start_shard=2, on_shard=on_shard) as writer:  # stale shards >= 2 cleared
        writer.add({"m": 9})
    assert [p.name for p in list_parquet_files(out)] == [f"data-{i:05d}.parquet" for i in range(3)]
    assert _read_all(out)[2].to_pylist() == [{"m": 9}] and published[-1] == "data-00002.parquet"
