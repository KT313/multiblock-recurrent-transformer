# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.storage.parquet: HF cache setup, hashing, token estimate, parquet shard I/O."""

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.storage.parquet import (
    configure_hf_cache,
    estimate_tokens,
    list_parquet_files,
    md5_hex,
    normalized_hash,
    normalized_text,
    shard_index,
    write_dict_rows,
    write_parquet_shards,
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


def test_md5_hex_matches_hashlib() -> None:
    assert md5_hex("hello") == hashlib.md5(b"hello").hexdigest()
    assert md5_hex("hello") != md5_hex("hello!")
    assert len(md5_hex("")) == 32


def test_md5_hex_unicode_is_stable() -> None:
    assert md5_hex("héllo wörld") == hashlib.md5("héllo wörld".encode()).hexdigest()
    # lone surrogates (possible in scraped text) are dropped instead of raising
    assert md5_hex("a\ud800b") == hashlib.md5(b"ab").hexdigest()


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


def test_normalized_text_and_hash() -> None:
    assert normalized_text("  Hello\t World\n\nfoo ") == "hello world foo"
    variants = ["Hello World", "hello   world", "\nHELLO\tworld\n", " hello world "]
    assert len({normalized_hash(v) for v in variants}) == 1
    assert normalized_hash("hello world") == md5_hex("hello world")
    assert normalized_hash("hello world") != normalized_hash("hello worlds")



# --- parquet shard writer -------------------------------------------------------------------------------------------


def _batches(sizes: list[int], start: int = 0) -> Iterator[pa.RecordBatch]:
    i = start
    for n in sizes:
        yield pa.RecordBatch.from_pydict({"x": list(range(i, i + n))})
        i += n


def test_write_parquet_shards_rechunks_to_exact_shard_size(tmp_path: Path) -> None:
    n = write_parquet_shards(_batches([7, 1, 9, 3]), tmp_path / "out", shard_size=5)
    tables = _read_all(tmp_path / "out")
    assert n == 4 and len(tables) == 4
    assert [t.num_rows for t in tables] == [5, 5, 5, 5]
    assert [p.name for p in list_parquet_files(tmp_path / "out")] == [f"data-{i:05d}.parquet" for i in range(4)]
    assert pa.concat_tables(tables)["x"].to_pylist() == list(range(20))


def test_write_parquet_shards_last_shard_is_remainder(tmp_path: Path) -> None:
    n = write_parquet_shards(_batches([4, 4, 4]), tmp_path / "out", shard_size=5)
    assert n == 3
    assert [t.num_rows for t in _read_all(tmp_path / "out")] == [5, 5, 2]


def test_write_parquet_shards_single_big_batch_split(tmp_path: Path) -> None:
    n = write_parquet_shards(_batches([23]), tmp_path / "out", shard_size=10)
    assert n == 3
    assert [t.num_rows for t in _read_all(tmp_path / "out")] == [10, 10, 3]
    assert pa.concat_tables(_read_all(tmp_path / "out"))["x"].to_pylist() == list(range(23))


def test_write_parquet_shards_accepts_tables_and_skips_empty(tmp_path: Path) -> None:
    items: list[pa.RecordBatch | pa.Table] = [
        pa.table({"x": [1, 2]}),
        pa.RecordBatch.from_pydict({"x": []}),
        pa.table({"x": [3]}),
    ]
    n = write_parquet_shards(items, tmp_path / "out", shard_size=100)
    assert n == 1
    assert _read_all(tmp_path / "out")[0]["x"].to_pylist() == [1, 2, 3]


def test_write_parquet_shards_empty_input_writes_nothing_but_creates_dir(tmp_path: Path) -> None:
    n = write_parquet_shards([], tmp_path / "out", shard_size=5)
    assert n == 0 and (tmp_path / "out").is_dir()
    assert list_parquet_files(tmp_path / "out") == []


def test_write_parquet_shards_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_size"):
        write_parquet_shards([], tmp_path / "out", shard_size=0)
    with pytest.raises(ValueError, match="start_shard"):
        write_parquet_shards([], tmp_path / "out", shard_size=1, start_shard=-1)


def test_write_parquet_shards_failure_leaves_out_dir_untouched(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_parquet_shards(_batches([3]), out, shard_size=5)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def failing() -> Iterator[pa.RecordBatch]:
        yield from _batches([6, 6])  # enough for two complete shards in the temp dir
        raise RuntimeError("stream broke")

    with pytest.raises(RuntimeError, match="stream broke"):
        write_parquet_shards(failing(), out, shard_size=5)
    assert not (tmp_path / "out.tmp").exists()
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not list(tmp_path.glob("**/*.parquet.tmp"))
    # a fresh directory is not created on failure either
    with pytest.raises(RuntimeError):
        write_parquet_shards(failing(), tmp_path / "fresh", shard_size=5)
    assert not (tmp_path / "fresh").exists() and not (tmp_path / "fresh.tmp").exists()


def test_write_parquet_shards_overwrite_clears_stale_shards(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert write_parquet_shards(_batches([20]), out, shard_size=5) == 4
    (out / "MANIFEST.json").write_text("{}")
    assert write_parquet_shards(_batches([7]), out, shard_size=5) == 2
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST.json", "data-00000.parquet", "data-00001.parquet"]
    assert pa.concat_tables(_read_all(out))["x"].to_pylist() == list(range(7))


def test_write_parquet_shards_append_keeps_lower_and_removes_higher(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert write_parquet_shards(_batches([20]), out, shard_size=5) == 4  # data-00000 .. data-00003
    n = write_parquet_shards(_batches([7], start=100), out, shard_size=5, start_shard=2)
    assert n == 2
    assert [p.name for p in list_parquet_files(out)] == [f"data-{i:05d}.parquet" for i in range(4)]
    assert pa.concat_tables(_read_all(out))["x"].to_pylist() == list(range(10)) + list(range(100, 107))
    # pure append: start at the current count, nothing is removed
    assert write_parquet_shards(_batches([2], start=500), out, shard_size=5, start_shard=4) == 1
    assert [p.name for p in list_parquet_files(out)] == [f"data-{i:05d}.parquet" for i in range(5)]
    assert _read_all(out)[4]["x"].to_pylist() == [500, 501]


def test_write_parquet_shards_clears_leftover_tmp_dir(tmp_path: Path) -> None:
    leftover = tmp_path / "out.tmp"
    leftover.mkdir()
    (leftover / "data-00009.parquet").write_bytes(b"garbage")
    (leftover / "junk.txt").write_bytes(b"")
    assert write_parquet_shards(_batches([3]), tmp_path / "out", shard_size=5) == 1
    assert not leftover.exists()
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["data-00000.parquet"]


def test_write_dict_rows(tmp_path: Path) -> None:
    rows = ({"text": f"t{i}", "n": i} for i in range(12))
    n = write_dict_rows(rows, tmp_path / "out", shard_size=5)
    tables = _read_all(tmp_path / "out")
    assert n == 3 and [t.num_rows for t in tables] == [5, 5, 2]
    merged = pa.concat_tables(tables)
    assert merged.column_names == ["text", "n"]
    assert merged["n"].to_pylist() == list(range(12))
    assert write_dict_rows(({"text": "x", "n": 99} for _ in range(1)), tmp_path / "out", 5, start_shard=3) == 1
    assert [p.name for p in list_parquet_files(tmp_path / "out")] == [f"data-{i:05d}.parquet" for i in range(4)]
