# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.common: CLI defaults, HF cache setup, hashing, token estimate, parquet shard I/O."""

import argparse
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib import common
from data_preparation.lib.common import (
    add_common_args,
    configure_hf_cache,
    estimate_tokens,
    iter_dataset_tables,
    list_parquet_files,
    md5_hex,
    select_dataset_dirs,
    write_dict_rows,
    write_parquet_shards,
)


def _read_all(out_dir: Path, prefix: str = "data") -> list[pa.Table]:
    return [pq.read_table(f) for f in list_parquet_files(out_dir, prefix)]


# --- CLI / environment ----------------------------------------------------------------------------------------------


def test_add_common_args_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([])
    assert args.dataset_dir == Path("dataset")
    assert args.cache_dir is None
    args = parser.parse_args(["--dataset_dir", "/x/y", "--cache_dir", "/c"])
    assert args.dataset_dir == Path("/x/y") and args.cache_dir == Path("/c")


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


def test_print_header(capsys: pytest.CaptureFixture[str]) -> None:
    common.print_header("Title", width=5)
    assert capsys.readouterr().out == "=====\nTitle\n=====\n"
    common.print_header("x")
    assert capsys.readouterr().out.splitlines()[0] == "=" * 80


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


def test_list_parquet_files_sorted_and_prefix_filtered(tmp_path: Path) -> None:
    assert list_parquet_files(tmp_path / "missing", "data") == []
    for name in ("data-00002.parquet", "data-00000.parquet", "shard-00000.parquet", "data-x.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_parquet_files(tmp_path, "data")] == ["data-00000.parquet", "data-00002.parquet"]
    assert [p.name for p in list_parquet_files(tmp_path, "shard")] == ["shard-00000.parquet"]


def test_select_dataset_dirs(tmp_path: Path) -> None:
    assert select_dataset_dirs(tmp_path / "missing", None) == []
    for name in ("b_src", "a_src", "github_python", "github_go"):
        (tmp_path / name).mkdir()
    (tmp_path / "not_a_dir.parquet").write_bytes(b"")
    assert [d.name for d in select_dataset_dirs(tmp_path, None)] == ["a_src", "b_src", "github_go", "github_python"]
    assert [d.name for d in select_dataset_dirs(tmp_path, ["github_*"])] == ["github_go", "github_python"]
    # duplicates across overlapping patterns are collapsed; order follows pattern order
    selected = select_dataset_dirs(tmp_path, ["b_src", "github_*", "github_go"])
    assert [d.name for d in selected] == ["b_src", "github_go", "github_python"]
    assert select_dataset_dirs(tmp_path, ["nope*"]) == []


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
    assert [p.name for p in list_parquet_files(tmp_path / "out", "data")] == [f"data-{i:05d}.parquet" for i in range(4)]
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
    assert list_parquet_files(tmp_path / "out", "data") == []


def test_write_parquet_shards_custom_prefix(tmp_path: Path) -> None:
    write_parquet_shards(_batches([3]), tmp_path / "out", shard_size=5, prefix="shard")
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["shard-00000.parquet"]


def test_write_dict_rows(tmp_path: Path) -> None:
    rows = ({"text": f"t{i}", "n": i} for i in range(12))
    n = write_dict_rows(rows, tmp_path / "out", shard_size=5)
    tables = _read_all(tmp_path / "out")
    assert n == 3 and [t.num_rows for t in tables] == [5, 5, 2]
    merged = pa.concat_tables(tables)
    assert merged.column_names == ["text", "n"]
    assert merged["n"].to_pylist() == list(range(12))


def test_iter_dataset_tables_honours_select_mapping() -> None:
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"x": list(range(10)), "y": [str(i) for i in range(10)]})
    subset = ds.select([9, 3, 5, 0, 7])
    tables = list(iter_dataset_tables(subset, batch_size=2))
    assert all(isinstance(t, pa.Table) for t in tables)
    assert [t.num_rows for t in tables] == [2, 2, 1]
    assert pa.concat_tables(tables)["x"].to_pylist() == [9, 3, 5, 0, 7]
    assert pa.concat_tables(tables).column_names == ["x", "y"]


def test_iter_dataset_tables_empty_dataset() -> None:
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"x": [1, 2]}).select([])
    assert list(iter_dataset_tables(ds)) == []
