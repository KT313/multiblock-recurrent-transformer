# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.storage.manifest."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.storage.manifest import (
    MANIFEST_NAME,
    Manifest,
    ShardInfo,
    library_versions,
    shard_rows,
    verify_shards,
)


def _manifest() -> Manifest:
    m = Manifest(source="src", source_hash="abc", stage="raw", rows_fetched=30, token_count="tokenizer",
                 tokenizer="llama-32k", versions={"python": "3.11"}, extra={"note": 1})  # fmt: skip
    m.add_shard("data-00000.parquet", 10, 100)
    m.add_shard("data-00001.parquet", 20, 250)
    return m


def test_round_trip_and_unknown_keys(tmp_path: Path) -> None:
    m = _manifest()
    path = m.save(tmp_path)
    assert path == tmp_path / MANIFEST_NAME and not list(tmp_path.glob("*.tmp"))
    loaded = Manifest.load(tmp_path)
    assert loaded == m
    payload = json.loads(path.read_text())
    payload["future_field"] = {"x": 1}
    payload["shards"][0]["future_shard_field"] = 2
    path.write_text(json.dumps(payload))
    assert Manifest.load(tmp_path) == m


def test_load_absent_or_unparsable(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert Manifest.load(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text("{not json")
    assert Manifest.load(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text("[1, 2]")
    assert Manifest.load(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"source": "s"}))  # missing required fields
    assert Manifest.load(tmp_path) is None
    assert sum("unparsable manifest" in r.message for r in caplog.records) == 3
    # next to shards an unreadable manifest is an error, never "absent" (that would restart the directory at shard 0)
    pq.write_table(pa.table({"text": ["a"]}), tmp_path / "data-00000.parquet")
    with pytest.raises(RuntimeError, match="unreadable manifest .* next to shards"):
        Manifest.load(tmp_path)


def test_stage_validated() -> None:
    with pytest.raises(ValueError, match="stage"):
        Manifest(source="s", source_hash="h", stage="bogus")


def test_rows_tokens_is_current() -> None:
    m = _manifest()
    assert m.rows() == 30 and m.tokens() == 350
    assert m.is_current("abc") and not m.is_current("abd")
    m.add_shard("data-00002.parquet", 5)  # no token count
    assert m.rows() == 35 and m.tokens() is None
    assert Manifest(source="s", source_hash="h", stage="raw").tokens() == 0


def test_add_shard_replaces_and_sorts() -> None:
    m = Manifest(source="s", source_hash="h", stage="processed")
    m.add_shard("data-00001.parquet", 2, 20)
    m.add_shard("data-00000.parquet", 1, 10)
    m.add_shard("data-00001.parquet", 3, 30)
    assert m.shards == [ShardInfo("data-00000.parquet", 1, 10), ShardInfo("data-00001.parquet", 3, 30)]


def test_shard_rows_and_verify(tmp_path: Path) -> None:
    pq.write_table(pa.table({"x": list(range(10))}), tmp_path / "data-00000.parquet")
    pq.write_table(pa.table({"x": list(range(20))}), tmp_path / "data-00001.parquet")
    assert shard_rows(tmp_path / "data-00000.parquet") == 10
    m = _manifest()
    assert verify_shards(tmp_path, m) == []
    m.add_shard("data-00001.parquet", 21, 250)
    m.add_shard("data-00002.parquet", 5, 50)
    problems = verify_shards(tmp_path, m)
    assert len(problems) == 2
    assert any("data-00001.parquet" in p and "21" in p and "20" in p for p in problems)
    assert any(p == "missing shard data-00002.parquet" for p in problems)
    (tmp_path / "data-00002.parquet").write_bytes(b"not parquet")
    assert any(p.startswith("unreadable shard data-00002.parquet") for p in verify_shards(tmp_path, m))


def test_library_versions() -> None:
    versions = library_versions()
    assert set(versions) >= {"python", "pyarrow"}
    assert all(isinstance(v, str) and v for v in versions.values())
    if "git" in versions:
        assert len(versions["git"]) == 40
