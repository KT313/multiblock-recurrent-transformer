# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.storage.manifest.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.storage.manifest import (
    MANIFEST_NAME,
    STAGES,
    Manifest,
    ShardInfo,
    library_versions,
    shard_problem,
    shard_rows,
)


def verify_shards(directory: Path, manifest: Manifest) -> list[str]:
    """
    The `shard_problem` of every shard of `manifest` that has one.
    """

    return [problem for shard in manifest.shards if (problem := shard_problem(directory, shard)) is not None]


def _manifest() -> Manifest:
    m = Manifest(source="src", source_hash="abc", stage="raw", rows_fetched=30, token_count="tokenizer",
                 tokenizer="llama-32k", truncated_at_tokens=2048, versions={"python": "3.11"}, extra={"note": 1})  # fmt: skip
    m.add_shard("data-00000.parquet", 10, 100)
    m.add_shard("data-00001.parquet", 20, 250)
    return m


def test_manifests_written_with_the_keys_under_extra_load_into_the_typed_fields(tmp_path: Path) -> None:
    """
    The on-disk layout keeps the stage's bookkeeping under `extra`; the fields are typed in memory only.
    """

    processed = {
        "source": "s", "source_hash": "h", "stage": "processed", "rows_fetched": 0, "shards": [{"name": "data-00000.parquet", "rows": 4, "tokens": 40}],
        "extra": {"input_shards": [["data-00000.parquet", 6]], "columns": ["text", "tokens"], "shuffled": True, "seed": 3, "stats": {"input_rows": 6}},
    }  # fmt: skip
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / MANIFEST_NAME).write_text(json.dumps(processed))
    m = Manifest.load(tmp_path / "p")
    assert m is not None and m.input_shards == [["data-00000.parquet", 6]] and m.columns == ["text", "tokens"]
    assert m.shuffled is True and m.shuffle_seed == 3 and m.stats == {"input_rows": 6} and m.extra == {}
    assert m.to_dict()["extra"] == processed["extra"], "saved again, the keys are where they were"
    raw = {"source": "s", "source_hash": "h", "stage": "raw", "rows_fetched": 9, "shards": [],
           "extra": {"exhausted": True, "check_limit": 9, "skipped_malformed": 2, "dropped_too_long": 1}}  # fmt: skip
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / MANIFEST_NAME).write_text(json.dumps(raw))
    m = Manifest.load(tmp_path / "r")
    assert m is not None and m.exhausted and m.check_limit_reached == 9 and (m.skipped_malformed, m.dropped_too_long) == (2, 1)
    assert m.extra == {} and m.to_dict()["extra"] == raw["extra"]
    m.exhausted, m.check_limit_reached = False, None
    assert m.to_dict()["extra"] == {"skipped_malformed": 2, "dropped_too_long": 1}, "unset flags are not written"
    tokenizer = {"source": "t", "source_hash": "h", "stage": "tokenizer", "extra": {"kind": "hf", "hf_id": "org/tok"}}
    assert Manifest.from_dict(tokenizer).extra == {"kind": "hf", "hf_id": "org/tok"}


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
    assert json.loads(path.read_text())["truncated_at_tokens"] == 2048


def test_legacy_token_cap_key_ignored_on_load(tmp_path: Path) -> None:
    payload = _manifest().to_dict()
    del payload["truncated_at_tokens"]
    payload["token_cap"] = 2048  # manifests written before the field was renamed
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(payload))
    loaded = Manifest.load(tmp_path)
    assert loaded is not None and loaded.truncated_at_tokens is None and not hasattr(loaded, "token_cap")
    assert loaded.rows() == 30 and loaded.tokenizer == "llama-32k"


def test_is_outdated_only_when_the_cap_is_raised() -> None:
    m = Manifest(source="s", source_hash="h", stage="raw", truncated_at_tokens=2048)
    assert m.is_outdated(4096), "raising the cap outdates raw"
    assert not m.is_outdated(2048) and not m.is_outdated(512), "the same or a lower cap never does"
    assert not Manifest(source="s", source_hash="h", stage="raw").is_outdated(4096), "no recorded cap: never outdated"
    assert not Manifest(source="s", source_hash="h", stage="processed").is_outdated(4096)


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
    assert STAGES == ("raw", "processed", "tokenizer")
    for stage in STAGES:
        assert Manifest(source="s", source_hash="h", stage=stage).stage == stage
    for stage in ("bogus", "validation", "instruct_mixture"):  # validation and mixtures no longer exist on disk
        with pytest.raises(ValueError, match="unknown manifest stage"):
            Manifest(source="s", source_hash="h", stage=stage)


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
    m.add_shard("data-100000.parquet", 1)
    m.add_shard("data-99999.parquet", 1)
    assert [shard.name for shard in m.shards][-2:] == ["data-99999.parquet", "data-100000.parquet"], "by index, not by name"


def test_a_raw_manifest_records_the_dataset_config_it_was_downloaded_under(tmp_path: Path) -> None:
    m = Manifest(source="s", source_hash="h", stage="raw", dataset_config="crow.yaml")
    assert m.to_dict()["extra"]["dataset_config"] == "crow.yaml"
    m.save(tmp_path)
    loaded = Manifest.load(tmp_path)
    assert loaded is not None and loaded.dataset_config == "crow.yaml" and loaded.extra == {}
    assert "dataset_config" not in Manifest(source="s", source_hash="h", stage="raw").to_dict()["extra"], "unknown: not written"


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
