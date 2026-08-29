# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the ``hf_files`` / ``github_code`` loaders (``sources/hub_files.py``): Hub access is stubbed with local
temp files and a download counter; offset/count/order across file boundaries, file skipping via the persisted index,
every supported file format, shared files between two language sources, and the error paths. Offline, fast."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from data_preparation.lib.schema.dataset_config import SourceConfig
from data_preparation.lib.sources import LOADERS, Row, hub_file_index
from data_preparation.lib.sources import hub_files
from data_preparation.lib.sources.hub_files import FileIndex, file_format, index_path, iter_file, read_rows

REPO = "org/name"
REV = "abc"


class FakeHub:
    """Stand-in for the Hub: `files` maps repo paths to local files; counts downloads and listings."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, Path] = {}
        self.downloads: list[str] = []
        self.listings = 0

    def add(self, name: str, rows: list[Row], fmt: str | None = None) -> None:
        path = self.root / name.replace("/", "__")
        fmt = file_format(name) if fmt is None else fmt
        if fmt == ".parquet":
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
        elif fmt == ".json":
            path.write_text(json.dumps(rows), encoding="utf-8")
        else:
            payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
            if fmt == ".jsonl.zst":
                path.write_bytes(zstandard.ZstdCompressor().compress(payload))
            elif fmt in (".jsonl.gz", ".json.gz"):
                with gzip.open(path, "wb") as fh:
                    fh.write(payload)
            else:
                path.write_bytes(payload)
        self.files[name] = path

    def list_repo_files(self, repo_id: str, revision: str | None, token: str | None) -> list[str]:
        assert (repo_id, revision) == (REPO, REV)
        self.listings += 1
        return sorted(self.files, reverse=True) + ["README.md"]

    def hub_download(self, repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
        assert (repo_id, revision) == (REPO, REV)
        self.downloads.append(filename)
        return self.files[filename]


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub(tmp_path / "hub")
    fake.root.mkdir()
    monkeypatch.setattr(hub_files, "list_repo_files", fake.list_repo_files)
    monkeypatch.setattr(hub_files, "hub_download", fake.hub_download)
    return fake


def _rows(prefix: str, n: int, language: Callable[[int], str] | None = None) -> list[Row]:
    return [
        {"id": f"{prefix}{i}", "text": f"{prefix} doc {i}", "language": language(i) if language else "x"}
        for i in range(n)
    ]


def _src(**kwargs: Any) -> SourceConfig:
    defaults: dict[str, Any] = {
        "kind": "pretrain",
        "loader": "hf_files",
        "hf_id": REPO,
        "revision": REV,
        "load_kwargs": {"data_files": "data/*.parquet"},
    }
    defaults.update(kwargs)
    return SourceConfig(**defaults)


def _ids(rows: Iterator[Row]) -> list[str]:
    return [r["id"] for r in rows]


# --- hf_files ---------------------------------------------------------------------------------------------------------


def test_order_offset_count_across_files(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/b.parquet", _rows("b", 3))
    hub.add("data/a.parquet", _rows("a", 5))
    hub.add("data/c.parquet", _rows("c", 4))
    hub.add("other/d.parquet", _rows("d", 2))  # not matched by the glob
    index_dir = tmp_path / "index"
    load = LOADERS["hf_files"]
    assert _ids(load(_src(), 0, 100, index_dir=index_dir)) == [f"a{i}" for i in range(5)] + ["b0", "b1", "b2"] + [f"c{i}" for i in range(4)]
    assert _ids(load(_src(), 4, 3, index_dir=index_dir)) == ["a4", "b0", "b1"]
    assert _ids(load(_src(), 12, 5, index_dir=index_dir)) == []
    assert _ids(load(_src(), 0, 0, index_dir=index_dir)) == []
    assert hub.listings == 1  # the file list is cached in the index


def test_second_fetch_skips_files_before_offset(hub: FakeHub, tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        hub.add(f"data/{name}.parquet", _rows(name, 4))
    index_dir = tmp_path / "index"
    load = LOADERS["hf_files"]
    assert _ids(load(_src(), 0, 5, index_dir=index_dir)) == ["a0", "a1", "a2", "a3", "b0"]
    assert hub.downloads == ["data/a.parquet", "data/b.parquet"]
    hub.downloads.clear()
    assert _ids(load(_src(), 9, 2, index_dir=index_dir)) == ["c1", "c2"]
    assert hub.downloads == ["data/c.parquet"]  # a and b are skipped by their row counts; c's footer says 4 rows
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    assert saved["files"] == ["data/a.parquet", "data/b.parquet", "data/c.parquet"]
    assert saved["rows"] == {"data/a.parquet": 4, "data/b.parquet": 4, "data/c.parquet": 4}


def test_jsonl_counts_known_only_after_full_read(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("f/a.jsonl", _rows("a", 3))
    hub.add("f/b.jsonl", _rows("b", 3))
    index_dir = tmp_path / "index"
    src = _src(load_kwargs={"data_files": "f/*.jsonl"})
    assert _ids(LOADERS["hf_files"](src, 0, 2, index_dir=index_dir)) == ["a0", "a1"]
    index = FileIndex.open(REPO, REV, "f/*.jsonl", index_dir, None)
    assert index.rows == {}  # a.jsonl was not read to its end
    assert _ids(LOADERS["hf_files"](src, 1, 3, index_dir=index_dir)) == ["a1", "a2", "b0"]
    assert FileIndex.open(REPO, REV, "f/*.jsonl", index_dir, None).rows == {"f/a.jsonl": 3}
    hub.downloads.clear()
    assert _ids(LOADERS["hf_files"](src, 3, 1, index_dir=index_dir)) == ["b0"]
    assert hub.downloads == ["f/b.jsonl"]


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl", ".jsonl.zst", ".jsonl.gz", ".json.gz", ".json"])
def test_every_format(hub: FakeHub, suffix: str) -> None:
    hub.add(f"x/one{suffix}", _rows("o", 5))
    src = _src(load_kwargs={"data_files": f"x/*{suffix}"})
    assert _ids(LOADERS["hf_files"](src, 2, 2)) == ["o2", "o3"]  # in-memory index (index_dir=None)
    assert _ids(LOADERS["hf_files"](src, 0, 9)) == [f"o{i}" for i in range(5)]


def test_unknown_format_and_missing_files(hub: FakeHub) -> None:
    with pytest.raises(ValueError, match="unsupported file format"):
        file_format("data/x.csv")
    hub.add("data/x.csv", _rows("x", 1), fmt=".jsonl")
    with pytest.raises(ValueError, match="unsupported file format"):
        list(LOADERS["hf_files"](_src(load_kwargs={"data_files": "data/*.csv"}), 0, 1))
    with pytest.raises(FileNotFoundError, match="no files match"):
        list(LOADERS["hf_files"](_src(load_kwargs={"data_files": "nothing/*.parquet"}), 0, 1))
    with pytest.raises(ValueError, match="non-negative"):
        list(LOADERS["hf_files"](_src(), -1, 1))


def test_plain_json_must_be_array(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="JSON array"):
        list(iter_file(path, "x.json"))


def test_on_file_and_token_passthrough(hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    hub.add("data/a.parquet", _rows("a", 2))
    seen_tokens: list[str | None] = []
    original = hub.hub_download

    def download(repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
        seen_tokens.append(token)
        return original(repo_id, filename, revision, token)

    monkeypatch.setattr(hub_files, "hub_download", download)
    opened: list[str] = []
    assert _ids(LOADERS["hf_files"](_src(), 0, 1, token="tok", on_file=opened.append)) == ["a0"]
    assert opened == ["data/a.parquet"] and seen_tokens == ["tok"]


def test_parquet_skips_row_groups(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    pq.write_table(pa.Table.from_pylist(_rows("r", 7)), path, row_group_size=3)
    assert _ids(iter_file(path, "x.parquet", skip=4)) == ["r4", "r5", "r6"]
    assert _ids(iter_file(path, "x.parquet", skip=7)) == []


def test_index_path_layout(tmp_path: Path) -> None:
    path = index_path(tmp_path, "org/name", "abc", "data/*.parquet")
    assert path.parent == tmp_path / "org--name@abc" and path.suffix == ".json"
    assert index_path(tmp_path, "org/name", None, "x").parent.name == "org--name@main"
    assert index_path(tmp_path, "org/name", "abc", "other") != path


# --- github_code ------------------------------------------------------------------------------------------------------


def _language(i: int) -> str:
    return "Python" if i % 3 == 0 else "Java"


def test_github_code_offsets_count_language_rows_and_share_files(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 6, _language))  # Python: a0 a3
    hub.add("data/b.parquet", _rows("b", 6, _language))  # Python: b0 b3
    hub.add("data/c.parquet", _rows("c", 6, _language))  # Python: c0 c3
    index_dir = tmp_path / "index"
    load = LOADERS["github_code"]
    python = _src(loader="github_code", language="Python", load_kwargs={})
    java = _src(loader="github_code", language="Java", load_kwargs={})
    assert _ids(load(python, 0, 3, index_dir=index_dir)) == ["a0", "a3", "b0"]
    assert _ids(load(python, 3, 2, index_dir=index_dir)) == ["b3", "c0"]
    assert _ids(load(java, 0, 5, index_dir=index_dir)) == ["a1", "a2", "a4", "a5", "b1"]
    assert _ids(load(java, 9, 5, index_dir=index_dir)) == ["c1", "c2", "c4", "c5"]
    # every file was downloaded (i.e. resolved) but the language counts let later fetches skip files
    assert hub.downloads.count("data/a.parquet") >= 1
    hub.downloads.clear()
    assert _ids(load(python, 5, 1, index_dir=index_dir)) == ["c3"]
    assert hub.downloads == ["data/c.parquet"]  # a and b: 2 Python rows each, known from the index
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    assert saved["counts"]["language=Python"] == {"data/a.parquet": 2, "data/b.parquet": 2}
    assert saved["counts"]["language=Java"]["data/a.parquet"] == 4
    assert saved["rows"] == {"data/a.parquet": 6, "data/b.parquet": 6, "data/c.parquet": 6}
    assert hub.listings == 1


def test_github_code_data_files_override_and_language_required(hub: FakeHub) -> None:
    hub.add("data/train-00000.parquet", _rows("a", 3, _language))
    hub.add("data/train-00001.parquet", _rows("b", 3, _language))
    src = _src(loader="github_code", language="Python", load_kwargs={"data_files": "data/train-00001.parquet"})
    assert _ids(LOADERS["github_code"](src, 0, 5)) == ["b0"]
    index = hub_file_index(_src(loader="github_code", language="Java", load_kwargs={}), "data/*.parquet", None, None)
    assert index.files == ["data/train-00000.parquet", "data/train-00001.parquet"]
    assert list(LOADERS["github_code"](src, 0, 0)) == []


def test_read_rows_match_without_key_records_totals(hub: FakeHub) -> None:
    hub.add("data/a.jsonl", _rows("a", 4, _language))
    index = FileIndex.open(REPO, REV, "data/*.jsonl", None, None)
    got = list(read_rows(index, 0, 10, token=None, key="k", match=lambda r: r["language"] == "Python"))
    assert _ids(iter(got)) == ["a0", "a3"]
    assert index.rows == {"data/a.jsonl": 4} and index.counts == {"k": {"data/a.jsonl": 2}}
