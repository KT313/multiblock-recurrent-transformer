# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the hf_files / github_code loaders (sources/hub_files.py): Hub access is stubbed with local
temp files and a download counter; offset/count/order across file boundaries, file skipping via the persisted index,
every supported file format, shared files between two language sources, and the error paths. Offline, fast.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.conftest import REPO, REV, FakeHub, RecordingFile
from data_preparation.dataset_config import SourceConfig
from data_preparation.lib.sources import hub_files
from data_preparation.lib.sources.hub_files import (
    HUB_REQUEST_TIMEOUT,
    FetchStats,
    FileIndex,
    HubFetcher,
    configure_hub_http,
    file_format,
    index_path,
    iter_file,
    iter_parquet,
    iter_row_batches,
    parquet_row_groups,
    read_rows,
    read_rows_multi,
    repo_listing,
)
from data_preparation.lib.sources.loaders import (
    LOADERS,
    GithubCodeRequest,
    Row,
    SharedLoaderParameters,
    hub_file_index,
    language_request,
    read_github_code_group,
)

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
    assert _ids(load(_src(), 0, 100, SharedLoaderParameters(index_dir=index_dir))) == [f"a{i}" for i in range(5)] + ["b0", "b1", "b2"] + [f"c{i}" for i in range(4)]
    assert _ids(load(_src(), 4, 3, SharedLoaderParameters(index_dir=index_dir))) == ["a4", "b0", "b1"]
    assert _ids(load(_src(), 12, 5, SharedLoaderParameters(index_dir=index_dir))) == []
    assert _ids(load(_src(), 0, 0, SharedLoaderParameters(index_dir=index_dir))) == []
    assert hub.listings == 1  # the file list is cached in the index


def test_second_fetch_skips_files_before_offset(hub: FakeHub, tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        hub.add(f"data/{name}.parquet", _rows(name, 4))
    index_dir = tmp_path / "index"
    load = LOADERS["hf_files"]
    assert _ids(load(_src(), 0, 5, SharedLoaderParameters(index_dir=index_dir))) == ["a0", "a1", "a2", "a3", "b0"]
    assert hub.downloads == ["data/a.parquet", "data/b.parquet"]
    hub.downloads.clear()
    assert _ids(load(_src(), 9, 2, SharedLoaderParameters(index_dir=index_dir))) == ["c1", "c2"]
    assert hub.downloads == ["data/c.parquet"]  # a and b are skipped by their row counts; c's footer says 4 rows
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    assert saved["files"] == ["data/a.parquet", "data/b.parquet", "data/c.parquet"]
    assert saved["rows"] == {"data/a.parquet": 4, "data/b.parquet": 4, "data/c.parquet": 4}


def test_jsonl_counts_known_only_after_full_read(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("f/a.jsonl", _rows("a", 3))
    hub.add("f/b.jsonl", _rows("b", 3))
    index_dir = tmp_path / "index"
    src = _src(load_kwargs={"data_files": "f/*.jsonl"})
    assert _ids(LOADERS["hf_files"](src, 0, 2, SharedLoaderParameters(index_dir=index_dir))) == ["a0", "a1"]
    index = FileIndex.open(REPO, REV, "f/*.jsonl", index_dir, None)
    assert index.row_counts == {}  # a.jsonl was not read to its end
    assert _ids(LOADERS["hf_files"](src, 1, 3, SharedLoaderParameters(index_dir=index_dir))) == ["a1", "a2", "b0"]
    assert FileIndex.open(REPO, REV, "f/*.jsonl", index_dir, None).row_counts == {"f/a.jsonl": 3}
    hub.downloads.clear()
    assert _ids(LOADERS["hf_files"](src, 3, 1, SharedLoaderParameters(index_dir=index_dir))) == ["b0"]
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
    with pytest.raises(ValueError, match="JSON array of rows \\(top-level map\\)"):
        list(iter_file(path, "x.json"))
    path.write_text("")
    with pytest.raises(ValueError, match="empty file"):
        list(iter_file(path, "x.json"))
    path.write_text(json.dumps([{"a": 1}, 2]))
    with pytest.raises(ValueError, match="element 1 .* not an object"):
        list(iter_file(path, "x.json"))


def test_json_gz_is_an_array_or_lines_by_its_content(tmp_path: Path) -> None:
    """
    Hub repos ship `.json.gz` in both shapes; the extension cannot tell them apart, the first byte can.
    """

    rows = _rows("a", 3)
    path = tmp_path / "x.json.gz"
    path.write_bytes(gzip.compress(b" \n" + json.dumps(rows).encode()))
    assert _ids(iter_file(path, "x.json.gz", skip=1)) == ["a1", "a2"]
    path.write_bytes(gzip.compress(b" \n" + "\n".join(json.dumps(r) for r in rows).encode()))
    assert _ids(iter_file(path, "x.json.gz", skip=1)) == ["a1", "a2"]


def _padded_rows(prefix: str, n: int, pad: int = 16 * 1024) -> list[Row]:
    """
    Rows of ~`pad` bytes each (ijson pulls 64 KB chunks, so a file of 50 rows is ~800 KB: 7 rows are one chunk).
    """

    return [{"id": f"{prefix}{i}", "text": "x" * pad, "score": i + 0.5} for i in range(n)]


def _bytes_read(handle: RecordingFile) -> int:
    return sum(end - start for start, end in handle.ranges)


def test_large_json_array_is_streamed_incrementally(hub: FakeHub, tmp_path: Path) -> None:
    """
    A `.json` array above the cache threshold is read remotely through ijson: `count` rows cost a prefix of the
    file, the row count is recorded only after a full read, and a top-up re-streams the file from its start.
    """

    hub.add("big/a.json", _padded_rows("a", 50))
    size = hub.files["big/a.json"].stat().st_size
    assert size > 50 * 16 * 1024
    index_dir = tmp_path / "index"
    stats = FetchStats()
    src = _src(load_kwargs={"data_files": "big/*.json", "max_cached_file_mb": 0.01})  # 10 KB: force the remote path
    opened: list[str] = []
    rows = list(LOADERS["hf_files"](src, 0, 7, SharedLoaderParameters(index_dir=index_dir, stats=stats, on_file=opened.append)))
    assert [r["id"] for r in rows] == [f"a{i}" for i in range(7)] and rows[0]["score"] == 0.5
    assert opened == ["big/a.json"] and hub.downloads == [] and hub.streams == ["big/a.json"]
    read = _bytes_read(hub.handles["big/a.json"])
    assert 0 < read < size * 0.25, (read, size)
    assert stats.bytes_fetched == read and stats.files_streamed == 1 and stats.files_downloaded == 0
    assert FileIndex.open(REPO, REV, "big/*.json", index_dir, None).row_counts == {}  # not read to the end
    # top-up at an offset re-streams from the start (prefix read: still far less than the file)
    assert _ids(LOADERS["hf_files"](src, 5, 4, SharedLoaderParameters(index_dir=index_dir))) == ["a5", "a6", "a7", "a8"]
    assert hub.streams == ["big/a.json", "big/a.json"]
    assert _bytes_read(hub.handles["big/a.json"]) < size * 0.3
    # reading to the end records the row count, after which a fetch past it opens nothing
    assert len(list(LOADERS["hf_files"](src, 40, 100, SharedLoaderParameters(index_dir=index_dir)))) == 10
    assert FileIndex.open(REPO, REV, "big/*.json", index_dir, None).row_counts == {"big/a.json": 50}
    hub.streams.clear()
    assert _ids(LOADERS["hf_files"](src, 50, 5, SharedLoaderParameters(index_dir=index_dir))) == []
    assert hub.streams == []


def test_large_json_non_array_errors_clearly(hub: FakeHub) -> None:
    (hub.root / "big__o.json").write_text(json.dumps({"rows": _padded_rows("o", 10)}), encoding="utf-8")
    hub.files["big/o.json"] = hub.root / "big__o.json"
    src = _src(load_kwargs={"data_files": "big/*.json", "max_cached_file_mb": 0.001})
    with pytest.raises(ValueError, match="big/o.json: plain .json must contain a JSON array"):
        list(LOADERS["hf_files"](src, 0, 1))


def test_small_json_array_uses_cache(hub: FakeHub) -> None:
    hub.add("s/a.json", _padded_rows("a", 5))
    src = _src(load_kwargs={"data_files": "s/*.json"})
    assert _ids(LOADERS["hf_files"](src, 1, 2)) == ["a1", "a2"]
    assert hub.downloads == ["s/a.json"] and hub.streams == []


def test_on_file_and_token_passthrough(hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    hub.add("data/a.parquet", _rows("a", 2))
    seen_tokens: list[str | None] = []
    original = hub.hub_download

    def download(repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
        seen_tokens.append(token)
        return original(repo_id, filename, revision, token)

    monkeypatch.setattr(hub_files, "hub_download", download)
    opened: list[str] = []
    assert _ids(LOADERS["hf_files"](_src(), 0, 1, SharedLoaderParameters(token="tok", on_file=opened.append))) == ["a0"]
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
    assert _ids(load(python, 0, 3, SharedLoaderParameters(index_dir=index_dir))) == ["a0", "a3", "b0"]
    assert _ids(load(python, 3, 2, SharedLoaderParameters(index_dir=index_dir))) == ["b3", "c0"]
    assert _ids(load(java, 0, 5, SharedLoaderParameters(index_dir=index_dir))) == ["a1", "a2", "a4", "a5", "b1"]
    assert _ids(load(java, 9, 5, SharedLoaderParameters(index_dir=index_dir))) == ["c2", "c4", "c5"]  # Java rows: a1 a2 a4 a5 b1 b2 b4 b5 c1 c2 ...
    # every file was downloaded (i.e. resolved) but the language counts let later fetches skip files
    assert hub.downloads.count("data/a.parquet") >= 1
    hub.downloads.clear()
    assert _ids(load(python, 5, 1, SharedLoaderParameters(index_dir=index_dir))) == ["c3"]
    assert hub.downloads == ["data/c.parquet"]  # a and b: 2 Python rows each, known from the index
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    # every language of a fully decoded file is counted: Java's read through c recorded Python's rows there too
    assert saved["counts"]["language=Python"] == {"data/a.parquet": 2, "data/b.parquet": 2, "data/c.parquet": 2}
    assert saved["counts"]["language=Java"]["data/a.parquet"] == 4
    assert saved["full_counts"] == {"data/a.parquet": 3, "data/b.parquet": 3, "data/c.parquet": 3}  # row groups of 2
    assert saved["rows"] == {"data/a.parquet": 6, "data/b.parquet": 6, "data/c.parquet": 6}
    assert hub.listings == 1


def test_keyed_offset_carries_across_files_with_unknown_counts(hub: FakeHub, tmp_path: Path) -> None:
    """
    A file with fewer matching rows than the remaining skip only shrinks the skip (nothing is known about the
    files yet, so none can be skipped without reading).
    """

    hub.add("data/a.parquet", _rows("a", 6, _language))  # Python: a0 a3
    hub.add("data/b.parquet", _rows("b", 6, _language))  # Python: b0 b3
    hub.add("data/c.parquet", _rows("c", 6, _language))  # Python: c0 c3
    python = _src(loader="github_code", language="Python", load_kwargs={})
    load = LOADERS["github_code"]
    assert _ids(load(python, 5, 1, SharedLoaderParameters(index_dir=tmp_path / "index"))) == ["c3"]
    assert _ids(load(python, 5, 1, SharedLoaderParameters(index_dir=tmp_path / "index"))) == ["c3"]  # now via the recorded counts
    assert _ids(load(python, 5, 1)) == ["c3"]  # in-memory index


def test_stream_offset_beyond_a_file_carries_over(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.jsonl", _rows("a", 3))
    hub.add("data/b.jsonl", _rows("b", 3))
    src = _src(load_kwargs={"data_files": "data/*.jsonl"})
    assert _ids(LOADERS["hf_files"](src, 4, 2, SharedLoaderParameters(index_dir=tmp_path / "index"))) == ["b1", "b2"]
    saved = json.loads(index_path(tmp_path / "index", REPO, REV, "data/*.jsonl").read_text())
    assert saved["rows"] == {"data/a.jsonl": 3}  # b was left in the middle: its total is not known yet
    assert _ids(LOADERS["hf_files"](src, 4, 2, SharedLoaderParameters(index_dir=tmp_path / "index"))) == ["b1", "b2"]


def test_github_code_data_files_override_and_language_required(hub: FakeHub) -> None:
    hub.add("data/train-00000.parquet", _rows("a", 3, _language))
    hub.add("data/train-00001.parquet", _rows("b", 3, _language))
    src = _src(loader="github_code", language="Python", load_kwargs={"data_files": "data/train-00001.parquet"})
    assert _ids(LOADERS["github_code"](src, 0, 5)) == ["b0"]
    index = hub_file_index(_src(loader="github_code", language="Java", load_kwargs={}), "data/*.parquet", SharedLoaderParameters())
    assert index.files == ["data/train-00000.parquet", "data/train-00001.parquet"]
    assert list(LOADERS["github_code"](src, 0, 0)) == []


def test_read_rows_match_without_key_records_totals(hub: FakeHub) -> None:
    hub.add("data/a.jsonl", _rows("a", 4, _language))
    index = FileIndex.open(REPO, REV, "data/*.jsonl", None, None)
    got = list(read_rows(index, 0, 10, token=None, key="k", match=lambda r: r["language"] == "Python"))
    assert _ids(iter(got)) == ["a0", "a3"]
    assert index.row_counts == {"data/a.jsonl": 4} and index.keyed_counts == {"k": {"data/a.jsonl": 2}}


# --- size-aware fetching (Hub cache vs. remote row groups / streams) ---------------------------------------------------

REMOTE = {"max_cached_file_mb": 0}  # every (non-empty) file counts as "large"


def _big_rows(prefix: str, n: int) -> list[Row]:
    """
    Rows of ~8 KB incompressible text, so row groups are large compared to pyarrow's 64 KB footer read.
    """

    rng = random.Random(0)
    return [{"id": f"{prefix}{i}", "text": f"{prefix}{i} " + rng.randbytes(4000).hex()} for i in range(n)]


def _group_spans(path: Path) -> list[tuple[int, int]]:
    """
    Byte span [start, end) of every row group's column chunks.
    """

    metadata = pq.ParquetFile(path).metadata
    spans: list[tuple[int, int]] = []
    for g in range(metadata.num_row_groups):
        group = metadata.row_group(g)
        starts, ends = [], []
        for c in range(group.num_columns):
            column = group.column(c)
            start = min(o for o in (column.dictionary_page_offset, column.data_page_offset) if o is not None)
            starts.append(start)
            ends.append(start + column.total_compressed_size)
        spans.append((min(starts), max(ends)))
    return spans


def _touched_groups(ranges: list[tuple[int, int]], spans: list[tuple[int, int]], size: int) -> set[int]:
    """
    Row groups intersected by the recorded reads, ignoring the footer read (pyarrow reads the file's tail once).
    """

    data_reads = [(a, b) for a, b in ranges if b != size]
    assert all(b - a < size // 2 for a, b in ranges)  # nothing ever reads the file whole
    return {g for g, (s, e) in enumerate(spans) if any(a < e and b > s for a, b in data_reads)}


def test_large_parquet_reads_only_the_needed_row_groups(hub: FakeHub, tmp_path: Path) -> None:
    path = tmp_path / "big.parquet"
    pq.write_table(pa.Table.from_pylist(_big_rows("r", 60)), path, row_group_size=10)  # 6 groups x 10 rows
    hub.files["data/big.parquet"] = path
    spans = _group_spans(path)
    size = path.stat().st_size
    assert size > 6 * 64 * 1024
    index_dir = tmp_path / "index"
    src = _src(load_kwargs={"data_files": "data/*.parquet", **REMOTE})
    # 12 rows from offset 3 end inside group 1, which is yielded to its end (rows 3..19)
    assert _ids(LOADERS["hf_files"](src, 3, 12, SharedLoaderParameters(index_dir=index_dir))) == [f"r{i}" for i in range(3, 20)]
    assert hub.downloads == [] and hub.streams == ["data/big.parquet"]
    assert _touched_groups(hub.handles["data/big.parquet"].ranges, spans, size) == {0, 1}
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    assert saved["row_groups"] == {"data/big.parquet": [10] * 6} and saved["rows"] == {"data/big.parquet": 60}
    assert saved["sizes"] == {"data/big.parquet": path.stat().st_size}
    # a top-up seeks straight to the right row group (and keeps it whole)
    assert _ids(LOADERS["hf_files"](src, 52, 5, SharedLoaderParameters(index_dir=index_dir))) == [f"r{i}" for i in range(52, 60)]
    assert hub.streams == ["data/big.parquet"] * 2 and hub.downloads == []
    assert _touched_groups(hub.handles["data/big.parquet"].ranges, spans, size) == {5}
    assert hub.size_lookups == 1 and hub.listings == 1
    # a fetch entirely past the known row count never opens the file
    hub.streams.clear()
    assert _ids(LOADERS["hf_files"](src, 60, 5, SharedLoaderParameters(index_dir=index_dir))) == []
    assert hub.streams == []


def test_small_files_use_the_hub_cache_and_threshold_is_per_source(hub: FakeHub) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    assert _ids(LOADERS["hf_files"](_src(), 0, 2)) == ["a0", "a1"]
    assert hub.downloads == ["data/a.parquet"] and hub.streams == []
    hub.downloads.clear()
    remote = _src(load_kwargs={"data_files": "data/*.parquet", **REMOTE})
    assert _ids(LOADERS["hf_files"](remote, 0, 2)) == ["a0", "a1"]
    assert hub.downloads == [] and hub.streams == ["data/a.parquet"]
    assert LOADERS["hf_files"] is not None and hub.size_lookups == 2  # in-memory indexes look sizes up each time


def test_fetcher_seams_and_stats(hub: FakeHub) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    hub.add("data/b.parquet", _rows("b", 3))
    index = FileIndex.open(REPO, REV, "data/*.parquet", None, None)
    opened: list[str] = []

    def remote(repo_id: str, filename: str, revision: str | None, token: str | None, block_size: int) -> BinaryIO:
        opened.append(filename)
        return hub.files[filename].open("rb")

    fetcher = HubFetcher(token="tok", max_cached_file_mb=0.0, remote=remote)
    assert fetcher.uses_cache(0) and not fetcher.uses_cache(1)
    assert _ids(read_rows(index, 2, 2, fetcher=HubFetcher(download=lambda repo, file, rev, tok: hub.files[file]))) == ["a2", "b0"]
    assert _ids(read_rows(index, 2, 2, fetcher=fetcher)) == ["a2", "b0", "b1"]  # b's first row group (2 rows) kept whole
    assert _ids(read_rows(index, 2, 2, fetcher=fetcher, align_to_row_group=False)) == ["a2", "b0"]
    assert opened == ["data/a.parquet", "data/b.parquet"] * 2 and hub.streams == []
    assert fetcher.stats.files_streamed == 4 and fetcher.stats.files_downloaded == 0
    assert fetcher.stats.bytes_fetched > 0
    cached = HubFetcher(download=lambda repo, file, rev, tok: hub.files[file])
    assert _ids(read_rows(index, 0, 1, fetcher=cached)) == ["a0"]
    assert cached.stats == FetchStats(files_downloaded=1) and hub.downloads == [], "a file already in the cache: no bytes fetched"


def test_every_hub_request_is_bounded_by_the_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The listing must pass the timeout itself: `dataset_info`'s own `timeout=None` default would disable the
    client's. The size lookup has no such parameter and inherits the client's, set once per process.
    """

    import httpx
    from huggingface_hub import HfApi, get_session

    calls: list[tuple[str, str | None, float | None]] = []

    def dataset_info(self: HfApi, repo_id: str, *, revision: str | None = None, timeout: float | None = None, **kwargs: Any) -> Any:
        calls.append((repo_id, revision, timeout))
        return SimpleNamespace(sha="abc", siblings=[SimpleNamespace(rfilename="data/a.parquet")])

    monkeypatch.setattr(HfApi, "dataset_info", dataset_info)
    assert repo_listing(REPO, REV, None) == (["data/a.parquet"], "abc")
    assert calls == [(REPO, REV, HUB_REQUEST_TIMEOUT)]
    assert get_session().timeout == httpx.Timeout(HUB_REQUEST_TIMEOUT)
    configure_hub_http()  # idempotent (cached): no second configuration
    assert configure_hub_http.cache_info().hits >= 1


def test_a_file_downloaded_whole_into_the_cache_counts_its_size(hub: FakeHub) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    index = FileIndex.open(REPO, REV, "data/*.parquet", None, None)
    path = hub.files["data/a.parquet"]
    fetcher = HubFetcher(download=lambda repo, file, rev, tok: path)
    os.utime(path, (fetcher.created + 1, fetcher.created + 1))  # the cache file is younger than the fetcher: downloaded now
    assert _ids(read_rows(index, 0, 1, fetcher=fetcher)) == ["a0"]
    assert fetcher.stats == FetchStats(bytes_fetched=path.stat().st_size, files_downloaded=1)


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zst", ".jsonl.gz", ".json.gz"])
def test_large_json_lines_stream_sequentially_and_reread_partial_files(hub: FakeHub, tmp_path: Path, suffix: str) -> None:
    hub.add(f"f/a{suffix}", _rows("a", 4))
    hub.add(f"f/b{suffix}", _rows("b", 4))
    index_dir = tmp_path / "index"
    src = _src(load_kwargs={"data_files": f"f/*{suffix}", **REMOTE})
    load = LOADERS["hf_files"]
    assert _ids(load(src, 0, 2, SharedLoaderParameters(index_dir=index_dir))) == ["a0", "a1"]
    assert hub.streams == [f"f/a{suffix}"] and hub.downloads == []
    assert FileIndex.open(REPO, REV, f"f/*{suffix}", index_dir, None).row_counts == {}  # not read to the end
    assert _ids(load(src, 1, 4, SharedLoaderParameters(index_dir=index_dir))) == ["a1", "a2", "a3", "b0"]  # a is re-streamed from its start
    assert hub.streams == [f"f/a{suffix}", f"f/a{suffix}", f"f/b{suffix}"]
    assert FileIndex.open(REPO, REV, f"f/*{suffix}", index_dir, None).row_counts == {f"f/a{suffix}": 4}
    hub.streams.clear()
    assert _ids(load(src, 5, 9, SharedLoaderParameters(index_dir=index_dir))) == ["b1", "b2", "b3"]
    assert hub.streams == [f"f/b{suffix}"]  # a is skipped by its recorded count; b read to the end
    assert FileIndex.open(REPO, REV, f"f/*{suffix}", index_dir, None).row_counts == {f"f/a{suffix}": 4, f"f/b{suffix}": 4}


def test_plain_json_streams_remotely_and_reads_from_cache(hub: FakeHub) -> None:
    hub.add("x/one.json", _rows("o", 5))
    assert _ids(LOADERS["hf_files"](_src(load_kwargs={"data_files": "x/*.json", **REMOTE}), 0, 1)) == ["o0"]
    assert hub.streams == ["x/one.json"] and hub.downloads == []
    assert _ids(LOADERS["hf_files"](_src(load_kwargs={"data_files": "x/*.json"}), 0, 1)) == ["o0"]
    assert hub.downloads == ["x/one.json"]


def test_github_code_streams_large_files(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 6, _language))
    src = _src(loader="github_code", language="Python", load_kwargs=REMOTE)
    assert _ids(LOADERS["github_code"](src, 0, 5, SharedLoaderParameters(index_dir=tmp_path / "index"))) == ["a0", "a3"]
    assert hub.streams == ["data/a.parquet"] and hub.downloads == []


def test_index_without_sizes_is_upgraded(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    index_dir = tmp_path / "index"
    old = index_path(index_dir, REPO, REV, "data/*.parquet")
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"files": ["data/a.parquet"], "rows": {}, "counts": {}}))
    index = FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)
    assert index.sizes == {"data/a.parquet": hub.files["data/a.parquet"].stat().st_size}
    assert hub.size_lookups == 1 and hub.listings == 0
    assert json.loads(old.read_text())["sizes"] == index.sizes and json.loads(old.read_text())["row_groups"] == {}
    FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)
    assert hub.size_lookups == 1


# --- the revision guard -------------------------------------------------------------------------------------------------


def _forget_open_indexes() -> None:
    """
    Simulate a fresh process: the next `FileIndex.open` loads from disk instead of the process-wide cache.
    """

    hub_files._OPEN_INDEXES.clear()


def test_index_records_the_resolved_commit_and_same_resolution_passes(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    index_dir = tmp_path / "index"
    fresh = FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)
    assert fresh.resolved_revision == hub.sha  # from the listing call itself, no extra resolution
    assert hub.resolutions == 0
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    assert saved["resolved_revision"] == hub.sha
    _forget_open_indexes()
    reloaded = FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)
    assert reloaded.resolved_revision == hub.sha
    assert hub.resolutions == 1 and hub.listings == 1  # loading resolved once, never re-listed


def test_the_hub_round_trip_of_an_open_runs_outside_the_process_wide_lock(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A stalled listing or revision check must not hold the lock every other `FileIndex.open` waits on.
    """

    lock_held: list[bool] = []

    def listing(*args: Any) -> tuple[list[str], str]:
        lock_held.append(hub_files._OPEN_INDEXES_LOCK.locked())
        return hub.repo_listing(*args)

    def resolution(*args: Any) -> str:
        lock_held.append(hub_files._OPEN_INDEXES_LOCK.locked())
        return hub.resolve_revision(*args)

    monkeypatch.setattr(hub_files, "repo_listing", listing)
    monkeypatch.setattr(hub_files, "resolve_revision", resolution)
    hub.add("data/a.parquet", _rows("a", 3))
    index_dir = tmp_path / "index"
    FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)  # fresh: listed
    _forget_open_indexes()
    FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)  # loaded: revision checked
    assert (hub.listings, hub.resolutions, lock_held) == (1, 1, [False, False])


def test_moved_repo_fails_the_loaded_index_with_the_pin_hint(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    index_dir = tmp_path / "index"
    load = LOADERS["hf_files"]
    assert _ids(load(_src(), 0, 2, SharedLoaderParameters(index_dir=index_dir))) == ["a0", "a1"]
    built_at = hub.sha
    _forget_open_indexes()
    hub.sha = "commit-2"
    with pytest.raises(RuntimeError) as error:
        list(load(_src(), 2, 1, SharedLoaderParameters(index_dir=index_dir)))
    message = str(error.value)
    assert built_at in message and "commit-2" in message
    assert f"revision: {built_at}" in message  # the pin that keeps the listed commit
    assert "downloaded again" in message and "raw fingerprint" in message  # honest: both ways out re-download the raw folder
    assert "stays valid" not in message  # the old text promised the raw data survives the pin; the fingerprint says otherwise
    assert str(index_path(index_dir, REPO, REV, "data/*.parquet")) in message
    assert hub.listings == 1  # a mismatch never re-lists
    # github_code shares the machinery (and here the very index): same error
    src = _src(loader="github_code", language="x", load_kwargs={"data_files": "data/*.parquet"})
    with pytest.raises(RuntimeError, match="file index was built at revision"):
        list(LOADERS["github_code"](src, 0, 1, SharedLoaderParameters(index_dir=index_dir)))


def test_legacy_index_without_revision_upgrades_once_then_guards(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    index_dir = tmp_path / "index"
    path = index_path(index_dir, REPO, REV, "data/*.parquet")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"files": ["data/a.parquet"], "rows": {}, "counts": {}}))  # written by the old code
    index = FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)  # no error: adopts the current resolution
    assert index.resolved_revision == hub.sha
    assert json.loads(path.read_text())["resolved_revision"] == hub.sha  # persisted: guarded from now on
    _forget_open_indexes()
    hub.sha = "commit-2"
    with pytest.raises(RuntimeError, match="file index was built at revision"):
        FileIndex.open(REPO, REV, "data/*.parquet", index_dir, None)


# --- the save clock -----------------------------------------------------------------------------------------------------


class _ManualClock:
    """
    Hand-advanced stand-in for time.monotonic (injected as FileIndex.clock).
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _index_on_manual_clock(hub: FakeHub, index_dir: Path, files: str) -> tuple[FileIndex, _ManualClock]:
    """
    An open index whose save throttle runs on a hand-advanced clock starting at 0.0.
    """

    clock = _ManualClock()
    for name in files:
        hub.add(f"f/{name}.jsonl", _rows(name, 2))
    index = FileIndex.open(REPO, REV, "f/*.jsonl", index_dir, None)
    index.clock = clock
    index.save()  # aligns the throttle with the manual clock (the last write is now at 0.0)
    return index, clock


def _spy_writes(monkeypatch: pytest.MonkeyPatch, clock: _ManualClock) -> list[float]:
    """
    The clock reading of every index write from here on.
    """

    writes: list[float] = []
    original = FileIndex._write

    def spy(self: FileIndex) -> None:
        writes.append(clock.now)
        original(self)

    monkeypatch.setattr(FileIndex, "_write", spy)
    return writes


def test_index_saves_once_per_read_within_the_save_interval(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, clock = _index_on_manual_clock(hub, tmp_path / "index", "abcd")
    writes = _spy_writes(monkeypatch, clock)
    assert len(_ids(read_rows(index, 0, 100))) == 8  # four files, each read to its end records its row count
    assert writes == [0.0]  # no per-file writes inside the interval; only the read's final save (the backstop)
    assert index.row_counts == {f"f/{n}.jsonl": 2 for n in "abcd"}
    _forget_open_indexes()
    assert FileIndex.open(REPO, REV, "f/*.jsonl", tmp_path / "index", None).row_counts == index.row_counts  # all persisted


def test_index_saves_between_files_once_the_save_interval_passed(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, clock = _index_on_manual_clock(hub, tmp_path / "index", "abc")
    writes = _spy_writes(monkeypatch, clock)
    step = hub_files.INDEX_SAVE_INTERVAL_SECONDS + 1.0

    def advance(file: str) -> None:
        clock.now += step

    assert len(_ids(read_rows(index, 0, 100, on_file=advance))) == 6
    # each file's record found the interval passed and wrote; the finally saved once more at the end
    assert writes == [step, 2 * step, 3 * step, 3 * step]


def test_every_supported_format_has_a_reader() -> None:
    """
    The dispatch table and the recognised suffixes must agree: a format added to one without the other either
    fails here or raises loudly (file_format / the dispatch), so no format can bypass the contract.
    """

    assert set(hub_files.FORMAT_READERS) == set(hub_files.FORMATS)


@pytest.mark.parametrize("suffix", hub_files.FORMATS)
def test_reading_contract_bounds_and_projects_every_format(hub: FakeHub, suffix: str) -> None:
    """
    Every supported format through the shared dispatch: batches of at most batch_size rows, every row
    projected to columns, order preserved across skip, columns=None keeps every column.
    """

    rows = [{"id": f"r{i}", "text": f"doc {i}", "extra": i} for i in range(7)]
    hub.add(f"rows{suffix}", rows)
    path = hub.files[f"rows{suffix}"]
    with path.open("rb") as handle:
        batches = list(iter_row_batches(handle, f"rows{suffix}", skip=1, columns=["id"], batch_size=3))
    assert all(len(batch) <= 3 for batch in batches)
    assert [row for batch in batches for row in batch] == [{"id": f"r{i}"} for i in range(1, 7)]
    with path.open("rb") as handle:
        assert [r for b in iter_row_batches(handle, f"rows{suffix}", batch_size=3) for r in b] == rows


def test_dispatch_enforces_the_contract_centrally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A format reader only decodes: the dispatch itself projects (a reader cannot forget it).
    """

    path = tmp_path / "x.jsonl"
    rows = [{"id": f"r{i}", "extra": i} for i in range(5)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def whole_file(handle: BinaryIO, name: str, skip: int, columns: list[str] | None, batch_size: int) -> Iterator[list[Row]]:
        yield [json.loads(line) for line in handle.read().decode().splitlines()]  # ignores columns AND the bound

    monkeypatch.setitem(hub_files.FORMAT_READERS, ".jsonl", whole_file)
    with path.open("rb") as handle:  # the dispatch projects for the reader
        assert list(iter_row_batches(handle, "x.jsonl", columns=["id"], batch_size=5)) == [[{"id": f"r{i}"} for i in range(5)]]


def test_parquet_iter_stops_before_later_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.parquet"
    pq.write_table(pa.Table.from_pylist(_rows("r", 9)), path, row_group_size=3)
    read = _spy_read_row_group(monkeypatch)
    parquet = pq.ParquetFile(path)
    rows: Generator[Row, None, None] = iter_parquet(parquet, skip=4)
    assert next(rows)["id"] == "r4"
    rows.close()
    assert [group for group, _ in read] == [1]
    assert parquet_row_groups(parquet) == [3, 3, 3]


def test_parquet_row_groups_are_decoded_in_bounded_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A row group is handed out in ROW_BATCH slices instead of one python list; the rows and their order are
    exactly those of a full read, and the projection still applies.
    """

    path = tmp_path / "big.parquet"
    rows = _rows("r", 25)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=10)  # 3 row groups: 10, 10, 5
    assert parquet_row_groups(pq.ParquetFile(path)) == [10, 10, 5]
    sizes: list[int] = []
    original = pq.ParquetFile.iter_batches

    def spy(self: pq.ParquetFile, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        for batch in original(self, **kwargs):
            sizes.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    monkeypatch.setattr(hub_files, "ROW_BATCH", 4)
    assert list(iter_file(path, "big.parquet")) == pq.read_table(path).to_pylist() == rows
    assert sizes == [4, 4, 2, 4, 4, 2, 4, 1]  # never a whole row group at once
    sizes.clear()
    assert list(iter_file(path, "big.parquet", skip=13)) == rows[13:]  # a skip landing inside a batch
    assert sizes == [4, 4, 2, 4, 1]  # group 0 is skipped whole, group 1 is decoded from its start
    assert list(iter_file(path, "big.parquet", skip=13, columns=["id"])) == [{"id": r["id"]} for r in rows[13:]]


# --- over-read: a remote row group is kept whole ------------------------------------------------------------------------


def _spy_read_row_group(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, list[str] | None]]:
    """
    Record every (row group, columns) pulled through ParquetFile.iter_batches.
    """

    calls: list[tuple[int, list[str] | None]] = []
    original = pq.ParquetFile.iter_batches

    def spy(self: pq.ParquetFile, row_groups: list[int], columns: list[str] | None = None, **kwargs: Any) -> Any:
        calls.append((row_groups[0], columns))
        return original(self, row_groups=row_groups, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    return calls


def test_remote_parquet_keeps_the_row_group_whole_and_never_rereads_it(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "books.parquet"
    pq.write_table(pa.Table.from_pylist(_big_rows("r", 300)), path, row_group_size=100)  # 3 groups x 100 rows
    hub.files["data/books.parquet"] = path
    spans = _group_spans(path)
    size = path.stat().st_size
    index_dir = tmp_path / "index"
    src = _src(load_kwargs={"data_files": "data/*.parquet", **REMOTE})
    calls = _spy_read_row_group(monkeypatch)
    load = LOADERS["hf_files"]
    # 11 rows wanted -> the whole first row group, and only that group was pulled
    assert _ids(load(src, 0, 11, SharedLoaderParameters(index_dir=index_dir))) == [f"r{i}" for i in range(100)]
    assert [g for g, _ in calls] == [0]
    assert _touched_groups(hub.handles["data/books.parquet"].ranges, spans, size) == {0}
    # the top-up starts at the boundary: group 0 is never read again
    calls.clear()
    assert _ids(load(src, 100, 3, SharedLoaderParameters(index_dir=index_dir))) == [f"r{i}" for i in range(100, 200)]
    assert [g for g, _ in calls] == [1]
    assert _touched_groups(hub.handles["data/books.parquet"].ranges, spans, size) == {1}
    # exact count on request
    calls.clear()
    assert _ids(load(src, 0, 11, SharedLoaderParameters(index_dir=index_dir, align_to_row_group=False))) == [f"r{i}" for i in range(11)]
    assert [g for g, _ in calls] == [0]
    assert _touched_groups(hub.handles["data/books.parquet"].ranges, spans, size) == {0}
    # cached files stay exact
    cached = _src(load_kwargs={"data_files": "data/*.parquet", "max_cached_file_mb": 1024})
    assert _ids(load(cached, 95, 11, SharedLoaderParameters(index_dir=index_dir))) == [f"r{i}" for i in range(95, 106)]


def test_columns_are_projected_for_parquet(hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    hub.add("data/a.parquet", _rows("a", 3))
    calls = _spy_read_row_group(monkeypatch)
    rows = list(LOADERS["hf_files"](_src(), 0, 2, SharedLoaderParameters(columns=["id"])))
    assert rows == [{"id": "a0"}, {"id": "a1"}]
    assert calls == [(0, ["id"])]
    calls.clear()
    remote = _src(load_kwargs={"data_files": "data/*.parquet", **REMOTE})
    assert list(LOADERS["hf_files"](remote, 2, 1, SharedLoaderParameters(columns=["text"]))) == [{"text": "a doc 2"}]
    assert calls == [(1, ["text"])]
    assert set(next(iter(LOADERS["hf_files"](_src(), 0, 1)))) == {"id", "text", "language"}  # None: every column


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zst", ".jsonl.gz", ".json.gz", ".json"])
def test_columns_are_projected_for_the_json_formats(hub: FakeHub, suffix: str) -> None:
    """
    The json family parses whole rows and drops the surplus columns afterwards, cached and streamed alike.
    """

    hub.add(f"f/a{suffix}", _rows("a", 4))
    for load_kwargs in ({"data_files": f"f/*{suffix}"}, {"data_files": f"f/*{suffix}", **REMOTE}):
        src = _src(load_kwargs=load_kwargs)
        assert list(LOADERS["hf_files"](src, 1, 2, SharedLoaderParameters(columns=["id"]))) == [{"id": "a1"}, {"id": "a2"}]
        assert list(LOADERS["hf_files"](src, 0, 1, SharedLoaderParameters(columns=["text", "id"]))) == [{"text": "a doc 0", "id": "a0"}]
        # a column the row does not have stays absent (the caller's own "row has no <column>" check still fires)
        assert list(LOADERS["hf_files"](src, 0, 1, SharedLoaderParameters(columns=["id", "missing"]))) == [{"id": "a0"}]
        assert set(next(iter(LOADERS["hf_files"](src, 0, 1)))) == {"id", "text", "language"}  # None: every column


def test_projection_keeps_a_mixed_type_surplus_column_out_of_the_shard_writer(hub: FakeHub, tmp_path: Path) -> None:
    """
    Why the json path must project: a surplus column whose type varies from row to row (a string here, a list
    there) makes the shard writer fail on every retry, so the source could never complete.
    """

    hub.add("f/a.jsonl", [
        {"id": "a0", "text": "a doc 0", "meta": "a string"},
        {"id": "a1", "text": "a doc 1", "meta": ["a", "list"]},
    ])
    src = _src(load_kwargs={"data_files": "f/*.jsonl"})
    projected = list(LOADERS["hf_files"](src, 0, 2, SharedLoaderParameters(columns=["text"])))
    assert projected == [{"text": "a doc 0"}, {"text": "a doc 1"}]
    pq.write_table(pa.Table.from_pylist(projected), tmp_path / "shard.parquet")
    with pytest.raises(pa.ArrowException):
        pa.Table.from_pylist(list(LOADERS["hf_files"](src, 0, 2)))


def test_github_code_keeps_matching_rows_of_the_row_group_and_seeks_by_group_counts(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.add("data/a.parquet", _rows("a", 8, _language))  # row groups of 2: [a0 a1] [a2 a3] [a4 a5] [a6 a7]; Python: a0 a3 a6
    index_dir = tmp_path / "index"
    src = _src(loader="github_code", language="Python", load_kwargs=REMOTE)
    calls = _spy_read_row_group(monkeypatch)
    load = LOADERS["github_code"]
    assert [r["id"] for r in load(src, 0, 2, SharedLoaderParameters(index_dir=index_dir, columns=["id"]))] == ["a0", "a3"]
    assert calls == [(0, ["id", "language"]), (1, ["id", "language"])]  # `language` is added for the match
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    assert saved["group_counts"] == {"language=Python": {"data/a.parquet": [1, 1]}, "language=Java": {"data/a.parquet": [1, 1]}}
    assert saved["full_counts"] == {"data/a.parquet": 2}  # both groups decoded whole: every language counted
    assert "language=Python" not in saved["counts"]  # the file is not finished
    calls.clear()
    assert _ids(load(src, 2, 1, SharedLoaderParameters(index_dir=index_dir))) == ["a6"]  # offset 2 = the two finished groups: seek to group 2
    assert [g for g, _ in calls] == [2, 3]  # group 2 has no Python row, group 3 finishes the file
    saved = json.loads(index_path(index_dir, REPO, REV, "data/*.parquet").read_text())
    assert saved["group_counts"]["language=Python"]["data/a.parquet"] == [1, 1, 0, 1]
    assert saved["counts"]["language=Python"] == {"data/a.parquet": 3} and saved["rows"] == {"data/a.parquet": 8}
    calls.clear()
    assert _ids(load(src, 1, 5, SharedLoaderParameters(index_dir=index_dir, align_to_row_group=False))) == ["a3", "a6"]
    assert [g for g, _ in calls] == [1, 2, 3]  # offset 1 lies in group 1: group 0 is skipped by its recorded count


# --- github_code group reads (several languages of one repo in one pass) --------------------------------------------


def _three_languages(i: int) -> str:
    return ("Python", "Java", "Go")[i % 3]


def _group_ids(pairs: Iterator[tuple[str, Row]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, row in pairs:
        out.setdefault(name, []).append(row["id"])
    return out


def test_group_read_reads_every_row_group_once_and_matches_separate_loads(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.add("data/a.parquet", _rows("a", 8, _three_languages))  # groups of 2; Python a0 a3 a6, Java a1 a4 a7, Go a2 a5
    hub.add("data/b.parquet", _rows("b", 8, _three_languages))
    sources = {lang: _src(loader="github_code", language=lang, load_kwargs=REMOTE) for lang in ("Python", "Java", "Go")}
    wanted = {"Python": (0, 4), "Java": (1, 2), "Go": (0, 5)}

    expected = {lang: _ids(LOADERS["github_code"](sources[lang], *wanted[lang], SharedLoaderParameters(index_dir=tmp_path / "separate"))) for lang in sources}
    assert expected == {"Python": ["a0", "a3", "a6", "b0"], "Java": ["a4", "a7"], "Go": ["a2", "a5", "b2", "b5"]}

    calls = _spy_read_row_group(monkeypatch)
    hub.streams.clear()
    requests = [GithubCodeRequest(lang, sources[lang], *wanted[lang]) for lang in sources]
    got = _group_ids(read_github_code_group(requests, SharedLoaderParameters(index_dir=tmp_path / "group", columns=["id"])))
    # every member gets its separate read as a prefix and reads on while the pass runs for Go (which wants more
    # than the repo has: read to the end), so Python and Java collect their surplus rows of every later group
    assert {lang: rows[: wanted[lang][1]] for lang, rows in got.items()} == expected
    assert got == {"Python": ["a0", "a3", "a6", "b0", "b3", "b6"], "Java": ["a4", "a7", "b1", "b4", "b7"], "Go": expected["Go"]}
    # each file opened once, each of its four row groups read once
    assert hub.streams == ["data/a.parquet", "data/b.parquet"]
    assert calls == [(g, ["id", "language"]) for g in range(4)] * 2
    # the shared index recorded every language's counts, per row group and per file
    saved = json.loads(index_path(tmp_path / "group", REPO, REV, "data/*.parquet").read_text())
    assert saved["counts"]["language=Python"] == {"data/a.parquet": 3, "data/b.parquet": 3}
    assert saved["group_counts"]["language=Java"]["data/a.parquet"] == [1, 0, 1, 1]
    assert saved["counts"]["language=Go"] == {"data/a.parquet": 2, "data/b.parquet": 2}
    assert saved["full_counts"] == {"data/a.parquet": 4, "data/b.parquet": 4}


def test_group_top_up_reads_only_the_row_groups_still_needed(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub.add("data/a.parquet", _rows("a", 8, _three_languages))
    hub.add("data/b.parquet", _rows("b", 8, _three_languages))
    index_dir = tmp_path / "index"
    python = _src(loader="github_code", language="Python", load_kwargs=REMOTE)
    java = _src(loader="github_code", language="Java", load_kwargs=REMOTE)
    first = [GithubCodeRequest("py", python, 0, 3), GithubCodeRequest("java", java, 0, 3)]
    assert _group_ids(read_github_code_group(first, SharedLoaderParameters(index_dir=index_dir))) == {"py": ["a0", "a3", "a6"], "java": ["a1", "a4", "a7"]}
    calls = _spy_read_row_group(monkeypatch)
    hub.streams.clear()
    # python continues at 3 (b0 ...), java is satisfied: only file b is opened, starting at its first group
    top_up = [GithubCodeRequest("py", python, 3, 1), GithubCodeRequest("java", java, 3, 0)]
    assert _group_ids(read_github_code_group(top_up, SharedLoaderParameters(index_dir=index_dir))) == {"py": ["b0"]}
    assert hub.streams == ["data/b.parquet"] and [g for g, _ in calls] == [0]


def test_group_exhausted_language_does_not_stop_the_others(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 6, _three_languages))  # Python a0 a3, Java a1 a4, Go a2 a5
    sources = {lang: _src(loader="github_code", language=lang, load_kwargs={}) for lang in ("Python", "Java", "Rust")}
    requests = [GithubCodeRequest(lang, src, 0, 2) for lang, src in sources.items()]
    got = _group_ids(read_github_code_group(requests, SharedLoaderParameters(index_dir=tmp_path / "index")))
    assert got == {"Python": ["a0", "a3"], "Java": ["a1", "a4"]}  # Rust: nothing, the others are complete
    saved = json.loads(index_path(tmp_path / "index", REPO, REV, "data/*.parquet").read_text())
    assert saved["counts"]["language=Rust"] == {"data/a.parquet": 0}


def test_group_rejects_mixed_repos_and_duplicate_languages(hub: FakeHub) -> None:
    hub.add("data/a.parquet", _rows("a", 3, _three_languages))
    python = _src(loader="github_code", language="Python", load_kwargs={})
    other_repo = _src(loader="github_code", language="Java", load_kwargs={"data_files": "data/a.parquet"})
    with pytest.raises(ValueError, match="share hf_id, revision and data_files"):
        list(read_github_code_group([GithubCodeRequest("p", python, 0, 1), GithubCodeRequest("j", other_repo, 0, 1)]))
    with pytest.raises(ValueError, match="distinct languages"):
        list(read_github_code_group([GithubCodeRequest("p", python, 0, 1), GithubCodeRequest("q", python, 0, 1)]))
    assert list(read_github_code_group([])) == []


def test_glob_regex_does_not_cross_directories() -> None:
    from data_preparation.lib.sources.hub_files import glob_regex

    assert glob_regex("data/*.parquet").fullmatch("data/x.parquet")
    assert not glob_regex("data/*.parquet").fullmatch("data/sub/x.parquet"), "fnmatch would match this"
    assert not glob_regex("*.parquet").fullmatch("data/x.parquet") and glob_regex("*.parquet").fullmatch("x.parquet")
    assert glob_regex("**/*.parquet").fullmatch("data/sub/x.parquet") and glob_regex("**/*.parquet").fullmatch("x.parquet")
    assert glob_regex("data/**").fullmatch("data/sub/x.parquet")
    assert glob_regex("sample/10BT/0?0_00000.parquet").fullmatch("sample/10BT/000_00000.parquet")
    assert not glob_regex("sample/10BT/0?0_00000.parquet").fullmatch("sample/10BT/0/0_00000.parquet")
    assert glob_regex("MetaMathQA-395K.json").fullmatch("MetaMathQA-395K.json") and not glob_regex("a.json").fullmatch("a_json")
    assert glob_regex("data/[ab]*.parquet").fullmatch("data/b1.parquet") and not glob_regex("data/[ab]*.parquet").fullmatch("data/c1.parquet")
    assert glob_regex("[!a]*").fullmatch("b.txt") and not glob_regex("[!a]*").fullmatch("a.txt"), "a negated class, not a literal !"


# --- passive members, discovery and the all-language counts -------------------------------------------------------------


def _github_sources(*languages: str) -> dict[str, SourceConfig]:
    return {lang: _src(loader="github_code", language=lang, load_kwargs=REMOTE) for lang in languages}


def test_group_passive_member_collects_only_from_groups_read_for_others(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A passive member takes its language's rows from every row group the active members read and never makes one
    be read; on a later pass it aligns by the recorded counts and collects on from where it stopped.
    """

    hub.add("data/a.parquet", _rows("a", 8, _three_languages))  # groups of 2; Python a0 a3 a6, Java a1 a4 a7, Go a2 a5
    sources = _github_sources("Python", "Java")
    index_dir = tmp_path / "index"
    calls = _spy_read_row_group(monkeypatch)
    first = [GithubCodeRequest("py", sources["Python"], 0, 2), GithubCodeRequest("java", sources["Java"], 0, 0, passive=True)]
    got = _group_ids(read_github_code_group(first, SharedLoaderParameters(index_dir=index_dir, columns=["id"])))
    assert got == {"py": ["a0", "a3"], "java": ["a1"]}  # Python is satisfied at the end of group 1: the pass ends there
    assert [g for g, _ in calls] == [0, 1]  # Java's rows in groups 2 and 3 caused no read
    calls.clear()
    second = [GithubCodeRequest("py", sources["Python"], 2, 1), GithubCodeRequest("java", sources["Java"], 1, 0, passive=True)]
    got = _group_ids(read_github_code_group(second, SharedLoaderParameters(index_dir=index_dir, columns=["id"])))
    assert got == {"py": ["a6"], "java": ["a4", "a7"]}  # Java's offset 1 lies at the start of group 2: aligned
    assert [g for g, _ in calls] == [2, 3]


def test_group_passive_member_is_detached_when_a_group_it_needs_is_skipped(hub: FakeHub, tmp_path: Path) -> None:
    """
    A passive member whose position lies in a row group the pass does not decode takes nothing (its rows must
    stay a contiguous prefix of the language's order), even from the groups decoded later.
    """

    hub.add("data/a.parquet", _rows("a", 8, _three_languages))
    sources = _github_sources("Python", "Java", "Go")
    index_dir = tmp_path / "index"
    first = [GithubCodeRequest("py", sources["Python"], 0, 1), GithubCodeRequest("java", sources["Java"], 0, 0, passive=True)]
    assert _group_ids(read_github_code_group(first, SharedLoaderParameters(index_dir=index_dir, columns=["id"]))) == {"py": ["a0"], "java": ["a1"]}
    # Go seeks past group 0 (its recorded count there is 0); Java at offset 0 would need group 0 again: detached
    second = [GithubCodeRequest("go", sources["Go"], 0, 2), GithubCodeRequest("java", sources["Java"], 0, 0, passive=True)]
    assert _group_ids(read_github_code_group(second, SharedLoaderParameters(index_dir=index_dir, columns=["id"]))) == {"go": ["a2", "a5"]}
    # at offset 1 (its position after group 0) Java is aligned with the seek and collects a4 from group 2
    third = [GithubCodeRequest("go", sources["Go"], 0, 2), GithubCodeRequest("java", sources["Java"], 1, 0, passive=True)]
    assert _group_ids(read_github_code_group(third, SharedLoaderParameters(index_dir=index_dir, columns=["id"]))) == {"go": ["a2", "a5"], "java": ["a4"]}


def test_group_discovers_languages_without_a_request(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 8, _three_languages))
    hub.add("data/b.parquet", _rows("b", 8, _three_languages))
    sources = _github_sources("Python")
    offered: list[str] = []

    def discover(language: str) -> Any:
        offered.append(language)
        return None if language == "Go" else language_request(f"extra:{language}", language, 0, 0, passive=True)

    requests = [GithubCodeRequest("py", sources["Python"], 0, 4)]
    got = _group_ids(read_github_code_group(requests, SharedLoaderParameters(index_dir=tmp_path / "index", columns=["id"]), discover=discover))
    # Python reads a whole and b's first group; Java is collected from every one of those groups, Go was declined
    assert got == {"py": ["a0", "a3", "a6", "b0"], "extra:Java": ["a1", "a4", "a7", "b1"]}
    assert offered == ["Java", "Go"]  # once per language


def test_group_discovered_language_aligns_on_a_later_pass_or_is_detached(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 8, _three_languages))
    hub.add("data/b.parquet", _rows("b", 8, _three_languages))
    sources = _github_sources("Python")
    index_dir = tmp_path / "index"

    def discover_at(offset: int) -> Any:
        return lambda language: language_request(f"extra:{language}", language, offset, 0, passive=True) if language == "Java" else None

    first = [GithubCodeRequest("py", sources["Python"], 0, 2)]
    assert _group_ids(read_github_code_group(first, SharedLoaderParameters(index_dir=index_dir, columns=["id"]), discover=discover_at(0))) == {
        "py": ["a0", "a3"], "extra:Java": ["a1"],
    }
    # Python seeks to group 2 of file a; Java at offset 1 (its count in groups 0 and 1 is recorded: 1, 0) is aligned
    second = [GithubCodeRequest("py", sources["Python"], 2, 2)]
    assert _group_ids(read_github_code_group(second, SharedLoaderParameters(index_dir=index_dir, columns=["id"]), discover=discover_at(1))) == {
        "py": ["a6", "b0"], "extra:Java": ["a4", "a7", "b1"],
    }
    # Python seeks into file b; Java at offset 0 would need file a again: detached at discovery, nothing collected
    third = [GithubCodeRequest("py", sources["Python"], 4, 1)]
    assert _group_ids(read_github_code_group(third, SharedLoaderParameters(index_dir=index_dir, columns=["id"]), discover=discover_at(0))) == {
        "py": ["b3"],
    }
    # at its recorded position (4 Java rows: the next one, b4, lies in b's third group) it waits: that pass only
    # decodes b's second group; the pass that reads the third group collects it
    fourth = [GithubCodeRequest("py", sources["Python"], 4, 1)]
    assert _group_ids(read_github_code_group(fourth, SharedLoaderParameters(index_dir=index_dir, columns=["id"]), discover=discover_at(4))) == {
        "py": ["b3"],
    }
    fifth = [GithubCodeRequest("py", sources["Python"], 5, 1)]  # b6 lies in the last group: the third and fourth are read
    assert _group_ids(read_github_code_group(fifth, SharedLoaderParameters(index_dir=index_dir, columns=["id"]), discover=discover_at(4))) == {
        "py": ["b6"], "extra:Java": ["b4", "b7"],
    }


def test_discover_needs_key_of_and_a_passive_keyed_request(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.parquet", _rows("a", 4, _three_languages))
    index = hub_file_index(_src(load_kwargs={"data_files": "data/*.parquet", **REMOTE}), None, SharedLoaderParameters())
    with pytest.raises(ValueError, match="discover needs key_of"):
        list(read_rows_multi(index, [language_request("py", "Python", 0, 1)], discover=lambda key: None))
    bad = read_rows_multi(
        index, [language_request("py", "Python", 0, 2)], key_of=lambda row: f"language={row['language']}",
        discover=lambda key: language_request("x", "Java", 0, 1),  # active, not passive
    )
    with pytest.raises(ValueError, match="must return a passive request"):
        list(bad)


def test_passive_requests_take_nothing_from_streamed_files(hub: FakeHub, tmp_path: Path) -> None:
    hub.add("data/a.jsonl", _rows("a", 6, _three_languages))
    index = hub_file_index(_src(load_kwargs={"data_files": "data/*.jsonl", **REMOTE}), None, SharedLoaderParameters())
    requests = [language_request("py", "Python", 0, 2), language_request("java", "Java", 0, 0, passive=True)]
    pairs = list(read_rows_multi(index, requests, key_of=lambda row: f"language={row['language']}"))
    assert [(name, row["id"]) for name, row in pairs] == [("py", "a0"), ("py", "a3")]


def test_index_records_every_key_of_a_classified_group_and_synthesises_zeros(tmp_path: Path) -> None:
    index = FileIndex(REPO, REV, "data/*.parquet", files=["f"], row_groups={"f": [2, 2, 2]}, path=tmp_path / "index.json")
    index.record_group_keys("f", 0, {"k": 1})
    index.record_group_keys("f", 1, {})
    index.record_group_keys("f", 2, {"k": 2, "z": 1})
    assert index.group_counts == {"k": {"f": [1, 0, 2]}, "z": {"f": [0, 0, 1]}} and index.full_counts == {"f": 3}
    assert index.known_group_counts("never", "f") == [0, 0, 0] and index.count("never", "f") == 0 and index.count("k", "f") == 3
    index.record_complete_file_counts("f", ["never"])
    assert index.keyed_counts == {"k": {"f": 3}, "z": {"f": 1}, "never": {"f": 0}}
    # a group that does not continue the classified prefix records only keys whose own prefix ends right before it
    other = FileIndex(REPO, REV, "data/*.parquet", files=["g"], row_groups={"g": [2, 2, 2]})
    other.record_group_keys("g", 2, {"k": 1})
    assert other.group_counts == {} and other.full_counts == {} and other.count("k", "g") is None
    other.record_group_keys("g", 0, {"k": 1})
    assert other.known_group_counts("k", "g") == [1] and other.count("k", "g") is None  # groups 1 and 2 unknown
    # persisted and loaded; an index file written before the section existed loads with none
    index.save()
    loaded = FileIndex._load(REPO, REV, "data/*.parquet", tmp_path / "index.json")
    assert loaded.full_counts == {"f": 3} and loaded.known_group_counts("never", "f") == [0, 0, 0]
    without = json.loads((tmp_path / "index.json").read_text())
    del without["full_counts"]
    (tmp_path / "old.json").write_text(json.dumps(without))
    assert FileIndex._load(REPO, REV, "data/*.parquet", tmp_path / "old.json").full_counts == {}
