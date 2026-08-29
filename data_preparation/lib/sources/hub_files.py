# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""File-by-file reading of a Hub dataset repo (the ``hf_files`` / ``github_code`` loaders).

Row order = the repo's files matching a glob (``load_kwargs.data_files``, relative to the repo root) sorted by path,
rows in file order. How a file is fetched depends on its size (known from the index, see below):

* **≤ ``max_cached_file_mb``** (default :data:`DEFAULT_MAX_CACHED_FILE_MB`; ``load_kwargs.max_cached_file_mb``
  overrides it per source): ``huggingface_hub.hf_hub_download`` into the Hub cache (``~/.cache/huggingface/hub`` or
  ``HF_HOME`` / ``--cache_dir``, never fetched twice), then read locally.
* **larger parquet files** are never downloaded whole: they are opened remotely (``HfFileSystem``, HTTP range
  requests) and only the row groups covering the requested rows are read — the footer once (its row-group row
  counts go into the index), then ``ParquetFile.read_row_group(i)`` for each needed group. A top-up at a larger
  offset therefore seeks straight to the right row group.
* **larger ``.jsonl`` / ``.jsonl.zst`` / ``.jsonl.gz`` / ``.json.gz`` files** are streamed sequentially from the
  same remote file object (through the zstd/gzip decompressor) and the stream is dropped as soon as enough rows
  were yielded. Their row count is only known once a file was read to its end, so a top-up that starts inside a
  partially consumed file re-streams that file from its start (bounded by one file). Plain ``.json`` arrays above
  the threshold cannot be streamed: use the ``hf_split`` loader for those.

A :class:`FileIndex` per ``(repo, revision, glob)`` remembers the file list, the file sizes (one batched
``HfApi.get_paths_info`` call), the row count of every file read so far and the row-group row counts of every
parquet footer seen, so a fetch at ``offset`` skips whole files without opening them. It is persisted as JSON under
``<index_dir>/<repo>@<revision>/<glob hash>.json`` when an ``index_dir`` is given (``dataset/hub_index/`` in a
build), else kept in memory for the loader call only. Extra per-file counters (``counts[key][file]``, e.g. rows of
one language for ``github_code``) share the index.

Hub access goes through the module-level functions :func:`list_repo_files`, :func:`paths_info`,
:func:`hub_download` and :func:`open_remote` (stubbed by the tests) or through the callables of a
:class:`HubFetcher`, which also holds the size threshold and the :class:`FetchStats` (bytes handed to the reader
by the remote file objects, files downloaded / streamed).
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import pyarrow.parquet as pq

Row = dict[str, Any]
OnFile = Callable[[str], None]

FORMATS: tuple[str, ...] = (".parquet", ".jsonl.zst", ".jsonl.gz", ".json.gz", ".jsonl", ".json")
STREAM_FORMATS: tuple[str, ...] = (".jsonl.zst", ".jsonl.gz", ".json.gz", ".jsonl")

DEFAULT_MAX_CACHED_FILE_MB = 256.0  # files up to this size go through the Hub cache, larger ones are read remotely
PARQUET_BLOCK_SIZE = 1 << 20  # fsspec read-ahead for remote parquet (random access: keep the over-read small)
STREAM_BLOCK_SIZE = 8 << 20  # fsspec read-ahead for sequential remote streams (fewer, larger range requests)
PATHS_INFO_BATCH = 500  # paths per `get_paths_info` request


# --- Hub access (module-level so tests can stub them) --------------------------------------------------------------


def list_repo_files(repo_id: str, revision: str | None, token: str | None) -> list[str]:
    """All file paths of a dataset repo at ``revision`` (``HfApi.list_repo_files``)."""
    from huggingface_hub import HfApi

    files: list[str] = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return files


def paths_info(repo_id: str, paths: list[str], revision: str | None, token: str | None) -> dict[str, int]:
    """Sizes in bytes of the given repo files (batched ``HfApi.get_paths_info``)."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=token)
    sizes: dict[str, int] = {}
    for start in range(0, len(paths), PATHS_INFO_BATCH):
        batch = paths[start : start + PATHS_INFO_BATCH]
        for entry in api.get_paths_info(repo_id, batch, repo_type="dataset", revision=revision):
            if isinstance(entry, RepoFile):
                sizes[entry.path] = int(entry.size)
    missing = [p for p in paths if p not in sizes]
    if missing:
        raise FileNotFoundError(f"{repo_id}@{revision or 'main'}: no size for {missing[:3]}{'...' if len(missing) > 3 else ''}")
    return sizes


def hub_download(repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
    """Download one repo file into the Hub cache (no-op if cached) and return its local path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision, token=token))


def open_remote(repo_id: str, filename: str, revision: str | None, token: str | None, block_size: int) -> BinaryIO:
    """Open one repo file for random-access reading over HTTP (``HfFileSystem``; nothing is cached on disk)."""
    from huggingface_hub import HfFileSystem

    at = f"@{revision}" if revision else ""
    fs = HfFileSystem(token=token)
    # fsspec's file classes derive from io.IOBase and are not declared BinaryIO in its stubs; they are binary files
    return cast(BinaryIO, fs.open(f"datasets/{repo_id}{at}/{filename}", "rb", block_size=block_size))


# --- file index --------------------------------------------------------------------------------------------------------


def index_path(index_dir: Path, repo_id: str, revision: str | None, pattern: str) -> Path:
    """``<index_dir>/<repo with / replaced>@<revision>/<sha256(pattern)[:16]>.json``."""
    repo = repo_id.replace("/", "--")
    return index_dir / f"{repo}@{revision or 'main'}" / f"{hashlib.sha256(pattern.encode()).hexdigest()[:16]}.json"


@dataclass
class FileIndex:
    repo_id: str
    revision: str | None
    pattern: str
    files: list[str] = field(default_factory=list)  # sorted repo paths matching `pattern`
    rows: dict[str, int] = field(default_factory=dict)  # file -> row count, once known
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # key -> file -> matching rows, once known
    sizes: dict[str, int] = field(default_factory=dict)  # file -> bytes (from the Hub listing)
    row_groups: dict[str, list[int]] = field(default_factory=dict)  # parquet file -> rows per row group
    path: Path | None = None  # where the index is persisted (None: in memory)

    @classmethod
    def open(
        cls, repo_id: str, revision: str | None, pattern: str, index_dir: Path | None, token: str | None
    ) -> FileIndex:
        """Load the persisted index or create it (listing the repo once); file list and sizes are cached in it."""
        path = None if index_dir is None else index_path(index_dir, repo_id, revision, pattern)
        if path is not None and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            index = cls(
                repo_id, revision, pattern, data["files"], data["rows"], data.get("counts", {}),
                data.get("sizes", {}), data.get("row_groups", {}), path,
            )
        else:
            files = sorted(f for f in list_repo_files(repo_id, revision, token) if fnmatch.fnmatchcase(f, pattern))
            if not files:
                raise FileNotFoundError(f"{repo_id}@{revision or 'main'}: no files match data_files={pattern!r}")
            index = cls(repo_id, revision, pattern, files, path=path)
        index.ensure_sizes(token)
        index.save()
        return index

    def ensure_sizes(self, token: str | None) -> None:
        """Fetch the sizes of files not yet in the index (one batched call; indexes written before sizes existed)."""
        missing = [f for f in self.files if f not in self.sizes]
        if missing:
            self.sizes.update(paths_info(self.repo_id, missing, self.revision, token))

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "pattern": self.pattern,
            "files": self.files,
            "rows": self.rows,
            "counts": self.counts,
            "sizes": self.sizes,
            "row_groups": self.row_groups,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def count(self, key: str | None, file: str) -> int | None:
        """Known row count of ``file`` (``key=None``: all rows; else the ``counts[key]`` counter), or None."""
        if key is None:
            return self.rows.get(file)
        return self.counts.get(key, {}).get(file)

    def record(self, key: str | None, file: str, value: int) -> None:
        if key is None:
            self.rows[file] = value
        else:
            self.counts.setdefault(key, {})[file] = value
        self.save()

    def record_row_groups(self, file: str, groups: list[int]) -> None:
        """Store a parquet file's row-group row counts (and thereby its total row count)."""
        self.row_groups[file] = groups
        self.rows[file] = sum(groups)
        self.save()


# --- fetching -----------------------------------------------------------------------------------------------------------


@dataclass
class FetchStats:
    bytes_read: int = 0  # bytes handed to the reader by remote file objects (read-ahead not included)
    files_downloaded: int = 0  # files fetched whole into the Hub cache (or already there)
    files_streamed: int = 0  # files opened remotely


class _CountingRaw(io.RawIOBase, BinaryIO):
    """Raw file over a binary file object that adds every byte read to ``stats.bytes_read``."""

    def __init__(self, inner: BinaryIO, stats: FetchStats) -> None:
        super().__init__()
        self._inner = inner
        self._stats = stats

    def readinto(self, buffer: Any) -> int:
        data = self._inner.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        self._stats.bytes_read += n
        return n

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return bool(self._inner.seekable())

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._inner.seek(offset, whence)

    def tell(self) -> int:
        return self._inner.tell()

    def close(self) -> None:
        if not self.closed:
            self._inner.close()
        super().close()


HubDownload = Callable[[str, str, str | None, str | None], Path]
OpenRemote = Callable[[str, str, str | None, str | None, int], BinaryIO]


@dataclass
class HubFetcher:
    """Decides per file between the Hub cache and remote reading and opens it; ``download`` / ``remote`` default to
    the module-level :func:`hub_download` / :func:`open_remote` (resolved at call time so tests can stub either)."""

    token: str | None = None
    max_cached_file_mb: float = DEFAULT_MAX_CACHED_FILE_MB
    download: HubDownload | None = None
    remote: OpenRemote | None = None
    stats: FetchStats = field(default_factory=FetchStats)

    def uses_cache(self, size: int) -> bool:
        return size <= self.max_cached_file_mb * 1024 * 1024

    @contextmanager
    def open(self, index: FileIndex, file: str, fmt: str) -> Iterator[BinaryIO]:
        """A binary, seekable file object for ``file``: the cached local copy or the remote file."""
        size = index.sizes[file]
        if self.uses_cache(size):
            path = (self.download or hub_download)(index.repo_id, file, index.revision, self.token)
            self.stats.files_downloaded += 1
            with path.open("rb") as fh:
                yield fh
            return
        if fmt == ".json":
            raise ValueError(
                f"{index.repo_id}: {file} is a plain .json array of {size / 2**20:.0f} MB, above max_cached_file_mb="
                f"{self.max_cached_file_mb:g} and it cannot be streamed; use the hf_split loader for this source"
                " or raise load_kwargs.max_cached_file_mb"
            )
        block_size = PARQUET_BLOCK_SIZE if fmt == ".parquet" else STREAM_BLOCK_SIZE
        raw = (self.remote or open_remote)(index.repo_id, file, index.revision, self.token, block_size)
        self.stats.files_streamed += 1
        counting = _CountingRaw(raw, self.stats)
        if fmt == ".parquet":  # random access: pyarrow reads exact column-chunk ranges, no extra buffering
            with counting:
                yield counting
            return
        with io.BufferedReader(counting, buffer_size=1 << 16) as buffered:  # sequential text decoding
            yield buffered


# --- per-format readers -----------------------------------------------------------------------------------------------


def file_format(name: str) -> str:
    """The recognised suffix of ``name`` (longest match of :data:`FORMATS`) or a clear error."""
    lower = name.lower()
    for suffix in FORMATS:
        if lower.endswith(suffix):
            return suffix
    raise ValueError(f"unsupported file format {name!r}; supported: {', '.join(FORMATS)}")


def parquet_row_groups(parquet: pq.ParquetFile) -> list[int]:
    """Rows per row group from the footer (no data read)."""
    return [int(parquet.metadata.row_group(i).num_rows) for i in range(parquet.num_row_groups)]


def iter_parquet(parquet: pq.ParquetFile, skip: int = 0) -> Generator[Row, None, None]:
    """Rows of an open parquet file in order, skipping the first ``skip``; row groups are read one at a time and
    only from the first one that holds a wanted row on (a consumer that stops early never touches later groups)."""
    for group, group_rows in enumerate(parquet_row_groups(parquet)):
        if skip >= group_rows:
            skip -= group_rows
            continue
        rows = parquet.read_row_group(group).to_pylist()
        yield from rows[skip:]
        skip = 0


def iter_stream(handle: BinaryIO, name: str, skip: int = 0) -> Iterator[Row]:
    """Rows of an open binary file in order, skipping the first ``skip`` (parquet skips whole row groups)."""
    fmt = file_format(name)
    if fmt == ".parquet":
        yield from iter_parquet(pq.ParquetFile(handle), skip)
        return
    if fmt == ".json":
        data = json.loads(handle.read().decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{name}: plain .json must contain a JSON array of rows")
        for row in data[skip:]:
            yield row
        return
    yield from _iter_json_lines(handle, fmt, skip)


def iter_file(path: Path, name: str, skip: int = 0) -> Iterator[Row]:
    """:func:`iter_stream` over a local file."""
    with path.open("rb") as handle:
        yield from iter_stream(handle, name, skip)


def _iter_json_lines(handle: BinaryIO, fmt: str, skip: int) -> Iterator[Row]:
    if fmt == ".jsonl.zst":
        import zstandard

        with zstandard.ZstdDecompressor().stream_reader(handle, closefd=False) as reader:
            yield from _lines(io.TextIOWrapper(reader, encoding="utf-8"), skip)
    elif fmt in (".jsonl.gz", ".json.gz"):
        with gzip.GzipFile(fileobj=handle, mode="rb") as unzipped:
            yield from _lines(io.TextIOWrapper(unzipped, encoding="utf-8"), skip)
    else:
        text = io.TextIOWrapper(handle, encoding="utf-8")
        try:
            yield from _lines(text, skip)
        finally:
            text.detach()  # the caller owns `handle`


def _lines(lines: Any, skip: int) -> Iterator[Row]:
    seen = 0
    for line in lines:
        if not line.strip():
            continue
        if seen < skip:
            seen += 1
            continue
        yield json.loads(line)


# --- the reader --------------------------------------------------------------------------------------------------------


def read_rows(
    index: FileIndex,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    on_file: OnFile | None = None,
    key: str | None = None,
    match: Callable[[Row], bool] | None = None,
    fetcher: HubFetcher | None = None,
) -> Iterator[Row]:
    """Rows ``offset..offset+count`` (counting rows that pass ``match``) across the index's files.

    Files whose known row count (``index.count(key, file)``) lies entirely before ``offset`` are skipped without
    being opened; every file read through to its end records its count (``key`` for the matching rows, and the
    total row count) so the next call can skip it. Parquet files record their row-group layout as soon as their
    footer was read, so a file that lies entirely before ``offset`` is skipped even if it was never read.
    ``fetcher`` (default: a :class:`HubFetcher` with ``token``) chooses cache vs. remote reading per file.
    """
    if count <= 0:
        return
    fetcher = HubFetcher(token=token) if fetcher is None else fetcher
    remaining_skip = offset
    taken = 0
    for file in index.files:
        known = index.count(key, file)
        if known is not None and remaining_skip >= known:
            remaining_skip -= known
            continue
        fmt = file_format(file)
        if on_file is not None:
            on_file(file)
        with fetcher.open(index, file, fmt) as handle:
            rows_iter: Iterator[Row]
            if fmt == ".parquet":
                parquet = pq.ParquetFile(handle)
                if index.row_groups.get(file) is None:
                    index.record_row_groups(file, parquet_row_groups(parquet))
                if known is None and key is None:
                    known = index.rows[file]
                    if remaining_skip >= known:
                        remaining_skip -= known
                        continue
                rows_iter = iter_parquet(parquet, remaining_skip if match is None else 0)
            else:
                rows_iter = iter_stream(handle, file, remaining_skip if match is None else 0)
            file_skip = remaining_skip if match is None else 0
            total_rows = 0
            matched = 0
            completed = True
            for row in rows_iter:
                total_rows += 1
                if match is not None and not match(row):
                    continue
                matched += 1
                if match is not None and remaining_skip > 0:
                    remaining_skip -= 1
                    continue
                remaining_skip = 0
                yield dict(row)
                taken += 1
                if taken >= count:
                    completed = False
                    break
        if completed:
            if match is None:
                index.record(None, file, file_skip + total_rows)
            else:
                index.record(None, file, total_rows)
                index.record(key, file, matched)
            remaining_skip = 0
        if taken >= count:
            return
