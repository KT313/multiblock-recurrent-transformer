# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""File-by-file reading of a Hub dataset repo (the ``hf_files`` / ``github_code`` loaders).

Row order = the repo's files matching a glob (``load_kwargs.data_files``, relative to the repo root) sorted by path,
rows in file order. How a file is fetched depends on its size (known from the index, see below):

* **≤ ``max_cached_file_mb``** (default :data:`DEFAULT_MAX_CACHED_FILE_MB`; ``load_kwargs.max_cached_file_mb``
  overrides it per source): ``huggingface_hub.hf_hub_download`` into the Hub cache (``~/.cache/huggingface/hub`` or
  ``HF_HOME`` / ``--cache_dir``, never fetched twice), then read locally.
* **larger parquet files** are never downloaded whole: they are opened remotely (``HfFileSystem``, HTTP range
  requests) and only the row groups covering the requested rows are read — the footer once (its row-group row
  counts go into the index), then ``ParquetFile.read_row_group(i, columns=...)`` for each needed group. A row
  group that was fetched is **kept whole**: ``count`` is a minimum and the reader keeps yielding until the end of
  the row group that satisfied it (``align_to_row_group=True``), so the rows a top-up needs next are already on disk
  and the same bytes are never downloaded twice (row groups can be large for book-like sources: gutenberg is
  ~300 MB per 1,000 rows). A top-up at a larger offset seeks straight to the right row group.
* **larger ``.jsonl`` / ``.jsonl.zst`` / ``.jsonl.gz`` / ``.json.gz`` / ``.json`` files** are streamed sequentially
  from the same remote file object (through the zstd/gzip decompressor; a plain ``.json`` array is parsed
  incrementally with ``ijson``, see :func:`iter_json_array`) and the stream is dropped as soon as exactly ``count``
  rows were yielded (a stream has no cheap unit to finish; the rest of the file could be gigabytes). Their row
  count is only known once a file was read to its end, so a top-up that starts inside a partially consumed file
  re-streams that file from its start (bounded by one file; for a ``.json`` array that is a sequential prefix
  read, so a top-up of the first few hundred rows of a 400 MB file costs a few MB, not the file).

A :class:`FileIndex` per ``(repo, revision, glob)`` remembers the file list, the file sizes (one batched
``HfApi.get_paths_info`` call), the row count of every file read so far and the row-group row counts of every
parquet footer seen, so a fetch at ``offset`` skips whole files without opening them. It is persisted as JSON under
``<index_dir>/<repo>@<revision>/<glob hash>.json`` when an ``index_dir`` is given (``dataset/hub_index/`` in a
build), else kept in memory for the loader call only. Extra per-file counters (``counts[key][file]``, e.g. rows of
one language for ``github_code``) share the index, together with the matching rows per row group of every parquet
row group read so far under that key (``group_counts[key][file]``), so a keyed fetch at an offset also seeks
straight to the right row group instead of re-reading the file from its start.

Files that go through the Hub cache (and local files) are read with exact ``count`` semantics: over-reading a
cached file costs nothing on the wire, so nothing needs to be kept.

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
import itertools
import json
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import pyarrow.parquet as pq

Row = dict[str, Any]
OnFile = Callable[[str], None]
RowFilter = Callable[[Row], bool]

FORMATS: tuple[str, ...] = (".parquet", ".jsonl.zst", ".jsonl.gz", ".json.gz", ".jsonl", ".json")
STREAM_FORMATS: tuple[str, ...] = (".jsonl.zst", ".jsonl.gz", ".json.gz", ".jsonl")

DEFAULT_MAX_CACHED_FILE_MB = 32.0  # files up to this size go through the Hub cache whole; larger ones are read remotely by row group / streamed (a 240 MB parquet file for 20 rows is not worth caching)
PARQUET_BLOCK_SIZE = 1 << 20  # fsspec read-ahead for remote parquet (random access: keep the over-read small)
STREAM_BLOCK_SIZE = 8 << 20  # fsspec read-ahead for sequential remote streams (fewer, larger range requests)
STREAM_BUFFER_SIZE = 1 << 16  # local buffer in front of a remote stream (json lines / ijson decode from it)
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
        shown = f"{missing[:3]}..." if len(missing) > 3 else f"{missing}"
        raise FileNotFoundError(f"{repo_id}@{revision or 'main'}: no size for {shown}")
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
    pattern_hash = hashlib.sha256(pattern.encode()).hexdigest()[:16]
    return index_dir / f"{repo}@{revision or 'main'}" / f"{pattern_hash}.json"


_OPEN_INDEXES: dict[Path, FileIndex] = {}  # persisted indexes opened in this process, by path
_OPEN_INDEXES_LOCK = threading.Lock()


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
    # key -> parquet file -> matching rows per row group, for the prefix of row groups read so far under `key`
    group_counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    path: Path | None = None  # where the index is persisted (None: in memory)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def open(
        cls, repo_id: str, revision: str | None, pattern: str, index_dir: Path | None, token: str | None
    ) -> FileIndex:
        """Load the persisted index or create it (listing the repo once); file list and sizes are cached in it.
        A persisted index is one process-wide instance per path, so concurrent readers of the same repo files
        (the build runs items in threads) share it and its saves never overwrite each other's counts."""
        path = None if index_dir is None else index_path(index_dir, repo_id, revision, pattern)
        if path is None:
            index = cls._from_repo_listing(repo_id, revision, pattern, None, token)
            index.ensure_sizes(token)
            return index
        with _OPEN_INDEXES_LOCK:
            cached = _OPEN_INDEXES.get(path)
            if cached is not None:
                index = cached
            elif path.is_file():
                index = cls._load(repo_id, revision, pattern, path)
            else:
                index = cls._from_repo_listing(repo_id, revision, pattern, path, token)
            _OPEN_INDEXES[path] = index
        index.ensure_sizes(token)
        index.save()
        return index

    @classmethod
    def _load(cls, repo_id: str, revision: str | None, pattern: str, path: Path) -> FileIndex:
        """The index persisted at ``path`` (older files may lack the optional sections; they default to empty)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            repo_id=repo_id,
            revision=revision,
            pattern=pattern,
            files=data["files"],
            rows=data["rows"],
            counts=data.get("counts", {}),
            sizes=data.get("sizes", {}),
            row_groups=data.get("row_groups", {}),
            group_counts=data.get("group_counts", {}),
            path=path,
        )

    @classmethod
    def _from_repo_listing(
        cls, repo_id: str, revision: str | None, pattern: str, path: Path | None, token: str | None
    ) -> FileIndex:
        """A fresh index: list the repo once and keep the files matching ``pattern``, sorted by path."""
        all_files = list_repo_files(repo_id, revision, token)
        files = sorted(f for f in all_files if fnmatch.fnmatchcase(f, pattern))
        if not files:
            raise FileNotFoundError(f"{repo_id}@{revision or 'main'}: no files match data_files={pattern!r}")
        return cls(repo_id, revision, pattern, files, path=path)

    def ensure_sizes(self, token: str | None) -> None:
        """Fetch the sizes of files not yet in the index (one batched call; indexes written before sizes existed)."""
        missing = [f for f in self.files if f not in self.sizes]
        if missing:
            self.sizes.update(paths_info(self.repo_id, missing, self.revision, token))

    def save(self) -> None:
        """Write the index to ``path`` atomically (no-op for an in-memory index); serialised per instance."""
        if self.path is None:
            return
        with self._lock:
            self._write()

    def _write(self) -> None:
        assert self.path is not None
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
            "group_counts": self.group_counts,
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
        """Store the row count of ``file`` (``key=None``: all rows; else the ``counts[key]`` counter) and save."""
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

    def known_group_counts(self, key: str | None, file: str) -> list[int]:
        """Rows per row group of ``file`` that count towards ``key`` (``key=None``: the footer's row counts; else the
        matching rows of the row groups already read under ``key``, a prefix of the file's groups)."""
        if key is None:
            return self.row_groups.get(file, [])
        return self.group_counts.get(key, {}).get(file, [])

    def record_group_counts(self, key: str, file: str, groups: list[int]) -> None:
        """Store the matching rows per row group read so far under ``key`` (a prefix of the file's groups) and save."""
        self.group_counts.setdefault(key, {})[file] = groups
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
        """Whether a file of ``size`` bytes is fetched whole into the Hub cache (else it is read remotely)."""
        return size <= self.max_cached_file_mb * 1024 * 1024

    @contextmanager
    def open(self, index: FileIndex, file: str, fmt: str) -> Iterator[BinaryIO]:
        """A binary, seekable file object for ``file``: the cached local copy or the remote file."""
        if self.uses_cache(index.sizes[file]):
            with self._open_cached(index, file) as handle:
                yield handle
        else:
            with self._open_remote(index, file, fmt) as handle:
                yield handle

    @contextmanager
    def _open_cached(self, index: FileIndex, file: str) -> Iterator[BinaryIO]:
        download = self.download or hub_download
        path = download(index.repo_id, file, index.revision, self.token)
        self.stats.files_downloaded += 1
        with path.open("rb") as handle:
            yield handle

    @contextmanager
    def _open_remote(self, index: FileIndex, file: str, fmt: str) -> Iterator[BinaryIO]:
        open_file = self.remote or open_remote
        block_size = PARQUET_BLOCK_SIZE if fmt == ".parquet" else STREAM_BLOCK_SIZE
        raw = open_file(index.repo_id, file, index.revision, self.token, block_size)
        self.stats.files_streamed += 1
        counting = _CountingRaw(raw, self.stats)
        if fmt == ".parquet":
            # random access: pyarrow reads exact column-chunk ranges itself, no extra buffering wanted
            with counting:
                yield counting
        else:
            # sequential decoding (json lines / ijson): read through a local buffer
            with io.BufferedReader(counting, buffer_size=STREAM_BUFFER_SIZE) as buffered:
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


def iter_parquet(parquet: pq.ParquetFile, skip: int = 0, columns: list[str] | None = None) -> Generator[Row, None, None]:
    """Rows of an open parquet file in order, skipping the first ``skip``; row groups are read one at a time and
    only from the first one that holds a wanted row on (a consumer that stops early never touches later groups).
    ``columns`` projects the read (None: every column)."""
    for group, group_rows in enumerate(parquet_row_groups(parquet)):
        if skip >= group_rows:  # every row of this group is skipped: do not read it
            skip -= group_rows
            continue
        rows = read_row_group(parquet, group, columns)
        yield from rows[skip:]
        skip = 0


def read_row_group(parquet: pq.ParquetFile, group: int, columns: list[str] | None) -> list[Row]:
    """``parquet.read_row_group(group, columns=columns)`` as dict rows (the single place that pulls row-group bytes)."""
    if columns is None:
        return parquet.read_row_group(group).to_pylist()
    return parquet.read_row_group(group, columns=columns).to_pylist()


def iter_stream(handle: BinaryIO, name: str, skip: int = 0) -> Iterator[Row]:
    """Rows of an open binary file in order, skipping the first ``skip`` (parquet skips whole row groups)."""
    fmt = file_format(name)
    if fmt == ".parquet":
        yield from iter_parquet(pq.ParquetFile(handle), skip)
    elif fmt == ".json":
        yield from iter_json_array(handle, name, skip)
    else:
        yield from _iter_json_lines(handle, fmt, skip)


def iter_file(path: Path, name: str, skip: int = 0) -> Iterator[Row]:
    """:func:`iter_stream` over a local file."""
    with path.open("rb") as handle:
        yield from iter_stream(handle, name, skip)


def iter_json_array(handle: BinaryIO, name: str, skip: int = 0) -> Iterator[Row]:
    """Elements of a top-level JSON array parsed incrementally with ``ijson`` (the C ``yajl2_c`` backend when it is
    installed, else ijson's pure-Python one), skipping the first ``skip``. Only the bytes up to the last element
    consumed are read, so a consumer that stops early never pulls the rest of the file; this is the single ``.json``
    code path for cached and remote files alike (a cached file is read from disk in 64 KB chunks the same way).
    A file whose top-level value is not an array (e.g. an object) is a clear error."""
    import ijson  # in the `data` extra; ijson ships no type information (mypy override in pyproject.toml)

    events = ijson.parse(handle, use_float=True)  # reads 64 KB chunks: the granularity of an early stop
    try:
        first_event = next(events)
    except ijson.IncompleteJSONError as error:  # empty or truncated file
        raise ValueError(f"{name}: plain .json must contain a JSON array of rows (empty file)") from error
    first_event_type = first_event[1]
    if first_event_type != "start_array":
        top_level = first_event_type.removeprefix("start_")
        raise ValueError(f"{name}: plain .json must contain a JSON array of rows (top-level {top_level})")
    all_events = itertools.chain([first_event], events)  # give the consumed event back to ijson
    for position, row in enumerate(ijson.items(all_events, "item")):
        if position < skip:
            continue
        if not isinstance(row, dict):
            raise ValueError(f"{name}: element {position} of the JSON array is not an object")
        yield cast(Row, row)


def _iter_json_lines(handle: BinaryIO, fmt: str, skip: int) -> Iterator[Row]:
    """Rows of a (possibly compressed) json-lines file, skipping the first ``skip`` non-empty lines."""
    if fmt == ".jsonl.zst":
        import zstandard

        with zstandard.ZstdDecompressor().stream_reader(handle, closefd=False) as decompressed:
            yield from _parse_json_lines(io.TextIOWrapper(decompressed, encoding="utf-8"), skip)
    elif fmt in (".jsonl.gz", ".json.gz"):
        with gzip.GzipFile(fileobj=handle, mode="rb") as decompressed:
            yield from _parse_json_lines(io.TextIOWrapper(decompressed, encoding="utf-8"), skip)
    else:  # plain .jsonl
        text = io.TextIOWrapper(handle, encoding="utf-8")
        try:
            yield from _parse_json_lines(text, skip)
        finally:
            text.detach()  # closing the wrapper would close `handle`, which the caller owns


def _parse_json_lines(lines: Any, skip: int) -> Iterator[Row]:
    """One JSON object per non-empty line, skipping the first ``skip`` of them."""
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        if skipped < skip:
            skipped += 1
            continue
        yield json.loads(line)


# --- the reader --------------------------------------------------------------------------------------------------------


@dataclass
class ReadRequest:
    """One consumer of :func:`read_rows_multi`: at least ``count`` rows passing ``match`` (all rows without it), the
    first ``offset`` such rows skipped. ``key`` is where the index records the per-file / per-row-group counts of
    matching rows (``counts[key]`` / ``group_counts[key]``); a request with ``match`` but no ``key`` records nothing
    and cannot skip files. ``name`` tags the rows it receives."""

    name: str
    offset: int
    count: int
    key: str | None = None
    match: RowFilter | None = None


@dataclass
class _Cursor:
    """Mutable position of one :class:`ReadRequest` while :func:`read_rows_multi` runs.

    ``remaining_skip`` counts rows still to skip before the first yielded row — plain rows without ``match``,
    matching rows with it; it carries across files (a file with fewer matching rows than the skip only shrinks it).
    """

    request: ReadRequest
    remaining_skip: int
    taken: int = 0  # rows yielded so far

    @property
    def name(self) -> str:
        return self.request.name

    @property
    def match(self) -> RowFilter | None:
        return self.request.match

    @property
    def record_key(self) -> str | None:
        """Where matching-row counts are recorded: ``key`` for filtered reads, nothing otherwise (plain row counts
        come from footers and full reads regardless of the request)."""
        return self.request.key if self.request.match is not None else None

    @property
    def satisfied(self) -> bool:
        return self.taken >= self.request.count

    def wants(self, row: Row) -> bool:
        return self.match is None or self.match(row)

    def skip_whole(self, rows: int) -> bool:
        """Skip a unit (file / row group) of ``rows`` (matching) rows if the skip position lies at or beyond its
        end; return whether it was skipped."""
        if self.remaining_skip < rows:
            return False
        self.remaining_skip -= rows
        return True

    def known_count(self, index: FileIndex, file: str) -> int | None:
        """The known number of rows of ``file`` that count for this request, or None."""
        if self.match is None:
            return index.count(None, file)
        if self.request.key is None:
            return None
        return index.count(self.request.key, file)

    def known_group_counts(self, index: FileIndex, file: str) -> list[int]:
        """Rows per row group of ``file`` that count for this request, for the prefix of groups whose count is
        known (every group from the footer for a plain read)."""
        if self.match is None:
            return index.known_group_counts(None, file)
        if self.request.key is None:
            return []
        return index.known_group_counts(self.request.key, file)


def read_rows(
    index: FileIndex,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    on_file: OnFile | None = None,
    key: str | None = None,
    match: RowFilter | None = None,
    fetcher: HubFetcher | None = None,
    columns: list[str] | None = None,
    align_to_row_group: bool = True,
) -> Iterator[Row]:
    """Rows from ``offset`` on (counting rows that pass ``match``) across the index's files: **at least** ``count``
    of them when the source has that many — :func:`read_rows_multi` with a single request.

    ``count`` is exact for files read from the Hub cache and for remote streams. For a parquet file read remotely
    with ``align_to_row_group`` (the default) the reader finishes the row group in which it reached ``count`` — the
    bytes were already fetched, so keeping the rows means a later fetch at the resulting offset never downloads
    them again; ``align_to_row_group=False`` stops at exactly ``count`` rows. ``columns`` projects parquet reads
    (other formats yield every column).

    Files whose known row count (``index.count(key, file)``) lies entirely before ``offset`` are skipped without
    being opened; every file read through to its end records its count (``key`` for the matching rows, and the
    total row count) so the next call can skip it. Parquet files record their row-group layout as soon as their
    footer was read, so a file that lies entirely before ``offset`` is skipped even if it was never read, and
    with ``key`` the matching rows of every row group read so far, so a keyed fetch seeks to the right row group
    too. ``fetcher`` (default: a :class:`HubFetcher` with ``token``) chooses cache vs. remote reading per file.
    """
    request = ReadRequest(name="", offset=offset, count=count, key=key, match=match)
    for _, row in read_rows_multi(
        index, [request], token=token, on_file=on_file, fetcher=fetcher, columns=columns,
        align_to_row_group=align_to_row_group,
    ):
        yield row


def read_rows_multi(
    index: FileIndex,
    requests: list[ReadRequest],
    *,
    token: str | None = None,
    on_file: OnFile | None = None,
    fetcher: HubFetcher | None = None,
    columns: list[str] | None = None,
    align_to_row_group: bool = True,
) -> Iterator[tuple[str, Row]]:
    """Serve several :class:`ReadRequest` in **one pass** over the index's files: every file and every parquet row
    group is opened / read at most once and each row is handed to every request that wants it, as
    ``(request.name, row)`` pairs. A file or row group is skipped when no request that still needs rows has to look
    at it (each request seeks by its own known counts, see :func:`read_rows`); a request that reached its
    ``count`` stops taking rows (at the end of the current remote row group with ``align_to_row_group``) while the
    others read on, and the pass ends when every request is satisfied or the files are exhausted. Everything else
    (``count`` semantics, what gets recorded in the index, ``columns``) is as for :func:`read_rows`, per request.
    """
    if fetcher is None:
        fetcher = HubFetcher(token=token)
    cursors = [_Cursor(request=r, remaining_skip=r.offset) for r in requests if r.count > 0]

    for file in index.files:
        readers = [c for c in cursors if not c.satisfied]
        if not readers:
            return
        readers = [c for c in readers if not _skip_file_if_count_known(index, c, file)]
        if not readers:
            continue
        fmt = file_format(file)
        if on_file is not None:
            on_file(file)
        with fetcher.open(index, file, fmt) as handle:
            if fmt == ".parquet":
                parquet = pq.ParquetFile(handle)
                if index.row_groups.get(file) is None:
                    index.record_row_groups(file, parquet_row_groups(parquet))  # the footer told us the row count
                # a plain request may find that the whole file lies before its offset after all (footer only)
                readers = [c for c in readers if c.match is not None or not _skip_file_if_count_known(index, c, file)]
                if not readers:
                    continue
                is_remote = not fetcher.uses_cache(index.sizes[file])
                finish_group = align_to_row_group and is_remote
                yield from _parquet_rows(parquet, index, file, readers, columns, finish_group)
            else:
                yield from _stream_rows(handle, index, file, readers)


def _skip_file_if_count_known(index: FileIndex, cursor: _Cursor, file: str) -> bool:
    """Skip ``file`` for ``cursor`` without opening it when its (matching) row count is known and lies entirely
    before the cursor's skip position; return whether it was skipped."""
    known = cursor.known_count(index, file)
    if known is None:
        return False
    return cursor.skip_whole(known)


@dataclass
class _ParquetReader:
    """One request's progress through one parquet file (see :func:`_parquet_rows`)."""

    cursor: _Cursor
    known_group_counts: list[int]  # (matching) rows per row group, the prefix known so far
    first_group: int  # the first row group this request has to read (earlier ones were skipped by known counts)
    matched_in_group: int = 0
    reading: bool = True  # False once the request stopped taking rows from this file

    def finished_group(self, index: FileIndex, file: str, group: int) -> None:
        """Record the matching rows of a row group read completely (extends the known prefix by one group)."""
        record_key = self.cursor.record_key
        if record_key is not None and group == len(self.known_group_counts):
            self.known_group_counts.append(self.matched_in_group)
            index.record_group_counts(record_key, file, self.known_group_counts)


def _parquet_rows(
    parquet: pq.ParquetFile,
    index: FileIndex,
    file: str,
    cursors: list[_Cursor],
    columns: list[str] | None,
    finish_group: bool,
) -> Iterator[tuple[str, Row]]:
    """Rows of one parquet file for several requests: a row group is read (once) when at least one request needs
    it, row groups that every request skips by its known counts are never read. With ``finish_group`` a request
    that reaches ``count`` still takes the rest of the row group. Keyed requests record the matching rows of every
    row group they read completely; a request that reads the file to its end records the file's count."""
    groups = index.row_groups[file]  # rows per row group, from the footer
    readers: list[_ParquetReader] = []
    for cursor in cursors:
        known = list(cursor.known_group_counts(index, file))
        first_group = 0
        for rows_in_group in known:
            if not cursor.skip_whole(rows_in_group):
                break
            first_group += 1
        readers.append(_ParquetReader(cursor, known, first_group))

    for group in range(min(r.first_group for r in readers), len(groups)):
        participants = [r for r in readers if r.reading and r.first_group <= group]
        if not participants:
            if not any(r.reading for r in readers):
                return
            continue  # the group lies before the first group of every request still reading

        for reader in participants:
            reader.matched_in_group = 0
        for row in read_row_group(parquet, group, columns):
            for reader in participants:
                cursor = reader.cursor
                if not reader.reading or not cursor.wants(row):
                    continue
                reader.matched_in_group += 1
                if cursor.remaining_skip > 0:
                    cursor.remaining_skip -= 1
                    continue
                yield cursor.name, dict(row)
                cursor.taken += 1
                if cursor.satisfied and not finish_group:
                    reader.reading = False  # stopped in the middle of the group: its count stays unknown

        is_last_group = group == len(groups) - 1
        for reader in participants:
            if not reader.reading:
                continue
            reader.finished_group(index, file, group)
            if reader.cursor.satisfied and not is_last_group:
                reader.reading = False  # done; later row groups of this file were not read
        if not any(r.reading for r in readers):
            return

    # Requests still reading went through the last group (or skipped every group by known counts): the file's
    # matching rows are known now.
    for reader in readers:
        record_key = reader.cursor.record_key
        if reader.reading and record_key is not None:
            index.record(record_key, file, sum(reader.known_group_counts))


def _stream_rows(handle: BinaryIO, index: FileIndex, file: str, cursors: list[_Cursor]) -> Iterator[tuple[str, Row]]:
    """Rows of one non-parquet file for several requests, each exactly up to its ``count``; the stream is dropped
    as soon as every request is satisfied. The file is decoded from its first row (a stream has no cheap way to
    skip, and only a full read tells how many rows it holds), so a file read to its end records its row count and,
    for every keyed request that read it through, its matching rows."""
    reading = list(cursors)
    matched = {c.name: 0 for c in cursors}
    rows_seen = 0
    for row in iter_stream(handle, file, 0):
        rows_seen += 1
        for cursor in list(reading):
            if not cursor.wants(row):
                continue
            matched[cursor.name] += 1
            if cursor.remaining_skip > 0:
                cursor.remaining_skip -= 1
                continue
            yield cursor.name, dict(row)
            cursor.taken += 1
            if cursor.satisfied:
                reading.remove(cursor)
        if not reading:
            return

    # The whole file was read: record what we learned about it.
    index.record(None, file, rows_seen)
    for cursor in reading:
        if cursor.record_key is not None:
            index.record(cursor.record_key, file, matched[cursor.name])
