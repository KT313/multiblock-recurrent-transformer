# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
File-by-file reading of a Hub dataset repo (the hf_files / github_code loaders).

Row order = the repo's files matching a glob (load_kwargs.data_files, relative to the repo root) sorted by path,
rows in file order. How a file is fetched depends on its size (known from the index):

* ≤ max_cached_file_mb (default :data:`DEFAULT_MAX_CACHED_FILE_MB`; load_kwargs.max_cached_file_mb
  overrides it per source): hf_hub_download into the Hub cache (never fetched twice), then read locally.
* larger parquet files are opened remotely (HfFileSystem, HTTP range requests) and only the row groups
  covering the requested rows are read, each decoded in :data:`ROW_BATCH` slices. A fetched row group is kept
  whole (count is a minimum, align_to_row_group=True), so the same bytes are never downloaded twice; a
  top-up at a larger offset seeks straight to the right row group.
* larger .jsonl / .jsonl.zst / .jsonl.gz / .json.gz / .json files are streamed from the
  start (a .json array incrementally with ijson, :func:`iter_json_array`; a .json.gz is an array or json lines,
  told apart by its first byte) and dropped after exactly count rows. Their row count is only known once read
  to the end, so a top-up inside a partially consumed file re-streams that file's prefix.

A :class:`FileIndex` per (repo, revision, glob) remembers the file list and sizes (one batched
HfApi.get_paths_info call), the row count of every file read so far and the row-group row counts of every
parquet footer seen, so a fetch at offset skips whole files without opening them. It records the commit the
file list was taken at; an index loaded from disk is only valid while the repo still resolves to that commit (a
moved repo is a hard error: offsets counted against the old listing would skip or duplicate rows). It is
persisted as JSON under <index_dir>/<repo>@<revision>/<glob hash>.json when an index_dir is given, else
kept in memory. Per-key counters (keyed_counts[key][file], group_counts[key][file], e.g. rows of one language
for github_code) share the index.

Files read from the Hub cache (and local files) use exact count semantics: over-reading them costs nothing.

Several requests share one pass (:func:`read_rows_multi`). A request that reached its count in a remote parquet
file turns *passive*: it keeps taking the rows it matches from every row group the pass reads for the other
requests (the bytes are fetched anyway) but never causes one to be read, and a request may be passive from the
start (a folder that is complete but should keep growing while the pass runs). A row whose key (`key_of`) no
request matches can create a passive request on first sight (`discover`). Passive rows are only taken while the
request is *aligned*, i.e. its position was carried through every earlier row group, by a known count or by
reading it in this pass; a passive request that would have to skip an unread row group or file is detached for
the rest of the pass, so a folder fed passively is always a contiguous prefix of its source order. With
`key_of`, a fully decoded row group records the rows of *every* key in the index (`group_counts`) and
`full_counts[file]` says how many leading groups of a file were classified that way, so a key absent from those
groups has a known count of zero there; that is what lets a passive request align on a later pass.

Hub access goes through :func:`repo_listing`, :func:`resolve_revision`, :func:`paths_info`, :func:`hub_download`
and :func:`open_remote` (stubbed by the tests) or the callables of a :class:`HubFetcher`, which also holds the
size threshold and the :class:`FetchStats`. Every Hub request is bounded by :data:`HUB_REQUEST_TIMEOUT`
(:func:`configure_hub_http`): huggingface_hub's shared HTTP client has no timeout of its own, and a connection
into a dead tunnel would otherwise wait forever.
"""

from __future__ import annotations

import gzip
import hashlib
import functools
import io
import itertools
import json
import re
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import pyarrow.parquet as pq

from data_preparation.lib.storage.atomic import write_atomically

Row = dict[str, Any]
RowBatch = list[Row]
OnFile = Callable[[str], None]
RowFilter = Callable[[Row], bool]

FORMATS: tuple[str, ...] = (".parquet", ".jsonl.zst", ".jsonl.gz", ".json.gz", ".jsonl", ".json")
STREAM_FORMATS: tuple[str, ...] = (".jsonl.zst", ".jsonl.gz", ".json.gz", ".jsonl")

DEFAULT_MAX_CACHED_FILE_MB = 32.0  # files up to this size go through the Hub cache whole; larger ones are read remotely by row group / streamed
PARQUET_BLOCK_SIZE = 1 << 20  # fsspec read-ahead for remote parquet (random access: keep the over-read small)
ROW_BATCH = 1000  # rows decoded from a parquet row group at a time (a whole group as python dicts can be hundreds of MB)
STREAM_BLOCK_SIZE = 8 << 20  # fsspec read-ahead for sequential remote streams (fewer, larger range requests)
STREAM_BUFFER_SIZE = 1 << 16  # local buffer in front of a remote stream (json lines / ijson decode from it)
PATHS_INFO_BATCH = 500  # paths per `get_paths_info` request
INDEX_SAVE_INTERVAL_SECONDS = 30.0  # how often at most a persisted FileIndex is rewritten while reading (`_write_if_due`)


# --- Hub access (module-level so tests can stub them) --------------------------------------------------------------


HUB_REQUEST_TIMEOUT = 30.0  # seconds to connect, and between two reads, of any Hub request (listing, sizes, downloads)


@functools.cache
def configure_hub_http() -> None:
    """
    Bound every request of huggingface_hub's shared HTTP client by :data:`HUB_REQUEST_TIMEOUT` (once per process).
    The library passes its own, shorter timeouts to range reads and cache downloads; the size lookup
    (HfApi.get_paths_info) relies on the client's, which is unset by default. The repo listing
    (:func:`repo_listing`) gets the timeout as an explicit argument instead: HfApi.dataset_info passes its own
    default timeout=None down to the client, and an explicit None disables the client's timeout in httpx.
    """

    from huggingface_hub import get_session

    get_session().timeout = HUB_REQUEST_TIMEOUT


def repo_listing(repo_id: str, revision: str | None, token: str | None) -> tuple[list[str], str]:
    """
    All file paths of a dataset repo at revision plus the commit hash that revision resolved to.

    One HfApi.dataset_info call gives both (the sibling list is the file list, sha the resolved commit), so
    recording the commit alongside the listing costs no extra request.
    """

    from huggingface_hub import HfApi

    configure_hub_http()
    info = HfApi(token=token).dataset_info(repo_id, revision=revision, timeout=HUB_REQUEST_TIMEOUT)
    if info.sha is None:
        raise RuntimeError(f"{repo_id}@{revision or 'main'}: the Hub returned no commit hash for the listing")
    return [sibling.rfilename for sibling in info.siblings or []], str(info.sha)


def resolve_revision(repo_id: str, revision: str | None, token: str | None) -> str:
    """
    The commit hash revision currently resolves to (the default branch's head when unset).
    """

    return repo_listing(repo_id, revision, token)[1]


def paths_info(repo_id: str, paths: list[str], revision: str | None, token: str | None) -> dict[str, int]:
    """
    Sizes in bytes of the given repo files (batched HfApi.get_paths_info).
    """

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    configure_hub_http()
    api = HfApi(token=token)
    sizes: dict[str, int] = {}
    for start in range(0, len(paths), PATHS_INFO_BATCH):
        batch = paths[start : start + PATHS_INFO_BATCH]
        for entry in api.get_paths_info(repo_id, batch, repo_type="dataset", revision=revision):
            if isinstance(entry, RepoFile):
                sizes[entry.path] = int(entry.size)
    missing = [path for path in paths if path not in sizes]
    if missing:
        shown = f"{missing[:3]}..." if len(missing) > 3 else f"{missing}"
        raise FileNotFoundError(f"{repo_id}@{revision or 'main'}: no size for {shown}")
    return sizes


def hub_download(repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
    """
    Download one repo file into the Hub cache (no-op if cached) and return its local path.
    """

    from huggingface_hub import hf_hub_download

    configure_hub_http()
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision, token=token))


def open_remote(repo_id: str, filename: str, revision: str | None, token: str | None, block_size: int) -> BinaryIO:
    """
    Open one repo file for random-access reading over HTTP (HfFileSystem; nothing is cached on disk).
    """

    from huggingface_hub import HfFileSystem

    configure_hub_http()
    revision_suffix = f"@{revision}" if revision else ""
    filesystem = HfFileSystem(token=token)
    # fsspec's file classes derive from io.IOBase and are not declared BinaryIO in its stubs; they are binary files
    return cast(BinaryIO, filesystem.open(f"datasets/{repo_id}{revision_suffix}/{filename}", "rb", block_size=block_size))


# --- file index --------------------------------------------------------------------------------------------------------


def index_path(index_dir: Path, repo_id: str, revision: str | None, pattern: str) -> Path:
    """
    <index_dir>/<repo with / replaced>@<revision>/<sha256(pattern)[:16]>.json.
    """

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
    row_counts: dict[str, int] = field(default_factory=dict)  # file -> row count, once known
    keyed_counts: dict[str, dict[str, int]] = field(default_factory=dict)  # key -> file -> matching rows, once known
    sizes: dict[str, int] = field(default_factory=dict)  # file -> bytes (from the Hub listing)
    row_groups: dict[str, list[int]] = field(default_factory=dict)  # parquet file -> rows per row group
    # key -> parquet file -> matching rows per row group, for the prefix of row groups read so far under `key`
    group_counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    # parquet file -> leading row groups whose rows were all classified by key (`record_group_keys`): a key with
    # a shorter `group_counts` prefix had zero rows in the groups it lacks (`known_group_counts` fills them in)
    full_counts: dict[str, int] = field(default_factory=dict)
    resolved_revision: str | None = None  # commit hash the file list was taken at (None: pre-recording index file)
    path: Path | None = None  # where the index is persisted (None: in memory)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)  # injected by tests
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _last_write: float = field(default=0.0, repr=False, compare=False)  # `clock()` at the last write

    @classmethod
    def open(
        cls, repo_id: str, revision: str | None, pattern: str, index_dir: Path | None, token: str | None
    ) -> FileIndex:
        """
        Load the persisted index or create it (listing the repo once); file list and sizes are cached in it.
        A persisted index is one process-wide instance per path, so concurrent readers of the same repo files
        (the build runs items in threads) share it; its mutations and saves are serialised by _lock. An index
        loaded from disk is checked against the repo's current revision resolution (a moved repo is an error, see
        :meth:`_check_revision`); the process-cached instance was checked when it was first opened.
        """

        path = None if index_dir is None else index_path(index_dir, repo_id, revision, pattern)
        if path is None:
            index = cls._from_repo_listing(repo_id, revision, pattern, None, token)
            index.ensure_sizes(token)
            return index
        with _OPEN_INDEXES_LOCK:
            cached = _OPEN_INDEXES.get(path)
        if cached is not None:
            index = cached
        else:
            # the Hub round-trip (the listing, or the revision check of a loaded index) runs outside the lock: one
            # stalled request must not hold up every other open of the process
            if path.is_file():
                index = cls._load(repo_id, revision, pattern, path)
                index._check_revision(token)
            else:
                index = cls._from_repo_listing(repo_id, revision, pattern, path, token)
            with _OPEN_INDEXES_LOCK:
                index = _OPEN_INDEXES.setdefault(path, index)  # another thread may have published it meanwhile
        index.ensure_sizes(token)
        index.save()
        return index

    @classmethod
    def _load(cls, repo_id: str, revision: str | None, pattern: str, path: Path) -> FileIndex:
        """
        The index persisted at path (older files may lack the optional sections; they default to empty).
        """

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            repo_id=repo_id,
            revision=revision,
            pattern=pattern,
            files=data["files"],
            row_counts=data["rows"],
            keyed_counts=data.get("counts", {}),
            sizes=data.get("sizes", {}),
            row_groups=data.get("row_groups", {}),
            group_counts=data.get("group_counts", {}),
            full_counts=data.get("full_counts", {}),
            resolved_revision=data.get("resolved_revision"),
            path=path,
        )

    def _check_revision(self, token: str | None) -> None:
        """
        Fail if the repo no longer resolves to the commit the file list was taken at.

        The file order and every per-file row count are only valid for that exact listing (raw-folder offsets
        were counted against it), so silently re-listing a moved repo would skip or duplicate rows. This is the
        one place a loaded index costs a network call. An index written before the commit was recorded stores
        none: it adopts the current resolution once without erroring (failing would break every existing
        dataset/hub_index/ tree) and is guarded from then on.
        """

        current = resolve_revision(self.repo_id, self.revision, token)
        if self.resolved_revision is None:
            self.resolved_revision = current  # one-time upgrade of a pre-recording index; persisted by open()'s save
            return
        if self.resolved_revision != current:
            # Both ways out re-download the source: `revision` is part of the raw fingerprint, so pinning it marks
            # the raw folder stale (the repair step deletes and downloads it again after confirmation); re-listing
            # at the new head needs the raw folder deleted as well, its offsets were counted against the old listing.
            raise RuntimeError(
                f"{self.repo_id}: the file index was built at revision {self.resolved_revision} but the repo now "
                f"resolves to {current}. Either way the source's raw folder is downloaded again: pin "
                f"`revision: {self.resolved_revision}` in the source config to keep the listed commit (revision is "
                f"part of the raw fingerprint, so raw/<source> becomes stale and the repair step re-downloads it "
                f"after confirmation), or delete {self.path} together with the source's raw folder to re-list at "
                f"{current} (the raw offsets were counted against the old listing)."
            )

    @classmethod
    def _from_repo_listing(
        cls, repo_id: str, revision: str | None, pattern: str, path: Path | None, token: str | None
    ) -> FileIndex:
        """
        A fresh index: list the repo once and keep the files matching pattern, sorted by path, together
        with the commit the listing resolved to (the same call yields both).
        """

        all_files, resolved = repo_listing(repo_id, revision, token)
        matcher = glob_regex(pattern)
        files = sorted(path for path in all_files if matcher.fullmatch(path))
        if not files:
            raise FileNotFoundError(f"{repo_id}@{revision or 'main'}: no files match data_files={pattern!r}")
        return cls(repo_id, revision, pattern, files, resolved_revision=resolved, path=path)

    def ensure_sizes(self, token: str | None) -> None:
        """
        Fetch the sizes of files not yet in the index (one batched call; indexes written before sizes existed).
        """

        missing = [file for file in self.files if file not in self.sizes]
        if missing:
            sizes = paths_info(self.repo_id, missing, self.revision, token)
            with self._lock:
                self.sizes.update(sizes)

    def save(self) -> None:
        """
        Write the index to path atomically (no-op for an in-memory index); serialised per instance.

        This is the unconditional backstop of the throttled :meth:`_write_if_due`: `open()` calls it, and so does
        the finally of :func:`read_rows_multi`, which also runs when a download completes or the stop flag
        makes the consumer close the row generator early.
        """

        if self.path is None:
            return
        with self._lock:
            self._write()

    def _write(self) -> None:
        """
        Write the JSON (caller holds _lock; no-op for an in-memory index).
        """

        if self.path is None:
            return
        payload = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "pattern": self.pattern,
            "files": self.files,
            "rows": self.row_counts,
            "counts": self.keyed_counts,
            "sizes": self.sizes,
            "row_groups": self.row_groups,
            "group_counts": self.group_counts,
            "full_counts": self.full_counts,
        }
        with write_atomically(self.path) as tmp:
            tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        self._last_write = self.clock()

    def _write_if_due(self) -> None:
        """
        Write only when :data:`INDEX_SAVE_INTERVAL_SECONDS` have passed since the last write (caller holds
        _lock).

        The index is a cache of learned row counts (several MB of JSON for a repo of ~12,000 files) and writing
        it after every finished file rewrote the whole file once per file. Batching on a clock loses at most the
        last interval's counts in a crash, and a lost count is only re-learned by re-reading that file: never
        wrong data, just a bounded re-read. :meth:`save` is the unconditional backstop at the end of every read.
        """

        if self.clock() - self._last_write >= INDEX_SAVE_INTERVAL_SECONDS:
            self._write()

    def count(self, key: str | None, file: str) -> int | None:
        """
        Known row count of file (key=None: all rows; else the keyed_counts[key] counter, or the sum over
        the row groups when every group of the file was classified by key, :attr:`full_counts`), or None.
        """

        if key is None:
            return self.row_counts.get(file)
        known = self.keyed_counts.get(key, {}).get(file)
        if known is not None:
            return known
        groups = self.row_groups.get(file)
        if groups is not None and self.full_counts.get(file, 0) >= len(groups):
            return sum(self.known_group_counts(key, file))
        return None

    def record_count(self, key: str | None, file: str, value: int) -> None:
        """
        Store the row count of file (key=None: all rows; else the keyed_counts[key] counter) and save on
        the clock (:meth:`_write_if_due`; the end of the read saves unconditionally).
        """

        with self._lock:
            if key is None:
                self.row_counts[file] = value
            else:
                self.keyed_counts.setdefault(key, {})[file] = value
            self._write_if_due()

    def record_row_groups(self, file: str, groups: list[int]) -> None:
        """
        Store a parquet file's row-group row counts (and thereby its total row count); save on the clock.
        """

        with self._lock:
            self.row_groups[file] = groups
            self.row_counts[file] = sum(groups)
            self._write_if_due()

    def known_group_counts(self, key: str | None, file: str) -> list[int]:
        """
        Rows per row group of file that count towards key (key=None: the footer's row counts; else the
        matching rows of the row groups already read under key, a prefix of the file's groups, extended with
        zeros up to :attr:`full_counts` [file]: groups classified by key in which key did not occur).
        """

        if key is None:
            return self.row_groups.get(file, [])
        prefix = list(self.group_counts.get(key, {}).get(file, []))
        classified = self.full_counts.get(file, 0)
        if len(prefix) < classified:
            prefix.extend([0] * (classified - len(prefix)))
        return prefix

    def record_group_counts(self, key: str, file: str, groups: list[int]) -> None:
        """
        Store (a copy of) the matching rows per row group read so far under key (a prefix of the file's
        groups) in memory; the index is written on the save clock (record / :meth:`_write_if_due`) and when a
        read ends (save), not once per row group: for a repo of hundreds of files and thousands of row groups
        that would rewrite the whole JSON thousands of times.
        """

        with self._lock:
            self.group_counts.setdefault(key, {})[file] = list(groups)

    def record_group_keys(self, file: str, group: int, counts: dict[str, int]) -> None:
        """
        Store the rows of every key of a row group decoded completely under a `key_of` function: each key's
        prefix grows by the group when it ends right before it (zeros first for the classified groups it lacks,
        :attr:`full_counts`), keys with a prefix ending at group but absent from it get a 0, and full_counts
        advances when group continues the classified prefix. In memory; saved on the clock / at the end.
        """

        with self._lock:
            classified = self.full_counts.get(file, 0)
            for key, matching in counts.items():
                prefix = list(self.group_counts.get(key, {}).get(file, []))
                if len(prefix) < group <= classified:
                    prefix.extend([0] * (group - len(prefix)))
                if len(prefix) == group:
                    prefix.append(matching)
                    self.group_counts.setdefault(key, {})[file] = prefix
            for key, per_file in self.group_counts.items():
                absent = per_file.get(file)
                if absent is not None and len(absent) == group and key not in counts:
                    absent.append(0)
            if classified == group:
                self.full_counts[file] = group + 1

    def record_complete_file_counts(self, file: str, keys: Iterable[str] = ()) -> None:
        """
        Record the keyed file count of every key whose per-group prefix covers the whole file (a read that
        went through the last row group made it complete), keys included (request keys that may never have
        occurred: their prefix is all synthesised zeros); saved on the clock.
        """

        groups = self.row_groups.get(file)
        if groups is None:
            return
        with self._lock:
            for key in [*self.group_counts, *keys]:
                if file in self.keyed_counts.get(key, {}):
                    continue
                prefix = self.known_group_counts(key, file)
                if len(prefix) == len(groups):
                    self.keyed_counts.setdefault(key, {})[file] = sum(prefix)
            self._write_if_due()


def glob_regex(pattern: str) -> re.Pattern[str]:
    """
    data_files glob as an anchored regex with Hub semantics: * and ? do not cross /, ** does
    (data/*.parquet matches data/x.parquet but not data/sub/x.parquet; fnmatch would match both).
    """

    parts: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**", i):
            parts.append(".*")
            i += 2
            if i < len(pattern) and pattern[i] == "/":
                parts[-1] = "(?:.*/)?"
                i += 1
            continue
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        elif char == "[":
            end = pattern.find("]", i + 1)
            if end == -1:
                parts.append(re.escape(char))
            else:
                body = pattern[i + 1 : end].replace("\\", "\\\\")
                if body[:1] in ("!", "^"):
                    body = "^" + body[1:]  # a glob negation ([!a]); copied literally, ! would only match itself
                parts.append("[" + body + "]")
                i = end
        else:
            parts.append(re.escape(char))
        i += 1
    return re.compile("".join(parts))


# --- fetching -----------------------------------------------------------------------------------------------------------


@dataclass
class FetchStats:
    """
    The download counters of one fetcher (the dashboard's download row reads bytes_fetched live).
    """

    bytes_fetched: int = 0  # remote reads (read-ahead excluded) + whole files this fetcher downloaded to the cache
    files_downloaded: int = 0  # files fetched whole into the Hub cache (or already there)
    files_streamed: int = 0  # files opened remotely


class _CountingRaw(io.RawIOBase, BinaryIO):
    """
    Raw file over a binary file object that adds every byte read to stats.bytes_fetched.
    """

    def __init__(self, inner: BinaryIO, stats: FetchStats) -> None:
        super().__init__()
        self._inner = inner
        self._stats = stats

    def readinto(self, buffer: Any) -> int:
        data = self._inner.read(len(buffer))
        bytes_read = len(data)
        buffer[:bytes_read] = data
        self._stats.bytes_fetched += bytes_read
        return bytes_read

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
    """
    Decides per file between the Hub cache and remote reading and opens it; download / remote default to
    the module-level :func:`hub_download` / :func:`open_remote` (resolved at call time so tests can stub either).
    """

    token: str | None = None
    max_cached_file_mb: float = DEFAULT_MAX_CACHED_FILE_MB
    download: HubDownload | None = None
    remote: OpenRemote | None = None
    stats: FetchStats = field(default_factory=FetchStats)
    created: float = field(default_factory=time.time)  # a cache file younger than this was downloaded by this fetcher

    def uses_cache(self, size: int) -> bool:
        """
        Whether a file of size bytes is fetched whole into the Hub cache (else it is read remotely).
        """

        return size <= self.max_cached_file_mb * 1024 * 1024

    @contextmanager
    def open(self, index: FileIndex, file: str, fmt: str) -> Iterator[BinaryIO]:
        """
        A binary, seekable file object for file: the cached local copy or the remote file.
        """

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
        status = path.stat()
        if status.st_mtime >= self.created:  # downloaded now, not found in the cache: its bytes were fetched
            self.stats.bytes_fetched += status.st_size
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
#
# One reading contract for every file format, enforced in one place: a per-format reader (FORMAT_READERS) only
# decodes bytes into batches of at most `batch_size` rows, and the shared dispatch `iter_row_batches`, the single
# entry every consumer goes through (`iter_stream` / `iter_file` here, the `local` loader in loaders.py), checks
# the bound and projects every row to the requested columns itself. A reader therefore cannot forget the
# projection (parquet passes `columns` down only so pyarrow prunes the read) and cannot materialise a whole file
# into one batch without the dispatch failing loudly.


def file_format(name: str) -> str:
    """
    The recognised suffix of name (longest match of :data:`FORMATS`) or a clear error.
    """

    lower = name.lower()
    for suffix in FORMATS:
        if lower.endswith(suffix):
            return suffix
    raise ValueError(f"unsupported file format {name!r}; supported: {', '.join(FORMATS)}")


def parquet_row_groups(parquet: pq.ParquetFile) -> list[int]:
    """
    Rows per row group from the footer (no data read).
    """

    return [int(parquet.metadata.row_group(group).num_rows) for group in range(parquet.num_row_groups)]


def iter_parquet(parquet: pq.ParquetFile, skip: int = 0, columns: list[str] | None = None) -> Generator[Row, None, None]:
    """
    Rows of :func:`parquet_batches` one at a time (kept for consumers that want rows, not batches).
    """

    for batch in parquet_batches(parquet, skip, columns, ROW_BATCH):
        yield from batch


def parquet_batches(
    parquet: pq.ParquetFile, skip: int, columns: list[str] | None, batch_size: int
) -> Iterator[RowBatch]:
    """
    Row batches of an open parquet file in order, skipping the first skip rows; row groups are read one at a
    time and only from the first one that holds a wanted row on (a consumer that stops early never touches later
    groups), each decoded in batch_size slices. columns prunes the read (None: every column).
    """

    for group, group_rows in enumerate(parquet_row_groups(parquet)):
        if skip >= group_rows:  # every row of this group is skipped: do not read it
            skip -= group_rows
            continue
        for rows in row_group_batches(parquet, group, columns, batch_size):
            if skip > 0:
                dropped = min(skip, len(rows))
                skip -= dropped
                rows = rows[dropped:]
            if rows:
                yield rows


def read_row_group(parquet: pq.ParquetFile, group: int, columns: list[str] | None) -> Iterator[Row]:
    """
    Rows of :func:`row_group_batches` one at a time, in :data:`ROW_BATCH` slices.
    """

    for batch in row_group_batches(parquet, group, columns, ROW_BATCH):
        yield from batch


def row_group_batches(parquet: pq.ParquetFile, group: int, columns: list[str] | None, batch_size: int) -> Iterator[RowBatch]:
    """
    Row batches of one row group as dicts (the single place that pulls row-group bytes), batch_size rows each.

    A whole row group as a python list is what a book-like source cannot afford (gutenberg row groups hold ~300 MB
    per 1,000 rows and several downloads run at once), so the group is decoded batch by batch
    (ParquetFile.iter_batches(row_groups=[group])) and only one batch of dicts is alive at a time. Rows and
    their order are exactly those of the row group; a consumer that stops early leaves the rest undecoded.
    """

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, row_groups=[group]):
        yield batch.to_pylist()


def project_row(row: Row, columns: list[str] | None) -> Row:
    """
    row reduced to columns (None: the row unchanged).

    A column the row does not have stays absent instead of becoming None, so a caller checking for its own
    column (download.py's text_row) still sees the row as the file had it; parquet raises for an unknown
    column at read time, so neither path invents data.
    """

    if columns is None:
        return row
    return {column: row[column] for column in columns if column in row}


FormatReader = Callable[[BinaryIO, str, int, list[str] | None, int], Iterator[RowBatch]]


def _parquet_batches(handle: BinaryIO, name: str, skip: int, columns: list[str] | None, batch_size: int) -> Iterator[RowBatch]:
    """
    :data:`FORMAT_READERS` entry for parquet: :func:`parquet_batches` over the opened file. columns is
    passed down so pyarrow prunes the read to those columns (and errors on an unknown one); the projection
    guarantee itself lives in :func:`iter_row_batches`.
    """

    yield from parquet_batches(pq.ParquetFile(handle), skip, columns, batch_size)


def _json_array_batches(handle: BinaryIO, name: str, skip: int, columns: list[str] | None, batch_size: int) -> Iterator[RowBatch]:
    """
    :data:`FORMAT_READERS` entry for .json arrays: one row per batch (the parse is sequential, so a
    single-row batch keeps an early stop as cheap as before); the dispatch projects.
    """

    for row in iter_json_array(handle, name, skip):
        yield [row]


def _json_lines_batches(handle: BinaryIO, name: str, skip: int, columns: list[str] | None, batch_size: int) -> Iterator[RowBatch]:
    """
    :data:`FORMAT_READERS` entry for the json-lines family: one row per batch (a remote stream is dropped the
    moment the consumer has enough rows, so nothing may be decoded ahead); the dispatch projects.
    """

    for row in _iter_json_lines(handle, file_format(name), skip):
        yield [row]


def _json_gz_batches(handle: BinaryIO, name: str, skip: int, columns: list[str] | None, batch_size: int) -> Iterator[RowBatch]:
    """
    :data:`FORMAT_READERS` entry for .json.gz, which Hub repos use for both shapes: a JSON array (`[` as the
    first non-blank byte, read like a .json) or json lines (read like a .jsonl.gz). One row per batch, as above.
    """

    with gzip.GzipFile(fileobj=handle, mode="rb") as decompressed:
        if decompressed.peek(64).lstrip().startswith(b"["):
            rows = iter_json_array(cast(BinaryIO, decompressed), name, skip)
        else:
            rows = _parse_json_lines(io.TextIOWrapper(decompressed, encoding="utf-8"), skip)
        for row in rows:
            yield [row]


FORMAT_READERS: dict[str, FormatReader] = {
    ".parquet": _parquet_batches,
    ".jsonl.zst": _json_lines_batches,
    ".jsonl.gz": _json_lines_batches,
    ".json.gz": _json_gz_batches,
    ".jsonl": _json_lines_batches,
    ".json": _json_array_batches,
}


def iter_row_batches(
    handle: BinaryIO, name: str, skip: int = 0, columns: list[str] | None = None, batch_size: int | None = None
) -> Iterator[RowBatch]:
    """
    The reading contract, the single dispatch every consumer reads files through: batches of about
    batch_size (default :data:`ROW_BATCH`) rows of the open binary file in order, skipping the first skip
    rows (parquet skips whole row groups), every row projected to columns (None: every column).

    The per-format reader (:data:`FORMAT_READERS` by :func:`file_format`) only decodes; this function applies the
    projection itself (:func:`project_row`: a requested column a row lacks stays absent, so a caller checking for
    its own column still sees the row as the file had it). Parquet additionally prunes the read to columns and
    raises for an unknown one at read time, so neither path invents data, and either way every surplus column
    stays out of what the caller stores, including one whose type varies from row to row and would make the shard
    writer fail.
    """

    fmt = file_format(name)
    reader = FORMAT_READERS.get(fmt)
    if reader is None:
        raise ValueError(f"{name}: no reader registered for format {fmt!r}; readers: {sorted(FORMAT_READERS)}")
    for batch in reader(handle, name, skip, columns, ROW_BATCH if batch_size is None else batch_size):
        yield batch if columns is None else [project_row(row, columns) for row in batch]


def iter_stream(handle: BinaryIO, name: str, skip: int = 0, columns: list[str] | None = None) -> Iterator[Row]:
    """
    Rows of :func:`iter_row_batches` one at a time (same contract: ordered, skip applied, projected).
    """

    for batch in iter_row_batches(handle, name, skip, columns):
        yield from batch


def iter_file(path: Path, name: str, skip: int = 0, columns: list[str] | None = None) -> Iterator[Row]:
    """
    :func:`iter_stream` over a local file.
    """

    with path.open("rb") as handle:
        yield from iter_stream(handle, name, skip, columns)


def iter_json_array(handle: BinaryIO, name: str, skip: int = 0) -> Iterator[Row]:
    """
    Elements of a top-level JSON array parsed incrementally with ijson (the C yajl2_c backend when it is
    installed, else ijson's pure-Python one), skipping the first skip. Only the bytes up to the last element
    consumed are read, so a consumer that stops early never pulls the rest of the file; this is the single .json
    code path for cached and remote files alike (a cached file is read from disk in 64 KB chunks the same way).
    A file whose top-level value is not an array (e.g. an object) is a clear error.
    """

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
    """
    Rows of a (possibly compressed) json-lines file, skipping the first skip non-empty lines.
    """

    if fmt == ".jsonl.zst":
        import zstandard

        with zstandard.ZstdDecompressor().stream_reader(handle, closefd=False) as decompressed:
            yield from _parse_json_lines(io.TextIOWrapper(decompressed, encoding="utf-8"), skip)
    elif fmt == ".jsonl.gz":
        with gzip.GzipFile(fileobj=handle, mode="rb") as decompressed:
            yield from _parse_json_lines(io.TextIOWrapper(decompressed, encoding="utf-8"), skip)
    else:  # plain .jsonl
        text = io.TextIOWrapper(handle, encoding="utf-8")
        try:
            yield from _parse_json_lines(text, skip)
        finally:
            text.detach()  # closing the wrapper would close `handle`, which the caller owns


def _parse_json_lines(lines: Any, skip: int) -> Iterator[Row]:
    """
    One JSON object per non-empty line, skipping the first skip of them.
    """

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
    """
    One consumer of :func:`read_rows_multi`: at least count rows passing match (all rows without it), the
    first offset such rows skipped. key is where the index records the per-file / per-row-group counts of
    matching rows (keyed_counts[key] / group_counts[key]); a request with match but no key records nothing
    and cannot skip files. name tags the rows it receives.

    A passive request (count 0 by convention; it is ignored) never causes a file or row group to be read: it takes
    the matching rows of every remote parquet row group the active requests read anyway, for as long as it is
    aligned (module docstring), and ends with the pass.
    """

    name: str
    offset: int
    count: int
    key: str | None = None
    match: RowFilter | None = None
    passive: bool = False


KeyOf = Callable[[Row, str], str]  # the key of a row of a file (`language=<value>`), for the all-key counts and discovery (row, file)
Discover = Callable[[str], "ReadRequest | None"]  # a passive request for a key no request matches, or None to ignore the key


@dataclass
class _Cursor:
    """
    Mutable position of one :class:`ReadRequest` while :func:`read_rows_multi` runs.

    remaining_skip counts rows still to skip before the first yielded row (plain rows without match,
    matching rows with it); it carries across files (a file with fewer matching rows than the skip only shrinks it).
    passive cursors (a passive request, or an active one that reached its count and reads on) only take rows
    from row groups read for active cursors; detached is the sticky "lost alignment" state of a passive cursor.
    """

    request: ReadRequest
    remaining_skip: int
    taken: int = 0  # rows yielded so far
    passive: bool = False
    detached: bool = False

    @property
    def name(self) -> str:
        return self.request.name

    @property
    def match(self) -> RowFilter | None:
        return self.request.match

    @property
    def record_key(self) -> str | None:
        """
        Where matching-row counts are recorded: key for filtered reads, nothing otherwise (plain row counts
        come from footers and full reads regardless of the request).
        """

        return self.request.key if self.request.match is not None else None

    @property
    def satisfied(self) -> bool:
        return self.taken >= self.request.count

    @property
    def active(self) -> bool:
        """
        Still makes files and row groups be read: not passive and short of its count.
        """

        return not self.passive and not self.satisfied

    @property
    def collecting(self) -> bool:
        """
        A passive cursor still aligned with the pass.
        """

        return self.passive and not self.detached

    def wants(self, row: Row) -> bool:
        return self.match is None or self.match(row)

    def skip_whole(self, rows: int) -> bool:
        """
        Skip a unit (file / row group) of rows (matching) rows if the skip position lies at or beyond its
        end; return whether it was skipped.
        """

        if self.remaining_skip < rows:
            return False
        self.remaining_skip -= rows
        return True

    def known_count(self, index: FileIndex, file: str) -> int | None:
        """
        The known number of rows of file that count for this request, or None.
        """

        if self.match is None:
            return index.count(None, file)
        if self.request.key is None:
            return None
        return index.count(self.request.key, file)

    def known_group_counts(self, index: FileIndex, file: str) -> list[int]:
        """
        Rows per row group of file that count for this request, for the prefix of groups whose count is
        known (every group from the footer for a plain read).
        """

        if self.match is None:
            return index.known_group_counts(None, file)
        if self.request.key is None:
            return []
        return index.known_group_counts(self.request.key, file)


def _cursor(request: ReadRequest) -> _Cursor:
    return _Cursor(request=request, remaining_skip=request.offset, passive=request.passive)


@dataclass
class _Pass:
    """
    The shared state of one :func:`read_rows_multi` call: every cursor (discovered ones are appended while a
    file is read), the key function and the discovery callback, and the keys already seen (matched by a
    request, discovered, or declined by discover).
    """

    cursors: list[_Cursor]
    key_of: KeyOf | None
    discover: Discover | None
    known_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.known_keys.update(cursor.request.key for cursor in self.cursors if cursor.request.key is not None)


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
    """
    Rows from offset on (counting rows that pass match) across the index's files: at least count
    of them when the source has that many. :func:`read_rows_multi` with a single request.

    count is exact for files read from the Hub cache and for remote streams. For a parquet file read remotely
    with align_to_row_group (the default) the reader finishes the row group in which it reached count: the
    bytes were already fetched, so keeping the rows means a later fetch at the resulting offset never downloads
    them again; align_to_row_group=False stops at exactly count rows. columns projects every yielded
    row (None: every column): parquet reads only those columns, the json formats drop the rest after parsing.

    Files whose known row count (index.count(key, file)) lies entirely before offset are skipped without
    being opened; every file read through to its end records its count (key for the matching rows, and the
    total row count) so the next call can skip it. Parquet files record their row-group layout as soon as their
    footer was read, so a file that lies entirely before offset is skipped even if it was never read, and
    with key the matching rows of every row group read so far, so a keyed fetch seeks to the right row group
    too. fetcher (default: a :class:`HubFetcher` with token) chooses cache vs. remote reading per file.
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
    key_of: KeyOf | None = None,
    discover: Discover | None = None,
) -> Iterator[tuple[str, Row]]:
    """
    Serve several :class:`ReadRequest` in one pass over the index's files: every file and every parquet row
    group is opened / read at most once and each row is handed to every request that wants it, as
    (request.name, row) pairs. A file or row group is skipped when no active request has to look at it (each
    request seeks by its own known counts, see :func:`read_rows`), and the pass ends when every active request is
    satisfied or the files are exhausted. Everything else (count semantics, what gets recorded in the index,
    columns) is as for :func:`read_rows`, per request.

    A request that reached its count in a remote parquet file (align_to_row_group) turns passive and, like a
    request passive from the start, keeps taking the rows it matches from the row groups the pass still reads,
    while aligned (module docstring; a passive request that would have to skip an unread row group or file is
    detached for the rest of the pass, and a file that is not parquet detaches every passive request). key_of
    gives the key of any row (called with the row and the file it came from): every fully decoded row group then
    records the rows of every key
    (:meth:`FileIndex.record_group_keys`), and a row whose key no request carries is offered to discover once,
    which may answer with a passive request for that key (its offset is the caller's business); the new
    request joins the pass at that row group if it is aligned there, else it is detached at once. The requests
    must have distinct names (the counts recorded per file are keyed by them).
    """

    if fetcher is None:
        fetcher = HubFetcher(token=token)
    if discover is not None and key_of is None:
        raise ValueError("discover needs key_of")
    names = [request.name for request in requests]
    if len(set(names)) != len(names):  # the per-request counts are keyed by name: a shared name records a wrong count
        raise ValueError(f"read_rows_multi needs distinct request names, got {names}")
    state = _Pass([_cursor(request) for request in requests if request.passive or request.count > 0], key_of, discover)

    try:
        for position, file in enumerate(index.files):
            active = [cursor for cursor in state.cursors if cursor.active]
            if not active:
                return
            active = [cursor for cursor in active if not _skip_file_if_count_known(index, cursor, file)]
            if not active:
                _skip_file_passively(index, state, file)
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
                    active = [
                        cursor for cursor in active if cursor.match is not None or not _skip_file_if_count_known(index, cursor, file)
                    ]
                    if not active:
                        _skip_file_passively(index, state, file)
                        continue
                    collecting = [
                        cursor for cursor in state.cursors if cursor.collecting and not _skip_file_if_count_known(index, cursor, file)
                    ]
                    is_remote = not fetcher.uses_cache(index.sizes[file])
                    finish_group = align_to_row_group and is_remote
                    yield from _parquet_rows(parquet, index, position, file, active + collecting, columns, finish_group, state)
                else:
                    for cursor in state.cursors:  # passive rows come from parquet row groups only
                        if cursor.collecting:
                            cursor.detached = True
                    yield from _stream_rows(handle, index, file, active, columns)
    finally:
        index.save()  # the row-group counts recorded in memory while reading


def _skip_file_if_count_known(index: FileIndex, cursor: _Cursor, file: str) -> bool:
    """
    Skip file for cursor without opening it when its (matching) row count is known and lies entirely
    before the cursor's skip position; return whether it was skipped.
    """

    known = cursor.known_count(index, file)
    if known is None:
        return False
    return cursor.skip_whole(known)


def _skip_file_passively(index: FileIndex, state: _Pass, file: str) -> None:
    """
    A file no active cursor opens: a collecting cursor passes it by its known count or loses alignment.
    """

    for cursor in state.cursors:
        if cursor.collecting and not _skip_file_if_count_known(index, cursor, file):
            cursor.detached = True


@dataclass
class _ParquetReader:
    """
    One request's progress through one parquet file (see :func:`_parquet_rows`).
    """

    cursor: _Cursor
    known_group_counts: list[int]  # (matching) rows per row group, the prefix known so far
    first_group: int  # the first row group this request has to read (earlier ones were skipped by known counts)
    matched_in_group: int = 0
    reading: bool = True  # False once an active request stopped taking rows from this file
    next_group: int = 0  # passive: the row group its position lies at the start of or inside; it must be the next one decoded

    @classmethod
    def for_cursor(cls, cursor: _Cursor, index: FileIndex, file: str) -> _ParquetReader:
        """
        Position cursor inside file: skip the leading row groups its known counts cover.
        """

        known = list(cursor.known_group_counts(index, file))
        first_group = 0
        for rows_in_group in known:
            if not cursor.skip_whole(rows_in_group):
                break
            first_group += 1
        return cls(cursor, known, first_group, next_group=first_group)

    @property
    def active(self) -> bool:
        return self.reading and self.cursor.active

    @property
    def taking(self) -> bool:
        """
        Takes rows from the group being decoded: an active reader that has not stopped inside the file (it
        finishes the group it reached its count in), or a collecting passive one.
        """

        return self.cursor.collecting if self.cursor.passive else self.reading

    def finished_group(self, index: FileIndex, file: str, group: int) -> None:
        """
        Record the matching rows of a row group read completely (extends the known prefix by one group).
        """

        record_key = self.cursor.record_key
        if record_key is not None and group == len(self.known_group_counts):
            self.known_group_counts.append(self.matched_in_group)
            index.record_group_counts(record_key, file, self.known_group_counts)


def _parquet_rows(
    parquet: pq.ParquetFile,
    index: FileIndex,
    position: int,
    file: str,
    cursors: list[_Cursor],
    columns: list[str] | None,
    finish_group: bool,
    state: _Pass,
) -> Iterator[tuple[str, Row]]:
    """
    Rows of one parquet file for several requests: a row group is read (once) when at least one active
    request needs it, row groups that every active request skips by its known counts are never read. With
    finish_group a request that reaches count still takes the rest of the row group and then turns passive.
    Passive cursors take rows from a decoded group only when it is the one their position lies in (`next_group`);
    one whose next group is skipped is detached. Keyed requests record the matching rows of every row group they
    read completely, or, with the pass's key_of, every key of every completely decoded group is recorded and
    unknown keys are offered to discover; a request that reads the file to its end records the file's count.
    """

    groups = index.row_groups[file]  # rows per row group, from the footer
    readers = [_ParquetReader.for_cursor(cursor, index, file) for cursor in cursors]
    if not any(reader.active for reader in readers):
        return
    run_start = min(reader.first_group for reader in readers if reader.active)

    for group in range(run_start, len(groups)):
        participants = [reader for reader in readers if reader.active and reader.first_group <= group]
        _detach_passed(readers, group)
        if not participants:
            if not any(reader.active for reader in readers):
                break
            continue  # the group lies before the first group of every request still reading
        participants.extend(reader for reader in readers if reader.cursor.collecting and reader.next_group == group)

        for reader in participants:
            reader.matched_in_group = 0
        counts: dict[str, int] = {}
        complete = True
        for row in read_row_group(parquet, group, columns):
            if state.key_of is not None:
                key = state.key_of(row, file)
                counts[key] = counts.get(key, 0) + 1
                if key not in state.known_keys:
                    discovered = _discover(state, key, index, position, file, group, run_start)
                    if discovered is not None:
                        readers.append(discovered)
                        if discovered.next_group == group:
                            participants.append(discovered)
            for reader in participants:
                cursor = reader.cursor
                if not reader.taking or not cursor.wants(row):
                    continue
                reader.matched_in_group += 1
                if cursor.remaining_skip > 0:
                    cursor.remaining_skip -= 1
                    continue
                yield cursor.name, dict(row)
                cursor.taken += 1
                if cursor.satisfied and not cursor.passive and not finish_group:
                    reader.reading = False  # stopped in the middle of the group: its count stays unknown
            if not any(reader.taking for reader in participants):
                complete = False
                break  # every request that wanted this group stopped inside it: leave the rest of it undecoded

        if complete:
            if state.key_of is None:
                for reader in participants:
                    if reader.taking:
                        reader.finished_group(index, file, group)
            else:
                index.record_group_keys(file, group, counts)
        for reader in participants:
            cursor = reader.cursor
            if cursor.collecting:
                reader.next_group = group + 1
            elif reader.reading and cursor.satisfied:
                reader.reading = False
                if finish_group:  # the group was taken whole: the position is its end, the request reads on passively
                    cursor.passive = True
                    reader.next_group = group + 1
        if not any(reader.active for reader in readers):
            break

    # Requests that went through the last group (or skipped every group by known counts) know the file's
    # matching rows now; with key_of every key with a complete prefix does.
    if state.key_of is not None:
        index.record_complete_file_counts(file, [key for key in (reader.cursor.record_key for reader in readers) if key is not None])
        return
    for reader in readers:
        record_key = reader.cursor.record_key
        went_through = reader.active or (reader.cursor.passive and reader.next_group == len(groups))
        if went_through and record_key is not None and len(reader.known_group_counts) == len(groups):
            index.record_count(record_key, file, sum(reader.known_group_counts))


def _detach_passed(readers: list[_ParquetReader], group: int) -> None:
    """
    A collecting cursor whose next group lies before group missed a row group the pass did not decode.
    """

    for reader in readers:
        if reader.cursor.collecting and reader.next_group < group:
            reader.cursor.detached = True


def _discover(state: _Pass, key: str, index: FileIndex, position: int, file: str, group: int, run_start: int) -> _ParquetReader | None:
    """
    Offer a key no request carries to discover (once). A returned passive request becomes a cursor of the
    pass; it joins the row group being decoded if it is aligned there: every earlier file passes by its known
    count and its known counts in this file position it at run_start or later (the groups from there to this
    one were decoded in this pass without the key, else it would have been discovered earlier); a position past
    this group waits for its group like any passive reader. Otherwise the cursor is detached at once and stays
    in the pass's list so the caller sees it.
    """

    state.known_keys.add(key)
    if state.discover is None:
        return None
    request = state.discover(key)
    if request is None:
        return None
    if not request.passive or request.key != key or request.match is None:
        raise ValueError(f"discover must return a passive request keyed {key!r} with a match, got {request!r}")
    cursor = _cursor(request)
    state.cursors.append(cursor)
    for earlier in index.files[:position]:
        if not _skip_file_if_count_known(index, cursor, earlier):
            cursor.detached = True
            return None
    reader = _ParquetReader.for_cursor(cursor, index, file)
    if reader.first_group < run_start:
        cursor.detached = True
        return None
    reader.next_group = max(reader.first_group, group)
    return reader


def _stream_rows(
    handle: BinaryIO, index: FileIndex, file: str, cursors: list[_Cursor], columns: list[str] | None = None
) -> Iterator[tuple[str, Row]]:
    """
    Rows of one non-parquet file for several active requests, each exactly up to its count and projected to
    columns; the stream is dropped as soon as every request is satisfied. The file is decoded from its first row
    (a stream has no cheap way to skip, and only a full read tells how many rows it holds), so a file read to its end
    records its row count and, for every keyed request that read it through, its matching rows. Passive requests
    take nothing here (the caller detaches them).
    """

    reading = list(cursors)
    matched = {cursor.name: 0 for cursor in cursors}
    rows_seen = 0
    for row in iter_stream(handle, file, 0, columns):
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
    index.record_count(None, file, rows_seen)
    for cursor in reading:
        if cursor.record_key is not None:
            index.record_count(cursor.record_key, file, matched[cursor.name])
