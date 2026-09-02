# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Row loaders: `LOADERS[name](source, offset, count, shared_parameters) -> Iterator[Row]` yields raw rows starting
at row `offset` of the source's deterministic order (`hf_files`, `hf_split`, `hf_stream`, `github_code`, `local`,
`synthetic`).

`count` is a **minimum**: a loader yields exactly `count` rows (fewer only when the source runs dry), except that
`hf_files` / `github_code` reading a large parquet file remotely finish the row group in which `count` was reached
(`align_to_row_group=True`, the default) so the rows that were downloaded anyway are kept and a later fetch at
the resulting offset never fetches those bytes again; `align_to_row_group=False` makes every loader exact. The
caller must consume everything yielded and advance its offset by the number of rows consumed.

:class:`SharedLoaderParameters` carries what the download stage hands every loader alike; each loader uses the
members that apply to it. `columns` projects the rows `hf_files` / `github_code` / `local` read, whatever the file
format (`github_code` adds `language`, its filter column) — they all read through `hub_files.iter_row_batches`, the
one reading contract; the other loaders yield every column. `index_dir` is where `hf_files` / `github_code` persist
their file index (None: in memory), `on_file` is called with every repo file they open (progress display) and
`stats` collects their download counters (`FetchStats`, bytes fetched). `datasets` is imported lazily so the HF
cache environment can be configured before import."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

from data_preparation.dataset_config import SourceConfig
from data_preparation.lib.sources.hub_files import (
    DEFAULT_MAX_CACHED_FILE_MB,
    FetchStats,
    FileIndex,
    HubFetcher,
    OnFile,
    ReadRequest,
    iter_file,
    read_rows,
    read_rows_multi,
)
from data_preparation.lib.sources.synthetic import synthetic_row

Row = dict[str, Any]


@dataclass(frozen=True)
class SharedLoaderParameters:
    """What the download stage hands every loader besides `(source, offset, count)`: the Hub token, where the
    Hub loaders keep their file index (None: in memory), the callback for every repo file opened, the download
    counters, the column projection (None: every column) and whether a remote parquet row group is finished
    whole once `count` is reached."""

    token: str | None = None
    index_dir: Path | None = None
    on_file: OnFile | None = None
    stats: FetchStats | None = None
    columns: list[str] | None = None
    align_to_row_group: bool = True


class Loader(Protocol):
    def __call__(
        self, source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
    ) -> Iterator[Row]: ...


GITHUB_CODE_DATA_FILES = "data/*.parquet"
MAX_CACHED_FILE_KEY = "max_cached_file_mb"  # `load_kwargs` knob of hf_files / github_code (not part of the source hash)


def _check_offset_count(offset: int, count: int) -> None:
    """Every loader's first line: reject negative arguments early with one clear message."""
    if offset < 0 or count < 0:
        raise ValueError(f"offset and count must be non-negative, got offset={offset}, count={count}")


def _take(rows: Iterable[Row], count: int) -> Iterator[Row]:
    """Yield at most `count` rows as fresh dicts (stops pulling from `rows` as soon as the quota is reached)."""
    for row in islice(rows, count):
        yield dict(row)


def _load_dataset(**kwargs: Any) -> Any:
    """`datasets.load_dataset`, imported lazily (the HF cache env must be configurable before import)."""
    from datasets import load_dataset

    return load_dataset(**kwargs)


def hub_load_kwargs(source: SourceConfig, token: str | None, **extra: Any) -> dict[str, Any]:
    """`load_dataset` kwargs for a Hub source.

    Plain repos: `path=hf_id, revision=..., **load_kwargs`. Repos that still ship a loading script (refused by
    `datasets` >= 4) set `load_kwargs.builder` (`json` / `parquet`) plus `data_files` (a glob relative to the repo):
    the files are then read through the generic builder from `hf://datasets/<hf_id>@<revision>/<data_files>`.
    """
    load_kwargs = dict(source.load_kwargs)
    load_kwargs.pop(MAX_CACHED_FILE_KEY, None)  # an hf_files knob, meaningless to `load_dataset`
    builder = load_kwargs.pop("builder", None)

    if builder is None:  # plain repo: `datasets` resolves the files itself
        return {"token": token, **extra, "path": source.hf_id, "revision": source.revision, **load_kwargs}

    # repo with a loading script: read its data files through the generic `json` / `parquet` builder instead
    data_files = load_kwargs.pop("data_files", None)
    if data_files is None:
        raise ValueError(f"load_kwargs.builder={builder!r} requires load_kwargs.data_files")
    at = f"@{source.revision}" if source.revision else ""
    hub_glob = f"hf://datasets/{source.hf_id}{at}/{data_files}"
    return {"token": token, **extra, "path": builder, "data_files": hub_glob, **load_kwargs}


def load_hf_split(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via `split[a:b]` slicing (materialised download, deterministic order)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    dataset = _load_dataset(**hub_load_kwargs(source, shared_parameters.token, split=f"{source.split}[{offset}:{offset + count}]"))
    yield from _take(dataset, count)


def load_hf_stream(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via streaming with `skip(offset)` (used for instruct sources)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    stream = _load_dataset(**hub_load_kwargs(source, shared_parameters.token, split=source.split, streaming=True))
    if offset:
        stream = stream.skip(offset)
    yield from _take(stream, count)


def hub_file_index(source: SourceConfig, default_pattern: str | None, shared_parameters: SharedLoaderParameters) -> FileIndex:
    """The :class:`FileIndex` of a `hf_files` / `github_code` source (`load_kwargs.data_files` or `default_pattern`)."""
    if not source.hf_id:  # validated by SourceConfig; repeated for the type checker
        raise ValueError(f"loader {source.loader} requires hf_id")
    pattern = source.load_kwargs.get("data_files", default_pattern)
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"loader {source.loader} requires load_kwargs.data_files (a glob relative to the repo root)")
    return FileIndex.open(source.hf_id, source.revision, pattern, shared_parameters.index_dir, shared_parameters.token)


def hub_fetcher(source: SourceConfig, shared_parameters: SharedLoaderParameters) -> HubFetcher:
    """The :class:`HubFetcher` of a `hf_files` / `github_code` source: `load_kwargs.max_cached_file_mb` (default
    `DEFAULT_MAX_CACHED_FILE_MB`) decides which files go through the Hub cache and which are read remotely."""
    threshold = source.load_kwargs.get(MAX_CACHED_FILE_KEY, DEFAULT_MAX_CACHED_FILE_MB)
    return HubFetcher(
        token=shared_parameters.token, max_cached_file_mb=float(threshold), stats=shared_parameters.stats or FetchStats()
    )


def load_hf_files(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """Rows `offset..offset+count` of the repo files matching `load_kwargs.data_files`, sorted by path.

    Files up to `load_kwargs.max_cached_file_mb` are downloaded one at a time into the Hub cache (never twice) and
    read locally (exactly `count` rows); larger parquet files are read remotely row group by row group and the
    last row group read is yielded whole unless `align_to_row_group=False`, larger json-lines files are streamed
    (exactly `count` rows). The per-(repo, revision, glob) file index under `index_dir` lets a later fetch skip whole files (see
    `sources/hub_files.py`).
    """
    _check_offset_count(offset, count)
    if count == 0:
        return
    index = hub_file_index(source, None, shared_parameters)
    yield from read_rows(
        index, offset, count, on_file=shared_parameters.on_file, fetcher=hub_fetcher(source, shared_parameters),
        columns=shared_parameters.columns, align_to_row_group=shared_parameters.align_to_row_group,
    )


def load_github_code(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """`hf_files` over `hf_id` (codeparrot/github-code-clean, default `data_files: data/*.parquet`) keeping rows of
    `source.language`.

    `offset` counts rows *of that language* already consumed, so an incremental fetch continues where the previous
    one stopped. The per-language row counts of fully read files (and per row group of partially read parquet
    files) are stored in the shared file index, so all language sources read the same cached files and skip files
    and row groups they have already consumed. A remote row group is kept whole like in `hf_files`, so the
    language offset it leaves behind is a row-group boundary in source rows. `columns` always includes `language`.
    Several languages of one repo are read together by :func:`read_github_code_group`.
    """
    _check_offset_count(offset, count)
    if count == 0:
        return
    request = GithubCodeRequest(name="", source=source, offset=offset, count=count)
    for _, row in read_github_code_group([request], shared_parameters):
        yield row


@dataclass(frozen=True)
class GithubCodeRequest:
    """One member of :func:`read_github_code_group`: rows of ``source.language`` from ``offset`` (in rows of that
    language) on, at least ``count`` of them."""

    name: str
    source: SourceConfig
    offset: int
    count: int


def github_code_repo_key(source: SourceConfig) -> tuple[str | None, str | None, str]:
    """What `github_code` sources must share to be read in one pass: ``(hf_id, revision, data_files)``."""
    return (source.hf_id, source.revision, str(source.load_kwargs.get("data_files", GITHUB_CODE_DATA_FILES)))


def read_github_code_group(
    requests: list[GithubCodeRequest], shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[tuple[str, Row]]:
    """Serve several `github_code` sources of **one repo** in a single pass over its files (every row group read
    at most once), yielding ``(request.name, row)``; per source exactly what `load_github_code` yields for the same
    ``offset`` / ``count``, including the shared index bookkeeping. The sources must agree on `hf_id`, `revision`
    and `data_files` (:func:`github_code_repo_key`) and have distinct languages; the first request's
    `max_cached_file_mb` decides cache vs. remote reading for all of them.
    """
    if not requests:
        return
    first = requests[0].source
    for request in requests:
        _check_offset_count(request.offset, request.count)
        if github_code_repo_key(request.source) != github_code_repo_key(first):
            raise ValueError(f"{request.name}: github_code group members must share hf_id, revision and data_files")
        if request.source.language is None:  # validated by SourceConfig; repeated for the type checker
            raise ValueError(f"{request.name}: github_code loader requires source.language")
    languages = [str(r.source.language) for r in requests]
    if len(set(languages)) != len(languages):
        raise ValueError(f"github_code group members must have distinct languages, got {languages}")

    index = hub_file_index(first, GITHUB_CODE_DATA_FILES, shared_parameters)
    columns = shared_parameters.columns
    if columns is not None and "language" not in columns:
        columns = [*columns, "language"]
    yield from read_rows_multi(
        index,
        [_language_request(r) for r in requests],
        on_file=shared_parameters.on_file,
        fetcher=hub_fetcher(first, shared_parameters),
        columns=columns,
        align_to_row_group=shared_parameters.align_to_row_group,
    )


def _language_request(request: GithubCodeRequest) -> ReadRequest:
    language = str(request.source.language)
    return ReadRequest(
        name=request.name,
        offset=request.offset,
        count=request.count,
        key=f"language={language}",
        match=lambda row: bool(row["language"] == language),
    )


def list_local_files(directory: Path) -> list[Path]:
    """`*.parquet` and `*.jsonl` files directly under `directory`, sorted by name (the source's row order)."""
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix in (".parquet", ".jsonl")]
    return sorted(files)


def _iter_local_rows(directory: Path, columns: list[str] | None = None) -> Iterator[Row]:
    """Every row of every file under `directory`, files in sorted order, projected to `columns` — through the
    same reading contract as the Hub loaders (`hub_files.iter_row_batches` via `iter_file`: bounded batches,
    empty `.jsonl` lines skipped)."""
    for file in list_local_files(directory):
        yield from iter_file(file, file.name, 0, columns)


def load_local(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """Rows `offset..offset+count` of the parquet/jsonl files under `source.path` (files in sorted order), each
    projected to `columns` (None: every column) whatever the file format."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    if source.path is None:  # validated by SourceConfig; repeated for the type checker
        raise ValueError("local loader requires source.path")
    directory = Path(source.path)
    if not directory.is_dir():
        raise FileNotFoundError(f"local source directory not found: {directory}")
    yield from islice(_iter_local_rows(directory, shared_parameters.columns), offset, offset + count)


def load_synthetic(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """Deterministic random-word rows seeded by `source.seed` (`{"text"}` for pretrain/validation, instruct triple)."""
    _check_offset_count(offset, count)
    for index in range(offset, offset + count):
        yield synthetic_row(source.kind, source.seed, index)


LOADERS: dict[str, Loader] = {
    "hf_files": load_hf_files,
    "hf_split": load_hf_split,
    "hf_stream": load_hf_stream,
    "github_code": load_github_code,
    "local": load_local,
    "synthetic": load_synthetic,
}


def get_loader(name: str) -> Loader:
    """The loader registered under `name` (a `SourceConfig.loader` value) or a clear error."""
    if name not in LOADERS:
        raise ValueError(f"unknown loader {name!r}; known loaders: {sorted(LOADERS)}")
    return LOADERS[name]
