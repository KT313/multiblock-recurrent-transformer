# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Row loaders: `LOADERS[name](source, offset, count, shared_parameters) -> Iterator[Row]` yields raw rows starting
at row `offset` of the source's deterministic order (`hf_files`, `hf_split`, `hf_stream`, `github_code`, `local`,
`synthetic`).

`count` is a minimum: a loader yields exactly `count` rows (fewer only when the source runs dry), except that
`hf_files` / `github_code` reading a large parquet file remotely finish the row group in which `count` was reached
(`align_to_row_group=True`, the default), so rows downloaded anyway are kept and never fetched again. The caller
must consume everything yielded and advance its offset by the number of rows consumed.

:class:`SharedLoaderParameters` carries what the download stage hands every loader alike. `columns` projects the
rows `hf_files` / `github_code` / `local` read (all through `hub_files.iter_row_batches`; `github_code` adds
`language`); the other loaders yield every column. `datasets` is imported lazily so the HF cache environment can be
configured before import.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

from data_preparation.dataset_config import DEFAULT_TOKENS_PER_ROW_ESTIMATE, SourceConfig
from data_preparation.lib.sources.hub_files import (
    DEFAULT_MAX_CACHED_FILE_MB,
    FORMATS,
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
    """
    What the download stage hands every loader besides `(source, offset, count)`: the Hub token, where the
    Hub loaders keep their file index (None: in memory), the callback for every repo file opened, the download
    counters, the column projection (None: every column) and whether a remote parquet row group is finished
    whole once `count` is reached.
    """

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
    """
    Every loader's first line: reject negative arguments early with one clear message.
    """

    if offset < 0 or count < 0:
        raise ValueError(f"offset and count must be non-negative, got offset={offset}, count={count}")


def _take(rows: Iterable[Row], count: int) -> Iterator[Row]:
    """
    Yield at most `count` rows as fresh dicts (stops pulling from `rows` as soon as the quota is reached).
    """

    for row in islice(rows, count):
        yield dict(row)


def _load_dataset(**kwargs: Any) -> Any:
    """
    `datasets.load_dataset`, imported lazily (the HF cache env must be configurable before import).
    """

    from datasets import load_dataset

    return load_dataset(**kwargs)


def hub_load_kwargs(source: SourceConfig, token: str | None, **extra: Any) -> dict[str, Any]:
    """
    `load_dataset` kwargs for a Hub source.

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
    revision_suffix = f"@{source.revision}" if source.revision else ""
    hub_glob = f"hf://datasets/{source.hf_id}{revision_suffix}/{data_files}"
    return {"token": token, **extra, "path": builder, "data_files": hub_glob, **load_kwargs}


def load_hf_split(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """
    Rows `offset..offset+count` of `hf_id` via `split[a:b]` slicing (materialised download, deterministic order).
    """

    _check_offset_count(offset, count)
    if count == 0:
        return
    dataset = _load_dataset(**hub_load_kwargs(source, shared_parameters.token, split=f"{source.split}[{offset}:{offset + count}]"))
    yield from _take(dataset, count)


def load_hf_stream(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """
    Rows `offset..offset+count` of `hf_id` via streaming with `skip(offset)` (used for instruct sources).
    """

    _check_offset_count(offset, count)
    if count == 0:
        return
    stream = _load_dataset(**hub_load_kwargs(source, shared_parameters.token, split=source.split, streaming=True))
    if offset:
        stream = stream.skip(offset)
    yield from _take(stream, count)


def hub_file_index(source: SourceConfig, default_pattern: str | None, shared_parameters: SharedLoaderParameters) -> FileIndex:
    """
    The :class:`FileIndex` of a `hf_files` / `github_code` source (`load_kwargs.data_files` or `default_pattern`).
    """

    if not source.hf_id:  # validated by SourceConfig; repeated for the type checker
        raise ValueError(f"loader {source.loader} requires hf_id")
    pattern = source.load_kwargs.get("data_files", default_pattern)
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"loader {source.loader} requires load_kwargs.data_files (a glob relative to the repo root)")
    return FileIndex.open(source.hf_id, source.revision, pattern, shared_parameters.index_dir, shared_parameters.token)


def hub_fetcher(source: SourceConfig, shared_parameters: SharedLoaderParameters) -> HubFetcher:
    """
    The :class:`HubFetcher` of a `hf_files` / `github_code` source: `load_kwargs.max_cached_file_mb` (default
    `DEFAULT_MAX_CACHED_FILE_MB`) decides which files go through the Hub cache and which are read remotely.
    """

    threshold = source.load_kwargs.get(MAX_CACHED_FILE_KEY, DEFAULT_MAX_CACHED_FILE_MB)
    return HubFetcher(
        token=shared_parameters.token, max_cached_file_mb=float(threshold), stats=shared_parameters.stats or FetchStats()
    )


def load_hf_files(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """
    Rows `offset..offset+count` of the repo files matching `load_kwargs.data_files`, sorted by path.

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
    """
    `hf_files` over `hf_id` (codeparrot/github-code-clean, default `data_files: data/*.parquet`) keeping rows of
    `source.language`.

    `offset` counts rows *of that language* already consumed, so an incremental fetch continues where the previous
    one stopped. The per-language row counts of fully read files (and per row group of partially read parquet
    files) are stored in the shared file index, so all language sources read the same cached files and skip files
    and row groups they have already consumed. A remote row group is kept whole like in `hf_files`, so the
    language offset it leaves behind is a row-group boundary in source rows. `columns` always includes `language`.
    Several languages of one repo are read together by :func:`read_github_code_group`; a single language read this
    way stops at its count (nothing else is reading on).
    """

    _check_offset_count(offset, count)
    if count == 0:
        return
    request = GithubCodeRequest(name="", source=source, offset=offset, count=count)
    for _, row in read_github_code_group([request], shared_parameters):
        yield row


@dataclass(frozen=True)
class GithubCodeRequest:
    """
    One member of :func:`read_github_code_group`: rows of source.language from offset (in rows of that
    language) on, at least count of them; a passive member (count 0) takes its language's rows from the row
    groups the others read, while aligned (`hub_files`).
    """

    name: str
    source: SourceConfig
    offset: int
    count: int
    passive: bool = False


LANGUAGE_KEY_PREFIX = "language="  # the index key of a language: `language=<label>` (`language_key`, `language_of_key`)
DiscoverLanguage = Callable[[str], ReadRequest | None]  # a passive request for a language no member carries, or None


def language_key(language: str) -> str:
    """
    The file-index key under which a language's per-file / per-row-group counts are recorded.
    """

    return f"{LANGUAGE_KEY_PREFIX}{language}"


def language_of_key(key: str) -> str:
    """
    The language label of a :func:`language_key`.
    """

    if not key.startswith(LANGUAGE_KEY_PREFIX):
        raise ValueError(f"not a language key: {key!r}")
    return key[len(LANGUAGE_KEY_PREFIX) :]


def language_slug(language: str) -> str:
    """
    A language label as a source-name suffix: lower case, `#` -> `sharp`, `+` -> `p`, any other run of
    characters outside [a-z0-9] -> `_` (`C#` -> `csharp`, `C++` -> `cpp`, `Objective-C` -> `objective_c`,
    `Jupyter Notebook` -> `jupyter_notebook`, `GO` -> `go`).
    """

    slug = language.lower().replace("#", "sharp").replace("+", "p")
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    if not slug:
        raise ValueError(f"language {language!r} leaves no slug")
    return slug


def github_code_extra_name(template: SourceConfig, language: str) -> str:
    """
    The source name a language without a source of its own is stored under: the repo id's tail with every run
    of characters outside [a-z0-9] as `_`, then `_<language_slug>` (`codeparrot/github-code-clean` + `C#` ->
    `github_code_clean_csharp`). A config entry of that name (same repo, revision, data_files and text_field,
    the `language`) adopts the folder.
    """

    if not template.hf_id:  # validated by SourceConfig; repeated for the type checker
        raise ValueError("github_code_extra_name needs a template with hf_id")
    repo = re.sub(r"[^a-z0-9]+", "_", template.hf_id.rsplit("/", 1)[-1].lower()).strip("_")
    return f"{repo}_{language_slug(language)}"


def github_code_extra_source(template: SourceConfig, language: str) -> SourceConfig:
    """
    The source a language without a source of its own is downloaded as: template (a member of the group: same
    repo, revision, data_files, text_field, ...) with `language` replaced and the row estimate at its default.
    Its raw hash is what the extra folder's manifest records.
    """

    return replace(template, language=language, describe_tokens_per_row=DEFAULT_TOKENS_PER_ROW_ESTIMATE)


def github_code_repo_key(source: SourceConfig) -> tuple[str | None, str | None, str]:
    """
    What `github_code` sources must share to be read in one pass: (hf_id, revision, data_files).
    """

    return (source.hf_id, source.revision, str(source.load_kwargs.get("data_files", GITHUB_CODE_DATA_FILES)))


def read_github_code_group(
    requests: list[GithubCodeRequest],
    shared_parameters: SharedLoaderParameters = SharedLoaderParameters(),
    *,
    discover: DiscoverLanguage | None = None,
) -> Iterator[tuple[str, Row]]:
    """
    Serve several `github_code` sources of one repo in a single pass over its files (every row group read
    at most once), yielding (request.name, row). Every member gets at least its count from its offset, exactly
    what `load_github_code` yields; a member that reached its count (or is passive from the start) keeps taking
    its language's rows from the row groups the pass still reads for the others, while aligned (`hub_files`), so
    its rows are always a contiguous prefix of the language's order. Every fully decoded row group records the
    rows of every language in the shared index, and a row of a language no member carries is offered to
    discover (once per language, the label as argument), which may answer with a passive `ReadRequest` for it
    (name, offset, `key=language_key(label)`, a match on the label; :func:`language_request` builds one). The
    sources must agree on `hf_id`, `revision` and `data_files` (:func:`github_code_repo_key`) and have distinct
    languages and names; the first request's `max_cached_file_mb` decides cache vs. remote reading for all of them.
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
    languages = [str(request.source.language) for request in requests]
    if len(set(languages)) != len(languages):
        raise ValueError(f"github_code group members must have distinct languages, got {languages}")
    names = [request.name for request in requests]
    if len(set(names)) != len(names):  # `read_rows_multi` keys its per-file counts by name
        raise ValueError(f"github_code group members must have distinct names, got {names}")

    index = hub_file_index(first, GITHUB_CODE_DATA_FILES, shared_parameters)
    columns = shared_parameters.columns
    if columns is not None and "language" not in columns:
        columns = [*columns, "language"]
    yield from read_rows_multi(
        index,
        [language_request(request.name, str(request.source.language), request.offset, request.count, passive=request.passive) for request in requests],
        on_file=shared_parameters.on_file,
        fetcher=hub_fetcher(first, shared_parameters),
        columns=columns,
        align_to_row_group=shared_parameters.align_to_row_group,
        key_of=_language_of_row,
        discover=None if discover is None else (lambda key: discover(language_of_key(key))),
    )


def _language_of_row(row: Row, file: str) -> str:
    """
    The key of a row of a `github_code` file. A file without the column cannot be sorted into languages at all
    (it used to raise a bare KeyError naming nothing), so say which file is missing it.
    """

    if "language" not in row:
        raise ValueError(f"{file}: no 'language' column (columns: {', '.join(map(str, row))}); it is what github_code sorts rows by")
    return language_key(str(row["language"]))


def language_request(name: str, language: str, offset: int, count: int, *, passive: bool = False) -> ReadRequest:
    """
    The `hub_files` request for the rows of one language: keyed :func:`language_key`, matched on the label.
    """

    return ReadRequest(
        name=name,
        offset=offset,
        count=count,
        key=language_key(language),
        match=lambda row: bool(row["language"] == language),
        passive=passive,
    )


def list_local_files(directory: Path) -> list[Path]:
    """
    The files directly under `directory` in a format the Hub reader knows (`hub_files.FORMATS`: parquet, json
    lines, plain or compressed, json arrays), sorted by name (the source's row order).
    """

    files = [path for path in directory.iterdir() if path.is_file() and path.name.lower().endswith(FORMATS)]
    return sorted(files)


def _iter_local_rows(directory: Path, columns: list[str] | None = None) -> Iterator[Row]:
    """
    Every row of every file under `directory`, files in sorted order, projected to `columns`, through the same
    reading contract as the Hub loaders (`hub_files.iter_row_batches` via `iter_file`: bounded batches, empty
    `.jsonl` lines skipped).
    """

    for file in list_local_files(directory):
        yield from iter_file(file, file.name, 0, columns)


def load_local(
    source: SourceConfig, offset: int, count: int, shared_parameters: SharedLoaderParameters = SharedLoaderParameters()
) -> Iterator[Row]:
    """
    Rows `offset..offset+count` of the parquet/jsonl files under `source.path` (files in sorted order), each
    projected to `columns` (None: every column) whatever the file format.
    """

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
    """
    Deterministic random-word rows seeded by `source.seed` (`{"text"}` for pretrain/validation, instruct triple).
    """

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
    """
    The loader registered under `name` (a `SourceConfig.loader` value) or a clear error.
    """

    if name not in LOADERS:
        raise ValueError(f"unknown loader {name!r}; known loaders: {sorted(LOADERS)}")
    return LOADERS[name]
