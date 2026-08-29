# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Row loaders: `LOADERS[name](source, offset, count, *, token, index_dir, on_file) -> Iterator[Row]` yields at most
`count` raw rows starting at row `offset` of the source's deterministic order (`hf_files`, `hf_split`, `hf_stream`,
`github_code`, `local`, `synthetic`). `index_dir` is where `hf_files` / `github_code` persist their file index
(None: in memory), `on_file` is called with every repo file they open (progress display). `datasets` is imported
lazily so the HF cache environment can be configured before import."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from data_preparation.lib.schema.dataset_config import SourceConfig
from data_preparation.lib.sources.hub_files import FileIndex, OnFile, read_rows
from data_preparation.lib.sources.synthetic import synthetic_row

Row = dict[str, Any]


class Loader(Protocol):
    def __call__(
        self,
        source: SourceConfig,
        offset: int,
        count: int,
        *,
        token: str | None = None,
        index_dir: Path | None = None,
        on_file: OnFile | None = None,
    ) -> Iterator[Row]: ...


GITHUB_CODE_DATA_FILES = "data/*.parquet"


def _check_offset_count(offset: int, count: int) -> None:
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
    builder = load_kwargs.pop("builder", None)
    kwargs: dict[str, Any] = {"token": token, **extra}
    if builder is None:
        kwargs.update(path=source.hf_id, revision=source.revision, **load_kwargs)
        return kwargs
    data_files = load_kwargs.pop("data_files", None)
    if data_files is None:
        raise ValueError(f"load_kwargs.builder={builder!r} requires load_kwargs.data_files")
    at = f"@{source.revision}" if source.revision else ""
    kwargs.update(path=builder, data_files=f"hf://datasets/{source.hf_id}{at}/{data_files}", **load_kwargs)
    return kwargs


def load_hf_split(
    source: SourceConfig,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    index_dir: Path | None = None,
    on_file: OnFile | None = None,
) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via `split[a:b]` slicing (materialised download, deterministic order)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    dataset = _load_dataset(**hub_load_kwargs(source, token, split=f"{source.split}[{offset}:{offset + count}]"))
    yield from _take(dataset, count)


def load_hf_stream(
    source: SourceConfig,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    index_dir: Path | None = None,
    on_file: OnFile | None = None,
) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via streaming with `skip(offset)` (used for instruct sources)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    stream = _load_dataset(**hub_load_kwargs(source, token, split=source.split, streaming=True))
    if offset:
        stream = stream.skip(offset)
    yield from _take(stream, count)


def iter_language(rows: Iterable[Row], language: str, limit: int, offset: int = 0) -> Iterator[Row]:
    """Yield up to `limit` rows whose `language` column equals `language`, skipping the first `offset` matches."""
    if limit <= 0:
        return
    seen = 0
    taken = 0
    for row in rows:
        if row["language"] != language:
            continue
        if seen < offset:
            seen += 1
            continue
        taken += 1
        yield dict(row)
        if taken >= limit:
            return


def hub_file_index(source: SourceConfig, default_pattern: str | None, index_dir: Path | None, token: str | None) -> FileIndex:
    """The :class:`FileIndex` of a `hf_files` / `github_code` source (`load_kwargs.data_files` or `default_pattern`)."""
    if not source.hf_id:  # validated by SourceConfig; repeated for the type checker
        raise ValueError(f"loader {source.loader} requires hf_id")
    pattern = source.load_kwargs.get("data_files", default_pattern)
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"loader {source.loader} requires load_kwargs.data_files (a glob relative to the repo root)")
    return FileIndex.open(source.hf_id, source.revision, pattern, index_dir, token)


def load_hf_files(
    source: SourceConfig,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    index_dir: Path | None = None,
    on_file: OnFile | None = None,
) -> Iterator[Row]:
    """Rows `offset..offset+count` of the repo files matching `load_kwargs.data_files`, sorted by path.

    Files are downloaded one at a time into the Hub cache (never twice) and read locally; the per-(repo, revision,
    glob) file index under `index_dir` lets a later fetch skip whole files (see `sources/hub_files.py`).
    """
    _check_offset_count(offset, count)
    if count == 0:
        return
    index = hub_file_index(source, None, index_dir, token)
    yield from read_rows(index, offset, count, token=token, on_file=on_file)


def load_github_code(
    source: SourceConfig,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    index_dir: Path | None = None,
    on_file: OnFile | None = None,
) -> Iterator[Row]:
    """`hf_files` over `hf_id` (codeparrot/github-code-clean, default `data_files: data/*.parquet`) keeping rows of
    `source.language`.

    `offset` counts rows *of that language* already consumed, so an incremental fetch continues where the previous
    one stopped. The per-language row counts of fully read files are stored in the shared file index, so all
    language sources read the same cached files and skip files they have already consumed.
    """
    _check_offset_count(offset, count)
    if count == 0:
        return
    if source.language is None:  # validated by SourceConfig; repeated for the type checker
        raise ValueError("github_code loader requires source.language")
    language = source.language
    index = hub_file_index(source, GITHUB_CODE_DATA_FILES, index_dir, token)
    yield from read_rows(
        index,
        offset,
        count,
        token=token,
        on_file=on_file,
        key=f"language={language}",
        match=lambda row: bool(row["language"] == language),
    )


def list_local_files(directory: Path) -> list[Path]:
    """`*.parquet` and `*.jsonl` files directly under `directory`, sorted by name (the source's row order)."""
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix in (".parquet", ".jsonl")]
    return sorted(files)


def _iter_local_file(path: Path) -> Iterator[Row]:
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_local(
    source: SourceConfig,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    index_dir: Path | None = None,
    on_file: OnFile | None = None,
) -> Iterator[Row]:
    """Rows `offset..offset+count` of the parquet/jsonl files under `source.path` (files in sorted order)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    if source.path is None:  # validated by SourceConfig; repeated for the type checker
        raise ValueError("local loader requires source.path")
    directory = Path(source.path)
    if not directory.is_dir():
        raise FileNotFoundError(f"local source directory not found: {directory}")

    def all_rows() -> Iterator[Row]:
        for file in list_local_files(directory):
            yield from _iter_local_file(file)

    yield from islice(all_rows(), offset, offset + count)


def load_synthetic(
    source: SourceConfig,
    offset: int,
    count: int,
    *,
    token: str | None = None,
    index_dir: Path | None = None,
    on_file: OnFile | None = None,
) -> Iterator[Row]:
    """Deterministic random-word rows seeded by `source.seed` (`{"text"}` for pretrain/holdout, instruct triple)."""
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
    if name not in LOADERS:
        raise ValueError(f"unknown loader {name!r}; known loaders: {sorted(LOADERS)}")
    return LOADERS[name]


def repeat_indices(num_rows: int, target: int) -> list[int]:
    """Indices that cycle through `range(num_rows)` until `target` rows are covered (`repeat_to_budget` sources)."""
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    full_copies, remainder = divmod(target, num_rows)
    return list(range(num_rows)) * full_copies + list(range(remainder))
