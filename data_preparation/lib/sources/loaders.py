# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Row loaders: `LOADERS[name](source, offset, count) -> Iterator[Row]` yields at most `count` raw rows starting at
row `offset` of the source's deterministic order (`hf_split`, `hf_stream`, `github_code`, `local`, `synthetic`).
`datasets` is imported lazily so the HF cache environment can be configured before import."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from data_preparation.lib.schema.dataset_config import SourceConfig
from data_preparation.lib.sources.synthetic import synthetic_row

Row = dict[str, Any]


class Loader(Protocol):
    def __call__(self, source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]: ...


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


def load_hf_split(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via `split[a:b]` slicing (materialised download, deterministic order)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    dataset = _load_dataset(
        path=source.hf_id,
        split=f"{source.split}[{offset}:{offset + count}]",
        revision=source.revision,
        token=token,
        **source.load_kwargs,
    )
    yield from _take(dataset, count)


def load_hf_stream(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via streaming with `skip(offset)` (used for instruct sources)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    stream = _load_dataset(
        path=source.hf_id,
        split=source.split,
        streaming=True,
        revision=source.revision,
        token=token,
        **source.load_kwargs,
    )
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


def load_github_code(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Stream `hf_id` (codeparrot/github-code-clean) and keep rows of `source.language`.

    `offset` counts rows *of that language* already consumed, so an incremental fetch continues where the previous
    one stopped (the stream is re-read from the start; the pinned `revision` keeps its order stable).
    """
    _check_offset_count(offset, count)
    if count == 0:
        return
    if source.language is None:  # validated by SourceConfig; repeated for the type checker
        raise ValueError("github_code loader requires source.language")
    load_kwargs = dict(source.load_kwargs)
    data_files = load_kwargs.pop("data_files", GITHUB_CODE_DATA_FILES)
    stream = _load_dataset(
        path=source.hf_id,
        split=source.split,
        streaming=True,
        revision=source.revision,
        data_files=data_files,
        token=token,
        **load_kwargs,
    )
    yield from iter_language(stream, source.language, count, offset)


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


def load_local(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
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


def load_synthetic(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Deterministic random-word rows seeded by `source.seed` (`{"text"}` for pretrain/holdout, instruct triple)."""
    _check_offset_count(offset, count)
    for index in range(offset, offset + count):
        yield synthetic_row(source.kind, source.seed, index)


LOADERS: dict[str, Loader] = {
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
