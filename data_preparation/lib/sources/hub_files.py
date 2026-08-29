# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""File-by-file reading of a Hub dataset repo (the ``hf_files`` / ``github_code`` loaders).

Row order = the repo's files matching a glob (``load_kwargs.data_files``, relative to the repo root) sorted by path,
rows in file order. Files are fetched **one at a time, on demand** with ``huggingface_hub.hf_hub_download`` — they
land in the Hub cache (``~/.cache/huggingface/hub`` or ``HF_HOME`` / ``--cache_dir``) and are never fetched twice —
and read locally: ``.parquet`` (row groups, never the whole file), ``.jsonl``, ``.jsonl.zst``, ``.jsonl.gz`` /
``.json.gz`` (one JSON object per line) and plain ``.json`` (a JSON array, loaded whole; small files only).

A :class:`FileIndex` per ``(repo, revision, glob)`` remembers the file list and the row count of every file read so
far (parquet: from the footer, so it is known before the file is read), so a fetch at ``offset`` skips whole files
without opening them. It is persisted as JSON under ``<index_dir>/<repo>@<revision>/<glob hash>.json`` when an
``index_dir`` is given (``dataset/hub_index/`` in a build), else kept in memory for the loader call only. Extra
per-file counters (``counts[key][file]``, e.g. rows of one language for ``github_code``) share the index.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

Row = dict[str, Any]
OnFile = Callable[[str], None]

FORMATS: tuple[str, ...] = (".parquet", ".jsonl.zst", ".jsonl.gz", ".json.gz", ".jsonl", ".json")


# --- Hub access (module-level so tests can stub them) --------------------------------------------------------------


def list_repo_files(repo_id: str, revision: str | None, token: str | None) -> list[str]:
    """All file paths of a dataset repo at ``revision`` (``HfApi.list_repo_files``)."""
    from huggingface_hub import HfApi

    files: list[str] = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return files


def hub_download(repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
    """Download one repo file into the Hub cache (no-op if cached) and return its local path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision, token=token))


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
    path: Path | None = None  # where the index is persisted (None: in memory)

    @classmethod
    def open(
        cls, repo_id: str, revision: str | None, pattern: str, index_dir: Path | None, token: str | None
    ) -> FileIndex:
        """Load the persisted index or create it (listing the repo once); the file list is cached in the index."""
        path = None if index_dir is None else index_path(index_dir, repo_id, revision, pattern)
        if path is not None and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(repo_id, revision, pattern, data["files"], data["rows"], data.get("counts", {}), path)
        files = sorted(f for f in list_repo_files(repo_id, revision, token) if fnmatch.fnmatchcase(f, pattern))
        if not files:
            raise FileNotFoundError(f"{repo_id}@{revision or 'main'}: no files match data_files={pattern!r}")
        index = cls(repo_id, revision, pattern, files, path=path)
        index.save()
        return index

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


# --- per-format readers -----------------------------------------------------------------------------------------------


def file_format(name: str) -> str:
    """The recognised suffix of ``name`` (longest match of :data:`FORMATS`) or a clear error."""
    lower = name.lower()
    for suffix in FORMATS:
        if lower.endswith(suffix):
            return suffix
    raise ValueError(f"unsupported file format {name!r}; supported: {', '.join(FORMATS)}")


def parquet_rows(path: Path) -> int:
    """Row count from the parquet footer (no data read)."""
    return int(pq.ParquetFile(path).metadata.num_rows)


def iter_file(path: Path, name: str, skip: int = 0) -> Iterator[Row]:
    """Rows of a local file in order, skipping the first ``skip`` (parquet skips whole row groups by metadata)."""
    fmt = file_format(name)
    if fmt == ".parquet":
        yield from _iter_parquet(path, skip)
        return
    if fmt == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{name}: plain .json must contain a JSON array of rows")
        for row in data[skip:]:
            yield row
        return
    yield from _iter_json_lines(path, fmt, skip)


def _iter_parquet(path: Path, skip: int) -> Iterator[Row]:
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        group_rows = parquet.metadata.row_group(group).num_rows
        if skip >= group_rows:
            skip -= group_rows
            continue
        for batch in parquet.iter_batches(row_groups=[group]):
            if skip >= batch.num_rows:
                skip -= batch.num_rows
                continue
            rows = batch.to_pylist()
            yield from rows[skip:]
            skip = 0


def _iter_json_lines(path: Path, fmt: str, skip: int) -> Iterator[Row]:
    if fmt == ".jsonl.zst":
        import io

        import zstandard

        with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            yield from _lines(io.TextIOWrapper(reader, encoding="utf-8"), skip)
    elif fmt in (".jsonl.gz", ".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from _lines(fh, skip)
    else:
        with path.open(encoding="utf-8") as fh:
            yield from _lines(fh, skip)


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
    token: str | None,
    on_file: OnFile | None = None,
    key: str | None = None,
    match: Callable[[Row], bool] | None = None,
) -> Iterator[Row]:
    """Rows ``offset..offset+count`` (counting rows that pass ``match``) across the index's files.

    Files whose known row count (``index.count(key, file)``) lies entirely before ``offset`` are skipped without
    being downloaded or opened; every file read through to its end records its count (``key`` for the matching
    rows, and the total row count) so the next call can skip it.
    """
    if count <= 0:
        return
    remaining_skip = offset
    taken = 0
    for file in index.files:
        known = index.count(key, file)
        if known is not None and remaining_skip >= known:
            remaining_skip -= known
            continue
        path = hub_download(index.repo_id, file, index.revision, token)
        if on_file is not None:
            on_file(file)
        if known is None and file_format(file) == ".parquet" and index.rows.get(file) is None:
            index.record(None, file, parquet_rows(path))
            if key is None:
                known = index.rows[file]
                if remaining_skip >= known:
                    remaining_skip -= known
                    continue
        file_skip = remaining_skip if match is None else 0
        total_rows = 0
        matched = 0
        completed = True
        for row in iter_file(path, file, skip=file_skip):
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
