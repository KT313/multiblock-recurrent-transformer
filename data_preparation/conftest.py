# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Area fixtures for data_preparation: keep the `datasets` cache out of the user's home and offline; builders for
small dataset configs over `synthetic` / `local` sources used by the stage and planner tests."""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
import zstandard

from data_preparation.dataset_config import (
    DatasetConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
)
from data_preparation.layout import DatasetLayout
from data_preparation.lib.sources import hub_files
from data_preparation.lib.sources.hub_files import file_format
from data_preparation.lib.stages.download import prepare_tokenizer

# Must happen before `datasets` is imported anywhere (its config reads the env at import time).
_CACHE = tempfile.mkdtemp(prefix="hf_datasets_cache_")
os.environ.setdefault("HF_DATASETS_CACHE", _CACHE)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

Row = dict[str, Any]
CfgFactory = Callable[..., DatasetConfig]


# --- a fake Hub for the hf_files / github_code loaders -----------------------------------------------------------------

REPO = "org/name"
REV = "abc"


class RecordingFile(io.FileIO):
    """A local file that records the byte range of every read (stand-in for the remote fsspec file object)."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, "rb")
        self.ranges: list[tuple[int, int]] = []

    def read(self, size: int | None = -1) -> bytes:
        start = self.tell()
        data = super().read(-1 if size is None else size)
        self.ranges.append((start, start + len(data)))
        return data

    def readinto(self, buffer: Any) -> int:
        start = self.tell()
        n = super().readinto(buffer)
        self.ranges.append((start, start + (n or 0)))
        return n or 0


class FakeHub:
    """Stand-in for the Hub: `files` maps repo paths to local files; counts downloads and listings."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, Path] = {}
        self.downloads: list[str] = []
        self.streams: list[str] = []
        self.handles: dict[str, RecordingFile] = {}
        self.listings = 0
        self.size_lookups = 0

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

    def paths_info(self, repo_id: str, paths: list[str], revision: str | None, token: str | None) -> dict[str, int]:
        assert (repo_id, revision) == (REPO, REV)
        self.size_lookups += 1
        return {p: self.files[p].stat().st_size for p in paths}

    def hub_download(self, repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
        assert (repo_id, revision) == (REPO, REV)
        self.downloads.append(filename)
        return self.files[filename]

    def open_remote(self, repo_id: str, filename: str, revision: str | None, token: str | None, block_size: int) -> BinaryIO:
        assert (repo_id, revision) == (REPO, REV)
        self.streams.append(filename)
        handle = RecordingFile(self.files[filename])
        self.handles[filename] = handle
        return handle


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub(tmp_path / "hub")
    fake.root.mkdir()
    monkeypatch.setattr(hub_files, "list_repo_files", fake.list_repo_files)
    monkeypatch.setattr(hub_files, "paths_info", fake.paths_info)
    monkeypatch.setattr(hub_files, "hub_download", fake.hub_download)
    monkeypatch.setattr(hub_files, "open_remote", fake.open_remote)
    return fake


@pytest.fixture
def hf_datasets() -> Iterator[ModuleType]:
    """The `datasets` module with caching disabled (map/filter results stay in temp files)."""
    datasets = pytest.importorskip("datasets")
    datasets.disable_caching()
    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bars()
    yield datasets
    datasets.enable_caching()


@pytest.fixture
def layout(tmp_path: Path) -> DatasetLayout:
    return DatasetLayout(tmp_path / "dataset")


TEST_BLOOM_MEMORY_MB = 1  # the dedup filter of every test config (the default 1024 MB is for real sources)


@pytest.fixture
def cfg_factory() -> CfgFactory:
    """`make(sources, processing=..., token_count=..., max_seq_length=..., block_size=..., tokens=...)` -> DatasetConfig.

    A `pretrain` stage trains on every pretrain source without `rows` (equal weights) and a `finetune` stage on
    every instruct source without `rows`; a source with `rows` is used only for validation (in the stage of its
    kind, or the other one). Without any trainable source a synthetic `_pretrain` source is added so the config
    validates. `tokens` is the per-stage budget; `block_size` defaults to 1 so the sequence budget of a trained
    source equals `tokens × weight` (rows). The dedup filter of every config is `TEST_BLOOM_MEMORY_MB` (also when
    `processing` is given).
    """

    def make(
        sources: dict[str, SourceConfig],
        *,
        processing: ProcessingConfig | None = None,
        token_count: str = "tokenizer",
        max_seq_length: int = 64,
        block_size: int = 1,
        tokens: int = 10_000,
        tokenizer: TokenizerConfig | None = None,
        name: str = "t",
    ) -> DatasetConfig:
        sources = dict(sources)
        trained = {kind: [n for n, s in sources.items() if s.kind == kind and s.rows is None] for kind in ("pretrain", "instruct")}
        val_only = {kind: [n for n, s in sources.items() if s.kind == kind and s.rows is not None] for kind in ("pretrain", "instruct")}
        if not trained["pretrain"] and not trained["instruct"]:
            sources["_pretrain"] = SourceConfig(kind="pretrain", loader="synthetic")
            trained["pretrain"] = ["_pretrain"]
        stages: list[StageConfig] = []
        for kind, stage_name in (("pretrain", "pretrain"), ("instruct", "finetune")):
            if not trained[kind]:
                continue
            other = "instruct" if kind == "pretrain" else "pretrain"
            val_names = val_only[kind] or trained[kind]
            if not trained[other]:
                val_names = val_names + val_only[other]  # nowhere else to validate on them
            stages.append(
                StageConfig(
                    name=stage_name,
                    tokens=tokens,
                    train={n: 1.0 / len(trained[kind]) for n in trained[kind]},
                    val={n: 1.0 / len(val_names) for n in val_names},
                )
            )
        processing = processing or ProcessingConfig(min_chars=1)
        processing = replace(processing, dedup=replace(processing.dedup, bloom_memory_mb=TEST_BLOOM_MEMORY_MB))
        return DatasetConfig(
            name=name,
            tokenizer=tokenizer or TokenizerConfig(name="synthetic", kind="synthetic"),
            sources=sources,
            stages=stages,
            block_size=block_size,
            max_seq_length=max_seq_length,
            token_count=token_count,  # type: ignore[arg-type]  # Literal narrowed by the caller
            processing=processing,
        )

    return make


@pytest.fixture
def write_local() -> Callable[[Path, list[Row], str], Path]:
    """`write_local(directory, rows, fmt)` writes one parquet (or jsonl) file of dict rows for the `local` loader."""

    def write(directory: Path, rows: list[Row], fmt: str = "parquet") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        existing = len(list(directory.iterdir()))
        if fmt == "parquet":
            path = directory / f"part-{existing:03d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
        else:
            import json

            path = directory / f"part-{existing:03d}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    return write


@pytest.fixture
def mtimes() -> Callable[[Path], dict[str, int]]:
    """`{file name: mtime_ns}` of every parquet shard in a directory (to prove a stage did not rewrite them)."""

    def collect(directory: Path) -> dict[str, int]:
        return {p.name: p.stat().st_mtime_ns for p in sorted(directory.glob("*.parquet"))}

    return collect


@pytest.fixture
def read_rows() -> Callable[[Path], list[Row]]:
    """All rows of the `data-*.parquet` shards of a directory, in shard order."""

    def read(directory: Path) -> list[Row]:
        rows: list[Row] = []
        for path in sorted(directory.glob("data-*.parquet")):
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    return read


@pytest.fixture
def with_tokenizer(layout: DatasetLayout) -> Callable[[DatasetConfig], DatasetConfig]:
    """Run the tokenizer stage for a config (needed before any token counting) and hand the config back."""

    def prepare(cfg: DatasetConfig) -> DatasetConfig:
        prepare_tokenizer(cfg, layout)
        return cfg

    return prepare


@pytest.fixture
def config_file(tmp_path: Path) -> Callable[[DatasetConfig], Path]:
    """`config_file(cfg)` writes the config as YAML — what `prepare` / `status` take — and returns the path
    (`<tmp_path>/<cfg.name>.yaml`; a second call with the same name overwrites it)."""

    def write(cfg: DatasetConfig) -> Path:
        path = tmp_path / f"{cfg.name}.yaml"
        path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
        return path

    return write
