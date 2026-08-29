# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Area fixtures for data_preparation: keep the `datasets` cache out of the user's home and offline; builders for
small dataset configs over `synthetic` / `local` sources used by the stage and planner tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.dataset_config import (
    DatasetConfig,
    MixtureConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
)
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.stages_shared import prepare_tokenizer

# Must happen before `datasets` is imported anywhere (its config reads the env at import time).
_CACHE = tempfile.mkdtemp(prefix="hf_datasets_cache_")
os.environ.setdefault("HF_DATASETS_CACHE", _CACHE)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

Row = dict[str, Any]
CfgFactory = Callable[..., DatasetConfig]


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


@pytest.fixture
def cfg_factory() -> CfgFactory:
    """`make(sources, mixtures=..., processing=..., token_count=..., max_seq_length=..., tokens=...)` -> DatasetConfig.

    One stage trains on every pretrain source (equal weights) and validates on the holdout sources (or, without
    any, on the pretrain sources); every mixture gets its own finetune stage; instruct sources outside every
    mixture are wrapped in an `auto` mixture. Without any trainable source a synthetic `_pretrain` source is added
    so the config validates. `tokens` is the per-stage budget.
    """

    def make(
        sources: dict[str, SourceConfig],
        *,
        mixtures: dict[str, MixtureConfig] | None = None,
        processing: ProcessingConfig | None = None,
        token_count: str = "tokenizer",
        max_seq_length: int = 64,
        tokens: int = 10_000,
        tokenizer: TokenizerConfig | None = None,
        name: str = "t",
    ) -> DatasetConfig:
        sources = dict(sources)
        mixtures = dict(mixtures or {})
        in_mixture = {src for m in mixtures.values() for src in m.sources}
        loose = [n for n, s in sources.items() if s.kind == "instruct" and n not in in_mixture]
        if loose:
            mixtures["auto"] = MixtureConfig(sources={n: 1.0 / len(loose) for n in loose})
        pretrain = [n for n, s in sources.items() if s.kind == "pretrain"]
        holdouts = [n for n, s in sources.items() if s.kind == "holdout"]
        if not pretrain and not mixtures:
            sources["_pretrain"] = SourceConfig(kind="pretrain", loader="synthetic")
            pretrain = ["_pretrain"]
        stages: list[StageConfig] = []
        if pretrain:
            val_names = holdouts or pretrain
            stages.append(
                StageConfig(
                    name="pretrain",
                    tokens=tokens,
                    train={n: 1.0 / len(pretrain) for n in pretrain},
                    val={n: 1.0 / len(val_names) for n in val_names},
                )
            )
        for mixture_name in mixtures:
            stages.append(
                StageConfig(
                    name=f"finetune_{mixture_name}",
                    tokens=tokens,
                    train={mixture_name: 1.0},
                    val={f"{mixture_name}/validation": 1.0},
                )
            )
        return DatasetConfig(
            name=name,
            tokenizer=tokenizer or TokenizerConfig(name="synthetic", kind="synthetic"),
            sources=sources,
            stages=stages,
            mixtures=mixtures,
            max_seq_length=max_seq_length,
            token_count=token_count,  # type: ignore[arg-type]  # Literal narrowed by the caller
            processing=processing or ProcessingConfig(min_chars=1),
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
