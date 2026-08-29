# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Where a dataset config's outputs live on disk (pure path arithmetic, no I/O).

    <root>/sources/<source>/{raw,filtered,processed}/   shared by every dataset config (append-only source cache)
    <root>/sources/<source>/holdout/                     held-out validation rows of a `holdout` source
    <root>/mixtures/<config name>/<mixture>/{train,validation}/
    <root>/tokenizers/<tokenizer name>/
    <root>/benchmarks/                                   cache of benchmark test sets used for decontamination
    <root>/hub_index/<repo>@<revision>/<glob hash>.json  file lists + row counts of `hf_files` / `github_code` repos
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SourceStage = Literal["raw", "filtered", "processed"]
MixtureSplit = Literal["train", "validation"]
SOURCE_STAGES: tuple[str, ...] = ("raw", "filtered", "processed")
MIXTURE_SPLITS: tuple[str, ...] = ("train", "validation")


@dataclass(frozen=True)
class DatasetLayout:
    root: Path = Path("dataset")

    def source_dir(self, name: str, stage: str) -> Path:
        if stage not in SOURCE_STAGES:
            raise ValueError(f"unknown source stage {stage!r}; expected one of {SOURCE_STAGES}")
        return self.root / "sources" / name / stage

    def holdout_dir(self, name: str) -> Path:
        return self.root / "sources" / name / "holdout"

    def mixture_dir(self, config_name: str, mixture: str, split: str) -> Path:
        if split not in MIXTURE_SPLITS:
            raise ValueError(f"unknown mixture split {split!r}; expected one of {MIXTURE_SPLITS}")
        return self.root / "mixtures" / config_name / mixture / split

    def tokenizer_dir(self, name: str) -> Path:
        return self.root / "tokenizers" / name

    def benchmark_cache_dir(self) -> Path:
        return self.root / "benchmarks"

    def hub_index_dir(self) -> Path:
        return self.root / "hub_index"
