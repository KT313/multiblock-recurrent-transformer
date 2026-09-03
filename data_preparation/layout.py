# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Where a dataset config's outputs live on disk (pure path arithmetic, no I/O).

The tree shows the download/build boundary: sources/ holds only downloaded data, processed/ only derived data.

    <root>/sources/<source>/raw/                         downloaded rows (text truncated to `max_seq_length` tokens,
                                                         + `tokens` column); append-only, shared by every dataset config;
                                                         deleted only when the source identity changes or the cap is
                                                         raised, after the user confirmed
    <root>/processed/<source>/                           cleaned rows (one flat folder per source, derived from raw,
                                                         cheap to rebuild, shared); what training reads
    <root>/tokenizers/<tokenizer name>/
    <root>/benchmarks/                                   cache of benchmark test sets used for decontamination
    <root>/hub_index/<repo>@<revision>/<glob hash>.json  file lists + row counts of `hf_files` / `github_code` repos
    <root>/.build.lock                                   one build per directory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# columns of a processed shard per source kind; `hash` (int64 exact-dedup key) lets the build append new shards
# instead of rewriting, and refills the dedup filter from disk
PRETRAIN_PROCESSED_COLUMNS: tuple[str, ...] = ("text", "source", "tokens", "hash")
INSTRUCT_PROCESSED_COLUMNS: tuple[str, ...] = ("instruction", "input", "output", "tokens", "hash")
# alias for the pretrain columns (kept for callers that only handle pretrain shards)
PROCESSED_COLUMNS: tuple[str, ...] = PRETRAIN_PROCESSED_COLUMNS


def processed_columns(kind: str) -> tuple[str, ...]:
    """
    Columns of a processed shard of a source of kind (pretrain or instruct).
    """

    if kind == "pretrain":
        return PRETRAIN_PROCESSED_COLUMNS
    if kind == "instruct":
        return INSTRUCT_PROCESSED_COLUMNS
    raise ValueError(f"unknown source kind {kind!r}; expected 'pretrain' or 'instruct'")


@dataclass(frozen=True)
class DatasetLayout:
    """
    Directory scheme under `root` (default `dataset/`); every method returns a path, none touches the disk.
    """

    root: Path = Path("dataset")

    def raw_dir(self, name: str) -> Path:
        """
        Downloaded rows of source name (the only tree the download step writes).
        """

        return self.root / "sources" / name / "raw"

    def processed_dir(self, name: str) -> Path:
        """
        Cleaned rows of source name (the only tree the build step writes).
        """

        return self.root / "processed" / name

    def tokenizer_dir(self, name: str) -> Path:
        return self.root / "tokenizers" / name

    def benchmark_cache_dir(self) -> Path:
        return self.root / "benchmarks"

    def hub_index_dir(self) -> Path:
        return self.root / "hub_index"
