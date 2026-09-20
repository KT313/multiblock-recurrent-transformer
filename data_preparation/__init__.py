# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Dataset preparation driven by a dataset config (config/datasets/<name>.yaml).

Entry point: python data_preparation/prepare.py tiny [--dataset_config F] [--dataset_dir dataset], run from the
repo root. CLI support lives in cli/; the config schema, layout, and implementation live in lib/:
sources (loaders, converters), stages (pipeline steps), storage
(manifests, parquet), build (planner, runner). Outputs:

    dataset/sources/<source>/raw/      rows as downloaded, shared by every dataset config
    dataset/processed/<source>/        cleaned rows (what training reads); the validation split is made at training time
    dataset/tokenizers/<name>/
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_preparation.lib.dataset_config import (
        DatasetConfig as DatasetConfig,
        SourceConfig as SourceConfig,
        StageConfig as StageConfig,
        TokenizerConfig as TokenizerConfig,
        load_dataset_config as load_dataset_config,
    )
    from data_preparation.lib.layout import DatasetLayout as DatasetLayout

__all__ = ["DatasetConfig", "DatasetLayout", "SourceConfig", "StageConfig", "TokenizerConfig", "load_dataset_config"]


def __getattr__(name: str) -> Any:
    """Load public exports on demand so importing the package does not initialize preparation dependencies."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "layout" if name == "DatasetLayout" else "dataset_config"
    value = getattr(import_module(f"{__name__}.lib.{module}"), name)
    globals()[name] = value  # subsequent imports reuse the same class or function
    return value
