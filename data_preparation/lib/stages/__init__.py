# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The pipeline steps as functions, one namespace: ``download`` / ``download_github_code_group`` (``stages/shared.py``,
with the tokenizer step and the helpers), ``build_source`` (``stages/build.py``, both source kinds); row-level
helpers in ``row_pipeline.py``, the exact-dedup filter in ``exact_dedup.py``, near-duplicate removal in
``fuzzy_dedup.py``, benchmark n-grams in ``benchmarks.py``, text truncation for the download in ``truncation.py``.
"""

from __future__ import annotations

from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.shared import (
    DEFAULT_SHARD_SIZE,
    TokenCounter,
    download,
    download_github_code_group,
    prepare_tokenizer,
    truncate_raw_to_good_prefix,
)

__all__ = [
    "DEFAULT_SHARD_SIZE",
    "TokenCounter",
    "build_source",
    "download",
    "download_github_code_group",
    "prepare_tokenizer",
    "truncate_raw_to_good_prefix",
]
