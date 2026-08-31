# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The pipeline steps as functions, one namespace: ``download`` / ``download_github_code_group`` (``stages/download.py``,
with the tokenizer step, the raw-manifest state helpers and the shared manifest helpers), ``build_source``
(``stages/build.py``, both source kinds); row-level helpers in ``row_pipeline.py``, the exact-dedup filter in
``exact_dedup.py``, near-duplicate removal in ``fuzzy_dedup.py``, benchmark n-grams in ``benchmarks.py``, text
truncation at the token cap for the download in ``truncation.py``.
"""

from __future__ import annotations

from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.download import (
    DEFAULT_SHARD_SIZE,
    RawFolderError,
    RawManifestState,
    TokenCounter,
    current_raw_manifest,
    download,
    download_github_code_group,
    prepare_tokenizer,
    raw_manifest_problem,
    raw_manifest_state,
)

__all__ = [
    "DEFAULT_SHARD_SIZE",
    "RawFolderError",
    "RawManifestState",
    "TokenCounter",
    "build_source",
    "current_raw_manifest",
    "download",
    "download_github_code_group",
    "prepare_tokenizer",
    "raw_manifest_problem",
    "raw_manifest_state",
]
