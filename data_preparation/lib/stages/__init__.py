# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The pipeline stages as functions, one namespace: ``download``, ``length_filter``, ``process``, ``validation``,
``build_instruct_mixture``, ``prepare_tokenizer``. Implementation split by source kind into ``stages/shared.py``
(tokenizer / download / validation + helpers), ``stages/pretrain.py`` and ``stages/instruct.py``; row-level helpers in
``row_pipeline.py``, near-duplicate removal in ``fuzzy_dedup.py``, benchmark n-grams in ``benchmarks.py``.
"""

from __future__ import annotations

from data_preparation.lib.stages.instruct import build_instruct_mixture
from data_preparation.lib.stages.pretrain import length_filter, process
from data_preparation.lib.stages.shared import (
    DEFAULT_SHARD_SIZE,
    TokenCounter,
    download,
    download_github_code_group,
    prepare_tokenizer,
    validation,
)

__all__ = [
    "DEFAULT_SHARD_SIZE",
    "TokenCounter",
    "build_instruct_mixture",
    "download",
    "download_github_code_group",
    "validation",
    "length_filter",
    "prepare_tokenizer",
    "process",
]
