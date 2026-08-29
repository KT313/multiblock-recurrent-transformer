# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The pipeline stages as functions, one namespace: ``download``, ``length_filter``, ``process``, ``holdout``,
``build_mixture``, ``prepare_tokenizer``. Implementation split by source kind into ``stages_shared.py``
(tokenizer / download / holdout + helpers), ``stages_pretrain.py`` and ``stages_instruct.py``.
"""

from __future__ import annotations

from data_preparation.lib.stages_instruct import build_mixture
from data_preparation.lib.stages_pretrain import length_filter, process
from data_preparation.lib.stages_shared import DEFAULT_SHARD_SIZE, TokenCounter, download, holdout, prepare_tokenizer

__all__ = [
    "DEFAULT_SHARD_SIZE",
    "TokenCounter",
    "build_mixture",
    "download",
    "holdout",
    "length_filter",
    "prepare_tokenizer",
    "process",
]
