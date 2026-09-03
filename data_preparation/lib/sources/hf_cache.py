# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The HuggingFace cache location, set from the CLI before any HF library is imported.
"""

from __future__ import annotations

import os
from pathlib import Path

from data_preparation.lib.log import get_logger

log = get_logger(__name__)


def configure_hf_cache(cache_dir: Path | None) -> None:
    """
    Point every HuggingFace cache at cache_dir; must run before datasets/transformers are imported.
    """

    if cache_dir is None:
        return
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        os.environ[var] = str(cache_dir)
    log.info("using HuggingFace cache %s", cache_dir)
