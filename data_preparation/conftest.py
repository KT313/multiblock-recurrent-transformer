# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Area fixtures: keep the `datasets` cache out of the user's home and offline."""

import os
import tempfile
from collections.abc import Iterator
from types import ModuleType

import pytest

# Must happen before `datasets` is imported anywhere (its config reads the env at import time).
_CACHE = tempfile.mkdtemp(prefix="hf_datasets_cache_")
os.environ.setdefault("HF_DATASETS_CACHE", _CACHE)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture
def hf_datasets() -> Iterator[ModuleType]:
    """The `datasets` module with caching disabled (map/filter results stay in temp files)."""
    datasets = pytest.importorskip("datasets")
    datasets.disable_caching()
    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bars()
    yield datasets
    datasets.enable_caching()
