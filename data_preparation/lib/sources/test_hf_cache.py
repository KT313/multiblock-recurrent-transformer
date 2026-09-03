# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.sources.hf_cache.
"""

import os
from pathlib import Path

import pytest

from data_preparation.lib.sources.hf_cache import configure_hf_cache


def test_configure_hf_cache_none_leaves_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        monkeypatch.delenv(var, raising=False)
    configure_hf_cache(None)
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        assert var not in os.environ


def test_configure_hf_cache_sets_all_vars_and_creates_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv *before* the call records the original value (or its absence) so monkeypatch restores it; a
    # delenv afterwards would "restore" the value the test itself set and leak the temp cache path into the
    # rest of the session, and delenv(raising=False) on an absent variable records nothing at all
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        monkeypatch.setenv(var, "placeholder")
    cache = tmp_path / "hf_cache"
    configure_hf_cache(cache)
    assert cache.is_dir()
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        assert os.environ[var] == str(cache.resolve())
