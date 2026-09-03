# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Shared fixtures of the training tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def console_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Every run that opens its dashboard through `RunLogger.open` gets the console fallback, also under `pytest -s`
    on a terminal; the dashboard tests that want the live display pass `enabled=True` themselves.
    """

    monkeypatch.setenv("TRAINING_DASHBOARD", "0")
