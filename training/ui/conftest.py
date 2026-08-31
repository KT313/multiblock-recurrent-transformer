# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Fixtures of the dashboard tests: the hand-advanced clock and an open dashboard on a StringIO console."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from training.ui.board import TrainingDashboard
from training.ui.testing import LOGGER_NAME, STAGES, STEPS, TOTAL, FakeClock, string_console


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def board(clock: FakeClock) -> Iterator[TrainingDashboard]:
    """An enabled dashboard rendering into a StringIO console (no real terminal needed)."""
    with TrainingDashboard.open(
        "tiny-run",
        STAGES,
        STEPS,
        TOTAL,
        details={"model": "crow-tiny", "dataset": "tiny", "device": "cpu", "precision": "32"},
        logger=logging.getLogger(LOGGER_NAME),
        console=string_console(),
        clock=clock,
    ) as dashboard:
        yield dashboard
