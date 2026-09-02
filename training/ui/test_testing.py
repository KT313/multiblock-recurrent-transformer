# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the helpers of the dashboard tests."""

from __future__ import annotations

from training.ui.testing import FakeClock, metrics


def test_fake_clock_advances_by_hand() -> None:
    clock = FakeClock()
    assert clock() == 0.0
    clock.advance(2.5)
    assert clock() == 2.5


def test_metrics_is_a_step_dict() -> None:
    step = metrics(3, loss=2.0, extra=1.0)
    assert step["loss"] == 2.0 and step["total_tokens"] == 3 * 8_192 and step["extra"] == 1.0
