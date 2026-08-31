# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the smoothed steps-per-second estimate and the ETA."""

from __future__ import annotations

import pytest

from training.ui.testing import FakeClock
from training.ui.throughput import Throughput


def test_throughput_eta_arithmetic(clock: FakeClock) -> None:
    throughput = Throughput(100, clock=clock)
    assert throughput.seconds_per_step is None and throughput.remaining(0) is None and throughput.steps_per_second is None
    clock.advance(10)
    throughput.record(10)  # 1 s/step sets the estimate
    assert throughput.seconds_per_step == pytest.approx(1.0)
    assert throughput.remaining(10) == pytest.approx(90.0)
    assert throughput.steps_per_second == pytest.approx(1.0)
    clock.advance(20)
    throughput.record(20)  # 2 s/step sample: 0.9 * 1 + 0.1 * 2
    assert throughput.seconds_per_step == pytest.approx(1.1)
    assert throughput.remaining(20) == pytest.approx(88.0)
    assert throughput.elapsed == pytest.approx(30.0)
    throughput.record(20)  # no progress: ignored
    throughput.record(5)  # going backwards: ignored
    assert throughput.seconds_per_step == pytest.approx(1.1)


def test_throughput_starts_at_the_resume_step(clock: FakeClock) -> None:
    throughput = Throughput(100, start_step=50, clock=clock)
    clock.advance(5)
    throughput.record(55)
    assert throughput.seconds_per_step == pytest.approx(1.0), "the 50 checkpointed steps are not counted as done now"
    assert throughput.remaining(55) == pytest.approx(45.0)
