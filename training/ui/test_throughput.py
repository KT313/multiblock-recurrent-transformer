# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the smoothed steps-per-second estimate and the ETA.
"""

from __future__ import annotations

import pytest

from training.ui.throughput import Throughput
from ui.testing import FakeClock


def test_throughput_eta_arithmetic(clock: FakeClock) -> None:
    throughput = Throughput(100, clock=clock)
    assert throughput.seconds_per_step is None and throughput.remaining(0) is None and throughput.steps_per_second is None
    clock.advance(10)
    throughput.record(10)  # 1 s/step sets the estimate
    assert throughput.seconds_per_step == pytest.approx(1.0)
    assert throughput.remaining(10) == pytest.approx(90.0)
    assert throughput.steps_per_second == pytest.approx(1.0)
    clock.advance(20)
    throughput.record(20)  # 2 s/step: the second sample replaces the provisional first one
    assert throughput.seconds_per_step == pytest.approx(2.0)
    assert throughput.remaining(20) == pytest.approx(160.0)
    assert throughput.elapsed == pytest.approx(30.0)
    clock.advance(10)
    throughput.record(30)  # 1 s/step sample: from here on the EMA, 0.96 * 2 + 0.04 * 1
    assert throughput.seconds_per_step == pytest.approx(1.96)
    throughput.record(30)  # no progress: ignored
    throughput.record(5)  # going backwards: ignored
    assert throughput.seconds_per_step == pytest.approx(1.96)


def test_throughput_forgets_the_first_interval(clock: FakeClock) -> None:
    """
    The first interval holds the compile and the loader start-up; the ETA must not carry it for dozens of steps.
    """

    throughput = Throughput(1000, clock=clock)
    clock.advance(60)
    throughput.record(1)  # 60 s: the compile step
    clock.advance(1)
    throughput.record(2)
    assert throughput.seconds_per_step == pytest.approx(1.0)
    assert throughput.remaining(2) == pytest.approx(998.0)


def test_discount_keeps_a_block_that_was_not_training_out_of_the_estimate(clock: FakeClock) -> None:
    """
    L-H1: a 30 s evaluation between two 0.5 s steps must not read as a 60x slowdown; `elapsed` stays wall time.
    """

    throughput = Throughput(100, clock=clock)
    clock.advance(0.5)
    throughput.record(1)
    clock.advance(30.0)
    throughput.discount(30.0)
    clock.advance(0.5)
    throughput.record(2)
    assert throughput.seconds_per_step == pytest.approx(0.5)
    assert throughput.remaining(2) == pytest.approx(49.0)
    assert throughput.elapsed == pytest.approx(31.0), "elapsed is the run's wall time, the evaluation included"


def test_discount_of_more_than_the_interval_never_makes_a_step_negative(clock: FakeClock) -> None:
    """
    The blocks are timed on the logger's clock and discounted on the dashboard's; a rounding difference must not
    turn into a negative sample.
    """

    throughput = Throughput(100, clock=clock)
    clock.advance(1.0)
    throughput.record(1)
    throughput.discount(5.0)
    clock.advance(1.0)
    throughput.record(2)
    assert throughput.seconds_per_step == pytest.approx(0.0)


def test_throughput_starts_at_the_resume_step(clock: FakeClock) -> None:
    throughput = Throughput(100, start_step=50, clock=clock)
    clock.advance(5)
    throughput.record(55)
    assert throughput.seconds_per_step == pytest.approx(1.0), "the 50 checkpointed steps are not counted as done now"
    assert throughput.remaining(55) == pytest.approx(45.0)
