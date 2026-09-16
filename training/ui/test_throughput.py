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
    throughput.record(10)  # the entire first reported interval contains startup and is discarded
    assert throughput.seconds_per_step is None and throughput.remaining(10) is None
    clock.advance(20)
    throughput.record(20)  # 2 s/step: first complete post-startup sample initializes the EMA
    assert throughput.seconds_per_step == pytest.approx(2.0)
    assert throughput.remaining(20) == pytest.approx(160.0)
    assert throughput.elapsed == pytest.approx(30.0)
    clock.advance(10)
    throughput.record(30)  # 1 s/step sample: 0.9 * 2 + 0.1 * 1
    assert throughput.seconds_per_step == pytest.approx(1.9)
    throughput.record(30)  # no progress: ignored
    throughput.record(5)  # going backwards: ignored
    assert throughput.seconds_per_step == pytest.approx(1.9)


def test_throughput_forgets_the_first_interval(clock: FakeClock) -> None:
    """
    The first interval holds the compile and the loader start-up; the ETA must not carry it for dozens of steps.
    """

    throughput = Throughput(1000, clock=clock)
    clock.advance(60)
    throughput.record(1)  # model compilation, but optimizer.step is skipped at step index 0
    assert throughput.remaining(1) is None
    clock.advance(40)  # first actual optimizer update can compile its own kernels
    throughput.record(2)
    assert throughput.remaining(2) is None and throughput.steps_per_second is None
    clock.advance(1)
    throughput.record(3)
    assert throughput.seconds_per_step == pytest.approx(1.0)
    assert throughput.remaining(3) == pytest.approx(997.0)
    assert throughput.elapsed == pytest.approx(101.0)


def test_discount_keeps_a_block_that_was_not_training_out_of_the_estimate(clock: FakeClock) -> None:
    """
    L-H1: a 30 s evaluation between two 0.5 s steps must not read as a 60x slowdown; `elapsed` stays wall time.
    """

    throughput = Throughput(100, start_step=10, clock=clock)
    clock.advance(0.5)
    throughput.record(11)
    clock.advance(30.0)
    throughput.discount(30.0)
    clock.advance(0.5)
    throughput.record(12)
    assert throughput.seconds_per_step == pytest.approx(0.5)
    assert throughput.remaining(12) == pytest.approx(44.0)
    assert throughput.elapsed == pytest.approx(31.0), "elapsed is the run's wall time, the evaluation included"


def test_discount_of_more_than_the_interval_never_makes_a_step_negative(clock: FakeClock) -> None:
    """
    The blocks are timed on the logger's clock and discounted on the dashboard's; a rounding difference must not
    turn into a negative sample.
    """

    throughput = Throughput(100, start_step=10, clock=clock)
    clock.advance(1.0)
    throughput.record(11)
    throughput.discount(5.0)
    clock.advance(1.0)
    throughput.record(12)
    assert throughput.seconds_per_step == pytest.approx(0.0)


def test_throughput_starts_at_the_resume_step(clock: FakeClock) -> None:
    throughput = Throughput(100, start_step=50, clock=clock)
    clock.advance(5)
    throughput.record(55)
    assert throughput.seconds_per_step is None, "the first interval after resume is warmup too"
    clock.advance(5)
    throughput.record(60)
    assert throughput.seconds_per_step == pytest.approx(1.0), "the 50 checkpointed steps are not counted as done now"
    assert throughput.remaining(60) == pytest.approx(40.0)


@pytest.mark.parametrize("start", [0, 1, 50])
def test_warmup_ends_after_first_actual_update_and_keeps_elapsed(clock: FakeClock, start: int) -> None:
    throughput = Throughput(100, start_step=start, clock=clock)
    first_update_done = max(start, 1) + 1
    for completed in range(start + 1, first_update_done + 1):
        clock.advance(100)
        throughput.record(completed)
        assert throughput.remaining(completed) is None
    elapsed = throughput.elapsed
    clock.advance(2)
    throughput.record(first_update_done + 1)
    assert throughput.seconds_per_step == 2
    assert throughput.elapsed == elapsed + 2
    clock.advance(4)
    throughput.record(first_update_done + 2)
    assert throughput.seconds_per_step == pytest.approx(2.2)
