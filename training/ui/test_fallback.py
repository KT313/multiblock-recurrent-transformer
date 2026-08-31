# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the console fallback of the training dashboard."""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import pytest

from training.ui.fallback import NoOpDashboard
from training.ui.format import TRANSITION_FLAG_KEY, TRANSITION_PROGRESS_KEY
from training.ui.testing import STAGES, STEPS, TOTAL, FakeClock, metrics


def test_fallback_logs_one_line_per_interval_and_writes_the_log_file(tmp_path: Path, clock: FakeClock) -> None:
    stream = io.StringIO()
    log_file = tmp_path / "train.log"
    real_out = sys.stdout
    # the fallback's lines are emitted on `training.ui.dashboard`: they reach the `training` logger `open()` attaches
    with NoOpDashboard.open("r", STAGES, STEPS, TOTAL, log_step_interval=5, log_file=log_file, stream=stream, clock=clock) as b:
        assert sys.stdout is real_out, "the fallback captures nothing"
        b.set_status("training")  # DEBUG: not printed
        b.note_event("no checkpoint found, starting from scratch")
        for step in range(1, 13):
            clock.advance(2)
            b.update_step(step, 0, metrics(step, loss=3.0))
        b.update_step(18, 0, metrics(18, **{TRANSITION_FLAG_KEY: 1.0, TRANSITION_PROGRESS_KEY: 0.5}))
        b.update_validation(10, {"val_loss_4": 3.25, "val_loss": 3.125})
        b.update_step(TOTAL, 1, metrics(TOTAL))  # the last step is always logged
        with b.suspended():
            pass
    lines = [line.split(": ", 1)[1] for line in stream.getvalue().splitlines()]
    assert lines[0] == "event: no checkpoint found, starting from scratch"
    step_lines = [line for line in lines if line.startswith("step ") and "/" in line.split(" | ")[0]]
    assert [line.split(" | ")[0] for line in step_lines] == ["step 5/30", "step 10/30", "step 30/30"]
    assert step_lines[0].split(" | ")[1:] == [
        "stage 0 pretrain", "loss 3.0000", f"ppl {math.exp(3.0):.2f}", "lr 3.00e-04", "grad norm 1.250",
        "tokens/s 12,346", "tokens 40.96k", "s/step 2.00s", "elapsed 0:00:10", "ETA 0:00:50",
    ]
    assert "step 10: validation val_loss_4 3.2500, val_loss 3.1250" in lines
    assert not any("status" in line for line in lines)
    assert log_file.read_text().splitlines()[0].endswith(lines[0]) and "step 30/30" in log_file.read_text()


def test_fallback_step_line_shows_the_transition(clock: FakeClock) -> None:
    stream = io.StringIO()
    with NoOpDashboard.open("r", STAGES, STEPS, TOTAL, stream=stream, clock=clock) as b:
        clock.advance(18)
        b.update_step(18, 0, {"loss": 2.0, TRANSITION_FLAG_KEY: 1.0, TRANSITION_PROGRESS_KEY: 0.5})
        b.update_step(TOTAL, 5, {})  # an unknown stage index renders as "?"
        b.update_validation(TOTAL, {})
    output = stream.getvalue()
    assert "step 18/30 | stage 0 pretrain | transition 50% | loss 2.0000 | s/step 1.00s | elapsed 0:00:18 | ETA 0:00:12" in output
    assert "step 30/30 | stage 5 ?" in output and "step 30: validation (no losses)" in output


def test_fallback_rejects_mismatched_stage_lists() -> None:
    with pytest.raises(ValueError, match="2 stage names for 1 step counts"):
        NoOpDashboard("r", STAGES, [5], 5)


def test_fallback_write_is_a_plain_line() -> None:
    stream = io.StringIO()
    NoOpDashboard("r", STAGES, STEPS, TOTAL, stream=stream).write("hello", keep=True)
    assert stream.getvalue() == "hello\n"
