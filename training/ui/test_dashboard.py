# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the public API of the training dashboard: the factory and the scripted 30-step run the way ``train()``
/ ``RunLogger`` will drive it, with everything that could print around the display firing."""

from __future__ import annotations

import io
import logging
import sys
import time
from collections import deque
from pathlib import Path

import pytest

from training.ui.dashboard import (
    TRANSITION_FLAG_KEY,
    TRANSITION_PROGRESS_KEY,
    NoOpDashboard,
    TrainingDashboard,
    training_dashboard,
)
from training.ui.testing import BOX_CHARACTERS, LOGGER_NAME, STAGES, STEPS, TOTAL, FakeClock, metrics, screen_text, string_console


def test_factory_picks_the_fallback_when_disabled(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    monkeypatch.setenv("TRAINING_DASHBOARD", "0")
    logger = logging.getLogger(LOGGER_NAME + ".factory")
    real_out = sys.stdout
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logger, stream=io.StringIO(), clock=clock) as b:
        assert isinstance(b, NoOpDashboard) and sys.stdout is real_out, "the fallback captures nothing"
    monkeypatch.setenv("TRAINING_DASHBOARD", "1")
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logger, stream=io.StringIO(), clock=clock) as b:
        assert isinstance(b, NoOpDashboard), "stdout is not a TTY"
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logger, enabled=True, console=string_console(), clock=clock) as b:
        assert isinstance(b, TrainingDashboard) and b._live is not None
    assert b._live is None


def test_factory_passes_final_frame_on(clock: FakeClock) -> None:
    console = string_console()
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logging.getLogger(LOGGER_NAME), enabled=True, final_frame=False, console=console, clock=clock) as b:
        b.update_step(3, 0, metrics(3))
    assert "overall" not in screen_text(console, 120)


# --- the scripted run -----------------------------------------------------------------------------------------------------


def _check_frame(frame: str, height: int, log_lines: int, event_lines: int) -> None:
    """One rendered frame: the header, the bars and the metrics table once, the panels bounded, the frame fits."""
    lines = frame.splitlines()
    assert len(lines) <= height, frame
    assert frame.count("overall") == 1 and frame.count("grad norm") == 1 and frame.count("▶") <= 1, frame
    assert lines[0].startswith("tiny  model=tiny"), frame
    events_panel = frame[frame.index("─ events") : frame.index("─ log")]
    assert sum(1 for line in events_panel.splitlines() if ": " in line) <= event_lines
    log_panel = frame[frame.index("─ log") :]
    assert sum(1 for line in log_panel.splitlines() if " INFO " in line or " WARNING " in line) <= log_lines
    for line in lines:
        if "steps/s" in line or "transition →" in line:
            assert "━" in line or "╺" in line, f"bar text outside the bar rows: {line!r}"


def test_scripted_thirty_step_run_drives_the_whole_api(tmp_path: Path, clock: FakeClock) -> None:
    """The way ``train()`` / ``RunLogger`` will drive the dashboard (task 10 of the training restructure plan), while
    log records, stray prints, a bare stderr write and a third-party logger with its own stderr handler fire."""
    logger = logging.getLogger("training")  # as in a run: `RunLogger` logs under `training`, the sinks too
    library = logging.getLogger("fake_datasets_library")
    library_handler = logging.StreamHandler(sys.stderr)
    library.addHandler(library_handler)
    log_file = tmp_path / "outputs" / "tiny" / "train.log"
    height, log_lines, event_lines = 40, 4, 3
    console = string_console(120, height=height)
    frames: list[str] = []
    try:
        with training_dashboard(
            "tiny",
            STAGES,
            STEPS,
            TOTAL,
            details={"model": "tiny", "dataset": "tiny", "device": "cpu", "precision": "32"},
            log_step_interval=5,
            log_file=log_file,
            logger=logger,
            enabled=True,
            console=console,
            clock=clock,
        ) as board:
            assert isinstance(board, TrainingDashboard)
            board._log_lines, board._lines = log_lines, deque(board._lines, maxlen=log_lines)  # smaller panels than the defaults
            board._events = deque(board._events, maxlen=event_lines)
            board.note_event("no checkpoint found, starting from scratch")  # RunLogger.log_fresh_start
            logger.info("Total training steps: %d", TOTAL, extra={"keep": True})  # RunLogger.open
            board.set_status("training")
            for step in range(1, TOTAL + 1):  # progress.step after advance()
                clock.advance(0.5)
                stage_index = 0 if step <= STEPS[0] else 1
                in_transition = 18 <= step <= 20
                step_metrics = metrics(step, loss=4.0 - step * 0.05)
                if in_transition:
                    step_metrics |= {TRANSITION_FLAG_KEY: 1.0, TRANSITION_PROGRESS_KEY: (step - 18) / 2}
                board.update_step(step, stage_index, step_metrics)  # RunLogger.log_step, every step
                logger.info("step %d: grad metrics", step)
                if step == 18:
                    board.note_event("starting transition 0 -> 1 (pretrain -> instruct)")
                if step % 10 == 0 or step == TOTAL:  # is_evaluation_step
                    board.set_status("evaluating")
                    board.update_validation(step, {"val_loss_4": 3.5, "val_loss": 3.4})
                    board.set_status("training")
                if step == 20:
                    board.note_event("transition complete, now in stage 1 (instruct)")
                    board.set_status("saving checkpoint")
                    board.note_event("saved checkpoint outputs/tiny/checkpoints/step-00000020-tiny-stage-0_end.pth")
                    board.set_status("training")
                if step == 7:
                    print("a stray print from somewhere")
                if step == 12:
                    sys.stderr.write("a bare stderr write\n")
                if step == 15:
                    library.warning("a third-party warning")
                if step == TOTAL:
                    board.note_event("saved checkpoint outputs/tiny/checkpoints/step-00000030-tiny.pth")
                frame = board.render_text(width=120, height=height)  # a snapshot under the lock, like a refresh
                frames.append(frame)
                _check_frame(frame, height, log_lines, event_lines)
            board.set_status("exporting")
            board.note_event("exported HuggingFace model to outputs/tiny/hf_export")
            logger.info("Training finished after %d steps", TOTAL, extra={"keep": True})  # RunLogger.close
            time.sleep(0.05)  # a few live frames on the console
            text = board.render_text(width=120, height=height)
            assert "a stray print from somewhere" not in screen_text(console, 120), "nothing is printed while the display is up"
    finally:
        library.removeHandler(library_handler)
    assert any("transition → instruct" in frame for frame in frames) and any("▶ instruct" in frame for frame in frames)
    assert "✓ pretrain" in text and "✓ instruct" in text, "every stage complete at the end"
    assert "20/20" in text and "10/10" in text and "30/30" in text and "100%" in text
    assert "2.00 steps/s" in text and "0:00:15 elapsed" in text and "ETA 0:00:00" in text
    assert "step 30" in text and "2.5000" in text  # 4.0 - 30 * 0.05
    assert "validation (step 30)" in text and "3.5000" in text and "3.4000" in text
    assert "exporting" in text
    assert "step-00000030-tiny.pth" in text and "hf_export" in text
    assert "Training finished after 30 steps" in text
    assert [task.completed for task in board.tasks] == [20, 10, 30]
    assert len(board.events()) == event_lines and board.events()[-1].endswith("exported HuggingFace model to outputs/tiny/hf_export")
    log_text = log_file.read_text()
    assert "Total training steps: 30" in log_text and "Training finished after 30 steps" in log_text
    assert "a stray print from somewhere" in log_text and "a bare stderr write" in log_text
    assert board._live is None
    screen = screen_text(console, 120)
    assert not any(character in screen for character in BOX_CHARACTERS), screen
    for kept in ("Total training steps: 30", "a bare stderr write", "WARNING fake_datasets_library: a third-party warning", "Training finished after 30 steps"):
        assert screen.count(kept) == 1, (kept, screen)
    assert "a stray print" not in screen and "grad metrics" not in screen, "INFO records are not kept"
    assert screen.count("overall") == 1 and screen.count("grad norm") == 1 and screen.count("30/30") == 1, screen
    assert screen.index("Training finished") < screen.index("overall"), "kept lines first, then the summary"
    assert screen.rstrip().splitlines()[-1].endswith("exported HuggingFace model to outputs/tiny/hf_export")
