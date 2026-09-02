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

from training.ui.board import TrainingDashboard
from training.ui.dashboard import RunDashboard, training_dashboard
from training.ui.fallback import NoOpDashboard
from training.ui.testing import (
    BOX_CHARACTERS,
    LOGGER_NAME,
    STAGES,
    STEPS,
    TOTAL,
    FakeClock,
    console_output,
    metrics,
    screen_text,
    string_console,
)


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


def test_factory_gives_the_live_display_the_fallback_stream(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    """A display that disables itself after an internal error must write its plain lines where a run that never got
    a display writes them (the CLI passes stderr) — it used to build its fallback on ``stream``, i.e. stdout, which
    is exactly the failure path ``fallback_stream`` was added for."""
    stream, fallback = io.StringIO(), io.StringIO()

    def broken(step: int, stage_index: int) -> None:
        raise RuntimeError("renderer broke")

    with training_dashboard(
        "r", STAGES, STEPS, TOTAL, enabled=True,  # the default `training` logger: the fallback's own lines reach it
        log_step_interval=1, console=string_console(), stream=stream, fallback_stream=fallback, clock=clock,
    ) as b:
        assert isinstance(b, TrainingDashboard)
        monkeypatch.setattr(b, "_refresh_bars", broken)
        b.update_step(1, 0, None, metrics(1))  # disables the display
        b.update_step(2, 0, None, metrics(2))  # from here the fallback logs the step lines
        b.note_event("saved checkpoint x.pth")
    assert "step 2/30" in fallback.getvalue() and "event: saved checkpoint x.pth" in fallback.getvalue()
    assert "step 2/30" not in stream.getvalue()


def test_factory_passes_final_frame_on(clock: FakeClock) -> None:
    console = string_console()
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logging.getLogger(LOGGER_NAME), enabled=True, final_frame=False, console=console, clock=clock) as b:
        b.update_step(3, 0, None, metrics(3))
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
                transition = (step - 18) / 2 if 18 <= step <= 20 else None
                board.update_step(step, stage_index, transition, metrics(step, loss=4.0 - step * 0.05))  # every step
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
    assert "step 5/30 | stage 0 pretrain | loss 3.7500" in log_text and "step 20/30 | stage 0 pretrain | transition 100%" in log_text
    assert "step 30: validation val_loss_4 3.5000, val_loss 3.4000" in log_text and "event: exported HuggingFace model" in log_text
    assert "step 5/30" not in console_output(console) and "event: " not in console_output(console), "the fallback's lines: the file only"
    assert board._live is None
    screen = screen_text(console, 120)
    assert not any(character in screen for character in BOX_CHARACTERS), screen
    for kept in ("Total training steps: 30", "a bare stderr write", "WARNING fake_datasets_library: a third-party warning", "Training finished after 30 steps"):
        assert screen.count(kept) == 1, (kept, screen)
    assert "a stray print" not in screen and "grad metrics" not in screen, "INFO records are not kept"
    assert screen.count("overall") == 1 and screen.count("grad norm") == 1 and screen.count("30/30") == 1, screen
    assert screen.index("Training finished") < screen.index("overall"), "kept lines first, then the summary"
    assert screen.rstrip().splitlines()[-1].endswith("exported HuggingFace model to outputs/tiny/hf_export")


# --- the log file under both dashboards --------------------------------------------------------------------------------------


def _drive_scripted_run(board: RunDashboard, clock: FakeClock, logger: logging.Logger) -> None:
    """The same calls on either dashboard, the way ``RunLogger`` makes them: the transition progress at every step,
    the metric dict at log steps only, ``{}`` at the others, validations, events and two records of the attached
    logger."""
    board.note_event("no checkpoint found, starting from scratch")
    logger.info("Total training steps: %d", TOTAL, extra={"keep": True})
    board.set_status("training")
    for step in range(1, TOTAL + 1):
        clock.advance(0.5)
        stage_index = 0 if step <= STEPS[0] else 1
        transition = (step - 18) / 2 if 18 <= step <= 20 else None
        if step % 5:
            board.update_step(step, stage_index, transition, {})
            continue
        board.update_step(step, stage_index, transition, metrics(step, loss=4.0 - step * 0.05))
        if step % 10 == 0:
            board.update_validation(step, {"val_loss_4": 3.5, "val_loss": 3.4})
        if step == 20:
            board.note_event("saved checkpoint outputs/tiny/checkpoints/step-00000020-tiny-stage-0_end.pth")
    board.note_event("exported HuggingFace model to outputs/tiny/hf_export")
    logger.warning("Training finished after %d steps", TOTAL)


# what the scripted run leaves in `train.log` (message part, in order; the step lines checked by their prefix)
_EXPECTED_LOG_MESSAGES = [
    "event: no checkpoint found, starting from scratch",
    "Total training steps: 30",
    "step 5/30 | stage 0 pretrain | loss 3.7500 | ppl",
    "step 10/30 | stage 0 pretrain | loss 3.5000 | ppl",
    "step 10: validation val_loss_4 3.5000, val_loss 3.4000",
    "step 15/30 | stage 0 pretrain | loss 3.2500 | ppl",
    "step 20/30 | stage 0 pretrain | transition 100% | loss 3.0000 | ppl",
    "step 20: validation val_loss_4 3.5000, val_loss 3.4000",
    "event: saved checkpoint outputs/tiny/checkpoints/step-00000020-tiny-stage-0_end.pth",
    "step 25/30 | stage 1 instruct | loss 2.7500 | ppl",
    "step 30/30 | stage 1 instruct | loss 2.5000 | ppl",
    "step 30: validation val_loss_4 3.5000, val_loss 3.4000",
    "event: exported HuggingFace model to outputs/tiny/hf_export",
    "Training finished after 30 steps",
]


def test_live_dashboard_writes_the_fallback_lines_to_the_log_file_only(tmp_path: Path, clock: FakeClock) -> None:
    logger = logging.getLogger("training")  # as in a run: the fallback's lines are logged under `training`
    console = string_console(120, height=40)
    log_file = tmp_path / "train.log"
    with training_dashboard("tiny", STAGES, STEPS, TOTAL, log_step_interval=5, log_file=log_file, logger=logger, enabled=True, console=console, clock=clock) as board:
        assert isinstance(board, TrainingDashboard)
        _drive_scripted_run(board, clock, logger)
        panel = board.lines()
    messages = [line.split(": ", 1)[1] for line in log_file.read_text().splitlines() if "training.ui.dashboard: status:" not in line]
    assert len(messages) == len(_EXPECTED_LOG_MESSAGES), messages
    for message, expected in zip(messages, _EXPECTED_LOG_MESSAGES):
        assert message.startswith(expected), (message, expected)
    assert all("INFO training.ui.dashboard: step 5/30" in line for line in log_file.read_text().splitlines() if "step 5/30" in line)
    raw = console_output(console)
    for text in ("step 5/30", "step 30/30", "validation val_loss_4", "event: "):
        assert text not in raw and not any(text in line for line in panel), f"{text!r} reached the terminal or the panel"


def test_live_and_fallback_dashboards_write_identical_log_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 1_756_000_000.0)  # `logging` stamps records with `time.time`: one asctime in both
    logger = logging.getLogger("training")
    files: dict[str, bytes] = {}
    for name, enabled in (("live", True), ("fallback", False)):
        clock = FakeClock()
        log_file = tmp_path / f"{name}.log"
        with training_dashboard(
            "tiny", STAGES, STEPS, TOTAL, log_step_interval=5, log_file=log_file, logger=logger, enabled=enabled,
            console=string_console(120, height=40), stream=io.StringIO(), clock=clock,
        ) as board:
            assert isinstance(board, TrainingDashboard if enabled else NoOpDashboard)
            _drive_scripted_run(board, clock, logger)
        files[name] = log_file.read_bytes()
    assert files["live"] == files["fallback"]
    assert files["live"].count(b"\n") >= len(_EXPECTED_LOG_MESSAGES) and b"step 20/30 | stage 0 pretrain | transition 100%" in files["live"]
