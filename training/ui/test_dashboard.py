# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the training dashboard (rich live view of stages, metrics, validation, events, logs) and its fallback."""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console

from training.ui.dashboard import (
    TRANSITION_FLAG_KEY,
    TRANSITION_PROGRESS_KEY,
    DashboardLogHandler,
    NoOpDashboard,
    Throughput,
    TrainingDashboard,
    dashboard_enabled,
    format_duration,
    format_metric,
    format_tokens,
    training_dashboard,
)

STAGES = ["pretrain", "instruct"]
STEPS = [20, 10]
TOTAL = 30
LOGGER_NAME = "training.test_dashboard"


class FakeClock:
    """A clock the tests advance by hand (injected as ``clock=``)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _console(width: int = 120) -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=width)


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
        console=_console(),
        clock=clock,
    ) as dashboard:
        dashboard._refresh_per_second = 50  # tests never wait for a frame; keep the Live thread quiet
        yield dashboard


def _metrics(step: int, loss: float = 3.0, **extra: float) -> dict[str, float]:
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "lr": 3e-4,
        "grad_norm": 1.25,
        "tokens/second": 12_345.6,
        "total_tokens": step * 8_192,
        **extra,
    }


# --- formatting helpers -----------------------------------------------------------------------------------------------------


def test_format_duration() -> None:
    assert format_duration(None) == "—"
    assert format_duration(float("inf")) == "—"
    assert format_duration(-1) == "—"
    assert format_duration(0) == "0:00:00"
    assert format_duration(3_725.9) == "1:02:05"
    assert format_duration(90_061) == "1d 01:01:01"


def test_format_tokens() -> None:
    assert format_tokens(999) == "999"
    assert format_tokens(1_500) == "1.50k"
    assert format_tokens(2_500_000) == "2.50M"
    assert format_tokens(1.234e9) == "1.23B"
    assert format_tokens(3e12) == "3.00T"


def test_format_metric() -> None:
    assert format_metric("loss", 3.14159) == "3.1416"
    assert format_metric("ppl", 23.1) == "23.10"
    assert format_metric("lr", 0.0003) == "3.00e-04"
    assert format_metric("grad_norm", 1.23456) == "1.235"
    assert format_metric("tokens/second", 12345.6) == "12,346"
    assert format_metric("seconds/step", 0.5) == "0.50s"
    assert format_metric("total_tokens", 2e9) == "2.00B"
    assert format_metric("other", 0.123456) == "0.1235"


# --- throughput / ETA -----------------------------------------------------------------------------------------------------


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


# --- rendering ------------------------------------------------------------------------------------------------------------


def test_header_shows_run_details_and_status(board: TrainingDashboard) -> None:
    text = board.render_text()
    assert "tiny-run" in text
    assert "model=crow-tiny" in text and "dataset=tiny" in text and "device=cpu" in text and "precision=32" in text
    assert "starting" in text
    board.set_status("evaluating")
    text = board.render_text()
    assert "evaluating" in text and "starting" not in text


def test_stage_bars_and_overall_bar(board: TrainingDashboard) -> None:
    text = board.render_text()
    assert "▶ pretrain" in text and "  instruct" in text and "overall" in text
    assert "0/20" in text and "0/10" in text and "0/30" in text
    board.update_step(7, 0, _metrics(7))
    text = board.render_text()
    assert "7/20" in text and "0/10" in text and "7/30" in text
    assert [task.completed for task in board.tasks] == [7, 0, 7]


def test_stage_transition_moves_the_highlight(board: TrainingDashboard) -> None:
    board.update_step(18, 0, _metrics(18, **{TRANSITION_FLAG_KEY: 1.0, TRANSITION_PROGRESS_KEY: 0.5}))
    text = board.render_text()
    assert "▶ pretrain" in text and "transition → instruct 50%" in text
    board.update_step(23, 1, _metrics(23))
    text = board.render_text()
    assert "✓ pretrain" in text and "▶ instruct" in text
    assert "20/20" in text and "3/10" in text and "23/30" in text
    assert "transition" not in text


def test_last_stage_shows_no_transition_note(board: TrainingDashboard) -> None:
    board.update_step(29, 1, _metrics(29, **{TRANSITION_FLAG_KEY: 1.0, TRANSITION_PROGRESS_KEY: 0.9}))
    assert "transition" not in board.render_text()


def test_stage_names_with_markup_characters_render_literally(clock: FakeClock) -> None:
    with TrainingDashboard.open("r", ["[bold]x[/bold]"], [1], 1, logger=logging.getLogger(LOGGER_NAME), console=_console(), clock=clock) as b:
        assert "[bold]x[/bold]" in b.render_text()


def test_overall_bar_eta_from_the_injected_clock(board: TrainingDashboard, clock: FakeClock) -> None:
    assert "ETA —" in board.render_text() and "? steps/s" in board.render_text()
    clock.advance(20)
    board.update_step(10, 0, _metrics(10))  # 2 s/step, 20 steps left
    text = board.render_text()
    assert "0.50 steps/s" in text and "0:00:20 elapsed" in text and "ETA 0:00:40" in text
    assert "0:00:20" in text and "0:00:40" in text  # the metrics table's elapsed / remaining columns


def test_metrics_table_shows_the_latest_step(board: TrainingDashboard) -> None:
    board.update_step(5, 0, _metrics(5, loss=2.5, **{"seconds/step": 0.25}))
    text = board.render_text()
    assert "step 5" in text
    assert "2.5000" in text  # loss
    assert f"{math.exp(2.5):.2f}" in text  # ppl
    assert "3.00e-04" in text  # lr
    assert "1.250" in text  # grad norm
    assert "12,346" in text  # tokens/s
    assert "0.25s" in text  # s/step from the step dict
    assert "40.96k" in text  # 5 * 8192 tokens
    assert board.latest_metrics["loss"] == 2.5


def test_metrics_missing_from_a_step_keep_their_last_value(board: TrainingDashboard) -> None:
    board.update_step(1, 0, _metrics(1, loss=4.0))
    board.update_step(2, 0, {"loss": 3.5})  # a non-log step: only the loss
    assert board.latest_metrics["loss"] == 3.5 and board.latest_metrics["lr"] == 3e-4
    assert "3.5000" in board.render_text()


def test_metrics_table_uses_the_loops_own_timing_without_seconds_per_step(board: TrainingDashboard, clock: FakeClock) -> None:
    clock.advance(4)
    board.update_step(2, 0, _metrics(2))
    assert "2.00s" in board.render_text()


def test_validation_losses_render_per_depth(board: TrainingDashboard) -> None:
    assert "validation" not in board.render_text()
    board.update_validation(10, {"val_loss_4": 3.4567, "val_loss_8": 3.2, "val_loss": 3.1})
    text = board.render_text()
    assert "validation (step 10)" in text
    assert "val_loss_4" in text and "3.4567" in text and "val_loss_8" in text and "3.2000" in text and "3.1000" in text


def test_events_list_keeps_the_last_lines(clock: FakeClock) -> None:
    with TrainingDashboard(
        "r", STAGES, STEPS, TOTAL, event_lines=2, console=_console(), clock=clock
    ) as b:
        assert "(no events yet)" in b.render_text()
        b.update_step(3, 0, _metrics(3))
        b.note_event("saved checkpoint outputs/r/checkpoints/step-00000003-r.pth")
        b.update_step(6, 0, _metrics(6))
        b.note_event("starting transition 0 -> 1")
        b.note_event("saved checkpoint step-00000006-r.pth")
        events = b.events()
        assert len(events) == 2
        assert events[0].endswith("step 6: starting transition 0 -> 1")
        assert events[1].endswith("step 6: saved checkpoint step-00000006-r.pth")
        text = b.render_text()
        assert "step-00000006-r.pth" in text and "step-00000003-r.pth" not in text


def test_bars_for_a_zero_length_stage_render(clock: FakeClock) -> None:
    with TrainingDashboard("r", ["a", "empty", "b"], [5, 0, 5], 10, console=_console(), clock=clock) as b:
        b.update_step(5, 2, _metrics(5))
        text = b.render_text()
        assert "✓ a" in text and "0/0" in text and "▶ b" in text


# --- logging capture ---------------------------------------------------------------------------------------------------------


def test_log_records_land_in_the_panel(board: TrainingDashboard) -> None:
    logging.getLogger(LOGGER_NAME + ".child").info("hello %d", 7)
    (line,) = board.lines()
    assert f"INFO {LOGGER_NAME}.child: hello 7" in line
    assert "hello 7" in board.render_text()


def test_log_panel_keeps_last_lines_and_splits_multi_line_records(clock: FakeClock) -> None:
    with TrainingDashboard("r", STAGES, STEPS, TOTAL, log_lines=3, console=_console(), clock=clock) as b:
        for i in range(5):
            b.write(f"line {i}")
        assert b.lines() == ["line 2", "line 3", "line 4"]
        b.write("Traceback:\n  File x\nValueError: boom")
        assert b.lines() == ["Traceback:", "  File x", "ValueError: boom"]
        b.write("path [bold]x[/bold]")
        assert "[bold]x[/bold]" in b.render_text(), "markup in log lines is not interpreted"


def _scrollback(console: Console) -> str:
    """What the console printed outside the live frames (panel / bar lines start with box characters)."""
    output = console.file.getvalue()  # type: ignore[attr-defined]  # StringIO console
    return "\n".join(
        line for line in output.splitlines() if not line.lstrip("\x1b[0123456789;m").startswith(("│", "╭", "╰", "▶", "✓", "overall"))
    )


def test_warnings_are_printed_into_the_scrollback(clock: FakeClock) -> None:
    console = _console()
    logger = logging.getLogger(LOGGER_NAME + ".keep")
    with TrainingDashboard.open("r", STAGES, STEPS, TOTAL, logger=logger, console=console, clock=clock) as b:
        b._refresh_per_second = 0.001
        logger.info("quiet")
        logger.warning("loud")
        logger.info("table:\n%s", "a  b  c" + " " * 200 + "end", extra={"keep": True})
        scrollback = _scrollback(console)
        assert [line.split(": ")[-1] for line in b.lines()[:2]] == ["quiet", "loud"]
    assert "WARNING training.test_dashboard.keep: loud" in scrollback and "quiet" not in scrollback
    assert "end" in scrollback, "kept multi-line records are printed unwrapped"


def test_attach_swaps_the_stream_handler_writes_the_log_file_and_restores(tmp_path: Path, clock: FakeClock) -> None:
    logger = logging.getLogger(LOGGER_NAME + ".attach")
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    stream_handler = logging.StreamHandler(io.StringIO())
    logger.addHandler(stream_handler)
    log_file = tmp_path / "out" / "train.log"
    try:
        with TrainingDashboard.open("r", STAGES, STEPS, TOTAL, logger=logger, log_file=log_file, console=_console(), clock=clock) as b:
            assert stream_handler not in logger.handlers and len(logger.handlers) == 2
            assert logger.level == logging.INFO, "lowered to INFO for the block: the dashboard lives on INFO records"
            logger.info("inside %d", 1)
            assert b.lines()[-1].endswith("inside 1")
        assert logger.handlers == [stream_handler] and logger.level == logging.WARNING
        assert "inside 1" in log_file.read_text()
        assert stream_handler.stream.getvalue() == ""
    finally:
        logger.removeHandler(stream_handler)
        logger.propagate = True


def test_attach_restores_handlers_when_the_body_raises(clock: FakeClock) -> None:
    logger = logging.getLogger(LOGGER_NAME + ".raise")
    logger.propagate = False
    stream_handler = logging.StreamHandler(io.StringIO())
    logger.addHandler(stream_handler)
    try:
        with pytest.raises(RuntimeError, match="boom"), TrainingDashboard.open(
            "r", STAGES, STEPS, TOTAL, logger=logger, console=_console(), clock=clock
        ) as b:
            assert stream_handler not in logger.handlers
            raise RuntimeError("boom")
        assert logger.handlers == [stream_handler] and b._live is None
    finally:
        logger.removeHandler(stream_handler)
        logger.propagate = True


# --- enabling / the fallback -----------------------------------------------------------------------------------------------


def test_enabled_follows_env_and_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAINING_DASHBOARD", "0")
    assert dashboard_enabled(io.StringIO()) is False
    monkeypatch.setenv("TRAINING_DASHBOARD", "1")
    assert dashboard_enabled(io.StringIO()) is False, "StringIO is not a TTY"
    monkeypatch.delenv("TRAINING_DASHBOARD")
    assert dashboard_enabled(io.StringIO()) is False

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert dashboard_enabled(Tty()) is True
    monkeypatch.setenv("TRAINING_DASHBOARD", "off")
    assert dashboard_enabled(Tty()) is False


def test_factory_picks_the_fallback_when_disabled(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    monkeypatch.setenv("TRAINING_DASHBOARD", "0")
    logger = logging.getLogger(LOGGER_NAME + ".factory")
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logger, stream=io.StringIO(), clock=clock) as b:
        assert isinstance(b, NoOpDashboard)
    monkeypatch.setenv("TRAINING_DASHBOARD", "1")
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logger, stream=io.StringIO(), clock=clock) as b:
        assert isinstance(b, NoOpDashboard), "stdout is not a TTY"
    with training_dashboard("r", STAGES, STEPS, TOTAL, logger=logger, enabled=True, console=_console(), clock=clock) as b:
        assert isinstance(b, TrainingDashboard) and b._live is not None
    assert b._live is None


def test_fallback_logs_one_line_per_interval_and_writes_the_log_file(tmp_path: Path, clock: FakeClock) -> None:
    stream = io.StringIO()
    log_file = tmp_path / "train.log"
    # the fallback's lines are emitted on `training.ui.dashboard`: they reach the `training` logger `open()` attaches
    with NoOpDashboard.open("r", STAGES, STEPS, TOTAL, log_step_interval=5, log_file=log_file, stream=stream, clock=clock) as b:
        b.set_status("training")  # DEBUG: not printed
        b.note_event("no checkpoint found, starting from scratch")
        for step in range(1, 13):
            clock.advance(2)
            b.update_step(step, 0, _metrics(step, loss=3.0))
        b.update_step(18, 0, _metrics(18, **{TRANSITION_FLAG_KEY: 1.0, TRANSITION_PROGRESS_KEY: 0.5}))
        b.update_validation(10, {"val_loss_4": 3.25, "val_loss": 3.125})
        b.update_step(TOTAL, 1, _metrics(TOTAL))  # the last step is always logged
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
    expected = "step 18/30 | stage 0 pretrain | transition 50% | loss 2.0000 | s/step 1.00s | elapsed 0:00:18 | ETA 0:00:12"
    assert expected in stream.getvalue()


def test_fallback_rejects_mismatched_stage_lists() -> None:
    with pytest.raises(ValueError, match="2 stage names for 1 step counts"):
        NoOpDashboard("r", STAGES, [5], 5)


# --- never raise -----------------------------------------------------------------------------------------------------------


def _boom(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("renderer broke")


def _is_enabled(board: TrainingDashboard) -> bool:
    return board.enabled  # read through a call so mypy does not narrow `board.enabled` across the failing update


def test_a_failing_update_disables_the_display_once_and_falls_back(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    stream = io.StringIO()
    with TrainingDashboard.open(
        "r", STAGES, STEPS, TOTAL, log_step_interval=1, console=_console(), stream=stream, clock=clock
    ) as b:
        monkeypatch.setattr(b._progress, "update", _boom)
        b.update_step(1, 0, _metrics(1))  # must not raise
        assert _is_enabled(b) is False and b._live is None
        b.update_step(2, 0, _metrics(2))  # the fallback now logs the step lines
        b.update_validation(2, {"val_loss": 3.0})
        b.note_event("saved checkpoint x.pth")
        b.set_status("training")
    output = stream.getvalue()
    assert output.count("training dashboard disabled after an internal error") == 1
    assert "RuntimeError('renderer broke')" in output
    assert "step 1/30" in output and "step 2/30" in output, "the step of the failed update is not lost"
    assert "step 2: validation val_loss 3.0000" in output and "event: saved checkpoint x.pth" in output


def test_a_failing_render_is_reported_and_disables_on_the_next_call(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    stream = io.StringIO()
    with TrainingDashboard.open("r", STAGES, STEPS, TOTAL, console=_console(), stream=stream, clock=clock) as b:
        monkeypatch.setattr(b, "_render_metrics", _boom)
        text = b.render_text()  # what the Live thread does: never raises, shows the error instead
        assert "training dashboard render failed" in text and _is_enabled(b) is True
        b.note_event("after the broken frame")
        assert _is_enabled(b) is False
    assert stream.getvalue().count("training dashboard disabled") == 1
    assert "event: after the broken frame" in stream.getvalue()


def test_log_handler_never_raises_on_a_broken_sink() -> None:
    class BrokenSink:
        def write(self, text: str, *, keep: bool = False) -> None:
            raise OSError("closed")

    handler = DashboardLogHandler(BrokenSink())
    handler.handleError = lambda record: None  # type: ignore[method-assign]  # silence the stderr report of the test
    handler.emit(logging.LogRecord("training.x", logging.INFO, __file__, 1, "msg", None, None))


# --- the scripted run -----------------------------------------------------------------------------------------------------


def test_scripted_thirty_step_run_drives_the_whole_api(tmp_path: Path, clock: FakeClock) -> None:
    """The way ``train()`` / ``RunLogger`` will drive the dashboard (task 10 of the training restructure plan)."""
    logger = logging.getLogger(LOGGER_NAME + ".scripted")
    logger.propagate = False
    log_file = tmp_path / "outputs" / "tiny" / "train.log"
    console = _console()
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
            board.note_event("no checkpoint found, starting from scratch")  # RunLogger.log_fresh_start
            logger.info("Total training steps: %d", TOTAL)  # RunLogger.open
            board.set_status("training")
            for step in range(1, TOTAL + 1):  # progress.step after advance()
                clock.advance(0.5)
                stage_index = 0 if step <= STEPS[0] else 1
                in_transition = 18 <= step <= 20
                metrics = _metrics(step, loss=4.0 - step * 0.05)
                if in_transition:
                    metrics |= {TRANSITION_FLAG_KEY: 1.0, TRANSITION_PROGRESS_KEY: (step - 18) / 2}
                board.update_step(step, stage_index, metrics)  # RunLogger.log_step, every step
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
                if step == TOTAL:
                    board.note_event("saved checkpoint outputs/tiny/checkpoints/step-00000030-tiny.pth")
            board.set_status("exporting")
            board.note_event("exported HuggingFace model to outputs/tiny/hf_export")
            logger.info("Training finished after %d steps", TOTAL)  # RunLogger.close
            text = board.render_text()
    finally:
        logger.propagate = True
    assert "✓ pretrain" in text and "▶ instruct" in text
    assert "20/20" in text and "10/10" in text and "30/30" in text and "100%" in text
    assert "2.00 steps/s" in text and "0:00:15 elapsed" in text and "ETA 0:00:00" in text
    assert "step 30" in text and "2.5000" in text  # 4.0 - 30 * 0.05
    assert "validation (step 30)" in text and "3.5000" in text and "3.4000" in text
    assert "exporting" in text
    assert "step-00000030-tiny.pth" in text and "hf_export" in text
    assert "Training finished after 30 steps" in text
    assert [task.completed for task in board.tasks] == [20, 10, 30]
    assert len(board.events()) == 6 and board.events()[0].endswith("no checkpoint found, starting from scratch")
    log_text = log_file.read_text()
    assert "Total training steps: 30" in log_text and "Training finished after 30 steps" in log_text
    assert board._live is None
