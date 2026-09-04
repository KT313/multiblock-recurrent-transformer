# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the live training dashboard: rendering, the captured terminal, kept lines, the clean scrollback.
"""

from __future__ import annotations

import io
import logging
import math
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console
from rich.live import Live

from ui.capture import LineSink
from training.ui.board import StageBar, TrainingDashboard
from training.ui.testing import BOX_CHARACTERS, LOGGER_NAME, STAGES, STEPS, TOTAL, live_board, metrics
from ui.testing import DyingFile, FakeClock, console_output, screen_text, string_console


def _live_of(board: TrainingDashboard) -> Live | None:
    return board._live  # through a call: mypy would otherwise keep the narrowing of an earlier assertion


def _is_enabled(board: TrainingDashboard) -> bool:
    return board.enabled  # read through a call so mypy does not narrow `board.enabled` across the failing update


def _boom(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("renderer broke")


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
    board.update_step(7, 0, None, metrics(7))
    text = board.render_text()
    assert "7/20" in text and "0/10" in text and "7/30" in text and " 35%" in text and " 23%" in text
    assert [task.completed for task in board.tasks] == [7, 0, 7]
    assert StageBar("x", 0).percentage == 100.0 and StageBar("x", 4, completed=9).percentage == 100.0


def test_stage_transition_moves_the_highlight(board: TrainingDashboard) -> None:
    board.update_step(18, 0, 0.5, metrics(18))
    text = board.render_text()
    assert "▶ pretrain" in text and "transition → instruct 50%" in text
    board.update_step(23, 1, None, metrics(23))
    text = board.render_text()
    assert "✓ pretrain" in text and "▶ instruct" in text
    assert "20/20" in text and "3/10" in text and "23/30" in text
    assert "transition" not in text


def test_transition_note_follows_the_transition_argument(board: TrainingDashboard) -> None:
    """
    The note shows the transition progress passed with every step (log step or not) and goes when None comes.
    """

    board.update_step(18, 0, 0.5, metrics(18))
    board.update_step(19, 0, 0.5, {})  # a non-log step: an empty dict, the transition still on
    assert "transition → instruct 50%" in board.render_text()
    board.update_step(20, 0, None, metrics(20))
    assert "transition" not in board.render_text(), "the transition is over"
    board.update_step(20, 0, 1.0, metrics(20))
    assert "transition → instruct 100%" in board.render_text()
    board.update_step(21, 1, None, {})  # the stage moved on
    assert "transition" not in board.render_text() and "▶ instruct" in board.render_text()


def test_last_stage_shows_no_transition_note(board: TrainingDashboard) -> None:
    board.update_step(29, 1, 0.9, metrics(29))
    assert "transition" not in board.render_text()


def test_stage_names_with_markup_characters_render_literally(clock: FakeClock) -> None:
    with live_board("r", ["[bold]x[/bold]"], [1], 1, logger=logging.getLogger(LOGGER_NAME), console=string_console(), clock=clock) as b:
        assert "[bold]x[/bold]" in b.render_text()


def test_overall_bar_eta_from_the_injected_clock(board: TrainingDashboard, clock: FakeClock) -> None:
    assert "ETA —" in board.render_text() and "? steps/s" in board.render_text()
    clock.advance(20)
    board.update_step(10, 0, None, metrics(10))  # 2 s/step, 20 steps left
    text = board.render_text()
    assert "0.50 steps/s" in text and "0:00:20 elapsed" in text and "ETA 0:00:40" in text
    assert "0:00:20" in text and "0:00:40" in text  # the metrics table's elapsed / remaining columns


def test_metrics_table_shows_the_latest_step(board: TrainingDashboard) -> None:
    board.update_step(5, 0, None, metrics(5, loss=2.5, **{"seconds/step": 0.25}))
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
    board.update_step(1, 0, None, metrics(1, loss=4.0))
    board.update_step(2, 0, None, {"loss": 3.5})  # a non-log step: only the loss
    assert board.latest_metrics["loss"] == 3.5 and board.latest_metrics["lr"] == 3e-4
    assert "3.5000" in board.render_text()


def test_metrics_table_uses_the_loops_own_timing_without_seconds_per_step(board: TrainingDashboard, clock: FakeClock) -> None:
    clock.advance(4)
    board.update_step(2, 0, None, metrics(2))
    assert "2.00s" in board.render_text()


def test_validation_losses_render_per_depth(board: TrainingDashboard) -> None:
    assert "validation" not in board.render_text()
    board.update_validation(10, {"val_loss_4": 3.4567, "val_loss_8": 3.2, "val_loss": 3.1})
    text = board.render_text()
    assert "validation (step 10)" in text
    assert "val_loss_4" in text and "3.4567" in text and "val_loss_8" in text and "3.2000" in text and "3.1000" in text
    board.update_validation(20, {"val_loss_4": 2.6, "val_loss": 2.5, "val_loss/pretrain-a": 2.75})  # per-source: log line only
    text = board.render_text()
    assert "validation (step 20)" in text and "2.5000" in text and "val_loss/pretrain-a" not in text and "2.7500" not in text


def test_events_list_keeps_the_last_lines(clock: FakeClock) -> None:
    with TrainingDashboard("r", STAGES, STEPS, TOTAL, event_lines=2, console=string_console(), clock=clock) as b:
        assert "(no events yet)" in b.render_text()
        b.update_step(3, 0, None, metrics(3))
        b.note_event("saved checkpoint outputs/r/checkpoints/step-00000003-r.pth")
        b.update_step(6, 0, None, metrics(6))
        b.note_event("starting transition 0 -> 1")
        b.note_event("saved checkpoint step-00000006-r.pth")
        events = b.events()
        assert len(events) == 2
        assert events[0].endswith("step 6: starting transition 0 -> 1")
        assert events[1].endswith("step 6: saved checkpoint step-00000006-r.pth")
        text = b.render_text()
        assert "step-00000006-r.pth" in text and "step-00000003-r.pth" not in text


def test_bars_for_a_zero_length_stage_render(clock: FakeClock) -> None:
    with TrainingDashboard("r", ["a", "empty", "b"], [5, 0, 5], 10, console=string_console(), clock=clock) as b:
        b.update_step(5, 2, None, metrics(5))
        text = b.render_text()
        assert "✓ a" in text and "✓ empty" in text and "0/0" in text and "▶ b" in text


def test_footer_names_the_log_file(tmp_path: Path, clock: FakeClock) -> None:
    log_file = tmp_path / "train.log"
    with live_board("r", STAGES, STEPS, TOTAL, logger=logging.getLogger(LOGGER_NAME), log_file=log_file, console=string_console(), clock=clock) as b:
        assert b.render_text().rstrip().splitlines()[-1].startswith(f"log: {log_file}")


def test_rows_never_wrap_so_the_frame_height_does_not_depend_on_the_width(clock: FakeClock) -> None:
    long_name = "a-very-long-stage-name-" * 4
    details = {"model": "crow-300m-final", "dataset": "crow_300m_final", "device": "cuda:0", "precision": "bf16-mixed"}
    with live_board(
        "run-" * 10, [long_name, "b"], STEPS, TOTAL, details=details, logger=logging.getLogger(LOGGER_NAME), console=string_console(), clock=clock
    ) as b:
        b.update_step(18, 0, 0.5, metrics(18))
        b.update_validation(10, {f"val_loss_{depth}": 3.0 for depth in range(1, 12)})
        b.note_event("saved checkpoint " + "outputs/very/long/path/" * 6 + "step-00000018-run.pth")
        b.write("a log line " * 30)
        wide, narrow = b.render_text(width=240), b.render_text(width=50)
    assert len(wide.splitlines()) == len(narrow.splitlines()), "no row wrapped: cells are cropped, not folded"
    assert all(len(line) <= 50 for line in narrow.splitlines())
    assert "▶ a-very-long-stage-name-a-ve…" in narrow and "run-run-run-run-run-run-run-run-run-run- …starting" in narrow


def test_short_terminal_shrinks_the_log_panel_then_the_events_panel(clock: FakeClock) -> None:
    b = TrainingDashboard("r", STAGES, STEPS, TOTAL, log_lines=12, event_lines=6, console=string_console(), clock=clock)
    with b.running(logging.getLogger(LOGGER_NAME)):
        for i in range(20):
            b.write(f"line {i}")
        for i in range(8):
            b.note_event(f"event {i}")
        b.update_validation(10, {"val_loss": 3.0})
        tall = b.render_text(width=100, height=60)
        short = b.render_text(width=100, height=30)  # header 1 + bars 3 + metrics 4 + validation 4 + footer 1 = 13 fixed
        shorter = b.render_text(width=100, height=23)  # 10 lines for both panels: 4 events + 2 log lines + 4 borders
        shortest = b.render_text(width=100, height=20)  # 7 lines: 1 event + 2 log lines + 4 borders
    assert "line 8" in tall and "line 19" in tall and "event 2" in tall and "event 7" in tall
    assert len(short.splitlines()) <= 30 and len(shorter.splitlines()) <= 23 and len(shortest.splitlines()) <= 20
    assert "event 7" in shortest and "event 6" not in shortest and "line 19" in shortest and "line 18" in shortest
    assert "line 19" in short and "line 8" not in short and "line 13" in short and "line 12" not in short, "the log panel keeps its newest lines"
    assert "event 2" in short and "event 7" in short, "the events panel is untouched while the log panel can shrink"
    assert "line 19" in shorter and "line 18" in shorter and "line 17" not in shorter, "the log panel at its minimum"
    assert "event 7" in shorter and "event 4" in shorter and "event 3" not in shorter, "then the events panel shrinks"


def test_a_dead_terminal_closes_the_display_and_training_continues(tmp_path: Path, clock: FakeClock) -> None:
    file = DyingFile()
    console = Console(file=file, force_terminal=True, width=100)
    log_file = tmp_path / "train.log"
    with live_board("r", STAGES, STEPS, TOTAL, logger=logging.getLogger("training"), log_file=log_file, console=console, clock=clock) as b:
        b.update_step(1, 0, None, metrics(1))
        file.die()
        live = _live_of(b)
        assert live is not None
        live.refresh()
        assert b.headless and not _is_enabled(b) and _live_of(b) is None
        b.update_step(2, 0, None, metrics(2))
        b.note_event("checkpoint written")
        b.set_status("evaluating")
        b.write("a kept line after the loss", keep=True)
    text = log_file.read_text()
    assert "terminal gone ([Errno 5] Input/output error): the display is closed, the run continues headless; its log: " in text
    assert "step 2/30" in text and "event: checkpoint written" in text
    assert "a kept line" not in file.getvalue() and "overall" not in file.getvalue()[-300:] and file.refused >= 1, "nothing reached the dead terminal"


def test_a_resized_terminal_gets_the_frame_redrawn_from_a_cleared_screen(clock: FakeClock) -> None:
    console = string_console(120, height=40)
    with live_board("tiny-run", STAGES, STEPS, TOTAL, logger=logging.getLogger(LOGGER_NAME), console=console, clock=clock) as b:
        live = _live_of(b)
        assert live is not None
        live.refresh()
        assert "\x1b[2J" not in console_output(console), "the same size: the previous frame is erased with cursor-up"
        console.size = (100, 30)
        live.refresh()
        live.refresh()
        assert console_output(console).count("\x1b[2J\x1b[H") == 1, "one clear per size change, right before the frame"
        assert screen_text(console, 100).count("tiny-run") == 1, "one frame on the screen: no leftovers of the wider one"


# --- log lines and kept records ------------------------------------------------------------------------------------------------


def test_log_records_land_in_the_panel_once(board: TrainingDashboard) -> None:
    logging.getLogger(LOGGER_NAME + ".child").info("hello %d", 7)  # attached logger: its handler, not the root one
    logging.getLogger("some_library").warning("careful")  # no handler of its own: the root one
    first, second = board.lines()
    assert f"INFO {LOGGER_NAME}.child: hello 7" in first and "WARNING some_library: careful" in second
    assert board.kept() == [second]
    assert "hello 7" in board.render_text()
    assert board.is_attached(LOGGER_NAME + ".child") and not board.is_attached("training")


def test_log_panel_keeps_last_lines_and_splits_multi_line_records(clock: FakeClock) -> None:
    with TrainingDashboard("r", STAGES, STEPS, TOTAL, log_lines=3, console=string_console(), clock=clock) as b:
        for i in range(5):
            b.write(f"line {i}")
        assert b.lines() == ["line 2", "line 3", "line 4"]
        b.write("Traceback:\n  File x\nValueError: boom")
        assert b.lines() == ["Traceback:", "  File x", "ValueError: boom"]
        b.write("path [bold]x[/bold]")
        assert "[bold]x[/bold]" in b.render_text(), "markup in log lines is not interpreted"


def test_kept_records_are_printed_once_after_the_display_closed_not_during(clock: FakeClock) -> None:
    console = string_console(100)
    logger = logging.getLogger(LOGGER_NAME + ".keep")
    table = "a  b  c" + " " * 200 + "end"
    with live_board("r", STAGES, STEPS, TOTAL, logger=logger, console=console, clock=clock) as b:
        b._refresh_per_second = 50
        logger.info("quiet")
        logger.warning("loud")
        logger.info("table:\n%s", table, extra={"keep": True})
        b.update_step(3, 0, None, metrics(3))
        time.sleep(0.1)  # a few live frames
        assert [line.split(": ")[-1] for line in b.lines()[:2]] == ["quiet", "loud"]
        assert len(b.kept()) == 2
        assert "loud" not in screen_text(console, 100) and "end" not in screen_text(console, 100), "nothing is printed while the display is up"
    screen = screen_text(console, 100)
    assert screen.count(f"WARNING {LOGGER_NAME}.keep: loud") == 1 and "quiet" not in screen
    assert console_output(console).count(table) == 1 and screen.count("end") == 1, "kept multi-line records are written unwrapped, once"
    assert not any(character in screen for character in BOX_CHARACTERS), "the display is transient: no panel is left behind"
    assert screen.count("overall") == 1 and screen.count("grad norm") == 1 and "3/30" in screen, "the static summary, once"
    assert screen.index("loud") < screen.index("end") < screen.index("overall"), "kept lines first, then the summary"
    assert b.kept() == []


def test_final_frame_can_be_turned_off(clock: FakeClock) -> None:
    console = string_console()
    with live_board("r", STAGES, STEPS, TOTAL, logger=logging.getLogger(LOGGER_NAME), final_frame=False, console=console, clock=clock) as b:
        b.update_step(3, 0, None, metrics(3))
        logging.getLogger(LOGGER_NAME).warning("only this")
    screen = screen_text(console, 120)
    assert "only this" in screen and "overall" not in screen and "grad norm" not in screen


def test_attach_swaps_the_stream_handler_writes_the_log_file_and_restores(tmp_path: Path, clock: FakeClock) -> None:
    logger = logging.getLogger(LOGGER_NAME + ".attach")
    logger.propagate = False
    stream_handler = logging.StreamHandler(io.StringIO())
    logger.addHandler(stream_handler)
    log_file = tmp_path / "out" / "train.log"
    try:
        with live_board("r", STAGES, STEPS, TOTAL, logger=logger, log_file=log_file, console=string_console(), clock=clock) as b:
            assert stream_handler not in logger.handlers and len(logger.handlers) == 2
            logger.info("inside %d", 1)
            assert b.lines()[-1].endswith("inside 1")
        assert logger.handlers == [stream_handler]
        assert "inside 1" in log_file.read_text() and stream_handler.stream.getvalue() == ""
    finally:
        logger.removeHandler(stream_handler)
        logger.propagate = True


# --- what else could reach the terminal -------------------------------------------------------------------------------------


def test_third_party_console_handlers_are_detached_while_the_display_is_up(clock: FakeClock) -> None:
    library = logging.getLogger("fake_transformers_library")  # like transformers / datasets: a StreamHandler on stderr
    library.setLevel(logging.WARNING)
    plain = logging.StreamHandler(sys.stderr)
    library.addHandler(plain)
    try:
        with TrainingDashboard("r", STAGES, STEPS, TOTAL, console=string_console(), clock=clock) as b:
            assert library.handlers == [], "the plain handler would print behind the display"
            library.warning("model card missing")
            assert b.lines()[-1].endswith("WARNING fake_transformers_library: model card missing")
            assert b.kept()[-1].endswith("model card missing")
        assert library.handlers == [plain]
    finally:
        library.removeHandler(plain)


def test_stdout_and_stderr_are_captured_while_the_display_is_up(tmp_path: Path, clock: FakeClock) -> None:
    console = string_console(100)
    real_out, real_err = sys.stdout, sys.stderr
    log_file = tmp_path / "train.log"
    # the `training` logger, as in a run: the sink loggers `training.stdout` / `training.stderr` sit under it
    with live_board("r", STAGES, STEPS, TOTAL, logger=logging.getLogger("training"), log_file=log_file, console=console, clock=clock) as b:
        streams: tuple[object, object] = (sys.stdout, sys.stderr)  # object: the stubs type them TextIO, the sink is not one
        assert all(isinstance(stream, LineSink) for stream in streams)
        print("stray print")
        sys.stderr.write("a bare stderr write\n")
        lines = b.lines()
        assert any(line.endswith("INFO training.stdout: stray print") for line in lines)
        assert any(line.endswith("WARNING training.stderr: a bare stderr write") for line in lines)
        assert b.kept() and all("stray print" not in text for text in b.kept()), "stdout lines are not kept, stderr lines are"
    assert sys.stdout is real_out and sys.stderr is real_err
    screen = screen_text(console, 100)
    assert screen.count("a bare stderr write") == 1 and "stray print" not in screen
    log_text = log_file.read_text()
    assert "stray print" in log_text and "a bare stderr write" in log_text, "the sinks log under `training`: stray lines reach train.log"


def test_suspended_clears_the_display_for_a_prompt_and_brings_it_back(clock: FakeClock) -> None:
    console = string_console(80)
    real_out = sys.stdout
    with TrainingDashboard("r", STAGES, STEPS, TOTAL, console=console, clock=clock) as b:
        b.update_step(3, 0, None, metrics(3))
        live = _live_of(b)
        assert live is not None
        with b.suspended():
            assert _live_of(b) is None and sys.stdout is real_out, "the terminal belongs to the prompt"
            assert "overall" not in screen_text(console, 80), "the frame is erased while suspended"
        stdout: object = sys.stdout
        restarted = _live_of(b)
        assert restarted is not None and restarted is not live and isinstance(stdout, LineSink)
        assert "3/30" in b.render_text()
    with b.suspended():
        pass  # closed: nothing to suspend


def test_an_exception_inside_the_block_leaves_a_clean_scrollback(clock: FakeClock) -> None:
    console = string_console(100)
    real_out, real_err = sys.stdout, sys.stderr
    logger = logging.getLogger(LOGGER_NAME + ".exception")
    with pytest.raises(RuntimeError, match="loop broke"), live_board("r", STAGES, STEPS, TOTAL, logger=logger, console=console, clock=clock) as b:
        b.update_step(7, 0, None, metrics(7))
        logger.warning("last words")
        raise RuntimeError("loop broke")
    assert sys.stdout is real_out and sys.stderr is real_err and b._live is None
    screen = screen_text(console, 100)
    assert screen.count("last words") == 1 and screen.count("overall") == 1 and "7/30" in screen
    assert not any(character in screen for character in BOX_CHARACTERS), screen


def test_close_is_idempotent_and_a_console_on_stdout_is_pinned(clock: FakeClock) -> None:
    console = string_console()
    b = TrainingDashboard("r", STAGES, STEPS, TOTAL, console=console, clock=clock)
    with b:
        b.close()
        assert b._live is None
    assert screen_text(console, 120).count("overall") == 1, "the summary is printed once"
    console.file = sys.stdout  # a console following sys.stdout: pinned to the current stream before the redirect
    with TrainingDashboard("r", STAGES, STEPS, TOTAL, console=console, clock=clock, final_frame=False):
        assert not isinstance(console.file, LineSink)


# --- never raise -----------------------------------------------------------------------------------------------------------


def test_a_failing_update_disables_the_display_once_and_falls_back(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    stream = io.StringIO()
    real_out = sys.stdout
    with live_board(
        "r", STAGES, STEPS, TOTAL, log_step_interval=1, console=string_console(), stream=stream, clock=clock
    ) as b:
        logging.getLogger(LOGGER_NAME).warning("before the failure")
        monkeypatch.setattr(b, "_refresh_bars", _boom)
        b.update_step(1, 0, None, metrics(1))  # must not raise
        assert _is_enabled(b) is False and b._live is None
        assert sys.stdout is real_out, "the terminal is restored the moment the display is disabled"
        b.update_step(2, 0, None, metrics(2))  # the fallback now logs the step lines
        b.update_validation(2, {"val_loss": 3.0})
        b.note_event("saved checkpoint x.pth")
        b.set_status("training")
    output = stream.getvalue()
    assert output.count("training dashboard disabled after an internal error") == 1
    assert "RuntimeError('renderer broke')" in output
    assert "step 1/30" in output and "step 2/30" in output, "the step of the failed update is not lost"
    assert "step 2: validation val_loss 3.0000" in output and "event: saved checkpoint x.pth" in output
    screen = screen_text(b._console, 120)
    assert screen.count("before the failure") == 1, "the kept lines so far are printed when the display goes"
    assert "overall" not in screen and not any(character in screen for character in BOX_CHARACTERS), "no frame, no summary for a disabled display"


def test_a_failing_render_is_reported_and_disables_on_the_next_call(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    stream = io.StringIO()
    with live_board("r", STAGES, STEPS, TOTAL, console=string_console(), stream=stream, clock=clock) as b:
        monkeypatch.setattr(b, "_render_metrics", _boom)
        text = b.render_text()  # what the Live thread does: never raises, shows the error instead
        assert "training dashboard render failed" in text and _is_enabled(b) is True
        b.note_event("after the broken frame")
        assert _is_enabled(b) is False
    assert stream.getvalue().count("training dashboard disabled") == 1
    assert "event: after the broken frame" in stream.getvalue()


def test_a_failing_start_disables_before_the_block_runs(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    stream = io.StringIO()
    real_out = sys.stdout
    monkeypatch.setattr(TrainingDashboard, "_start_live", _boom)
    with live_board("r", STAGES, STEPS, TOTAL, console=string_console(), stream=stream, clock=clock) as b:
        assert _is_enabled(b) is False and sys.stdout is real_out
        b.write("plain", keep=True)
    assert "training dashboard disabled" in stream.getvalue() and "plain\n" in stream.getvalue()
