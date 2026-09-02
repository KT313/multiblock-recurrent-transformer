# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the logging handler, the run's log-file handlers, and the terminal capture around the live display."""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings
from pathlib import Path

import pytest

from data_preparation.lib.ui.capture import DashboardLogHandler, LineSink, attach_logger
from training.ui.capture import (
    QUIET_ENV,
    STDERR_LOGGER,
    STDOUT_LOGGER,
    WANDB_QUIET_SETTINGS,
    WARNINGS_LOGGER,
    TerminalCapture,
    format_warning,
    run_log_handlers,
)
from training.ui.common import lines_log
from training.ui.testing import LOGGER_NAME


class RecordingSink:
    """A ``LogSink`` remembering what it was given."""

    def __init__(self) -> None:
        self.written: list[tuple[str, bool]] = []

    def write(self, text: str, *, keep: bool = False) -> None:
        self.written.append((text, keep))

    def texts(self) -> list[str]:
        return [text for text, _keep in self.written]


def _streams() -> tuple[object, object]:
    """``(sys.stdout, sys.stderr)`` through a call: mypy would otherwise keep the narrowing of an earlier assertion."""
    return sys.stdout, sys.stderr


def _redirected(capture: TerminalCapture) -> bool:
    """``capture.streams_redirected`` through a call, for the same reason (mypy narrows attribute expressions)."""
    return capture.streams_redirected


def _record(name: str, level: int, message: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(name, level, __file__, 1, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# --- the handler ------------------------------------------------------------------------------------------------------------


def test_handler_formats_and_keeps_warnings_and_marked_records() -> None:
    sink = RecordingSink()
    handler = DashboardLogHandler(sink)
    handler.emit(_record("training.x", logging.INFO, "plain"))
    handler.emit(_record("training.x", logging.WARNING, "loud"))
    handler.emit(_record("training.x", logging.INFO, "table", keep=True))
    assert [text.split(": ")[-1] for text in sink.texts()] == ["plain", "loud", "table"]
    assert [keep for _text, keep in sink.written] == [False, True, True]
    assert sink.texts()[0].endswith("INFO training.x: plain")


def test_handler_skips_the_loggers_it_is_told_to() -> None:
    sink = RecordingSink()
    handler = DashboardLogHandler(sink, skip=lambda name: name.startswith("training"))
    handler.emit(_record("training.x", logging.INFO, "handled elsewhere"))
    handler.emit(_record("some_library", logging.INFO, "mine"))
    assert sink.texts() == [sink.texts()[0]] and sink.texts()[0].endswith("some_library: mine")


def test_handler_never_raises_on_a_broken_sink() -> None:
    class BrokenSink:
        def write(self, text: str, *, keep: bool = False) -> None:
            raise OSError("closed")

    handler = DashboardLogHandler(BrokenSink())
    handler.handleError = lambda record: None  # type: ignore[method-assign]  # silence the stderr report of the test
    handler.emit(_record("training.x", logging.INFO, "msg"))


# --- the run's log-file and console handlers ------------------------------------------------------------------------------------


def test_run_log_handlers_share_one_file_handler_between_the_logger_and_the_lines(tmp_path: Path) -> None:
    logger = logging.getLogger(LOGGER_NAME + ".handlers")
    logger.setLevel(logging.INFO)
    lines_log.setLevel(logging.INFO)  # in a run it inherits the attached `training` logger's level
    before = list(lines_log.handlers)  # pytest puts its capture handlers on every non-propagating logger
    stream = io.StringIO()
    log_file = tmp_path / "out" / "train.log"
    try:
        with run_log_handlers(logger, log_file, stream):
            [file_handler] = logger.handlers
            assert isinstance(file_handler, logging.FileHandler) and file_handler.baseFilename == str(log_file)
            added = [handler for handler in lines_log.handlers if handler not in before]
            assert added[0] is file_handler and len(added) == 2, "the same handler instance, plus the stream handler"
            logger.info("a record")
            lines_log.info("a line")
        assert logger.handlers == [] and lines_log.handlers == before
    finally:
        lines_log.setLevel(logging.NOTSET)
    assert [line.split(": ", 1)[1] for line in log_file.read_text().splitlines()] == ["a record", "a line"], "one file, in logging order"
    assert stream.getvalue().endswith("INFO training.ui.lines: a line\n") and "a record" not in stream.getvalue()


def test_run_log_handlers_without_a_file_or_a_stream_add_nothing() -> None:
    logger = logging.getLogger(LOGGER_NAME + ".nothing")
    before = list(lines_log.handlers)
    with run_log_handlers(logger, None, None):
        assert logger.handlers == [] and lines_log.handlers == before


# --- the terminal capture -----------------------------------------------------------------------------------------------------


def test_third_party_console_handlers_are_detached_while_captured() -> None:
    sink = RecordingSink()
    library = logging.getLogger("fake_transformers_library")  # like transformers / datasets: a StreamHandler on stderr
    library.setLevel(logging.WARNING)
    plain = logging.StreamHandler(sys.stderr)
    library.addHandler(plain)
    root = logging.getLogger()
    root_handlers = list(root.handlers)
    capture = TerminalCapture(sink)
    try:
        capture.start()
        assert library.handlers == [], "the plain handler would print behind the display"
        assert len(root.handlers) == len(root_handlers) + 1
        library.warning("model card missing")
        assert sink.written[-1][0].endswith("WARNING fake_transformers_library: model card missing") and sink.written[-1][1]
        capture.stop()
        capture.stop()  # idempotent
        assert library.handlers == [plain] and root.handlers == root_handlers
    finally:
        capture.stop()
        library.removeHandler(plain)


def test_root_handler_skips_what_an_attached_handler_covers() -> None:
    sink = RecordingSink()
    logger = logging.getLogger(LOGGER_NAME + ".covered")
    capture = TerminalCapture(sink, skip=lambda name: name.startswith(LOGGER_NAME))
    capture.start()
    try:
        with attach_logger(sink, logger, None, ensure_info_level=True):
            logger.info("once")
            logging.getLogger("elsewhere").warning("root")
    finally:
        capture.stop()
    assert [text.split(": ")[-1] for text in sink.texts()] == ["once", "root"]


def test_stdout_and_stderr_are_captured_while_captured() -> None:
    sink = RecordingSink()
    real_out, real_err = sys.stdout, sys.stderr
    capture = TerminalCapture(sink)
    capture.start()
    try:
        streams: tuple[object, object] = (sys.stdout, sys.stderr)  # object: the stubs type them TextIO, the sink is not one
        assert all(isinstance(stream, LineSink) for stream in streams) and _redirected(capture)
        assert not sys.stdout.isatty()
        print("stray print")
        sys.stderr.write("\rbar 10%\rbar 100%\n")
        # what a foreign `warnings.showwarning` (one installed inside the block) writes to sys.stderr
        sys.stderr.write(warnings.formatwarning("careful", UserWarning, "x.py", 1))
        print("partial", end="")  # no newline: flushed when the capture ends
        texts = sink.texts()
        assert texts[0].endswith(f"INFO {STDOUT_LOGGER}: stray print") and not sink.written[0][1], "stdout lines are not kept"
        assert texts[1].endswith(f"WARNING {STDERR_LOGGER}: bar 100%") and sink.written[1][1], "a carriage return discards the line so far; stderr lines are kept"
        assert f"WARNING {STDERR_LOGGER}: x.py:1: UserWarning: careful" in texts[2]
        assert len(texts) == 3
    finally:
        capture.stop()
    assert _streams() == (real_out, real_err) and not _redirected(capture)
    assert sink.texts()[-1].endswith(f"INFO {STDOUT_LOGGER}: partial")


def test_format_warning_puts_the_message_before_the_location_on_one_line() -> None:
    text = format_warning("careful", UserWarning, "/a/very/long/path/module.py", 12)
    assert text == "UserWarning: careful (/a/very/long/path/module.py:12)"
    assert format_warning(DeprecationWarning("old"), DeprecationWarning, "x.py", 1) == "DeprecationWarning: old (x.py:1)"
    assert "\n" not in format_warning("a", UserWarning, "x.py", 1), "the source line of `warnings.formatwarning` is left out"


def test_warnings_become_one_kept_record_while_captured() -> None:
    sink = RecordingSink()
    previous = warnings.showwarning
    capture = TerminalCapture(sink)
    capture.start()
    try:
        assert warnings.showwarning is not previous
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("careful", stacklevel=1)
        [(text, kept)] = sink.written
        assert f"WARNING {WARNINGS_LOGGER}: UserWarning: careful ({__file__}:" in text and text.endswith(")") and kept
        capture.stop()
        capture.stop()  # idempotent
        assert warnings.showwarning is previous
    finally:
        capture.stop()


def test_a_showwarning_installed_inside_the_block_stays_and_a_stale_hook_forwards() -> None:
    sink = RecordingSink()
    seen: list[tuple[str, str]] = []

    def before(message: Warning | str, category: type[Warning], filename: str, lineno: int, file: object = None, line: str | None = None) -> None:
        seen.append(("before", str(message)))

    def inside(message: Warning | str, category: type[Warning], filename: str, lineno: int, file: object = None, line: str | None = None) -> None:
        seen.append(("inside", str(message)))

    with warnings.catch_warnings():  # restores `warnings.showwarning` afterwards
        warnings.showwarning = before
        capture = TerminalCapture(sink)
        capture.start()
        try:
            ours = warnings.showwarning
            warnings.showwarning = inside  # like `logging.captureWarnings(True)` called inside the block
        finally:
            capture.stop()
        assert warnings.showwarning is inside, "a hook installed inside the block is left in place"
        ours("late", UserWarning, "x.py", 1)  # a stale reference (`logging.captureWarnings(False)` restoring it) forwards
        assert seen == [("before", "late")] and sink.written == []


def test_streams_can_be_released_for_a_prompt_and_redirected_again() -> None:
    sink = RecordingSink()
    real_out = sys.stdout
    capture = TerminalCapture(sink)
    capture.start()
    try:
        capture.release_streams()
        assert _streams()[0] is real_out and not _redirected(capture)
        capture.redirect_streams()
        capture.redirect_streams()  # idempotent: the real streams stay saved
        assert _streams()[0] is not real_out and _redirected(capture)
        print("back")
    finally:
        capture.stop()
    assert _streams()[0] is real_out and sink.texts()[-1].endswith("back")


def test_wandb_is_quieted_through_the_environment_while_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_SILENT", raising=False)
    monkeypatch.setenv("WANDB_CONSOLE", "wrap")
    capture = TerminalCapture(RecordingSink())
    capture.start()
    try:
        assert os.environ["WANDB_SILENT"] == "true" and os.environ["WANDB_CONSOLE"] == "off"
    finally:
        capture.stop()
    assert "WANDB_SILENT" not in os.environ and os.environ["WANDB_CONSOLE"] == "wrap"
    assert QUIET_ENV == {"WANDB_CONSOLE": "off", "WANDB_SILENT": "true"}
    assert WANDB_QUIET_SETTINGS == {"console": "off", "silent": True}
