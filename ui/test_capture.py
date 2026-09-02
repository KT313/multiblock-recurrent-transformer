# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the console capture both live dashboards install: the line sink (splitting, the re-entrancy guard that
breaks the logging recursion, `fileno`), the dashboard log handler (formatting, keeping, `handleError`), attaching a
logger, the logger walk and the stream / logging captures."""

from __future__ import annotations

import io
import logging
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ui.capture import (
    DashboardLogHandler,
    LineSink,
    LoggingCapture,
    StreamCapture,
    attach_logger,
    existing_loggers,
    is_console_handler,
)

LOGGER_NAME = "data_preparation.test_capture"
STDOUT_LOGGER = f"{LOGGER_NAME}.stdout"
STDERR_LOGGER = f"{LOGGER_NAME}.stderr"


class RecordingSink:
    """A ``LogSink`` remembering what it was given."""

    def __init__(self) -> None:
        self.written: list[tuple[str, bool]] = []

    def write(self, text: str, *, keep: bool = False) -> None:
        self.written.append((text, keep))

    def texts(self) -> list[str]:
        return [text for text, _keep in self.written]


class FailingStream(io.StringIO):
    """A log file whose device is full: every write raises, as a ``FileHandler`` stream on ENOSPC would."""

    def write(self, s: str, /) -> int:
        raise OSError(28, "No space left on device")


def _redirected(capture: StreamCapture) -> bool:
    """``capture.redirected`` through a call: mypy would otherwise keep the narrowing of an earlier assertion."""
    return capture.redirected


def _record(name: str, level: int, message: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(name, level, __file__, 1, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


@pytest.fixture
def isolated_logger() -> Iterator[logging.Logger]:
    """A logger of its own, with the handlers a test added to it removed again afterwards.

    ``propagate`` is left alone: pytest's own capture handler attaches itself to every *non*-propagating logger at
    the start of every phase, which would show up in the handler lists these tests assert on."""
    logger = logging.getLogger(LOGGER_NAME)
    before, level = list(logger.handlers), logger.level
    logger.setLevel(logging.INFO)
    yield logger
    for handler in list(logger.handlers):
        if handler not in before:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(level)


# --- the line sink ------------------------------------------------------------------------------------------------------


def test_line_sink_splits_lines_and_flushes_the_rest() -> None:
    emitted: list[str] = []
    sink = LineSink(emitted.append)
    assert sink.write("a\nb") == 3
    sink.write("c\n\n\rd")
    assert emitted == ["a", "bc", ""]
    sink.close_flush()
    sink.close_flush()
    assert emitted == ["a", "bc", "", "d"] and sink.encoding == "utf-8" and sink.writable()
    assert not sink.isatty()


def test_line_sink_fileno_is_the_replaced_stream_s(tmp_path: Path) -> None:
    """T-M11: a bare ``io.TextIOBase`` has no ``fileno()``, so ``sys.stdout.fileno()`` used to raise for a whole run."""
    with (tmp_path / "real").open("w", encoding="utf-8") as real:
        sink = LineSink(lambda _line: None, real_stream=real)
        assert sink.fileno() == real.fileno()
        assert LineSink(lambda _line: None, real_stream=real).real_stream is real
    process_stderr = sys.__stderr__
    assert process_stderr is not None
    assert LineSink(lambda _line: None).fileno() == process_stderr.fileno(), "no stream given: the process' own stderr"


def test_line_sink_without_a_real_stream_reports_no_fileno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "__stderr__", None)
    with pytest.raises(io.UnsupportedOperation):
        LineSink(lambda _line: None).fileno()


def test_line_sink_sends_a_re_entrant_write_to_the_real_stream() -> None:
    """H9: a write caused by the sink's own emit must not be fed back in — that is the recursion bomb."""
    real = io.StringIO()
    lines: list[str] = []

    def emit(line: str) -> None:
        lines.append(line)
        sys.stdout.write("from inside the emit\n")  # what a failing handler's `handleError` does

    sink = LineSink(emit, real_stream=real)
    saved = sys.stdout
    sys.stdout = sink
    try:
        sink.write("outer\n")
    finally:
        sys.stdout = saved
    assert lines == ["outer"], "the re-entrant write never became a second emit"
    assert real.getvalue() == "from inside the emit\n"


def test_line_sink_guard_is_thread_local() -> None:
    """The guard must not silence another thread's lines while one thread is inside its emit."""
    real = io.StringIO()
    lines: list[str] = []
    inside = threading.Event()
    done = threading.Event()

    def emit(line: str) -> None:
        lines.append(line)
        if line == "blocking":
            inside.set()
            done.wait(5)

    sink = LineSink(emit, real_stream=real)
    thread = threading.Thread(target=lambda: sink.write("blocking\n"))
    thread.start()
    try:
        assert inside.wait(5)
        sink.write("other thread\n")
    finally:
        done.set()
        thread.join(5)
    assert lines == ["blocking", "other thread"] and real.getvalue() == ""


def test_a_failing_log_handler_does_not_recurse_into_the_sink(isolated_logger: logging.Logger, tmp_path: Path) -> None:
    """H9 end to end: the train.log handler on a full disk must not end the run with a ``RecursionError``.

    ``logging.Handler.handleError`` writes its report to ``sys.stderr`` — the sink — which logs it — which reaches
    the same failing handler. The thread-local guard sends that second write to the real stream instead."""
    real_err = io.StringIO()
    file_handler = logging.StreamHandler(FailingStream())  # a `FileHandler` whose device filled up behaves like this
    isolated_logger.addHandler(file_handler)
    stderr_logger = logging.getLogger(STDERR_LOGGER)  # under the failing logger: its records reach that handler again
    stderr_logger.setLevel(logging.INFO)
    sink = LineSink(stderr_logger.warning, real_stream=real_err)
    saved = sys.stderr
    sys.stderr = sink
    try:
        isolated_logger.info("the run continues")  # must not raise
    finally:
        sys.stderr = saved
    assert "No space left on device" in real_err.getvalue(), "the error text reached the real stderr"
    assert "--- Logging error ---" in real_err.getvalue()


# --- the handler ----------------------------------------------------------------------------------------------------------


def test_handler_formats_and_keeps_warnings_and_marked_records() -> None:
    sink = RecordingSink()
    handler = DashboardLogHandler(sink)
    handler.emit(_record("data_preparation.x", logging.INFO, "plain"))
    handler.emit(_record("data_preparation.x", logging.WARNING, "loud"))
    handler.emit(_record("data_preparation.x", logging.INFO, "table", keep=True))
    assert [text.split(": ")[-1] for text in sink.texts()] == ["plain", "loud", "table"]
    assert [keep for _text, keep in sink.written] == [False, True, True]


def test_handler_skips_the_loggers_it_is_told_to() -> None:
    sink = RecordingSink()
    handler = DashboardLogHandler(sink, skip=lambda name: name.startswith("data_preparation"))
    handler.emit(_record("data_preparation.x", logging.INFO, "handled elsewhere"))
    handler.emit(_record("some_library", logging.INFO, "mine"))
    assert len(sink.texts()) == 1 and sink.texts()[0].endswith("some_library: mine")


def test_handler_reports_a_broken_sink_on_the_saved_stderr_and_never_through_logging() -> None:
    """The other end of H9: ``handleError`` must not write to ``sys.stderr`` (a sink while a display is up)."""

    class BrokenSink:
        def write(self, text: str, *, keep: bool = False) -> None:
            raise OSError("closed")

    errors = io.StringIO()
    handler = DashboardLogHandler(BrokenSink(), error_stream=errors)
    written: list[str] = []
    saved = sys.stderr
    sys.stderr = LineSink(written.append)
    try:
        handler.emit(_record("data_preparation.x", logging.INFO, "msg"))  # must not raise
    finally:
        sys.stderr = saved
    assert "--- Logging error ---" in errors.getvalue() and "OSError: closed" in errors.getvalue()
    assert "Logged from file" in errors.getvalue()
    assert written == [], "nothing went through sys.stderr, so nothing could come back as a record"


def test_handler_error_is_silent_when_logging_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenSink:
        def write(self, text: str, *, keep: bool = False) -> None:
            raise OSError("closed")

    errors = io.StringIO()
    monkeypatch.setattr(logging, "raiseExceptions", False)
    DashboardLogHandler(BrokenSink(), error_stream=errors).emit(_record("x", logging.INFO, "msg"))
    assert errors.getvalue() == ""


def test_handler_error_survives_a_broken_error_stream() -> None:
    class BrokenSink:
        def write(self, text: str, *, keep: bool = False) -> None:
            raise OSError("closed")

    DashboardLogHandler(BrokenSink(), error_stream=FailingStream()).emit(_record("x", logging.INFO, "msg"))


# --- console handlers, the logger walk ------------------------------------------------------------------------------------


def test_is_console_handler_tells_stream_from_file_handlers(tmp_path: Path) -> None:
    file_handler = logging.FileHandler(tmp_path / "x.log", encoding="utf-8")
    try:
        assert is_console_handler(logging.StreamHandler(io.StringIO()))
        assert not is_console_handler(file_handler)
        assert not is_console_handler(DashboardLogHandler(RecordingSink()))
    finally:
        file_handler.close()


def test_existing_loggers_lists_the_root_logger_and_every_named_one() -> None:
    logging.getLogger("data_preparation.test_capture.walked")
    loggers = existing_loggers()
    assert loggers[0] is logging.getLogger(), "the root logger first: it carries the dashboard's own handler"
    assert "data_preparation.test_capture.walked" in {logger.name for logger in loggers}
    assert all(isinstance(logger, logging.Logger) for logger in loggers), "placeholders are not loggers"


def test_existing_loggers_copies_the_dict_while_holding_the_logging_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-M16: wandb's background threads create loggers, so an unguarded walk can raise "dictionary changed size"."""
    locked = [False]
    held: list[bool] = []
    # the module lock helpers are private, so typeshed does not declare them; `existing_loggers` looks them up the same way
    real_acquire: Callable[[], None] = getattr(logging, "_acquireLock")  # noqa: B009
    real_release: Callable[[], None] = getattr(logging, "_releaseLock")  # noqa: B009

    def acquire() -> None:
        real_acquire()
        locked[0] = True

    def release() -> None:
        locked[0] = False
        real_release()

    class Watched(dict[str, Any]):
        def values(self) -> Any:
            held.append(locked[0])
            return super().values()

    monkeypatch.setattr(logging, "_acquireLock", acquire)
    monkeypatch.setattr(logging, "_releaseLock", release)
    monkeypatch.setattr(logging.root.manager, "loggerDict", Watched(logging.root.manager.loggerDict))
    assert logging.getLogger() in existing_loggers()
    assert held == [True] and not locked[0], "copied once, under the lock, and the lock is released again"


# --- attaching a logger ---------------------------------------------------------------------------------------------------


def test_attach_swaps_the_stream_handler_writes_the_log_file_and_restores(
    isolated_logger: logging.Logger, tmp_path: Path
) -> None:
    sink = RecordingSink()
    before = list(isolated_logger.handlers)
    stream_handler = logging.StreamHandler(io.StringIO())
    isolated_logger.addHandler(stream_handler)
    log_file = tmp_path / "out" / "build.log"
    with attach_logger(sink, isolated_logger, log_file) as file_handler:
        assert stream_handler not in isolated_logger.handlers, "a plain stream handler would print behind the display"
        assert isinstance(file_handler, logging.FileHandler) and file_handler.baseFilename == str(log_file)
        isolated_logger.info("inside %d", 1)
        assert sink.texts()[-1].endswith("inside 1")
    assert isolated_logger.handlers == [*before, stream_handler]
    assert "inside 1" in log_file.read_text(encoding="utf-8")
    assert stream_handler.stream.getvalue() == ""


def test_attach_leaves_the_level_alone_unless_asked(isolated_logger: logging.Logger) -> None:
    isolated_logger.setLevel(logging.WARNING)
    with attach_logger(RecordingSink(), isolated_logger, None) as file_handler:
        assert isolated_logger.level == logging.WARNING and file_handler is None
    with attach_logger(RecordingSink(), isolated_logger, None, ensure_info_level=True):
        assert isolated_logger.level == logging.INFO, "the training dashboard lives on INFO records"
    assert isolated_logger.level == logging.WARNING


def test_attach_restores_handlers_when_the_body_raises(isolated_logger: logging.Logger) -> None:
    before = list(isolated_logger.handlers)
    stream_handler = logging.StreamHandler(io.StringIO())
    isolated_logger.addHandler(stream_handler)
    with pytest.raises(RuntimeError, match="boom"), attach_logger(RecordingSink(), isolated_logger, None):
        assert stream_handler not in isolated_logger.handlers
        raise RuntimeError("boom")
    assert isolated_logger.handlers == [*before, stream_handler]


# --- the logging and stream captures --------------------------------------------------------------------------------------


def test_logging_capture_detaches_console_handlers_and_routes_the_root_logger() -> None:
    sink = RecordingSink()
    library = logging.getLogger("fake_hub_library_shared")  # like huggingface_hub: a StreamHandler on stderr
    library.setLevel(logging.WARNING)
    plain = logging.StreamHandler(sys.stderr)
    library.addHandler(plain)
    root = logging.getLogger()
    root_handlers = list(root.handlers)
    capture = LoggingCapture(sink)
    try:
        capture.start()
        capture.start()  # idempotent
        assert library.handlers == [] and capture.active
        library.warning("repo card missing")
        assert sink.written[-1][0].endswith("WARNING fake_hub_library_shared: repo card missing") and sink.written[-1][1]
        capture.stop()
        capture.stop()  # idempotent
        assert library.handlers == [plain] and root.handlers == root_handlers and not capture.active
    finally:
        capture.stop()
        library.removeHandler(plain)


def test_stream_capture_redirects_releases_and_is_idempotent() -> None:
    real_out, real_err = sys.stdout, sys.stderr
    real_out_fd = real_out.fileno()
    lines: list[logging.LogRecord] = []

    class Recording(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record)

    handler = Recording()
    root = logging.getLogger()
    root.addHandler(handler)
    capture = StreamCapture(STDOUT_LOGGER, STDERR_LOGGER)
    try:
        with capture:
            capture.redirect()  # idempotent: the real streams stay saved
            stream: object = sys.stdout  # object: the stubs type it TextIO, the sink is not one
            assert _redirected(capture) and isinstance(stream, LineSink)
            assert stream.fileno() == real_out_fd, "T-M11: the descriptor of the stream the sink replaced"
            replaced: object = stream.real_stream  # through a name (and last): `is` would narrow `real_out` too
            assert replaced is real_out
            print("stray print")
            print("partial", end="")
        out_after: object = sys.stdout  # through names: the stubs type the streams TextIO, the sinks are not
        err_after: object = sys.stderr
        assert not _redirected(capture)
        assert out_after is real_out and err_after is real_err
        capture.release()  # idempotent
    finally:
        root.removeHandler(handler)
    messages = [record.getMessage() for record in lines if record.name == STDOUT_LOGGER]
    assert messages == ["stray print", "partial"], "the pending line is flushed when the streams are released"
