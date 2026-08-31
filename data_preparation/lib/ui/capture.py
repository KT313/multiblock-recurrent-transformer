# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""What both live dashboards — the data-prep one (:mod:`data_preparation.lib.ui.dashboard`) and the training one
(:mod:`training.ui.dashboard`) — do to keep the terminal to themselves, in one place.

A dashboard is a :class:`LogSink`: something with ``write(text, *, keep=False)``. While its display is up

* :class:`LoggingCapture` puts a :class:`DashboardLogHandler` on the root logger and detaches every plain console
  ``StreamHandler`` of every existing logger (the ones ``huggingface_hub`` / ``datasets`` / ``transformers`` install
  on theirs at import) — they would write to the real streams behind the display,
* :class:`StreamCapture` replaces ``sys.stdout`` / ``sys.stderr`` with :class:`LineSink`\\ s that turn what is
  written to them into records on two loggers, and
* :func:`attach_logger` routes one logger (``data_preparation`` / ``training``) into the sink directly and appends
  every record to a log file.

Three things a live display must survive, fixed here once for both dashboards:

* **the recursion bomb.** A handler that raises while emitting (the log file on a full disk) lands in
  ``logging.Handler.handleError``, which writes to ``sys.stderr`` — the line sink — which logs — which reaches the
  same failing handler again. :class:`LineSink` therefore keeps a *thread-local* re-entrancy flag: a write that
  arrives while that thread is already inside the sink's ``emit`` goes straight to the saved real stream, so the
  chain ends at its second link instead of raising ``RecursionError`` into the caller's ``logger.info(...)``.
  :class:`DashboardLogHandler` closes the other end: its ``handleError`` writes to the saved real stderr and never
  re-enters ``logging``.
* **``sys.stdout.fileno()``.** A bare ``io.TextIOBase`` has none, so anything asking the current stdout for its file
  descriptor used to fail for the whole run; :meth:`LineSink.fileno` answers with the saved real stream's.
* **loggers created while the walk runs.** wandb's background threads create loggers, so
  ``logging.root.manager.loggerDict`` may change size during iteration; :func:`existing_loggers` snapshots it while
  holding the ``logging`` module's own lock.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TextIO, TypeGuard

from data_preparation.lib.log import LOG_FORMAT


class LogSink(Protocol):
    """What :class:`DashboardLogHandler` writes to: a dashboard's log panel (or, disabled, its stream)."""

    def write(self, text: str, *, keep: bool = False) -> None: ...


def is_console_handler(handler: logging.Handler) -> TypeGuard[logging.StreamHandler[Any]]:
    """A plain ``StreamHandler`` (a terminal writer) rather than a file / dashboard / library handler."""
    return isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)


def existing_loggers() -> list[logging.Logger]:
    """Every logger that exists right now, plus the root logger.

    ``logging.root.manager.loggerDict`` is mutated by every ``logging.getLogger(name)`` of a name seen the first
    time — wandb's background threads do that while a run is up — so the dict is copied while holding the lock the
    ``logging`` module itself takes around that mutation; iterating it directly can raise "dictionary changed size
    during iteration"."""
    acquire: Callable[[], None] | None = getattr(logging, "_acquireLock", None)
    release: Callable[[], None] | None = getattr(logging, "_releaseLock", None)
    if acquire is None or release is None:  # pragma: no cover - the private lock helpers exist in every CPython 3.x
        values = list(logging.root.manager.loggerDict.values())
    else:
        acquire()
        try:
            values = list(logging.root.manager.loggerDict.values())
        finally:
            release()
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers.extend(logger for logger in values if isinstance(logger, logging.Logger))
    return loggers


class LineSink(io.TextIOBase):
    """A ``sys.stdout`` / ``sys.stderr`` replacement: complete lines go to ``emit``; a carriage return discards the
    line so far (a tqdm-style bar only delivers its final state); the rest is emitted on :meth:`close_flush`.

    ``real_stream`` is the stream this one replaced (``sys.__stderr__`` when it is not given): where :meth:`fileno`
    points and where a *re-entrant* write goes — one arriving while this thread is already inside ``emit``, i.e. a
    write caused by the logging of an earlier write. Without that escape a handler failing inside ``emit`` would
    report the failure to this very sink, forever (see the module docstring)."""

    def __init__(self, emit: Callable[[str], None], *, real_stream: TextIO | None = None) -> None:
        super().__init__()
        self._emit = emit
        self._pending = ""
        self._lock = threading.Lock()
        self._real_stream = real_stream
        self._state = threading.local()

    @property
    def real_stream(self) -> TextIO | None:
        """The stream this sink replaced — the escape hatch of the re-entrancy guard and of :meth:`fileno`."""
        return self._real_stream if self._real_stream is not None else sys.__stderr__

    def writable(self) -> bool:
        return True

    def write(self, s: str, /) -> int:
        if getattr(self._state, "emitting", False):  # a write caused by our own emit: never feed it back in
            stream = self.real_stream
            if stream is not None:
                stream.write(s)
                stream.flush()
            return len(s)
        with self._lock:
            self._pending += s
            *complete, self._pending = self._pending.split("\n")
        self._emit_lines(complete)
        return len(s)

    def _emit_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        self._state.emitting = True
        try:
            for line in lines:
                self._emit(line.rsplit("\r", 1)[-1])
        finally:
            self._state.emitting = False

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        """The file descriptor of the stream this sink replaced — a library asking the current ``sys.stdout`` for
        its descriptor gets the terminal's, instead of ``io.UnsupportedOperation`` for the whole run."""
        stream = self.real_stream
        if stream is None:
            raise io.UnsupportedOperation("fileno")
        return stream.fileno()

    @property
    def encoding(self) -> str:  # type: ignore[override]  # TextIOBase declares a plain attribute; libraries only read it
        return "utf-8"

    def close_flush(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, ""
        if pending.strip():
            self._emit_lines([pending])


class DashboardLogHandler(logging.Handler):
    """``logging.Handler`` whose records land in ``sink`` — a dashboard's log panel (or its stream when disabled).

    Records of ``keep_level`` and above (default WARNING), and records logged with ``extra={"keep": True}`` (the
    plan / status tables, stage summaries, the final report), are *kept*: the live dashboards print them once,
    unwrapped, after the display closed, where they survive the run in the terminal's history — the panel only shows
    the last few lines. ``skip`` names the loggers that reach the sink through another handler already (the handler
    :func:`attach_logger` installed): the root handler of a :class:`LoggingCapture` passes it so a record is not
    written twice."""

    def __init__(
        self,
        sink: LogSink,
        level: int = logging.NOTSET,
        keep_level: int = logging.WARNING,
        *,
        skip: Callable[[str], bool] | None = None,
        error_stream: TextIO | None = None,
    ) -> None:
        super().__init__(level)
        self._sink = sink
        self._keep_level = keep_level
        self._skip = skip
        self._error_stream = error_stream
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._skip is not None and self._skip(record.name):
                return
            text = self.format(record)
            keep = record.levelno >= self._keep_level or bool(getattr(record, "keep", False))
            self._sink.write(text, keep=keep)
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Report a failing :meth:`emit` on the *saved real* stderr, never through ``logging`` or ``sys.stderr``.

        ``logging.Handler.handleError`` writes to ``sys.stderr``, which is a :class:`LineSink` while a display is
        up: the report would become a record, reach this handler again and recurse. Writing to the stream the
        dashboard replaced ends there."""
        if not logging.raiseExceptions:
            return
        stream = self._error_stream if self._error_stream is not None else sys.__stderr__
        if stream is None:
            return
        try:
            stream.write("--- Logging error ---\n")
            traceback.print_exception(*sys.exc_info(), file=stream)
            stream.write(f"Logged from file {record.filename}, line {record.lineno}\n")
            stream.flush()
        except Exception:  # noqa: S110  # the reporting stream is gone too: there is nowhere left to complain
            pass


@contextmanager
def attach_logger(
    sink: LogSink, logger: logging.Logger, log_file: Path | None, *, ensure_info_level: bool = False
) -> Iterator[logging.FileHandler | None]:
    """Route ``logger`` into ``sink`` for the duration of the block (the body of both dashboards' ``attach``).

    The plain stream handlers a CLI installed are detached (their lines would print twice and garble the live
    display) and restored afterwards; with ``log_file`` every record is also appended to that file. With
    ``ensure_info_level`` a logger whose effective level is above INFO is lowered to INFO for the block — the
    training dashboard lives on INFO records — and restored afterwards. Yields the file handler (None without
    ``log_file``): the training dashboard hands it the lines the fallback would log (step, validation, event), so
    they reach the file and only the file."""
    detached: list[logging.Handler] = [handler for handler in logger.handlers if is_console_handler(handler)]
    added: list[logging.Handler] = [DashboardLogHandler(sink)]
    file_handler: logging.FileHandler | None = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        added.append(file_handler)
    for handler in detached:
        logger.removeHandler(handler)
    for handler in added:
        logger.addHandler(handler)
    previous_level = logger.level
    if ensure_info_level and logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield file_handler
    finally:
        logger.setLevel(previous_level)
        for handler in added:
            logger.removeHandler(handler)
            handler.close()
        for handler in detached:
            logger.addHandler(handler)


class LoggingCapture:
    """Every ``logging`` record into ``sink`` while :meth:`start`\\ ed: a :class:`DashboardLogHandler` on the root
    logger (``skip`` names the loggers an attached handler already covers), while every plain console
    ``StreamHandler`` of every logger — the ones ``transformers`` / ``datasets`` / ``huggingface_hub`` install on
    theirs at import — is detached until :meth:`stop`. Both are idempotent."""

    def __init__(self, sink: LogSink, *, skip: Callable[[str], bool] | None = None) -> None:
        self._sink = sink
        self._skip = skip
        self._detached: list[tuple[logging.Logger, logging.Handler]] = []
        self._root_handler: DashboardLogHandler | None = None

    @property
    def active(self) -> bool:
        return self._root_handler is not None

    def start(self) -> None:
        if self._root_handler is not None:
            return
        console_streams = {stream for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__) if stream is not None}
        error_stream = sys.__stderr__
        for logger in existing_loggers():
            for handler in list(logger.handlers):
                if is_console_handler(handler) and handler.stream in console_streams:
                    logger.removeHandler(handler)
                    self._detached.append((logger, handler))
        self._root_handler = DashboardLogHandler(self._sink, skip=self._skip, error_stream=error_stream)
        logging.getLogger().addHandler(self._root_handler)

    def stop(self) -> None:
        if self._root_handler is not None:
            logging.getLogger().removeHandler(self._root_handler)
            self._root_handler.close()
            self._root_handler = None
        detached, self._detached = self._detached, []
        for logger, handler in detached:
            logger.addHandler(handler)


class StreamCapture:
    """``sys.stdout`` / ``sys.stderr`` as :class:`LineSink`\\ s logging (INFO / WARNING) what is written to them, on
    the two given loggers — so stray prints, bare stderr writes and the final line of a tqdm bar reach the log panel
    instead of the terminal behind the display. :meth:`redirect` and :meth:`release` are idempotent and can be used
    around a terminal prompt; :meth:`release` logs what a sink still holds without a newline."""

    def __init__(self, stdout_logger_name: str, stderr_logger_name: str) -> None:
        self._stdout_logger_name = stdout_logger_name
        self._stderr_logger_name = stderr_logger_name
        self._saved: tuple[TextIO, TextIO] | None = None
        self._sinks: tuple[LineSink, LineSink] | None = None

    @property
    def redirected(self) -> bool:
        return self._sinks is not None

    def redirect(self) -> None:
        if self._sinks is not None:
            return
        real_out, real_err = sys.stdout, sys.stderr
        self._saved = (real_out, real_err)
        stdout_logger = logging.getLogger(self._stdout_logger_name)
        stderr_logger = logging.getLogger(self._stderr_logger_name)
        stdout_logger.setLevel(logging.INFO)  # whatever the package logger is set to, a stray line is never dropped
        stderr_logger.setLevel(logging.INFO)
        self._sinks = (
            LineSink(stdout_logger.info, real_stream=real_out),
            LineSink(stderr_logger.warning, real_stream=real_err),
        )
        sys.stdout, sys.stderr = self._sinks

    def release(self) -> None:
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved
            self._saved = None
        if self._sinks is not None:
            sinks, self._sinks = self._sinks, None
            for sink in sinks:
                sink.close_flush()

    def __enter__(self) -> StreamCapture:
        self.redirect()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback_: TracebackType | None
    ) -> None:
        self.release()
