# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
How both live dashboards keep the terminal to themselves.

A dashboard is a :class:`LogSink`: something with write(text, *, keep=False). While its display is up:

* :class:`LoggingCapture` puts a :class:`DashboardLogHandler` on the root logger and detaches every plain console
  StreamHandler (the ones huggingface_hub / datasets / transformers install at import), which would
  write behind the display,
* :class:`StreamCapture` replaces sys.stdout / sys.stderr with :class:`LineSink`\\ s that turn writes into
  log records,
* :func:`attach_logger` routes one logger (data_preparation / training) into the sink directly and appends
  its records to a log file.

Three pitfalls handled here once:

* Recursion. A handler that fails while emitting reports to sys.stderr, which is the line sink, which logs
  again. :class:`LineSink` keeps a thread-local re-entrancy flag and sends such writes to the saved real stream;
  :meth:`DashboardLogHandler.handleError` writes to the real stderr, never through logging.
* sys.stdout.fileno(). A bare io.TextIOBase has none; :meth:`LineSink.fileno` answers with the real
  stream's.
* Loggers created during iteration. wandb's threads create loggers while a run is up; :func:`existing_loggers`
  snapshots loggerDict under the logging lock.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import traceback
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TextIO, TypeGuard

from ui.log_format import LOG_FORMAT


class LogSink(Protocol):
    """
    What :class:`DashboardLogHandler` writes to: a dashboard's log panel (or, disabled, its stream).
    """

    def write(self, text: str, *, keep: bool = False) -> None: ...


def is_console_handler(handler: logging.Handler) -> TypeGuard[logging.StreamHandler[Any]]:
    """
    A plain StreamHandler (a terminal writer) rather than a file / dashboard / library handler.
    """

    return isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)


def existing_loggers() -> list[logging.Logger]:
    """
    Every logger that exists right now, plus the root logger.

    The dict is copied under the logging module's own lock: wandb's background threads create loggers while a
    run is up, and iterating it directly can raise "dictionary changed size during iteration".

    The lock is private to `logging` and named differently across versions: `logging._lock` (a context manager,
    3.13 dropped the helpers around it), else the `_acquireLock` / `_releaseLock` pair of 3.11 and 3.12. Without
    either the copy is taken unguarded, which is what iterating would do anyway.
    """

    lock = getattr(logging, "_lock", None)
    acquire: Callable[[], None] | None = getattr(logging, "_acquireLock", None)
    release: Callable[[], None] | None = getattr(logging, "_releaseLock", None)
    if lock is not None:
        with lock:
            entries = list(logging.root.manager.loggerDict.values())
    elif acquire is not None and release is not None:
        acquire()
        try:
            entries = list(logging.root.manager.loggerDict.values())
        finally:
            release()
    else:  # no CPython release so far is without both
        entries = list(logging.root.manager.loggerDict.values())
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers.extend(entry for entry in entries if isinstance(entry, logging.Logger))
    return loggers


_sinks: weakref.WeakSet[LineSink] = weakref.WeakSet()
_fork_hook_registered = False  # `register_at_fork` cannot be undone, so the hook is registered once


def _reset_in_child() -> None:
    """
    A fresh lock for every sink in a forked child (a DataLoader worker): the copied lock may be held by a
    parent thread that does not exist in the child, and the worker's first print would block on it forever.
    """

    for sink in _sinks:
        sink._lock = threading.Lock()


def _register_fork_hook() -> None:
    global _fork_hook_registered
    if not _fork_hook_registered and hasattr(os, "register_at_fork"):
        _fork_hook_registered = True
        os.register_at_fork(after_in_child=_reset_in_child)


class LineSink(io.TextIOBase):
    """
    A sys.stdout / sys.stderr replacement: complete lines go to emit, a carriage return discards the
    line so far (a tqdm bar delivers only its final state), the rest is emitted on :meth:`close_flush`.

    real_stream is the stream this one replaced (sys.__stderr__ when not given): where :meth:`fileno` points
    and where a re-entrant write goes, one arriving while this thread is already inside emit (see the module
    docstring).
    """

    def __init__(self, emit: Callable[[str], None], *, real_stream: TextIO | None = None) -> None:
        super().__init__()
        self._emit = emit
        self._pending = ""
        self._lock = threading.Lock()
        self._real_stream = real_stream
        self._thread_local = threading.local()
        _register_fork_hook()
        _sinks.add(self)

    @property
    def real_stream(self) -> TextIO | None:
        """
        The stream this sink replaced: the target of re-entrant writes and of :meth:`fileno`.
        """

        return self._real_stream if self._real_stream is not None else sys.__stderr__

    def writable(self) -> bool:
        return True

    def write(self, text: str, /) -> int:
        if getattr(self._thread_local, "emitting", False):  # caused by our own emit: do not feed it back in
            stream = self.real_stream
            if stream is not None:
                stream.write(text)
                stream.flush()
            return len(text)
        with self._lock:
            self._pending += text
            *complete, rest = self._pending.split("\n")
            # only what follows the last "\r" (a trailing one stays, it opens the next frame): a bar that redraws
            # itself with "\r" alone never grows the remainder
            self._pending = rest[rest.rfind("\r", 0, -1) + 1 :]
        self._emit_lines(complete)
        return len(text)

    def _emit_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        self._thread_local.emitting = True
        try:
            for line in lines:
                self._emit(line.rstrip("\r").rsplit("\r", 1)[-1])  # rstrip: a CRLF line, or a bar closed with "\r\n"
        finally:
            self._thread_local.emitting = False

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        """
        The descriptor of the replaced stream, so a library asking sys.stdout for one gets the terminal's.
        """

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
    """
    logging.Handler whose records land in sink, a dashboard's log panel (or its stream when disabled).

    Records at keep_level and above (default WARNING) and records logged with extra={"keep": True} are
    *kept*: the dashboards print them once after the display closed, so they survive in the terminal's history.
    already_attached names loggers that reach the sink through another handler already, so a record is not written twice.
    on_new_logger is called with each logger name the first time a record of it arrives (the capture detaches
    the console handler a library imported under the display put on its logger).
    """

    def __init__(
        self,
        sink: LogSink,
        level: int = logging.NOTSET,
        keep_level: int = logging.WARNING,
        *,
        already_attached: Callable[[str], bool] | None = None,
        error_stream: TextIO | None = None,
        on_new_logger: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(level)
        self._sink = sink
        self._keep_level = keep_level
        self._already_attached = already_attached
        self._error_stream = error_stream
        self._on_new_logger = on_new_logger
        self._seen: set[str] = set()
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._already_attached is not None and self._already_attached(record.name):
                return
            if self._on_new_logger is not None and record.name not in self._seen:
                self._seen.add(record.name)
                self._on_new_logger(record.name)
            text = self.format(record)
            keep = record.levelno >= self._keep_level or bool(getattr(record, "keep", False))
            self._sink.write(text, keep=keep)
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """
        Report a failing :meth:`emit` on the saved real stderr, never through logging or sys.stderr (a
        :class:`LineSink` while a display is up, which would recurse).
        """

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
        except Exception:  # noqa: S110  # the reporting stream is gone too
            pass


@contextmanager
def attach_logger(
    sink: LogSink, logger: logging.Logger, log_file: Path | None, *, ensure_info_level: bool = False
) -> Iterator[logging.FileHandler | None]:
    """
    Route logger into sink for the block (the body of both dashboards' attach).

    The logger's plain stream handlers are detached for the block (their lines would garble the display) and
    restored afterwards. With log_file every record is also appended to that file. With ensure_info_level a
    logger above INFO is lowered to INFO for the block. Yields the file handler (None without log_file); the
    training dashboard writes its step / validation / event lines to it directly.
    """

    detached_handlers: list[logging.Handler] = [handler for handler in logger.handlers if is_console_handler(handler)]
    added_handlers: list[logging.Handler] = [DashboardLogHandler(sink)]
    file_handler: logging.FileHandler | None = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        added_handlers.append(file_handler)
    for handler in detached_handlers:
        logger.removeHandler(handler)
    for handler in added_handlers:
        logger.addHandler(handler)
    previous_level = logger.level
    if ensure_info_level and logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield file_handler
    finally:
        logger.setLevel(previous_level)
        for handler in added_handlers:
            logger.removeHandler(handler)
            handler.close()
        for handler in detached_handlers:
            logger.addHandler(handler)


def _writes_to_the_terminal(handler: logging.Handler) -> bool:
    """
    A console handler on the process's stdout / stderr, or on a :class:`LineSink` standing in for one.
    """

    return is_console_handler(handler) and (
        isinstance(handler.stream, LineSink) or handler.stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    )


class LoggingCapture:
    """
    Every logging record into sink between :meth:`start` and :meth:`stop`: a :class:`DashboardLogHandler`
    on the root logger (already_attached names loggers an attached handler already covers) while every plain console
    StreamHandler of every logger is detached. A library imported under the display (huggingface_hub, datasets)
    puts a StreamHandler bound to the line sink on its logger; it is detached at the logger's first record and,
    like every handler still bound to a sink, pointed at the real stream on :meth:`stop`. Both methods are idempotent.
    """

    def __init__(self, sink: LogSink, *, already_attached: Callable[[str], bool] | None = None) -> None:
        self._sink = sink
        self._already_attached = already_attached
        self._detached_handlers: list[tuple[logging.Logger, logging.Handler]] = []
        self._root_handler: DashboardLogHandler | None = None

    @property
    def active(self) -> bool:
        return self._root_handler is not None

    def start(self) -> None:
        if self._root_handler is not None:
            return
        for logger in existing_loggers():
            self._detach_console_handlers(logger)
        self._root_handler = DashboardLogHandler(
            self._sink, already_attached=self._already_attached, error_stream=sys.__stderr__, on_new_logger=self._on_new_logger
        )
        logging.getLogger().addHandler(self._root_handler)

    def stop(self) -> None:
        if self._root_handler is not None:
            logging.getLogger().removeHandler(self._root_handler)
            self._root_handler.close()
            self._root_handler = None
        detached_handlers, self._detached_handlers = self._detached_handlers, []
        for logger, handler in detached_handlers:
            logger.addHandler(handler)
        for logger in existing_loggers():  # a handler created under the capture holds the sink, dead from now on
            for handler in logger.handlers:
                if is_console_handler(handler) and isinstance(handler.stream, LineSink):
                    real_stream = handler.stream.real_stream
                    if real_stream is not None:
                        handler.setStream(real_stream)

    def _detach_console_handlers(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            if _writes_to_the_terminal(handler):
                logger.removeHandler(handler)
                self._detached_handlers.append((logger, handler))

    def _on_new_logger(self, name: str) -> None:
        self._detach_console_handlers(logging.getLogger(name))


class StreamCapture:
    """
    sys.stdout / sys.stderr replaced by :class:`LineSink`\\ s that log what is written (INFO / WARNING) on
    the two given loggers, so stray prints and tqdm's final line reach the log panel. :meth:`redirect` lowers the
    two loggers to INFO for the block and :meth:`release` puts their levels back. Both are idempotent;
    :meth:`release` logs what a sink still holds without a newline.
    """

    def __init__(self, stdout_logger_name: str, stderr_logger_name: str) -> None:
        self._stdout_logger_name = stdout_logger_name
        self._stderr_logger_name = stderr_logger_name
        self._real_streams: tuple[TextIO, TextIO] | None = None
        self._sinks: tuple[LineSink, LineSink] | None = None
        self._levels: tuple[int, int] | None = None  # the two stream loggers' levels before `redirect` lowered them

    @property
    def redirected(self) -> bool:
        return self._sinks is not None

    def redirect(self) -> None:
        if self._sinks is not None:
            return
        real_out, real_err = sys.stdout, sys.stderr
        self._real_streams = (real_out, real_err)
        stdout_logger = logging.getLogger(self._stdout_logger_name)
        stderr_logger = logging.getLogger(self._stderr_logger_name)
        self._levels = (stdout_logger.level, stderr_logger.level)  # put back by `release`
        stdout_logger.setLevel(logging.INFO)  # whatever the package logger is set to, a stray line is never dropped
        stderr_logger.setLevel(logging.INFO)
        self._sinks = (
            LineSink(stdout_logger.info, real_stream=real_out),
            LineSink(stderr_logger.warning, real_stream=real_err),
        )
        sys.stdout, sys.stderr = self._sinks

    def release(self) -> None:
        if self._real_streams is not None:
            sys.stdout, sys.stderr = self._real_streams
            self._real_streams = None
        if self._sinks is not None:
            sinks, self._sinks = self._sinks, None
            for sink in sinks:
                sink.close_flush()
        if self._levels is not None:  # after the flush above: its records are the ones the low level exists for
            stdout_level, stderr_level = self._levels
            self._levels = None
            logging.getLogger(self._stdout_logger_name).setLevel(stdout_level)
            logging.getLogger(self._stderr_logger_name).setLevel(stderr_level)

    def __enter__(self) -> StreamCapture:
        self.redirect()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback_: TracebackType | None
    ) -> None:
        self.release()
