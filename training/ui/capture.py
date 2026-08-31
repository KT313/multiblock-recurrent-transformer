# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Everything that could print around the live display, routed into it: the ``logging`` handler of the dashboards,
:func:`attach_logger` (one logger plus the log file), and :class:`TerminalCapture` — the root-logger handler, the
detached third-party console handlers, the ``warnings.showwarning`` hook, the ``sys.stdout`` / ``sys.stderr`` line
sinks and the wandb environment variables, all for the duration of the display. The line sink and the console-handler
check are the data-prep dashboard's (``data_preparation/lib/ui/dashboard.py``)."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, TextIO

from data_preparation.lib.log import LOG_FORMAT
from data_preparation.lib.ui.dashboard import _is_console_handler, _LineSink  # generic, shared with the data-prep dashboard
from training.ui.common import TRAINING_LOGGER_NAME

STDOUT_LOGGER = f"{TRAINING_LOGGER_NAME}.stdout"  # lines written to sys.stdout while the display is up (INFO)
STDERR_LOGGER = f"{TRAINING_LOGGER_NAME}.stderr"  # lines written to sys.stderr while the display is up (WARNING: kept)
WARNINGS_LOGGER = f"{TRAINING_LOGGER_NAME}.warnings"  # `warnings.warn` calls while the display is up (WARNING: kept)

# `wandb.init(settings=wandb.Settings(**WANDB_QUIET_SETTINGS))`: no console wrapping, no banner lines on stderr
WANDB_QUIET_SETTINGS: dict[str, object] = {"console": "off", "silent": True}
# the same as environment variables, set while the display is up (wandb reads them at `init`)
QUIET_ENV: dict[str, str] = {f"WANDB_{key.upper()}": str(value).lower() for key, value in WANDB_QUIET_SETTINGS.items()}


def format_warning(message: Warning | str, category: type[Warning], filename: str, lineno: int) -> str:
    """One line per warning with the message first: ``UserWarning: the text (/abs/path/module.py:12)``.

    ``warnings.formatwarning`` puts the (absolute, environment-dependent) path first and adds the source line as a
    second line; here the location trails, so a long path never pushes the message past the terminal width, where the
    terminal's soft-wrap would split it, and a warning is exactly one kept line."""
    return f"{category.__name__}: {message} ({filename}:{lineno})"


class _ShowWarning(Protocol):
    """The signature of ``warnings.showwarning``."""

    def __call__(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None: ...


class LogSink(Protocol):
    """What :class:`DashboardLogHandler` writes to (the live dashboard and the fallback)."""

    def write(self, text: str, *, keep: bool = False) -> None: ...


class DashboardLogHandler(logging.Handler):
    """``logging.Handler`` whose records land in the dashboard's log panel (or on its stream, for the fallback).

    Records of ``keep_level`` and above (default WARNING), and records logged with ``extra={"keep": True}`` (stage
    summaries, the final report), are *kept*: the live dashboard prints them once, unwrapped, after the display
    closed, where they survive the run in the terminal's history — the panel only shows the last few lines. The
    handler :class:`TerminalCapture` installs on the root logger passes ``skip``: records of loggers
    :func:`attach_logger` handles directly are dropped there (they reach the panel through the attached handler)."""

    def __init__(
        self,
        sink: LogSink,
        level: int = logging.NOTSET,
        keep_level: int = logging.WARNING,
        *,
        skip: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(level)
        self._sink = sink
        self._keep_level = keep_level
        self._skip = skip
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


@contextmanager
def attach_logger(sink: LogSink, logger: logging.Logger, log_file: Path | None) -> Iterator[logging.FileHandler | None]:
    """Route ``logger`` into ``sink`` for the duration of the block (the body of both dashboards' ``attach``).

    The plain stream handlers a CLI installed are detached (their lines would print twice and garble the live
    display) and restored afterwards; with ``log_file`` every record is also appended to that file. A logger whose
    effective level is above INFO is lowered to INFO for the block — the dashboard lives on INFO records — and
    restored afterwards. Yields the file handler (None without ``log_file``): the live dashboard hands it the lines
    the fallback would log (step, validation, event), so they reach the file and only the file."""
    detached: list[logging.Handler] = [h for h in logger.handlers if _is_console_handler(h)]
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
    if logger.getEffectiveLevel() > logging.INFO:
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


class TerminalCapture:
    """While :meth:`start`\\ed, nothing but the live display reaches the terminal:

    * every ``logging`` record goes to ``sink`` through a handler on the root logger (``skip`` names the loggers
      an attached handler already covers), while every plain console ``StreamHandler`` of every logger — the ones
      ``transformers`` / ``datasets`` / ``huggingface_hub`` install on theirs at import — is detached,
    * ``warnings.showwarning`` is replaced: every ``warnings.warn`` becomes one WARNING (kept) record on
      :data:`WARNINGS_LOGGER`, :func:`format_warning`\\ ed — whether the process would otherwise write warnings to
      stderr (the default) or route them through ``logging.captureWarnings`` (``py.warnings``), the hook is what the
      ``warnings`` module calls, so the text and the routing are the dashboard's. The previous hook is restored
      afterwards; one installed *inside* the block (a library calling ``logging.captureWarnings(True)``) is left in
      place — its records reach the panel through the root handler, its stderr lines through the sink below — and
      a stale reference to the dashboard's hook forwards to the hook the display found,
    * ``sys.stdout`` / ``sys.stderr`` are replaced by line sinks logging on :data:`STDOUT_LOGGER` (INFO) and
      :data:`STDERR_LOGGER` (WARNING), so stray prints, bare stderr writes and the final line of a tqdm bar land in
      the panel (the sink loggers and :data:`WARNINGS_LOGGER` sit under ``training``: the lines also reach the
      attached log file),
    * :data:`QUIET_ENV` is set for a ``wandb.init`` inside the block.

    :meth:`stop` undoes all of it (idempotent; each part on its own, so a failure in one still restores the rest).
    :meth:`release_streams` / :meth:`redirect_streams` hand the streams back around a terminal prompt.
    """

    def __init__(self, sink: LogSink, *, skip: Callable[[str], bool] | None = None) -> None:
        self._sink = sink
        self._skip = skip
        self._detached_handlers: list[tuple[logging.Logger, logging.Handler]] = []
        self._root_handler: DashboardLogHandler | None = None
        self._saved_streams: tuple[TextIO, TextIO] | None = None
        self._sinks: tuple[_LineSink, _LineSink] | None = None
        self._saved_env: dict[str, str | None] | None = None
        self._previous_showwarning: _ShowWarning | None = None  # what `warnings.showwarning` was when the capture started
        self._warnings_captured = False

    def start(self) -> None:
        self._quiet_environment()
        self._capture_logging()
        self._capture_warnings()
        self.redirect_streams()

    def stop(self) -> None:
        try:
            self.release_streams()
        finally:
            try:
                self._release_warnings()
            finally:
                try:
                    self._release_logging()
                finally:
                    self._restore_environment()

    @property
    def streams_redirected(self) -> bool:
        return self._sinks is not None

    def _quiet_environment(self) -> None:
        if self._saved_env is None:
            self._saved_env = {name: os.environ.get(name) for name in QUIET_ENV}
            os.environ.update(QUIET_ENV)

    def _restore_environment(self) -> None:
        saved, self._saved_env = self._saved_env, None
        for name, value in (saved or {}).items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _capture_logging(self) -> None:
        if self._root_handler is not None:
            return
        console_streams = {stream for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__) if stream is not None}
        loggers: list[logging.Logger] = [logging.getLogger()]
        loggers.extend(logger for logger in logging.root.manager.loggerDict.values() if isinstance(logger, logging.Logger))
        for logger in loggers:
            for handler in list(logger.handlers):
                if _is_console_handler(handler) and handler.stream in console_streams:
                    logger.removeHandler(handler)
                    self._detached_handlers.append((logger, handler))
        self._root_handler = DashboardLogHandler(self._sink, skip=self._skip)
        logging.getLogger().addHandler(self._root_handler)

    def _release_logging(self) -> None:
        if self._root_handler is not None:
            logging.getLogger().removeHandler(self._root_handler)
            self._root_handler.close()
            self._root_handler = None
        detached, self._detached_handlers = self._detached_handlers, []
        for logger, handler in detached:
            logger.addHandler(handler)

    def _capture_warnings(self) -> None:
        if self._warnings_captured:
            return
        self._previous_showwarning = warnings.showwarning
        self._warnings_captured = True
        logging.getLogger(WARNINGS_LOGGER).setLevel(logging.INFO)  # whatever the `training` logger is set to
        warnings.showwarning = self._show_warning

    def _release_warnings(self) -> None:
        if not self._warnings_captured:
            return
        self._warnings_captured = False
        if warnings.showwarning == self._show_warning and self._previous_showwarning is not None:  # else: replaced inside the block, theirs stays
            warnings.showwarning = self._previous_showwarning

    def _show_warning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        """The ``warnings.showwarning`` of the block: one kept record per warning on :data:`WARNINGS_LOGGER`."""
        if self._warnings_captured:
            logging.getLogger(WARNINGS_LOGGER).warning(format_warning(message, category, filename, lineno))
        elif self._previous_showwarning is not None:  # a stale reference after the display closed: behave like the hook it replaced
            self._previous_showwarning(message, category, filename, lineno, file, line)

    def redirect_streams(self) -> None:
        """``sys.stdout`` / ``sys.stderr`` become line sinks that log (INFO / WARNING) what is written to them."""
        if self._sinks is not None:
            return
        self._saved_streams = (sys.stdout, sys.stderr)
        stdout_logger, stderr_logger = logging.getLogger(STDOUT_LOGGER), logging.getLogger(STDERR_LOGGER)
        stdout_logger.setLevel(logging.INFO)  # whatever the `training` logger is set to, a stray line is never dropped
        stderr_logger.setLevel(logging.INFO)
        self._sinks = (_LineSink(stdout_logger.info), _LineSink(stderr_logger.warning))
        sys.stdout, sys.stderr = self._sinks

    def release_streams(self) -> None:
        """The real streams back; what a sink still holds without a newline is logged now."""
        if self._saved_streams is not None:
            sys.stdout, sys.stderr = self._saved_streams
            self._saved_streams = None
        if self._sinks is not None:
            sinks, self._sinks = self._sinks, None
            for sink in sinks:
                sink.close_flush()
