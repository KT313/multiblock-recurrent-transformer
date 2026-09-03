# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Everything that could print around the training display, routed into it: :class:`TerminalCapture` (the
root-logger handler, the detached third-party console handlers, the ``warnings.showwarning`` hook, the stdout /
stderr line sinks and the wandb environment variables) for the duration of the display.

The generic half lives in :mod:`ui.capture`, shared with the data-prep dashboard; this module adds what only a
training run needs: the ``training.*`` logger names, the run's log handlers (:func:`run_log_handlers`), the
``warnings`` hook and the wandb environment."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, TextIO

from data_preparation.lib.log import LOG_FORMAT
from ui.capture import LoggingCapture, LogSink, StreamCapture  # generic, shared with the data-prep dashboard
from training.ui.common import TRAINING_LOGGER_NAME, lines_log

STDOUT_LOGGER = f"{TRAINING_LOGGER_NAME}.stdout"  # lines written to sys.stdout while the display is up (INFO)
STDERR_LOGGER = f"{TRAINING_LOGGER_NAME}.stderr"  # lines written to sys.stderr while the display is up (WARNING: kept)
WARNINGS_LOGGER = f"{TRAINING_LOGGER_NAME}.warnings"  # `warnings.warn` calls while the display is up (WARNING: kept)

# `wandb.init(settings=wandb.Settings(**WANDB_QUIET_SETTINGS))`: no console wrapping, no banner lines on stderr
WANDB_QUIET_SETTINGS: dict[str, object] = {"console": "off", "silent": True}
# the same as environment variables, set while the display is up (wandb reads them at `init`)
QUIET_ENV: dict[str, str] = {f"WANDB_{key.upper()}": str(value).lower() for key, value in WANDB_QUIET_SETTINGS.items()}


def format_warning(message: Warning | str, category: type[Warning], filename: str, lineno: int) -> str:
    """One line per warning with the message first: ``UserWarning: the text (/abs/path/module.py:12)``.

    ``warnings.formatwarning`` puts the absolute path first and adds the source line as a second line; here the
    location trails, so a warning is exactly one kept line."""
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


def line_handler(stream_or_file: TextIO | Path) -> logging.Handler:
    """A handler for the dashboards' lines (:data:`~training.ui.common.lines_log`): a ``FileHandler`` appending to a
    path or a ``StreamHandler`` on a stream, both in the ``LOG_FORMAT`` the ``training`` records are written in, so
    ``train.log`` and the console read as one log."""
    handler: logging.Handler
    if isinstance(stream_or_file, Path):
        stream_or_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(stream_or_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(stream_or_file)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


@contextmanager
def run_log_handlers(logger: logging.Logger, log_file: Path | None, stream: TextIO | None) -> Iterator[None]:
    """The run's log file and console for the block: ONE ``FileHandler`` on ``log_file``, shared by ``logger`` and
    :data:`~training.ui.common.lines_log` so ``train.log`` is written in logging order, and a ``StreamHandler`` on
    ``lines_log`` when ``stream`` is given. None leaves a destination out. Handlers are removed and closed
    afterwards."""
    added: list[tuple[logging.Logger, logging.Handler]] = []
    if log_file is not None:
        file_handler = line_handler(log_file)
        added += [(logger, file_handler), (lines_log, file_handler)]
    if stream is not None:
        added.append((lines_log, line_handler(stream)))
    for target, handler in added:
        target.addHandler(handler)
    try:
        yield
    finally:
        for target, handler in added:
            target.removeHandler(handler)
        for handler in {id(handler): handler for _target, handler in added}.values():
            handler.close()


class TerminalCapture:
    """While :meth:`start`\\ed, nothing but the live display reaches the terminal:

    * every ``logging`` record goes to ``sink`` through a root-logger handler (``already_attached`` names the loggers an
      attached handler already covers); the plain console ``StreamHandler``\\s of every logger are detached,
    * ``warnings.showwarning`` is replaced: every ``warnings.warn`` becomes one kept record on
      :data:`WARNINGS_LOGGER` (:func:`format_warning`). The previous hook is restored afterwards; one installed
      inside the block stays, and a stale reference to the dashboard's hook forwards to the hook the display found,
    * ``sys.stdout`` / ``sys.stderr`` become line sinks logging on :data:`STDOUT_LOGGER` (INFO) and
      :data:`STDERR_LOGGER` (WARNING), under ``training`` so the lines also reach the log file,
    * :data:`QUIET_ENV` is set for a ``wandb.init`` inside the block.

    :meth:`stop` undoes all of it (idempotent, each part on its own). :meth:`release_streams` /
    :meth:`redirect_streams` hand the streams back around a terminal prompt.
    """

    def __init__(self, sink: LogSink, *, already_attached: Callable[[str], bool] | None = None) -> None:
        self._logging = LoggingCapture(sink, already_attached=already_attached)
        self._streams = StreamCapture(STDOUT_LOGGER, STDERR_LOGGER)
        self._saved_env: dict[str, str | None] | None = None
        self._previous_showwarning: _ShowWarning | None = None  # what `warnings.showwarning` was when the capture started
        self._warnings_captured = False

    def start(self) -> None:
        self._quiet_environment()
        self._logging.start()
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
                    self._logging.stop()
                finally:
                    self._restore_environment()

    @property
    def streams_redirected(self) -> bool:
        return self._streams.redirected

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
        # replaced inside the block: theirs stays
        if warnings.showwarning == self._show_warning and self._previous_showwarning is not None:
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
        elif self._previous_showwarning is not None:  # a stale reference after the display closed
            self._previous_showwarning(message, category, filename, lineno, file, line)

    def redirect_streams(self) -> None:
        """``sys.stdout`` / ``sys.stderr`` become line sinks that log (INFO / WARNING) what is written to them."""
        self._streams.redirect()

    def release_streams(self) -> None:
        """The real streams back; what a sink still holds without a newline is logged now."""
        self._streams.release()
