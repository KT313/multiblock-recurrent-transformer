# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Command line of a training run — the mirror of `data_preparation/prepare.py`.

    python training/train.py --config config/crow_300m_final.yaml [--key value ...]

Parses the settings (`--config` YAML plus `--key value` overrides), installs the run's stop request, calls
`training.run.train` and prints its report. Ctrl-C (or SIGTERM) once: the run finishes the current optimizer step,
saves a checkpoint and exits 130 — during the in-process dataset build (`auto_prepare`) it stops at the next shard
instead, everything published kept (`BuildAborted`, also 130); a second Ctrl-C aborts right away as usual.
`resume: true` continues a stopped run from its last checkpoint.

The console: this module attaches one stderr handler to the `training` and `data_preparation` logger hierarchies
(`configure_console_logging`). For the run itself `RunLogger` opens the terminal dashboard of `training/ui/` — the
live display on a TTY, the one-line-per-`log_step_interval` fallback when piped or with `TRAINING_DASHBOARD=0`,
`<out_dir>/train.log` in both cases — which swaps that `training` handler out for the duration and puts it back, so
nothing prints twice; a first Ctrl-C / SIGTERM and an exception both leave through the dashboard's `__exit__`
(frame erased, kept lines and the static summary printed), and the report's summary is printed after that.

Exit codes: 0 finished, 1 failed (logged with its traceback), 130 interrupted.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python training/train.py` from the repo root

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.log import LOG_FORMAT, ProgressStreamHandler, configure_logging
from training.run import train
from training.settings import parse_settings
from training.ui.dashboard import TRAINING_LOGGER_NAME  # `training`: the hierarchy `RunLogger` and the dashboard log on

EXIT_INTERRUPTED = 130
KEEP = {"keep": True}  # `extra=` of the records that must survive in the terminal scrollback under a live dashboard

log = logging.getLogger(f"{TRAINING_LOGGER_NAME}.train")


class StopRequest:
    """The one stop request of a run (a `StopCheck`: calling it answers "stop?"). `train()` polls it after every
    optimizer step, the in-process dataset build between shards."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_stop(self) -> None:
        self._event.set()

    def __call__(self) -> bool:
        return self._event.is_set()


@contextmanager
def stop_on_interrupt() -> Iterator[StopRequest]:
    """Install the Ctrl-C / SIGTERM handling of a run and yield its stop request; the previous handlers are put back
    on exit.

    The first signal sets the request, logs it and hands both signals back to their default handlers — so a second
    Ctrl-C raises `KeyboardInterrupt` as usual (and a second SIGTERM kills). Signal handlers can only be installed
    from the main thread; elsewhere the request is yielded unarmed.
    """
    request = StopRequest()
    if threading.current_thread() is not threading.main_thread():
        yield request
        return

    def on_signal(signum: int, frame: FrameType | None) -> None:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        request.request_stop()
        log.warning(
            "%s received: stopping after this step, saving a checkpoint (press Ctrl-C again to abort right away)",
            signal.Signals(signum).name,
            extra=KEEP,
        )

    previous = {signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield request
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def configure_console_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach one stderr stream handler each to the `training` and the `data_preparation` logger hierarchies, so
    `RunLogger`'s lines and the dataset resolver's (which logs under `data_preparation`) reach the terminal; idempotent.

    The CLI's job — library code does not configure logging. Both hierarchies get the same handler type and line
    format (`data_preparation.lib.log.configure_logging` for the data-prep one). The dashboard `RunLogger` opens
    swaps the `training` stream handler out for the run and restores it afterwards. Returns the `training` logger.
    """
    configure_logging(level)  # the `data_preparation` hierarchy: the resolver's status table, split and build lines
    training_logger = logging.getLogger(TRAINING_LOGGER_NAME)
    training_logger.setLevel(level)
    handler = next((h for h in training_logger.handlers if isinstance(h, ProgressStreamHandler)), None)
    if handler is None:
        handler = ProgressStreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        training_logger.addHandler(handler)
    handler.setLevel(level)
    return training_logger


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` (default: the command line), run, print the report; returns the exit code (module docstring)."""
    started_at = time.time()
    configure_console_logging()
    settings = parse_settings(argv)
    with stop_on_interrupt() as should_stop:
        try:
            report = train(settings, should_stop=should_stop, started_at=started_at)
        except (KeyboardInterrupt, BuildAborted):
            log.warning("training interrupted; checkpoints and published dataset shards are kept, rerun to resume")
            return EXIT_INTERRUPTED
        except Exception:
            log.exception("training failed")
            return 1
    print(report.summary())
    return EXIT_INTERRUPTED if report.stopped else 0


if __name__ == "__main__":
    sys.exit(main())
