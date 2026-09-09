# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Command line of a training run, the mirror of `data_preparation/prepare.py`.

    python training/train.py --config config/crow_300m_final.yaml [--key value ...]

Parses the settings (`--config` YAML plus `--key value` overrides), installs the stop request, calls
`training.run.train` and prints its report. Ctrl-C (or SIGTERM) once: the run finishes the current optimizer step,
saves a checkpoint and exits 130; during the in-process dataset build it stops at the next shard instead
(`BuildAborted`, also 130). A second Ctrl-C aborts right away. `resume: true` continues a stopped run.

Console: one stderr handler on the `training` and `data_preparation` logger hierarchies (`configure_console_logging`).
Under torchrun (`backend: ddp`, `make training-ddp`) every rank runs this CLI; rank 0 logs, shows the dashboard and
prints the report, the other ranks (`RANK` in the environment) keep WARNING and above with a `[rank N]` prefix, so a
failure on any rank is seen. `torchrun --redirects 3 --local-ranks-filter 0` silences them completely.
`RunLogger` opens the terminal dashboard of `training/ui/` for the run (the live display on a TTY, the one-line
fallback when piped or with `TRAINING_DASHBOARD=0`, `<out_dir>/<run_name>/train.log` in both cases, and the report as
`train_report.json` next to it); it swaps the `training`
handler out for the duration, so nothing prints twice.

Exit codes: 0 finished, 1 failed (traceback logged), 3 another run holds the lock (the message names its pid and
start time; `data_preparation/lib/build/lock.py`), 130 interrupted.
"""

from __future__ import annotations

import logging
import os
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
from data_preparation.lib.build.lock import RunLocked
from data_preparation.lib.log import LOG_FORMAT, ProgressStreamHandler, configure_logging
from training.run import train
from training.settings import parse_settings
from training.ui.common import KEEP, TRAINING_LOGGER_NAME  # `training`: the hierarchy `RunLogger` and the dashboard log on

EXIT_INTERRUPTED = 130
EXIT_ALREADY_RUNNING = 3

log = logging.getLogger(f"{TRAINING_LOGGER_NAME}.train")


class StopRequest:
    """
    The one stop request of a run (a `StopCheck`: calling it answers "stop?"). `train()` polls it after every
    optimizer step, the in-process dataset build between shards.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_stop(self) -> None:
        self._event.set()

    def __call__(self) -> bool:
        return self._event.is_set()


@contextmanager
def stop_on_interrupt() -> Iterator[StopRequest]:
    """
    Install the Ctrl-C / SIGTERM handling of a run and yield its stop request; the previous handlers are put back
    on exit.

    The first signal sets the request, logs it and hands both signals back to their default handlers, so a second
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


def launch_rank() -> int:
    """
    This process's rank under torchrun (`RANK` in the environment), 0 for a plain launch.
    """

    return int(os.environ.get("RANK", "0"))


def configure_console_logging(level: int = logging.INFO, rank: int = 0) -> logging.Logger:
    """
    Attach one stderr handler each to the `training` and `data_preparation` logger hierarchies (same handler type
    and line format); idempotent. Returns the `training` logger. The CLI's job: library code configures no logging.

    `rank` above 0 (a non-main rank under torchrun) logs WARNING and above only, every line prefixed with
    `[rank N]`: the main rank tells the story of the run, the others only speak up when something is wrong.
    """

    if rank > 0:
        level = max(level, logging.WARNING)
    line_format = f"[rank {rank}] {LOG_FORMAT}" if rank > 0 else LOG_FORMAT
    data_logger = configure_logging(level)  # the `data_preparation` hierarchy: the status table, split and build lines
    training_logger = logging.getLogger(TRAINING_LOGGER_NAME)
    training_logger.setLevel(level)
    handler = next(
        (existing for existing in training_logger.handlers if isinstance(existing, ProgressStreamHandler)), None
    )
    if handler is None:
        handler = ProgressStreamHandler(sys.stderr)
        training_logger.addHandler(handler)
    handler.setLevel(level)
    for owner in (training_logger, data_logger):
        for existing in owner.handlers:
            if isinstance(existing, ProgressStreamHandler):
                existing.setFormatter(logging.Formatter(line_format))
    return training_logger


def main(argv: list[str] | None = None) -> int:
    """
    Parse `argv` (default: the command line), run, print the report (the main rank; module docstring); returns the
    exit code.
    """

    started_at = time.time()
    rank = launch_rank()
    configure_console_logging(rank=rank)
    settings = parse_settings(argv)
    with stop_on_interrupt() as should_stop:
        try:
            report = train(settings, should_stop=should_stop, started_at=started_at)
        except RunLocked as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_ALREADY_RUNNING
        except (KeyboardInterrupt, BuildAborted):
            log.warning("training interrupted; checkpoints and published dataset shards are kept, rerun to resume")
            return EXIT_INTERRUPTED
        except Exception:
            log.exception("training failed")
            return 1
    if rank == 0:
        print(report.summary())
    return EXIT_INTERRUPTED if report.stopped else 0


if __name__ == "__main__":
    sys.exit(main())
