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
failure on any rank is seen. `torchrun --redirects 3 --local-ranks-filter 0` silences them completely. Ctrl-C reaches
torchrun, which forwards the signal to every rank and kills what is still running after `--shutdown-timeout` seconds
(30 by default, too short for a step plus a checkpoint write; `make training-ddp` passes 1800). torchrun then exits 1
with a `SignalException` traceback even when every rank stopped cleanly; rank 0's summary and checkpoint are the
confirmation.
`RunLogger` opens the terminal dashboard of `training/ui/` for the run (the live display on a TTY, the one-line
fallback when piped or with `TRAINING_DASHBOARD=0`, `<out_dir>/<run_name>/train.log` in both cases, and the report as
`train_report.json` next to it); it swaps the `training`
handler out for the duration, so nothing prints twice.

Exit codes: 0 finished, 1 failed (traceback logged), 3 another run holds the lock (the message names its pid and
start time; `data_preparation/lib/build/lock.py`), 130 interrupted.
Supervised DDP workers report fatal errors directly to stderr and exit immediately before run-level cleanup;
torchrun terminates their peers. Success, cooperative stops and library/single-device callers retain cleanup.
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python training/train.py` from the repo root

from training.cli import (
    configure_console_logging,
    finish_training_command,
    get_launch_rank,
    run_with_error_handling,
    select_fatal_error_handler,
    stop_on_interrupt,
)
from training.run import train
from training.settings import parse_settings


def main(argv: list[str] | None = None) -> int:
    """
    Parse `argv` (default: the command line), run, print the report (the main rank; module docstring); returns the
    exit code.
    """

    # set up logging and parse settings
    started_at = time.time()
    rank = get_launch_rank()
    configure_console_logging(rank=rank)
    settings = parse_settings(argv)
    on_fatal_error = select_fatal_error_handler(settings.backend)

    # run training with interruption and failure handling
    with stop_on_interrupt() as should_stop:
        run = partial(train, settings, should_stop=should_stop, started_at=started_at, on_fatal_error=on_fatal_error)
        result = run_with_error_handling(run, on_fatal_error)

    # report the result after restoring signal handlers
    return finish_training_command(result, rank)


if __name__ == "__main__":
    sys.exit(main())
