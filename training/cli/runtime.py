# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Launch policy, failure reporting, and result handling for the training CLI."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.lock import RunLocked
from training.failure import FatalHandler, exit_failed_worker, handle_fatal_error
from training.logger import TrainingReport
from training.ui.common import TRAINING_LOGGER_NAME

EXIT_INTERRUPTED = 130
EXIT_ALREADY_RUNNING = 3

log = logging.getLogger(f"{TRAINING_LOGGER_NAME}.train")


def get_launch_rank() -> int:
    """
    This process's rank under torchrun (`RANK` in the environment), 0 for a plain launch.
    """

    return int(os.environ.get("RANK", "0"))


def select_fatal_error_handler(backend: str) -> FatalHandler | None:
    """Opt into immediate fatal exit only for supervised DDP CLI workers."""

    return exit_failed_worker if backend == "ddp" and "TORCHELASTIC_RUN_ID" in os.environ else None


def run_with_error_handling(run: Callable[[], TrainingReport], on_fatal_error: FatalHandler | None) -> TrainingReport | int:
    """Run training and return either its report or the existing CLI failure code."""

    try:
        return run()
    except RunLocked as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ALREADY_RUNNING
    except (KeyboardInterrupt, BuildAborted):
        log.warning("training interrupted; checkpoints and published dataset shards are kept, rerun to resume")
        return EXIT_INTERRUPTED
    except Exception as error:
        handle_fatal_error(on_fatal_error, error)  # failures before the run's cleanup scopes exist
        log.exception("training failed")
        return 1


def finish_training_command(result: TrainingReport | int, rank: int) -> int:
    """Return a failure code, or print the main rank's report and return its completion code."""

    if isinstance(result, int):
        return result
    if rank == 0:
        print(result.summary())
    return EXIT_INTERRUPTED if result.stopped else 0
