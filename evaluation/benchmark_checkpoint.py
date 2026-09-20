# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Immediately run the training config's benchmarks using a saved checkpoint, on one GPU or under torchrun.

    uv run python evaluation/benchmark_checkpoint.py --config config/my_run.yaml
    uv run torchrun --standalone --nproc_per_node=8 evaluation/benchmark_checkpoint.py --config config/my_run.yaml

The config selects the backend. All normal training config overrides are accepted, including --benchmark_limit.
Defaults to the newest regular checkpoint, even if resume is false or resume_checkpoint_path selects an older one.
Use --checkpoint for an explicit file. Results go to a new <run>/benchmark_checks/step-XXXXXXXX-*/ directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.cli.benchmark_checkpoint import parse_benchmark_settings, run_checkpoint_benchmark_check
from training.cli import configure_console_logging, get_launch_rank, select_fatal_error_handler, stop_on_interrupt
from training.failure import fatal_errors


def main(argv: list[str] | None = None) -> int:
    # load the same settings and launcher failure policy as training
    configure_console_logging(rank=get_launch_rank())
    settings, options = parse_benchmark_settings(argv)
    on_fatal_error = select_fatal_error_handler(settings.backend)

    # start benchmarks immediately, with cooperative interruption between inference jobs
    try:
        with (
            stop_on_interrupt(message="stopping after the current benchmark work") as should_stop,
            fatal_errors(on_fatal_error),
        ):
            return run_checkpoint_benchmark_check(settings, options, should_stop=should_stop, on_fatal_error=on_fatal_error)
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.getLogger("training.benchmark_checkpoint").exception("benchmark check failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
