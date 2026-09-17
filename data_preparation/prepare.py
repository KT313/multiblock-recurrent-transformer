# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Entry point for dataset preparation.

    python data_preparation/prepare.py prepare  --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
                                                [--sources S ...] [--steps tokenizer download build] [--reopen S ...] [--yes] [--dry_run]
                                                [--allow_foreign_raw]
                                                [--num_workers N] [--pass_workers N] [--tokenizer_threads N] [--max_parallel_downloads N]
                                                [--hf_token T] [--cache_dir DIR]
                                                [--debug [SECONDS]] [--debug-file PATH]
    python data_preparation/prepare.py download --dataset_config config/datasets/<name>.yaml [same options; --steps tokenizer download]
    python data_preparation/prepare.py status   --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
    python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml   # Markdown to stdout
    python data_preparation/prepare.py tiny     # = prepare --dataset_config config/datasets/tiny.yaml

prepare materialises a dataset config: tokenizer, repair, (download + build) rounds, status table
(lib/build/runner.py). download is prepare without the build step (tokenizer + raw shards), complete when every
source has its raw rows (no processed folder is expected after it; `make prepare` builds them later).
Stale or outdated raw folders (deleted and downloaded again) and processed folders whose
manifest cannot be parsed (deleted and rebuilt) go only after a confirmation on the terminal; --yes answers it,
and without a terminal the command prints the list and exits 2 with nothing changed. A raw folder downloaded under
another dataset config (raw folders are shared by source name) goes only with --allow_foreign_raw on top. --reopen
clears the exhausted flag of the named sources first (a loader that yielded fewer rows than asked is latched
exhausted; say so when it has more rows now). status prints what the repair step would do plus the status table
and exits 0 iff the dataset is complete. describe renders the config
as Markdown (docs/data_mixture.md is generated with it). --cache_dir relocates the HuggingFace caches. prepare
turns the tokenizer's thread pool on with up to TOKENIZER_POOL_THREADS threads (TOKENIZERS_PARALLELISM=true and
RAYON_NUM_THREADS=8 unless set in the environment): this process never forks after loading the tokenizer, and
downloads tokenize every row on that one pool, whatever their number. --tokenizer_threads above 8 adds separate
tokenizer processes of up to 8 threads each (lib/stages/tokenizer_pool.py: one encode call stops scaling past
that, processes add up), which every download job's token worker feeds.

Exit codes: 0 ok, 1 failure (a broken config or a repair that cannot decide safely is logged as one line,
anything else with its traceback; a failed source is a failed build), 2 an unconfirmed
repair, 3 another data preparation is still running (lib/build/lock.py; the message names its pid and start
time), 130 interrupted (Ctrl-C or SIGTERM: every running step stops at its next shard, everything published is
kept; a second Ctrl-C ends the process without waiting for the running transfer). On a terminal the run shows the
live dashboard of lib/ui/dashboard.py; the log lines it kept (warnings, the tables) and the final status table are
printed once it closed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

if __name__ == "__main__":  # allow `python data_preparation/prepare.py` without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.cli.arguments import build_parser  # noqa: E402
from data_preparation.cli.commands import (  # noqa: E402
    DOWNLOAD_STEPS,
    check_completion_for_full_run,
    check_dataset_complete,
    check_download_complete,
    configure_preparation_environment,
    open_preparation_dashboard,
    prepare_requested_steps,
)
from data_preparation.cli.runtime import configure_interrupt_handling, handle_command_errors  # noqa: E402
from data_preparation.lib.layout import DatasetLayout  # noqa: E402
from data_preparation.lib.download_profile import profile_downloads  # noqa: E402
from data_preparation.lib.download_debug import log_download_debug  # noqa: E402
from data_preparation.lib.build.planner import DatasetReport  # noqa: E402
from data_preparation.lib.build.runner import STEPS  # noqa: E402
from data_preparation.lib.log import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

def run_prepare(args: argparse.Namespace) -> None:
    _materialise(args, STEPS, partial(check_dataset_complete, log=log))


def run_download(args: argparse.Namespace) -> None:
    _materialise(args, DOWNLOAD_STEPS, partial(check_download_complete, log=log))


def _materialise(
    args: argparse.Namespace,
    all_steps: tuple[str, ...],
    completeness: Callable[[argparse.Namespace, DatasetReport, DatasetLayout], None],
) -> None:
    """
    Run prepare with the command's steps (all_steps, or the --steps subset of them) and, unless the run was
    partial (--dry_run, --sources or a strict --steps subset: the tree is not expected to be complete then),
    apply the command's completeness check to the report (a RuntimeError there is exit 1).
    """

    configure_preparation_environment(args)
    layout = DatasetLayout(args.dataset_dir)

    steps = all_steps if args.steps is None else tuple(args.steps)
    debug_interval = args.debug if args.debug is not None else (5.0 if args.debug_file is not None else None)
    with (
        profile_downloads(dry_run=args.dry_run, debug=debug_interval) as profile,
        open_preparation_dashboard(args, layout),
        log_download_debug(profile, debug_interval, debug_file=args.debug_file),
    ):
        report = prepare_requested_steps(args, steps, layout, log=log)
        check_completion_for_full_run(args, steps, all_steps, report, layout, completeness)


def main(argv: list[str] | None = None) -> None:
    """
    Parse argv (default sys.argv), dispatch, and map failures to exit codes (module docstring).
    """

    # setup and parse args
    configure_logging()
    args = build_parser(description=__doc__, log=log, on_prepare=run_prepare, on_download=run_download).parse_args(argv)
    configure_interrupt_handling()

    # run the selected command
    with handle_command_errors(args.command, log=log):
        args.run(args)


if __name__ == "__main__":
    main()
