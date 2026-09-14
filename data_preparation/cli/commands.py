# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset preparation CLI commands and their execution helpers."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from data_preparation.lib.dataset_config import load_dataset_config
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.build.describe import describe, leading_comment
from data_preparation.lib.build.planner import DatasetReport
from data_preparation.lib.build.runner import prepare, status
from data_preparation.lib.log import ROOT_LOGGER_NAME
from data_preparation.lib.sources.hf_cache import configure_hf_cache
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, DataDashboard

DOWNLOAD_STEPS = ("tokenizer", "download")  # what the download command runs: STEPS without the build
TOKENIZER_POOL_THREADS = 8  # threads of the tokenizer's Rust pool (one per process, shared by every download job)


def check_dataset_complete(args: argparse.Namespace, report: DatasetReport, layout: DatasetLayout, *, log: logging.Logger) -> None:
    if not report.complete:
        raise RuntimeError(f"dataset {args.dataset_config} still incomplete after preparing: {report.missing()}")
    log.info("done: %s", layout.root, extra={"keep": True})


def check_download_complete(args: argparse.Namespace, report: DatasetReport, layout: DatasetLayout, *, log: logging.Logger) -> None:
    """
    The download command's verdict is about the raw side only: no processed folder exists after it.
    """

    missing = report.missing_raw_rows()
    if missing:
        raise RuntimeError(f"download incomplete: {', '.join(missing)} (the status table above says why)")
    log.info("download complete: %s", layout.root, extra={"keep": True})


def configure_preparation_environment(args: argparse.Namespace) -> None:
    """Configure tokenizer threads and HF caches before acquiring a tokenizer."""

    # The tokenizer's Rust thread pool: on here, off by library default (`_auto_tokenizer` in lib/stages/download.py).
    # The guard exists for a training run that prepares data in-process and then forks DataLoader workers; this
    # process never forks after the tokenizer is loaded (the cleaning passes use spawn pools), and a download
    # batch tokenizes several times faster on several cores. The pool is one per process, shared by every download
    # job, and sized TOKENIZER_POOL_THREADS (Rayon's default is every core: measured, past 8 threads a 256-row
    # batch barely gets faster while the CPU time keeps growing). Explicit values in the environment win.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(TOKENIZER_POOL_THREADS))
    configure_hf_cache(args.cache_dir, create=not args.dry_run)


@contextmanager
def open_preparation_dashboard(args: argparse.Namespace, layout: DatasetLayout) -> Iterator[None]:
    """Keep the dashboard and its log attachment open for the preparation run."""

    log_file = None if args.dry_run else layout.root / BUILD_LOG_NAME  # a dry run writes nothing
    with (
        DataDashboard(title=f"{args.command} {args.dataset_config}") as dashboard,
        dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=log_file),
    ):
        yield


def prepare_requested_steps(
    args: argparse.Namespace, steps: tuple[str, ...], layout: DatasetLayout, *, log: logging.Logger,
) -> DatasetReport:
    """Run the selected preparation steps with the CLI's resource and repair options."""

    log.info("preparing dataset config %s under %s", args.dataset_config, layout.root)
    return prepare(
        args.dataset_config,
        args.dataset_dir,
        num_workers=args.num_workers,
        pass_workers=args.pass_workers,
        max_parallel_downloads=args.max_parallel_downloads,
        assume_yes=args.yes,
        dry_run=args.dry_run,
        allow_foreign_raw=args.allow_foreign_raw,
        steps=steps,
        sources=args.sources,
        reopen=args.reopen,
        hf_token=args.hf_token,
    )


def check_completion_for_full_run(
    args: argparse.Namespace,
    steps: tuple[str, ...],
    all_steps: tuple[str, ...],
    report: DatasetReport,
    layout: DatasetLayout,
    completeness: Callable[[argparse.Namespace, DatasetReport, DatasetLayout], None],
) -> None:
    """Apply the command's completion check only when it requested the entire dataset."""

    partial = args.dry_run or args.sources is not None or set(steps) != set(all_steps)
    if partial:
        return  # the tree is not expected to be complete after a partial run
    completeness(args, report, layout)


def run_status(args: argparse.Namespace, *, log: logging.Logger) -> None:
    configure_hf_cache(args.cache_dir, create=False)
    report = status(args.dataset_config, args.dataset_dir)
    print(report.describe())
    if not report.complete:
        log.warning("missing: %s", ", ".join(report.missing()))
        raise SystemExit(1)


def run_describe(args: argparse.Namespace) -> None:
    dataset_config = load_dataset_config(args.dataset_config)
    sys.stdout.write(describe(dataset_config, args.dataset_config, notes=leading_comment(args.dataset_config)))
