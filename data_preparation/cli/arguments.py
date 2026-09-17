# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Parser construction and argument definitions for the dataset preparation CLI."""

from __future__ import annotations

import argparse
import math
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

from data_preparation.cli.commands import DOWNLOAD_STEPS, run_describe, run_status
from data_preparation.lib.build.runner import (
    DEFAULT_MAX_PARALLEL_DOWNLOADS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PASS_WORKERS,
    STEPS,
)

DEFAULT_DATASET_DIR = Path("dataset")
TINY_DATASET_CONFIG = Path("config/datasets/tiny.yaml")


def build_parser(
    *,
    description: str | None,
    log: logging.Logger,
    on_prepare: Callable[[argparse.Namespace], None],
    on_download: Callable[[argparse.Namespace], None],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    prepare_cmd = subparsers.add_parser("prepare", help="materialise a dataset config (missing parts only)")
    add_dataset_options(prepare_cmd, config_default=None)
    add_prepare_options(prepare_cmd)
    prepare_cmd.set_defaults(run=on_prepare)

    download_cmd = subparsers.add_parser("download", help="download only (tokenizer + raw shards): prepare without the build step")
    add_dataset_options(download_cmd, config_default=None)
    add_prepare_options(download_cmd, steps=DOWNLOAD_STEPS)
    download_cmd.set_defaults(run=on_download)

    status_cmd = subparsers.add_parser("status", help="print the status table; exit 0 iff the dataset is complete")
    add_dataset_options(status_cmd, config_default=None)
    status_cmd.set_defaults(run=partial(run_status, log=log))

    describe_cmd = subparsers.add_parser("describe", help="print the dataset config as a Markdown document")
    add_dataset_config_argument(describe_cmd)
    describe_cmd.set_defaults(run=run_describe)

    tiny_cmd = subparsers.add_parser("tiny", help=f"prepare {TINY_DATASET_CONFIG} (alias of prepare)")
    add_dataset_options(tiny_cmd, config_default=TINY_DATASET_CONFIG)
    add_prepare_options(tiny_cmd)
    tiny_cmd.set_defaults(run=on_prepare)

    return parser


def add_dataset_config_argument(sub: argparse.ArgumentParser, *, config_default: Path | None = None) -> None:
    """Require a dataset config unless the command supplies a default."""

    sub.add_argument(
        "--dataset_config", type=Path, default=config_default, required=config_default is None, help="dataset config YAML",
    )


def add_dataset_options(sub: argparse.ArgumentParser, *, config_default: Path | None) -> None:
    """
    --dataset_config (required unless config_default is given), --dataset_dir, --cache_dir.
    """

    add_dataset_config_argument(sub, config_default=config_default)
    sub.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR, help="root of all prepared data")
    sub.add_argument("--cache_dir", type=Path, default=None, help="HuggingFace cache directory (default: HF defaults)")


def debug_interval(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("debug interval must be a positive finite number of seconds") from error
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("debug interval must be a positive finite number of seconds")
    return interval


def add_prepare_options(sub: argparse.ArgumentParser, *, steps: tuple[str, ...] = STEPS) -> None:
    """
    The options prepare and download share; steps are the ones the command runs (all of them by default).
    """

    sub.add_argument("--debug", nargs="?", const=5.0, default=None, type=debug_interval, metavar="SECONDS",
                     help="log pipeline sections, ongoing waits and process CPU every SECONDS (default: 5); includes worker processes")
    sub.add_argument("--sources", nargs="+", default=None, metavar="NAME", help="only these sources")
    sub.add_argument("--steps", nargs="+", default=None, choices=steps, metavar="STEP", help=f"only these steps of {steps}")
    sub.add_argument("--reopen", nargs="+", default=None, metavar="NAME", help="clear the exhausted flag of these sources before planning (their loader has more rows now)")
    sub.add_argument("--yes", "-y", action="store_true", help="answer the repair confirmation (stale / outdated raw folders, unparsable processed manifests) without asking")
    sub.add_argument("--dry_run", action="store_true", help="print what would be repaired and downloaded, write nothing")
    sub.add_argument("--allow_foreign_raw", action="store_true", help="let the repair step delete stale / outdated raw folders that another dataset config downloaded (they are shared by source name)")
    sub.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="sources built at a time (build threads)")
    sub.add_argument("--pass_workers", type=int, default=DEFAULT_PASS_WORKERS, help="worker processes of EACH build's optional cleaning passes (decontamination / minhash; 1 = in-process)")
    sub.add_argument("--max_parallel_downloads", type=int, default=DEFAULT_MAX_PARALLEL_DOWNLOADS, help="sources downloading at a time")
    sub.add_argument("--download_prefetch_mb", type=int, default=None, help="remote read-ahead block size in MiB per download (up to two blocks buffered); 0 disables; defaults to dataset config, otherwise 0")
    sub.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated sources")
