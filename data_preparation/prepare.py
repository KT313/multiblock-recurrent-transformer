# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Entry point for dataset preparation.

    python data_preparation/prepare.py build  --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
                                              [--sources S ...] [--steps tokenizer download process validation instruct_mixtures]
                                              [--num_workers N] [--hf_token T] [--dry_run]
    python data_preparation/prepare.py status --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
    python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml   # Markdown to stdout
    python data_preparation/prepare.py tiny   # = build --dataset_config config/datasets/tiny.yaml

``build`` materialises a dataset config (tokenizer -> pretrain sources -> validation sources -> instruct mixtures; see
``lib/build/runner.py``), ``status`` prints the plan and exits 0 iff the dataset is complete, ``describe`` renders
the config as Markdown (``docs/data_mixture.md`` is generated with it). ``--cache_dir`` relocates the HuggingFace
caches. Any failure logs the exception and exits 1.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __name__ == "__main__":  # allow `python data_preparation/prepare.py` without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.lib.build import DEFAULT_MAX_PARALLEL_DOWNLOADS, STEPS, build, describe, leading_comment, status  # noqa: E402
from data_preparation.lib.storage.parquet import configure_hf_cache  # noqa: E402
from data_preparation.lib.schema.dataset_config import load_dataset_config  # noqa: E402
from data_preparation.lib.schema.layout import DatasetLayout  # noqa: E402
from data_preparation.lib.log import ROOT_LOGGER_NAME, configure_logging, get_logger  # noqa: E402
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, Dashboard  # noqa: E402

log = get_logger(__name__)

TINY_DATASET_CONFIG = Path("config/datasets/tiny.yaml")
DEFAULT_DATASET_DIR = Path("dataset")


# --- command line ------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    build_cmd = subparsers.add_parser("build", help="materialise a dataset config (missing parts only)")
    _add_dataset_options(build_cmd, config_default=None)
    _add_build_options(build_cmd)
    build_cmd.set_defaults(run=run_build)

    status_cmd = subparsers.add_parser("status", help="print the plan; exit 0 iff the dataset is complete")
    _add_dataset_options(status_cmd, config_default=None)
    status_cmd.set_defaults(run=run_status)

    describe_cmd = subparsers.add_parser("describe", help="print the dataset config as a Markdown document")
    describe_cmd.add_argument("--dataset_config", type=Path, required=True, help="dataset config YAML")
    describe_cmd.set_defaults(run=run_describe)

    tiny_cmd = subparsers.add_parser("tiny", help=f"build {TINY_DATASET_CONFIG} (alias of build)")
    _add_dataset_options(tiny_cmd, config_default=TINY_DATASET_CONFIG)
    _add_build_options(tiny_cmd)
    tiny_cmd.set_defaults(run=run_build)

    return parser


def _add_dataset_options(sub: argparse.ArgumentParser, *, config_default: Path | None) -> None:
    """``--dataset_config`` (required unless ``config_default`` is given), ``--dataset_dir``, ``--cache_dir``."""
    sub.add_argument("--dataset_config", type=Path, default=config_default, required=config_default is None, help="dataset config YAML")
    sub.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR, help="root of all prepared data")
    sub.add_argument("--cache_dir", type=Path, default=None, help="HuggingFace cache directory (default: HF defaults)")


def _add_build_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--sources", nargs="+", default=None, metavar="NAME", help="only these sources / instruct mixtures")
    sub.add_argument("--steps", nargs="+", default=None, choices=STEPS, metavar="STEP", help=f"only these steps of {STEPS}")
    sub.add_argument("--num_workers", type=int, default=1, help="worker processes for processing stages (and sources processed at a time)")
    sub.add_argument("--max_parallel_downloads", type=int, default=DEFAULT_MAX_PARALLEL_DOWNLOADS, help="sources downloading at a time")
    sub.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated sources")
    sub.add_argument("--dry_run", action="store_true", help="print the plan, write nothing")


# --- commands ----------------------------------------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> None:
    configure_hf_cache(args.cache_dir)
    cfg = load_dataset_config(args.dataset_config)
    layout = DatasetLayout(args.dataset_dir)
    log_file = None if args.dry_run else layout.root / BUILD_LOG_NAME  # a dry run writes nothing
    with Dashboard() as dashboard, dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=log_file):
        log.info("building dataset config %s (%s) under %s", cfg.name, args.dataset_config, layout.root)
        result = build(
            cfg,
            layout,
            sources=args.sources,
            steps=None if args.steps is None else set(args.steps),
            num_workers=args.num_workers,
            hf_token=args.hf_token,
            dry_run=args.dry_run,
            max_parallel_downloads=args.max_parallel_downloads,
        )
        partial_build = args.dry_run or args.sources is not None or args.steps is not None
        if partial_build:
            return  # the dataset is not expected to be complete after a partial build
        if not result.complete:
            log.error("dataset %s still incomplete after the build: %s", cfg.name, result.missing())
            raise RuntimeError(f"dataset {cfg.name} still incomplete after the build: {result.missing()}")
        log.info("done: %s", layout.root)


def run_status(args: argparse.Namespace) -> None:
    configure_hf_cache(args.cache_dir)
    cfg = load_dataset_config(args.dataset_config)
    result = status(cfg, DatasetLayout(args.dataset_dir))
    print(result.summary())
    if not result.complete:
        for line in result.missing():
            log.warning("missing: %s", line)
        raise SystemExit(1)


def run_describe(args: argparse.Namespace) -> None:
    cfg = load_dataset_config(args.dataset_config)
    sys.stdout.write(describe(cfg, args.dataset_config, notes=leading_comment(args.dataset_config)))


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``), dispatch; any exception is logged and turned into exit code 1."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except SystemExit:
        raise
    except Exception:
        log.exception("%s failed", args.command)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
