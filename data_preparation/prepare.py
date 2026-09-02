# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Entry point for dataset preparation.

    python data_preparation/prepare.py prepare  --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
                                                [--sources S ...] [--steps tokenizer download build] [--yes] [--dry_run]
                                                [--num_workers N] [--pass_workers N] [--max_parallel_downloads N]
                                                [--hf_token T] [--cache_dir DIR]
    python data_preparation/prepare.py status   --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
    python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml   # Markdown to stdout
    python data_preparation/prepare.py tiny     # = prepare --dataset_config config/datasets/tiny.yaml

``prepare`` materialises a dataset config: tokenizer → repair → (download + build) rounds → status table
(``lib/build/runner.py``). Stale or outdated raw folders are deleted and downloaded again only after a confirmation
on the terminal; ``--yes`` answers it, and without a terminal the command prints the list and exits 2 — nothing is
changed. ``status`` prints what the repair step would do and the status table and exits 0 iff the dataset is
complete. ``describe`` renders the config as Markdown (``docs/data_mixture.md`` is generated with it). ``--cache_dir``
relocates the HuggingFace caches.

Exit codes: 0 ok, 1 failure (logged with its traceback; a failed source is a failed build), 2 an unconfirmed raw
deletion, 130 interrupted (Ctrl-C or SIGTERM: every running step stops at its next shard, everything published is
kept). On a terminal the run shows the live dashboard of ``lib/ui/dashboard.py``; the log lines it kept (warnings,
the tables) and the final status table are printed once it closed.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

if __name__ == "__main__":  # allow `python data_preparation/prepare.py` without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.dataset_config import load_dataset_config  # noqa: E402
from data_preparation.layout import DatasetLayout  # noqa: E402
from data_preparation.lib.abort import BuildAborted  # noqa: E402
from data_preparation.lib.build.describe import describe, leading_comment  # noqa: E402
from data_preparation.lib.build.repair import ConfirmationRequired  # noqa: E402
from data_preparation.lib.build.runner import (  # noqa: E402
    DEFAULT_MAX_PARALLEL_DOWNLOADS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PASS_WORKERS,
    STEPS,
    prepare,
    status,
)
from data_preparation.lib.storage.parquet import configure_hf_cache  # noqa: E402
from data_preparation.lib.log import ROOT_LOGGER_NAME, configure_logging, get_logger  # noqa: E402
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, Dashboard  # noqa: E402

log = get_logger(__name__)

TINY_DATASET_CONFIG = Path("config/datasets/tiny.yaml")
DEFAULT_DATASET_DIR = Path("dataset")

EXIT_CONFIRMATION_REQUIRED = 2
EXIT_INTERRUPTED = 130


# --- command line ------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    prepare_cmd = subparsers.add_parser("prepare", help="materialise a dataset config (missing parts only)")
    _add_dataset_options(prepare_cmd, config_default=None)
    _add_prepare_options(prepare_cmd)
    prepare_cmd.set_defaults(run=run_prepare)

    status_cmd = subparsers.add_parser("status", help="print the status table; exit 0 iff the dataset is complete")
    _add_dataset_options(status_cmd, config_default=None)
    status_cmd.set_defaults(run=run_status)

    describe_cmd = subparsers.add_parser("describe", help="print the dataset config as a Markdown document")
    describe_cmd.add_argument("--dataset_config", type=Path, required=True, help="dataset config YAML")
    describe_cmd.set_defaults(run=run_describe)

    tiny_cmd = subparsers.add_parser("tiny", help=f"prepare {TINY_DATASET_CONFIG} (alias of prepare)")
    _add_dataset_options(tiny_cmd, config_default=TINY_DATASET_CONFIG)
    _add_prepare_options(tiny_cmd)
    tiny_cmd.set_defaults(run=run_prepare)

    return parser


def _add_dataset_options(sub: argparse.ArgumentParser, *, config_default: Path | None) -> None:
    """``--dataset_config`` (required unless ``config_default`` is given), ``--dataset_dir``, ``--cache_dir``."""
    sub.add_argument("--dataset_config", type=Path, default=config_default, required=config_default is None, help="dataset config YAML")
    sub.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR, help="root of all prepared data")
    sub.add_argument("--cache_dir", type=Path, default=None, help="HuggingFace cache directory (default: HF defaults)")


def _add_prepare_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--sources", nargs="+", default=None, metavar="NAME", help="only these sources")
    sub.add_argument("--steps", nargs="+", default=None, choices=STEPS, metavar="STEP", help=f"only these steps of {STEPS}")
    sub.add_argument("--yes", "-y", action="store_true", help="delete stale / outdated raw folders without asking")
    sub.add_argument("--dry_run", action="store_true", help="print what would be repaired and downloaded, write nothing")
    sub.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="sources built at a time (build threads)")
    sub.add_argument("--pass_workers", type=int, default=DEFAULT_PASS_WORKERS, help="worker processes of EACH build's optional cleaning passes (decontamination / minhash; 1 = in-process)")
    sub.add_argument("--max_parallel_downloads", type=int, default=DEFAULT_MAX_PARALLEL_DOWNLOADS, help="sources downloading at a time")
    sub.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated sources")


# --- commands ----------------------------------------------------------------------------------------------------------


def run_prepare(args: argparse.Namespace) -> None:
    configure_hf_cache(args.cache_dir)
    layout = DatasetLayout(args.dataset_dir)
    log_file = None if args.dry_run else layout.root / BUILD_LOG_NAME  # a dry run writes nothing
    with Dashboard(title=f"prepare {args.dataset_config}") as dashboard, dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=log_file):
        log.info("preparing dataset config %s under %s", args.dataset_config, layout.root)
        report = prepare(
            args.dataset_config,
            args.dataset_dir,
            num_workers=args.num_workers,
            pass_workers=args.pass_workers,
            max_parallel_downloads=args.max_parallel_downloads,
            assume_yes=args.yes,
            dry_run=args.dry_run,
            steps=STEPS if args.steps is None else args.steps,
            sources=args.sources,
            hf_token=args.hf_token,
        )
        partial = args.dry_run or args.sources is not None or args.steps is not None
        if partial:
            return  # the dataset is not expected to be complete after a partial run
        if not report.complete:
            raise RuntimeError(f"dataset {args.dataset_config} still incomplete after preparing: {report.missing()}")
        log.info("done: %s", layout.root, extra={"keep": True})


def run_status(args: argparse.Namespace) -> None:
    configure_hf_cache(args.cache_dir)
    report = status(args.dataset_config, args.dataset_dir)
    print(report.describe())
    if not report.complete:
        log.warning("missing: %s", ", ".join(report.missing()))
        raise SystemExit(1)


def run_describe(args: argparse.Namespace) -> None:
    cfg = load_dataset_config(args.dataset_config)
    sys.stdout.write(describe(cfg, args.dataset_config, notes=leading_comment(args.dataset_config)))


def _interrupt_on_sigterm(signum: int, frame: FrameType | None) -> None:
    """``kill`` / a byte-capped run (``tools/capped_download.sh``) end like Ctrl-C: stop at the next shard, exit 130."""
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``), dispatch, and map failures to exit codes (module docstring)."""
    configure_logging()
    args = build_parser().parse_args(argv)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        args.run(args)
    except SystemExit:
        raise
    except (KeyboardInterrupt, BuildAborted):
        log.warning("%s interrupted; everything published so far is kept, rerun to resume", args.command)
        raise SystemExit(EXIT_INTERRUPTED) from None
    except ConfirmationRequired as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_CONFIRMATION_REQUIRED) from None
    except Exception:
        log.exception("%s failed", args.command)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
