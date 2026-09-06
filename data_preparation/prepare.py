# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Entry point for dataset preparation.

    python data_preparation/prepare.py prepare  --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
                                                [--sources S ...] [--steps tokenizer download build] [--reopen S ...] [--yes] [--dry_run]
                                                [--allow_foreign_raw] [--num_workers N] [--pass_workers N] [--max_parallel_downloads N]
                                                [--hf_token T] [--cache_dir DIR]
    python data_preparation/prepare.py status   --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
    python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml   # Markdown to stdout
    python data_preparation/prepare.py tiny     # = prepare --dataset_config config/datasets/tiny.yaml

prepare materialises a dataset config: tokenizer, repair, (download + build) rounds, status table
(lib/build/runner.py). Stale or outdated raw folders (deleted and downloaded again) and processed folders whose
manifest cannot be parsed (deleted and rebuilt) go only after a confirmation on the terminal; --yes answers it,
and without a terminal the command prints the list and exits 2 with nothing changed. A raw folder downloaded under
another dataset config (raw folders are shared by source name) goes only with --allow_foreign_raw on top. --reopen
clears the exhausted flag of the named sources first (a loader that yielded fewer rows than asked is latched
exhausted; say so when it has more rows now). status prints what the repair step would do plus the status table
and exits 0 iff the dataset is complete. describe renders the config
as Markdown (docs/data_mixture.md is generated with it). --cache_dir relocates the HuggingFace caches. prepare
turns the tokenizer's thread pool on with TOKENIZER_POOL_THREADS threads (TOKENIZERS_PARALLELISM=true and
RAYON_NUM_THREADS=8 unless set in the environment): this process never forks after loading the tokenizer, and
downloads tokenize every row on that one pool, whatever their number.

Exit codes: 0 ok, 1 failure (logged with its traceback; a failed source is a failed build), 2 an unconfirmed
repair, 3 another data preparation is still running (lib/build/lock.py; the message names its pid and start
time), 130 interrupted (Ctrl-C or SIGTERM: every running step stops at its next shard, everything published is
kept; a second Ctrl-C ends the process without waiting for the running transfer). On a terminal the run shows the
live dashboard of lib/ui/dashboard.py; the log lines it kept (warnings, the tables) and the final status table are
printed once it closed.
"""

from __future__ import annotations

import argparse
import logging
import os
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
from data_preparation.lib.build.lock import RunLocked  # noqa: E402
from data_preparation.lib.build.repair import ConfirmationRequired  # noqa: E402
from data_preparation.lib.build.runner import (  # noqa: E402
    DEFAULT_MAX_PARALLEL_DOWNLOADS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PASS_WORKERS,
    STEPS,
    prepare,
    status,
)
from data_preparation.lib.log import ROOT_LOGGER_NAME, configure_logging, get_logger  # noqa: E402
from data_preparation.lib.sources.hf_cache import configure_hf_cache  # noqa: E402
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, DataDashboard  # noqa: E402

log = get_logger(__name__)

TINY_DATASET_CONFIG = Path("config/datasets/tiny.yaml")
TOKENIZER_POOL_THREADS = 8  # threads of the tokenizer's Rust pool (one per process, shared by every download job)
DEFAULT_DATASET_DIR = Path("dataset")

EXIT_CONFIRMATION_REQUIRED = 2
EXIT_ALREADY_RUNNING = 3
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
    """
    --dataset_config (required unless config_default is given), --dataset_dir, --cache_dir.
    """

    sub.add_argument("--dataset_config", type=Path, default=config_default, required=config_default is None, help="dataset config YAML")
    sub.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR, help="root of all prepared data")
    sub.add_argument("--cache_dir", type=Path, default=None, help="HuggingFace cache directory (default: HF defaults)")


def _add_prepare_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--sources", nargs="+", default=None, metavar="NAME", help="only these sources")
    sub.add_argument("--steps", nargs="+", default=None, choices=STEPS, metavar="STEP", help=f"only these steps of {STEPS}")
    sub.add_argument("--reopen", nargs="+", default=None, metavar="NAME", help="clear the exhausted flag of these sources before planning (their loader has more rows now)")
    sub.add_argument("--yes", "-y", action="store_true", help="answer the repair confirmation (stale / outdated raw folders, unparsable processed manifests) without asking")
    sub.add_argument("--dry_run", action="store_true", help="print what would be repaired and downloaded, write nothing")
    sub.add_argument("--allow_foreign_raw", action="store_true", help="let the repair step delete stale / outdated raw folders that another dataset config downloaded (they are shared by source name)")
    sub.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="sources built at a time (build threads)")
    sub.add_argument("--pass_workers", type=int, default=DEFAULT_PASS_WORKERS, help="worker processes of EACH build's optional cleaning passes (decontamination / minhash; 1 = in-process)")
    sub.add_argument("--max_parallel_downloads", type=int, default=DEFAULT_MAX_PARALLEL_DOWNLOADS, help="sources downloading at a time")
    sub.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated sources")


# --- commands ----------------------------------------------------------------------------------------------------------


def run_prepare(args: argparse.Namespace) -> None:
    # The tokenizer's Rust thread pool: on here, off by library default (`_auto_tokenizer` in lib/stages/download.py).
    # The guard exists for a training run that prepares data in-process and then forks DataLoader workers; this
    # process never forks after the tokenizer is loaded (the cleaning passes use spawn pools), and a download
    # batch tokenizes several times faster on several cores. The pool is one per process, shared by every download
    # job, and sized TOKENIZER_POOL_THREADS (Rayon's default is every core: measured, past 8 threads a 256-row
    # batch barely gets faster while the CPU time keeps growing). Explicit values in the environment win.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(TOKENIZER_POOL_THREADS))
    configure_hf_cache(args.cache_dir)
    layout = DatasetLayout(args.dataset_dir)
    log_file = None if args.dry_run else layout.root / BUILD_LOG_NAME  # a dry run writes nothing

    with DataDashboard(title=f"prepare {args.dataset_config}") as dashboard, dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=log_file):
        log.info("preparing dataset config %s under %s", args.dataset_config, layout.root)
        report = prepare(
            args.dataset_config,
            args.dataset_dir,
            num_workers=args.num_workers,
            pass_workers=args.pass_workers,
            max_parallel_downloads=args.max_parallel_downloads,
            assume_yes=args.yes,
            dry_run=args.dry_run,
            allow_foreign_raw=args.allow_foreign_raw,
            steps=STEPS if args.steps is None else args.steps,
            sources=args.sources,
            reopen=args.reopen,
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
    dataset_config = load_dataset_config(args.dataset_config)
    sys.stdout.write(describe(dataset_config, args.dataset_config, notes=leading_comment(args.dataset_config)))


def _interrupt_on_sigterm(signum: int, frame: FrameType | None) -> None:
    """
    kill ends like Ctrl-C: stop at the next shard, exit 130.
    """

    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    """
    Parse argv (default sys.argv), dispatch, and map failures to exit codes (module docstring).
    """

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
    except RunLocked as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_ALREADY_RUNNING) from None
    except Exception:
        log.exception("%s failed", args.command)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
