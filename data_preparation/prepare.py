# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Entry point for all dataset preparation steps.

    python data_preparation/prepare.py <command> [options]      (from the repo root)

Commands, in pipeline order: download, filter, process, fineweb-validation, flan-mixture, tokenizer; plus tiny
(synthetic smoke data). ``<command> --help`` lists the options of each; the implementation is in ``lib/``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

if __name__ == "__main__":  # allow `python data_preparation/prepare.py` without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.lib import (  # noqa: E402
    download_pretraining,
    filter_pretraining,
    make_tiny_dataset,
    prepare_fineweb_validation,
    prepare_flan_mixture,
    prepare_tokenizer,
    process_pretraining,
)

COMMANDS: dict[str, ModuleType] = {
    "download": download_pretraining,
    "filter": filter_pretraining,
    "process": process_pretraining,
    "fineweb-validation": prepare_fineweb_validation,
    "flan-mixture": prepare_flan_mixture,
    "tokenizer": prepare_tokenizer,
    "tiny": make_tiny_dataset,
}


def build_parser() -> argparse.ArgumentParser:
    """Parser with one subcommand per module in ``COMMANDS``; ``args.run`` is the module's ``run`` function."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name, module in COMMANDS.items():
        doc = (module.__doc__ or "").strip()
        sub = subparsers.add_parser(
            name, help=doc.splitlines()[0], description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        add_arguments: Callable[[argparse.ArgumentParser], None] = module.add_arguments
        add_arguments(sub)
        sub.set_defaults(run=module.run)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``) and dispatch to the selected command."""
    args = build_parser().parse_args(argv)
    run: Callable[[argparse.Namespace], None] = args.run
    run(args)


if __name__ == "__main__":
    main()
