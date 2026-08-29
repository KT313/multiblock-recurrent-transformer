# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Download the 32k-vocabulary Llama tokenizer used for training to ``dataset/tokenizer``."""

from __future__ import annotations

import argparse

from data_preparation.lib.common import add_common_args, configure_hf_cache

TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register this command's options on ``parser`` (used by ``prepare.py`` and ``build_parser``)."""
    add_common_args(parser)


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for this command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the command with parsed ``args``."""
    configure_hf_cache(args.cache_dir)
    from transformers import AutoTokenizer

    out_dir = args.dataset_dir / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True).save_pretrained(out_dir)
    print(f"Saved tokenizer to: {out_dir}")


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``) and run."""
    run(build_parser().parse_args(argv))
