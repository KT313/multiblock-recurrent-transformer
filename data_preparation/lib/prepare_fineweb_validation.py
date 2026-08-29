# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Build the pretraining validation set at ``dataset/fineweb-edu/validation/data-*.parquet``.

Loads the ``sample-10BT`` subset of ``HuggingFaceFW/fineweb-edu``, shuffles it with seed 42 and holds out
50,000 documents (``train_test_split``, seed 42) as validation. Rows keep the original fineweb-edu columns
(``text`` among them).
"""

from __future__ import annotations

import argparse

from data_preparation.lib.common import (
    RANDOM_SEED,
    add_common_args,
    configure_hf_cache,
    iter_dataset_tables,
    write_parquet_shards,
)

N_VAL = 50_000
SHARD_SIZE = 10_000


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
    from datasets import load_dataset

    out_dir = args.dataset_dir / "fineweb-edu" / "validation"
    train = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    splits = train.shuffle(seed=RANDOM_SEED).train_test_split(test_size=N_VAL, seed=RANDOM_SEED)
    validation = splits["test"]
    num_shards = write_parquet_shards(iter_dataset_tables(validation), out_dir, SHARD_SIZE)
    print(f"Saved {len(validation):,} validation documents in {num_shards} shards to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``) and run."""
    run(build_parser().parse_args(argv))
