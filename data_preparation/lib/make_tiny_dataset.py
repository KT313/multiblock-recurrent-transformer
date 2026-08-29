# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Generate the synthetic dataset used by `config/tiny.yaml` and the tests.

    python data_preparation/prepare.py tiny [--out dataset/tiny]

Writes a minimal WordLevel tokenizer (<pad>, <bos>, <eos> + tok_0..tok_255) and parquet files with a "text"
column of random tok_i words for pretrain/{train,val} and finetune/{train,val}. Deterministic; a few hundred KB.
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPECIALS = ["<pad>", "<bos>", "<eos>"]  # ids 0, 1, 2
N_WORD_TOKENS = 256  # tok_0..tok_255 -> ids 3..258
VOCAB_SIZE = len(SPECIALS) + N_WORD_TOKENS  # 259; the tiny model preset pads its vocab to 512
SPLITS = {"pretrain": {"train": (4, 64), "val": (2, 16)}, "finetune": {"train": (2, 32), "val": (2, 16)}}


def build_tokenizer(path: Path) -> None:
    """A WordLevel tokenizer.json written by hand so there is no magic."""
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for i in range(N_WORD_TOKENS):
        vocab[f"tok_{i}"] = len(vocab)
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab[t],
                "content": t,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for t in SPECIALS
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<pad>"},
    }
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "model_max_length": 1_000_000,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.json").write_text(json.dumps(tokenizer_json, indent=2))
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))
    (path / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}, indent=2)
    )


def write_split(path: Path, n_files: int, rows_per_file: int, rng: random.Random) -> int:
    """Parquet files of random-word documents, 64-384 words each; returns the number of words written."""
    path.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(n_files):
        rows = []
        for _ in range(rows_per_file):
            n = rng.randint(64, 384)
            rows.append(" ".join(f"tok_{rng.randrange(N_WORD_TOKENS)}" for _ in range(n)))
            total += n
        pq.write_table(pa.table({"text": rows}), path / f"part_{i:03d}.parquet")
    return total


def make_tiny_dataset(out: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    build_tokenizer(out / "tokenizer")
    for name, splits in SPLITS.items():
        for split, (n_files, rows) in splits.items():
            write_split(out / name / split, n_files, rows, rng)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register this command's options on ``parser`` (used by ``prepare.py`` and ``build_parser``)."""
    parser.add_argument("--out", type=Path, default=Path("dataset/tiny"), help="Output directory (default: dataset/tiny)")


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for this command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the command with parsed ``args``."""
    make_tiny_dataset(args.out)
    print(f"tiny dataset written to {args.out} ({VOCAB_SIZE} tokenizer entries)")


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``) and run."""
    run(build_parser().parse_args(argv))
