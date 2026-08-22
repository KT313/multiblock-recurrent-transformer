#!/usr/bin/env python
"""Generate the synthetic inputs for the training smoke run (tests/train_smoke/run.sh).

Creates, under dev/train_smoke/data/:
  tokenizer/            a minimal WordLevel tokenizer: <pad>, <bos>, <eos> + tok_0..tok_255
  pretrain/{train,val}  parquet files with a "text" column of random tok_i words —
  finetune/{train,val}  the same format as the real pretraining data: the hfds
                        pipeline reads text and tokenizes at collate time
                        (data_loading_utils.pass_text), bos/eos added there.

Everything is deterministic (fixed seed). Total size is a few hundred KB.
"""

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SPECIALS = ["<pad>", "<bos>", "<eos>"]  # ids 0, 1, 2
N_WORD_TOKENS = 256                     # tok_0..tok_255 -> ids 3..258
VOCAB_SIZE = len(SPECIALS) + N_WORD_TOKENS  # 259; model vocab_size in the yaml is 512


def build_tokenizer(path: Path):
    """A WordLevel tokenizer.json, written by hand so there is no magic."""
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for i in range(N_WORD_TOKENS):
        vocab[f"tok_{i}"] = len(vocab)

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": vocab[t], "content": t, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
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
    special_tokens_map = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}

    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.json").write_text(json.dumps(tokenizer_json, indent=2))
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))
    (path / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2))


def write_split(path: Path, n_files: int, rows_per_file: int, rng: random.Random) -> int:
    """Parquet files of random-word 'documents', 64-384 words each."""
    path.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    for i in range(n_files):
        rows = []
        for _ in range(rows_per_file):
            n = rng.randint(64, 384)
            text = " ".join(f"tok_{rng.randrange(N_WORD_TOKENS)}" for _ in range(n))
            rows.append(text)
            total_tokens += n
        pq.write_table(pa.table({"text": rows}), path / f"part_{i:03d}.parquet")
    return total_tokens


def main():
    rng = random.Random(0)
    build_tokenizer(DATA / "tokenizer")
    print(f"tokenizer written ({VOCAB_SIZE} tokens)")
    # the smoke run consumes ~20K tokens total; generate a comfortable multiple
    for name, splits in {
        "pretrain": {"train": (4, 64), "val": (2, 16)},
        "finetune": {"train": (2, 32), "val": (2, 16)},
    }.items():
        for split, (n_files, rows) in splits.items():
            n_tok = write_split(DATA / name / split, n_files, rows, rng)
            print(f"{name}/{split}: {n_files} parquet files, {n_files * rows} rows, {n_tok} tokens")


if __name__ == "__main__":
    main()
