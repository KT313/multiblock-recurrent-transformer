# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Synthetic data for the tiny run and the tests: a hand-written WordLevel tokenizer (<pad>, <bos>, <eos>,
tok_0..tok_255) and deterministic random-word rows (`synthetic_row(kind, seed, index)`), reproducible per row."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

Row = dict[str, Any]


SPECIALS = ["<pad>", "<bos>", "<eos>"]  # ids 0, 1, 2
N_WORD_TOKENS = 256  # tok_0..tok_255 -> ids 3..258
VOCAB_SIZE = len(SPECIALS) + N_WORD_TOKENS  # 259; the tiny model preset pads its vocab to 512
SYNTHETIC_DOC_WORDS = (64, 384)  # pretrain/holdout documents: words per row (inclusive)
SYNTHETIC_INSTRUCTION_WORDS = (4, 32)
SYNTHETIC_OUTPUT_WORDS = (8, 64)





def _synthetic_words(rng: random.Random, low: int, high: int) -> str:
    return " ".join(f"tok_{rng.randrange(N_WORD_TOKENS)}" for _ in range(rng.randint(low, high)))


def synthetic_row(kind: str, seed: int, index: int) -> Row:
    """Row `index` of a synthetic source; depends only on `(seed, index)` so any offset yields the same rows."""
    rng = random.Random(f"{seed}:{index}")
    if kind == "instruct":
        return {
            "instruction": _synthetic_words(rng, *SYNTHETIC_INSTRUCTION_WORDS),
            "input": "",
            "output": _synthetic_words(rng, *SYNTHETIC_OUTPUT_WORDS),
        }
    return {"text": _synthetic_words(rng, *SYNTHETIC_DOC_WORDS)}


def write_synthetic_tokenizer(path: Path) -> None:
    """A WordLevel tokenizer.json (<pad>=0, <bos>=1, <eos>=2, tok_i=3+i) written by hand so there is no magic."""
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
