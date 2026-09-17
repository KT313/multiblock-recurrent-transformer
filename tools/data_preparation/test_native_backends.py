# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Exact prefix/count parity for the optional native experiments."""

import os
import random
from pathlib import Path

import pytest

from tools.data_preparation.replay import make_counter


@pytest.mark.parametrize("variant", ["builtin", "rust"])
def test_native_prefixes_and_counts_match_original(tiny_tokenizer_dir: Path, variant: str) -> None:
    if variant == "rust" and "PREPARER_NATIVE_LIBRARY" not in os.environ:
        pytest.skip("optional native adapter not built; set PREPARER_NATIVE_LIBRARY to test it")
    original = make_counter(tiny_tokenizer_dir, "original")
    native = make_counter(tiny_tokenizer_dir, variant)
    rng = random.Random(21)
    pieces = ["tok_1", "tok_2", "日本語", "😀", "e\u0301", "<user>", "</s>", " ", "\n", "a" * 200]
    texts = ["", "  ", "😀 tok_1", "tok_1 tok_2 tok_3", "a" * 10000]
    texts += [" ".join(rng.choices(pieces, k=80)) for _ in range(30)]
    for cap in (0, 1, 2, 7, 64, 256):
        assert native.truncate_many(texts, cap) == original.truncate_many(texts, cap)
    assert native.count_many(texts) == original.count_many(texts)
    assert native.truncate_many([], 0) == []
    assert native.count_many([]) == []
    with pytest.raises(ValueError):
        native.truncate_many(["a"], -1)
