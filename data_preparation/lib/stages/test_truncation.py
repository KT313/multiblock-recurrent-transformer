# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.stages.truncation: token-boundary text truncation in tokenizer and estimate mode.

Two offline tokenizers: the synthetic WordLevel one of the tiny config (whitespace words, what the rest of the suite
uses) and a small BPE with a Metaspace pre-tokenizer trained here in milliseconds (llama-style "▁" tokens whose
offsets overlap the first word, the boundary subtlety the module handles)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, cast

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from data_preparation.lib.stages import truncation
from data_preparation.lib.stages.truncation import (
    CHARS_PER_TOKEN_ESTIMATE,
    PRE_CUT_CHARS_PER_TOKEN,
    estimate_tokens,
    truncate_many,
)


def truncate_to_token_cap(text: str, max_tokens: int, tokenizer: PreTrainedTokenizerFast | None) -> tuple[str, int]:
    """`truncate_many` for one text."""
    return truncate_many([text], max_tokens, tokenizer)[0]

CORPUS = [
    "the quick brown fox jumps over the lazy dog " * 20,
    "hello world hello there general kenobi " * 20,
    "日本語のテキストです 😀 " * 10,
    "a" * 200,
    " " * 100,
]
PROSE = "the quick brown fox jumps over the lazy dog, hello there general kenobi. "
UNICODE = "日本語 😀 naïve café … über 😀😀 tok_1 tok_2 "


def _count(tokenizer: PreTrainedTokenizerFast, text: str) -> int:
    """The count definition of ``TokenCounter``: no special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _words(n: int) -> str:
    return " ".join(f"tok_{i % 256}" for i in range(n))


@pytest.fixture(scope="module")
def wordlevel(tiny_tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    tok = AutoTokenizer.from_pretrained(str(tiny_tokenizer_dir))
    assert isinstance(tok, PreTrainedTokenizerFast)
    return tok


@pytest.fixture(scope="module")
def metaspace_bpe() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first")
    tok.decoder = decoders.Metaspace()
    alphabet = [chr(i) for i in range(32, 127)] + list("äöüß日本語のテキストです😀ïé…")
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]  # tokenizers ships unannotated constructors
        vocab_size=300, special_tokens=["<unk>", "<s>", "</s>"], initial_alphabet=alphabet
    )
    tok.train_from_iterator(CORPUS, trainer)
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # transformers 5: unannotated __init__
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )


@pytest.fixture(params=["wordlevel", "metaspace_bpe"])
def tokenizer(request: pytest.FixtureRequest) -> PreTrainedTokenizerFast:
    tok = request.getfixturevalue(request.param)
    assert isinstance(tok, PreTrainedTokenizerFast)
    return tok


def _check_invariants(tokenizer: PreTrainedTokenizerFast, text: str, max_tokens: int) -> tuple[str, int]:
    cut, count = truncate_to_token_cap(text, max_tokens, tokenizer)
    assert text.startswith(cut)
    assert count <= max_tokens
    assert count == _count(tokenizer, cut)
    return cut, count


# --- tokenizer mode ---------------------------------------------------------------------------------------------------


def test_short_text_is_unchanged_with_its_real_count(tokenizer: PreTrainedTokenizerFast) -> None:
    text = PROSE * 2
    assert truncate_to_token_cap(text, 10_000, tokenizer) == (text, _count(tokenizer, text))


def test_long_text_is_cut_at_a_token_boundary(tokenizer: PreTrainedTokenizerFast) -> None:
    text = PROSE * 50
    cut, count = _check_invariants(tokenizer, text, 37)
    assert len(cut) < len(text)
    # a cut at token 37's start normally keeps exactly 37 tokens; the recount may only fall short on a boundary merge
    assert count == 37 or _count(tokenizer, cut + text[len(cut)]) > 37


def test_exact_boundary_wordlevel(wordlevel: PreTrainedTokenizerFast) -> None:
    exact = _words(16)
    assert truncate_to_token_cap(exact, 16, wordlevel) == (exact, 16)
    one_more = _words(17)
    assert truncate_to_token_cap(one_more, 16, wordlevel) == (_words(16) + " ", 16)
    assert truncate_to_token_cap(exact, 15, wordlevel) == (_words(15) + " ", 15)


def test_exact_boundary_bpe(metaspace_bpe: PreTrainedTokenizerFast) -> None:
    text = "the lazy dog hello world there"
    n = _count(metaspace_bpe, text)
    assert truncate_to_token_cap(text, n, metaspace_bpe) == (text, n)
    cut, count = _check_invariants(metaspace_bpe, text, n - 1)
    assert len(cut) < len(text) and count == n - 1


def test_empty_string(tokenizer: PreTrainedTokenizerFast) -> None:
    assert truncate_to_token_cap("", 5, tokenizer) == ("", 0)
    assert truncate_to_token_cap("", 0, tokenizer) == ("", 0)


def test_max_tokens_one_and_zero(tokenizer: PreTrainedTokenizerFast) -> None:
    for text in (PROSE, "tok_3 tok_4", "a" * 50, "  leading spaces", UNICODE):
        cut, count = _check_invariants(tokenizer, text, 1)
        assert count <= 1
        assert truncate_to_token_cap(text, 0, tokenizer) == ("", 0)


def test_wordlevel_max_tokens_one_keeps_first_word(wordlevel: PreTrainedTokenizerFast) -> None:
    assert truncate_to_token_cap("tok_3 tok_4", 1, wordlevel) == ("tok_3 ", 1)


def test_negative_cap_is_rejected(wordlevel: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        truncate_to_token_cap("x", -1, wordlevel)


def test_metaspace_prefix_token_overlapping_the_first_word(metaspace_bpe: PreTrainedTokenizerFast) -> None:
    """'aaaa…' tokenizes to ['▁' (0, 1), 'aaaa…' (0, 32), …]: token 1 starts at offset 0, so a cap of 1 keeps
    nothing (a single 'a' would already be two tokens) and the recount (0) is what is returned."""
    text = "a" * 60
    assert _count(metaspace_bpe, text) > 2
    assert truncate_to_token_cap(text, 1, metaspace_bpe) == ("", 0)
    cut, count = _check_invariants(metaspace_bpe, text, 2)
    assert cut and count <= 2


def test_random_texts_satisfy_the_invariants(tokenizer: PreTrainedTokenizerFast) -> None:
    rng = random.Random(0)
    pieces = PROSE.split() + UNICODE.split() + ["aaaa", "   ", "\n\n", "tok_9", "😀"]
    for _ in range(60):
        text = "".join(rng.choice(pieces) + rng.choice(["", " ", "  ", "\n"]) for _ in range(rng.randint(0, 40)))
        max_tokens = rng.randint(0, 25)
        _check_invariants(tokenizer, text, max_tokens)


# --- boundary merge: re-tokenizing the cut text can exceed the cap, the loop cuts again ---------------------------------


class _ViterbiLikeStub:
    """One token per character, except that "xy" is one token when something follows it (an end-of-word dependent
    segmentation, as a Unigram model can produce): cutting "xyz" after token 0 ("xy") gives "xy", which alone is two
    tokens, more than the cap of 1, so the module must cut again."""

    calls: list[list[str]]

    def __init__(self) -> None:
        self.calls = []

    def _one(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            if text[i : i + 2] == "xy" and i + 2 < len(text):
                ids.append(1)
                offsets.append((i, i + 2))
                i += 2
            else:
                ids.append(2)
                offsets.append((i, i + 1))
                i += 1
        return ids, offsets

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return self._one(text)[0]

    def __call__(self, texts: list[str], add_special_tokens: bool, return_offsets_mapping: bool) -> dict[str, Any]:
        assert not add_special_tokens and return_offsets_mapping
        self.calls.append(list(texts))
        encoded = [self._one(t) for t in texts]
        return {"input_ids": [e[0] for e in encoded], "offset_mapping": [e[1] for e in encoded]}


def test_recount_above_the_cap_cuts_again_until_it_fits() -> None:
    stub = _ViterbiLikeStub()
    tokenizer = cast(PreTrainedTokenizerFast, stub)  # duck-typed stand-in: the module only uses __call__
    assert stub.encode("xyz", False) == [1, 2] and stub.encode("xy", False) == [2, 2]
    assert truncate_to_token_cap("xyz", 1, tokenizer) == ("x", 1)
    assert [len(c) for c in stub.calls] == [1, 1, 1]  # "xyz" -> "xy" (2 > 1) -> "x"
    assert truncate_many(["xyz", "xyzw", "ab", ""], 1, tokenizer) == [("x", 1), ("x", 1), ("a", 1), ("", 0)]


# --- pre-cut ---------------------------------------------------------------------------------------------------------


def test_pre_cut_bounds_the_tokenized_text_and_keeps_the_invariants(tokenizer: PreTrainedTokenizerFast) -> None:
    max_tokens = 4
    text = "a" * 100_000  # one whitespace word / long merged runs: far more than 32 chars per token
    cut, count = _check_invariants(tokenizer, text, max_tokens)
    assert 0 < len(cut) <= PRE_CUT_CHARS_PER_TOKEN * max_tokens
    long_prose = PROSE * 5_000
    cut, count = _check_invariants(tokenizer, long_prose, 100)
    assert len(cut) <= PRE_CUT_CHARS_PER_TOKEN * 100 and count == 100


def test_pre_cut_does_not_change_the_result_of_ordinary_text(
    tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = PROSE * 300
    with_pre_cut = truncate_to_token_cap(text, 64, tokenizer)
    monkeypatch.setattr(truncation, "PRE_CUT_CHARS_PER_TOKEN", 10**9)
    assert truncate_to_token_cap(text, 64, tokenizer) == with_pre_cut


# --- estimate mode -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), [("", 0), ("abc", 0), ("abcd", 1), ("a" * 4000, 1000), ("a" * 4003, 1000)])
def test_estimate_tokens_is_chars_div_4(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


def test_estimate_mode_cuts_at_chars_per_token_times_cap() -> None:
    short = "hello world"
    assert truncate_to_token_cap(short, 100, None) == (short, estimate_tokens(short))
    long = PROSE * 20
    cut, count = truncate_to_token_cap(long, 30, None)
    assert cut == long[: 30 * CHARS_PER_TOKEN_ESTIMATE] and count == 30 == estimate_tokens(cut)
    exact = "x" * (7 * CHARS_PER_TOKEN_ESTIMATE)
    assert truncate_to_token_cap(exact, 7, None) == (exact, 7)
    assert truncate_to_token_cap(exact + "y", 7, None) == (exact, 7)
    assert truncate_to_token_cap("", 7, None) == ("", 0)
    assert truncate_to_token_cap("abcdefgh", 1, None) == ("abcd", 1)
    assert truncate_to_token_cap("abc", 0, None) == ("", 0)


def test_estimate_mode_many_equals_single() -> None:
    texts = ["", "abc", PROSE * 20, UNICODE * 30]
    assert truncate_many(texts, 12, None) == [truncate_to_token_cap(t, 12, None) for t in texts]
    assert truncate_many([], 12, None) == []


# --- batches and unicode ------------------------------------------------------------------------------------------------


def test_truncate_many_equals_per_text(tokenizer: PreTrainedTokenizerFast) -> None:
    texts = ["", "tok_1", PROSE * 40, UNICODE * 40, "a" * 5_000, "  ", PROSE]
    for max_tokens in (0, 1, 7, 50):
        assert truncate_many(texts, max_tokens, tokenizer) == [truncate_to_token_cap(t, max_tokens, tokenizer) for t in texts]
    assert truncate_many([], 5, tokenizer) == []


def test_unicode_offsets_are_characters_not_bytes(wordlevel: PreTrainedTokenizerFast) -> None:
    # every emoji is one whitespace word (an <unk> token); 10 tokens end after 10 x 2 characters (40 bytes in UTF-8)
    text = "😀 " * 50
    assert truncate_to_token_cap(text, 10, wordlevel) == ("😀 " * 10, 10)
    mixed = UNICODE * 20
    cut, count = _check_invariants(wordlevel, mixed, 13)
    assert cut == " ".join(mixed.split()[:13]) + " "


def test_unicode_text_with_bpe(metaspace_bpe: PreTrainedTokenizerFast) -> None:
    text = UNICODE * 20
    for max_tokens in (1, 2, 5, 33):
        cut, _ = _check_invariants(metaspace_bpe, text, max_tokens)
        assert cut.encode("utf-8").decode("utf-8") == cut  # cut between code points, never inside one
