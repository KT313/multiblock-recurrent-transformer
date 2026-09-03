# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Truncate document text at a token boundary so the stored text has at most max_tokens tokens.

The download step cuts the text itself instead of storing the whole document with a capped count, so the stored
count is the true count of the stored text and storage is bounded. Tokens are counted like TokenCounter in
stages/download.py: the config tokenizer with add_special_tokens=False, or len(text) // 4 in
token_count: estimate mode.

Invariants of :func:`truncate_many` (tested): the returned text is a prefix of the input, its count is the
tokenizer's count of that text, and the count is <= max_tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # transformers is imported lazily by the tokenizer stage (HF cache env must be settable first)
    from transformers import PreTrainedTokenizerFast

CHARS_PER_TOKEN_ESTIMATE = 4  # token_count: estimate counts len(text) // CHARS_PER_TOKEN_ESTIMATE

# A text is cut at PRE_CUT_CHARS_PER_TOKEN * max_tokens characters before tokenizing, so the tokenizer cost per
# document is bounded (a Gutenberg book is millions of characters). English prose runs about 4 characters per token,
# so the pre-cut normally lies far beyond the cut point. A text averaging more than 32 characters per token (long
# runs of one character) is cut earlier than its tokens require; the invariants still hold, the lost tail is accepted.
PRE_CUT_CHARS_PER_TOKEN = 32

Offsets = list[tuple[int, int]]


def estimate_tokens(text: str) -> int:
    """
    The token_count: estimate count: characters / :data:`CHARS_PER_TOKEN_ESTIMATE`.
    """

    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def truncate_many(texts: list[str], max_tokens: int, tokenizer: PreTrainedTokenizerFast | None) -> list[tuple[str, int]]:
    """
    (prefix, count) per text: the longest prefix found with at most max_tokens tokens and its count. One
    batched tokenizer call per round instead of one per text.

    tokenizer None is the token_count: estimate mode: each text is cut at CHARS_PER_TOKEN_ESTIMATE *
    max_tokens characters and counted as len // CHARS_PER_TOKEN_ESTIMATE.

    Tokenizer mode: each text is pre-cut (:data:`PRE_CUT_CHARS_PER_TOKEN`), tokenized with offsets and, when it has
    more than max_tokens tokens, cut where token max_tokens (0-based) starts. The cut text is re-tokenized
    and its own count returned, because tokenizing a prefix does not always reproduce the prefix of the tokenization
    (Metaspace "▁" handling, normalizers, Unigram re-segmentation). A recount below max_tokens is accepted; a
    recount above cuts again until the count fits. Every round strictly shortens the text, so the loop ends.
    """

    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
    if not texts:
        return []  # HF fast tokenizers choke on an empty batch
    if tokenizer is None:
        limit = CHARS_PER_TOKEN_ESTIMATE * max_tokens
        return [(text[:limit], estimate_tokens(text[:limit])) for text in texts]

    pre_cut = PRE_CUT_CHARS_PER_TOKEN * max_tokens
    current = [text[:pre_cut] for text in texts]
    result: list[tuple[str, int] | None] = [None] * len(texts)
    pending = list(range(len(texts)))
    while pending:
        batch = [current[index] for index in pending]
        encoded = tokenizer(batch, add_special_tokens=False, return_offsets_mapping=True)
        ids: list[list[int]] = encoded["input_ids"]
        offsets: list[Offsets] = encoded["offset_mapping"]
        still_pending: list[int] = []
        for index, token_ids, token_offsets in zip(pending, ids, offsets, strict=True):
            if len(token_ids) <= max_tokens:
                result[index] = (current[index], len(token_ids))
            else:
                current[index] = _cut_before_token(current[index], token_offsets, max_tokens)
                still_pending.append(index)
        pending = still_pending
    return [entry for entry in result if entry is not None]


def _cut_before_token(text: str, offsets: Offsets, index: int) -> str:
    """
    text cut where token index starts; always strictly shorter than text so the recount loop ends.
    """

    start, _ = offsets[index]
    return text[: min(start, len(text) - 1)]
