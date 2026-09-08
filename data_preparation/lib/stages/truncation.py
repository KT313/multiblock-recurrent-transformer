# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Truncate document text at a token boundary so the stored text has at most max_tokens tokens.

The download step cuts the text itself instead of storing the whole document with a capped count, so the stored
count is the true count of the stored text and storage is bounded. Tokens are counted like TokenCounter in
stages/download.py: the config tokenizer without special tokens, or len(text) // 4 in
token_count: estimate mode. The trainer adds :data:`NUMBER_OF_SPECIAL_TOKENS` around every row, so the download asks for
dataset_max_sequence_length - NUMBER_OF_SPECIAL_TOKENS here and stores count + NUMBER_OF_SPECIAL_TOKENS as the row's tokens (the length the
trainer sees, never above dataset_max_sequence_length).

Invariants of :func:`truncate_many` (tested): the returned text is a prefix of the input, its count is the
tokenizer's count of that text, and the count is <= max_tokens.
"""

from __future__ import annotations

from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer

CHARS_PER_TOKEN_ESTIMATE = 4  # token_count: estimate counts len(text) // CHARS_PER_TOKEN_ESTIMATE

NUMBER_OF_SPECIAL_TOKENS = 2  # BOS and EOS the trainer adds around every row (training/data/formats.py); part of every stored count
TOKEN_RULE = "with_specials"  # names the counting rule in DatasetConfig.raw_hash, so raw folders counted otherwise are stale

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


def truncate_many(texts: list[str], max_tokens: int, tokenizer: SavedTokenizer | None) -> list[tuple[str, int]]:
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
    if tokenizer is None:
        limit = CHARS_PER_TOKEN_ESTIMATE * max_tokens
        return [(text[:limit], estimate_tokens(text[:limit])) for text in texts]

    pre_cut = PRE_CUT_CHARS_PER_TOKEN * max_tokens
    current = [text[:pre_cut] for text in texts]
    result: list[tuple[str, int] | None] = [None] * len(texts)
    pending = list(range(len(texts)))
    while pending:
        batch = [current[index] for index in pending]
        encoded = tokenizer.encode_batch(batch)  # an empty batch encodes to an empty list
        still_pending: list[int] = []
        for index, encoding in zip(pending, encoded, strict=True):
            if len(encoding.ids) <= max_tokens:
                result[index] = (current[index], len(encoding.ids))
            else:
                current[index] = _cut_before_token(current[index], encoding.offsets, max_tokens)
                still_pending.append(index)
        pending = still_pending
    return [entry for entry in result if entry is not None]


def _cut_before_token(text: str, offsets: Offsets, index: int) -> str:
    """
    text cut where token index starts; always strictly shorter than text so the recount loop ends.
    """

    start, _ = offsets[index]
    return text[: min(start, len(text) - 1)]
