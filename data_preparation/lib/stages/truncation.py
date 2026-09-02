# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Truncate document text at a token boundary so the stored text has at most ``max_tokens`` tokens.

Raw shards used to store the full text of every document and only cap the *count* at ``max_seq_length``; the
download step now cuts the text itself, so the stored count is the true count of the stored text and storage is
bounded. The token definition is the one of ``TokenCounter`` in ``stages/download.py``: the config tokenizer with
``add_special_tokens=False`` (no bos/eos), or ``len(text) // 4`` in ``token_count: estimate`` mode.

Invariants of :func:`truncate_many` (tested): the returned text is a prefix of the
input; the returned count equals the tokenizer's count of the returned text; the count is ``<= max_tokens``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from data_preparation.lib.storage.parquet import estimate_tokens

if TYPE_CHECKING:  # transformers is imported lazily by the tokenizer stage (HF cache env must be settable first)
    from transformers import PreTrainedTokenizerFast

# ``token_count: estimate`` counts ``len(text) // CHARS_PER_TOKEN_ESTIMATE``. Must match ``estimate_tokens`` in
# ``storage/parquet.py`` (asserted in the tests).
CHARS_PER_TOKEN_ESTIMATE = 4

# Before tokenizing, a text is cut at ``PRE_CUT_CHARS_PER_TOKEN * max_tokens`` characters so the tokenizer cost per
# document is bounded (a Gutenberg book is millions of characters, the cap a few thousand tokens). English prose
# runs ~4 characters per token with the llama tokenizer, so the pre-cut normally only touches text far beyond the
# cut point. It is *not* a guarantee: a single token may cover many characters (runs of one repeated character,
# whitespace runs merged into one token), so a text averaging more than 32 characters per token over the pre-cut
# window is cut earlier than its tokens alone would require. The result still satisfies every invariant (a shorter
# prefix, with its true count <= max_tokens); the lost tail in that pathological case is accepted.
PRE_CUT_CHARS_PER_TOKEN = 32

Offsets = list[tuple[int, int]]


def truncate_many(texts: list[str], max_tokens: int, tokenizer: PreTrainedTokenizerFast | None) -> list[tuple[str, int]]:
    """``(prefix, count)`` per text: the longest prefix this function finds with at most ``max_tokens`` tokens and its
    count (one batched tokenizer call per round instead of one per text).

    ``tokenizer`` None is the ``token_count: estimate`` mode: each text is cut at ``CHARS_PER_TOKEN_ESTIMATE *
    max_tokens`` characters and counted as ``len // CHARS_PER_TOKEN_ESTIMATE``.

    Tokenizer mode: each text is pre-cut (:data:`PRE_CUT_CHARS_PER_TOKEN`), tokenized with character offsets and,
    when it has more than ``max_tokens`` tokens, cut where token number ``max_tokens`` (0-based) starts. The cut
    text is then re-tokenized and its own count returned, because tokenizing a prefix is not guaranteed to
    reproduce the prefix of the tokenization: Metaspace / SentencePiece-style tokenizers attach the prepended
    "▁" to the first word (its offsets overlap the word's), normalizers and Unigram/Viterbi segmentation can
    re-segment the last word. The recount usually equals ``max_tokens``, may be lower (a boundary merge fell
    away — accepted, the prefix is kept as is), and in rare cases higher; then the cut text is cut again at its own
    token ``max_tokens`` and recounted until the count fits. Every round strictly shortens the text, so the loop
    ends (an empty text has zero tokens).
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
    if not texts:
        return []  # HF fast tokenizers choke on an empty batch
    if tokenizer is None:
        limit = CHARS_PER_TOKEN_ESTIMATE * max_tokens
        return [(t[:limit], estimate_tokens(t[:limit])) for t in texts]

    pre_cut = PRE_CUT_CHARS_PER_TOKEN * max_tokens
    current = [t[:pre_cut] for t in texts]
    result: list[tuple[str, int] | None] = [None] * len(texts)
    pending = list(range(len(texts)))
    while pending:
        batch = [current[i] for i in pending]
        encoded = tokenizer(batch, add_special_tokens=False, return_offsets_mapping=True)
        ids: list[list[int]] = encoded["input_ids"]
        offsets: list[Offsets] = encoded["offset_mapping"]
        still_pending: list[int] = []
        for i, token_ids, token_offsets in zip(pending, ids, offsets, strict=True):
            if len(token_ids) <= max_tokens:
                result[i] = (current[i], len(token_ids))
            else:
                current[i] = _cut_before_token(current[i], token_offsets, max_tokens)
                still_pending.append(i)
        pending = still_pending
    return [r for r in result if r is not None]


def _cut_before_token(text: str, offsets: Offsets, index: int) -> str:
    """``text`` cut where token ``index`` starts — always strictly shorter than ``text`` so the recount loop ends."""
    start, _ = offsets[index]
    return text[: min(start, len(text) - 1)]


__all__ = ["CHARS_PER_TOKEN_ESTIMATE", "PRE_CUT_CHARS_PER_TOKEN", "truncate_many"]
