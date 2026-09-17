# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Frozen pre-optimization algorithms, used only as benchmark/parity oracles."""

from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer


def truncate_original(texts: list[str], max_tokens: int, tokenizer: SavedTokenizer | None) -> list[tuple[str, int]]:
    """Keep the original allocating implementation independent of production helpers."""
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
    if tokenizer is None:
        return [(text[:4 * max_tokens], len(text[:4 * max_tokens]) // 4) for text in texts]
    current = [text[:32 * max_tokens] for text in texts]
    result: list[tuple[str, int] | None] = [None] * len(texts)
    pending = list(range(len(texts)))
    while pending:
        encoded = tokenizer.encode_batch([current[index] for index in pending])
        remaining: list[int] = []
        for index, encoding in zip(pending, encoded, strict=True):
            if len(encoding.ids) <= max_tokens:
                result[index] = current[index], len(encoding.ids)
            else:
                start, _ = encoding.offsets[max_tokens]
                current[index] = current[index][:min(start, len(current[index]) - 1)]
                remaining.append(index)
        pending = remaining
    return [entry for entry in result if entry is not None]
