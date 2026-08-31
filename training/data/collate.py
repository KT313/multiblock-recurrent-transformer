# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Batch collation: tokenize/format rows, pad, shift labels by one, and mask padding with the ignore index."""

from typing import Any

import torch

from training.data.formats import apply_formatting
from training.data.tokenizer import Tokenizer

IGNORE_INDEX = -100  # label value of positions without a loss (padding, out-of-vocab); the model defaults to it too


def find_multiple(n: int, k: int) -> int:
    """Smallest multiple of ``k`` that is >= ``n``."""
    return n if n % k == 0 else n + k - (n % k)


def shift_inputs_and_labels(
    inputs: torch.Tensor, labels: torch.Tensor, tokenizer: Tokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token shift: inputs drop the last position, labels drop the first.

    Trailing pad ids in the inputs are replaced by EOS so they are valid embedding indices; the labels keep their
    pad ids so the caller can turn them into the ignore index.
    """
    seq_len = inputs.shape[1]
    input_ids = inputs[:, : seq_len - 1].contiguous().long()
    label_ids = labels[:, 1:seq_len].contiguous().long()
    if tokenizer.eos_id is not None:
        input_ids[input_ids == tokenizer.pad_id] = tokenizer.eos_id
    return input_ids, label_ids


def collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: Tokenizer,
    block_size: int,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    add_bos: bool = True,
    add_eos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Turn dataset rows into ``(input_ids, labels, data_ids)``.

    Rows are formatted/tokenized, truncated to ``block_size + 1`` tokens, padded to the longest row in the batch
    (rounded up to ``padding_multiple``), then shifted so ``labels[t] == input_ids[t + 1]``. Pad and out-of-vocab
    label positions become ``ignore_index``.
    """
    cap = block_size + 1
    rows = [apply_formatting(row, tokenizer, add_bos, add_eos) for row in batch]
    data_ids = [row["data_id"] for row in batch]

    max_len = max(len(x) for row in rows for x in row)
    local = min(find_multiple(max_len, padding_multiple) if padding_multiple else max_len, cap)

    pad_id = tokenizer.pad_id
    inputs = torch.full((len(rows), local), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), local), pad_id, dtype=torch.long)
    for i, (inp, lab) in enumerate(rows):
        inputs[i, : min(len(inp), local)] = inp[:local]
        labels[i, : min(len(lab), local)] = lab[:local]

    # Tensor == Optional[int] resolves to object.__eq__ in the torch stubs; at runtime it is an elementwise compare.
    all_eos = bool(torch.all(labels == tokenizer.eos_id))  # type: ignore[call-overload]
    if all_eos or bool(torch.all(labels == pad_id)):
        raise StopIteration("All tokens in batch are padding tokens.")

    input_ids, label_ids = shift_inputs_and_labels(inputs, labels, tokenizer)
    label_ids[label_ids == pad_id] = ignore_index
    label_ids[(label_ids < 0) | (label_ids >= tokenizer.vocab_size)] = ignore_index
    return input_ids, label_ids, data_ids
