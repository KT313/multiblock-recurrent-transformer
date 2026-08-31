# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Batch collation in two halves: `collate_samples` tokenizes rows into unpadded samples (the expensive part, run in
the dataloader workers) and `pad_and_shift` turns a list of samples into one padded, shifted micro-batch (the cheap
part, run in the main process once the world batch is assembled).

`collate_fn` is the composition of the two and is what a padded loader (validation) uses. Splitting them is what lets
`training.data.loader.world_batch_micro_batches` group a world batch into micro-batches BEFORE anything is padded,
instead of re-cutting already padded batches: every micro-batch is padded exactly once, to its own longest sample.
"""

from typing import Any

import torch

from training.data.formats import apply_formatting
from training.data.tokenizer import Tokenizer

IGNORE_INDEX = -100  # label value of positions without a loss (padding, out-of-vocab); the model defaults to it too

Sample = tuple[torch.Tensor, torch.Tensor, str]  # one unpadded, unshifted row: (input_ids, labels, data_id)
Batch = tuple[torch.Tensor, torch.Tensor, list[str]]  # a padded, shifted micro-batch: (input_ids, labels, data_ids)


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


def has_supervised_label(labels: torch.Tensor, tokenizer: Tokenizer) -> bool:
    """Whether the shift of this unpadded ``labels`` row leaves a single position with a loss.

    The shift drops ``labels[0]`` and the collation masks pad ids and out-of-vocab ids, so the row is trainable iff
    some ``labels[1:]`` is a valid, non-pad id. False for a row that is one token long, for a row of pure padding
    (unknown tokens) and for an instruct row whose masked prompt alone fills ``block_size + 1``.
    """
    tail = labels[1:]
    if tail.numel() == 0:
        return False
    valid = (tail != tokenizer.pad_id) & (tail >= 0) & (tail < tokenizer.vocab_size)
    return bool(valid.any())


def collate_samples(
    batch: list[dict[str, Any]],
    tokenizer: Tokenizer,
    block_size: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[Sample]:
    """Format and tokenize dataset rows into unpadded ``(input_ids, labels, data_id)`` samples.

    Rows are truncated to ``block_size + 1`` tokens (the shift turns that into ``block_size`` positions); rows that
    keep no supervised label (`has_supervised_label`) are DROPPED. Dropping is what replaces the thesis
    `StopIteration("All tokens in batch are padding tokens.")`, which torch's worker loop read as "this worker is
    done" and the single-process loop as "the epoch is over" — an unusable row now costs one row, not a loader.
    """
    cap = block_size + 1
    samples: list[Sample] = []
    for row in batch:
        input_ids, labels = apply_formatting(row, tokenizer, add_bos, add_eos)
        input_ids, labels = input_ids[:cap], labels[:cap]
        if has_supervised_label(labels, tokenizer):
            samples.append((input_ids, labels, row["data_id"]))
    return samples


def pad_and_shift(
    samples: list[Sample],
    tokenizer: Tokenizer,
    block_size: int,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Batch:
    """Pad `samples` to one width and shift them into a ``(input_ids, labels, data_ids)`` micro-batch.

    The width is the longest sample of THIS micro-batch, rounded up to ``padding_multiple`` and capped at
    ``block_size + 1``; the shift then drops one position from it. Pad positions become EOS in the inputs and
    ``ignore_index`` in the labels, as do labels outside the tokenizer's vocabulary.
    """
    if not samples:
        raise ValueError("pad_and_shift needs at least one sample; empty micro-batches are never assembled")
    cap = block_size + 1
    max_len = max(max(inp.shape[0], lab.shape[0]) for inp, lab, _ in samples)
    local = min(find_multiple(max_len, padding_multiple) if padding_multiple else max_len, cap)

    pad_id = tokenizer.pad_id
    inputs = torch.full((len(samples), local), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), local), pad_id, dtype=torch.long)
    for i, (inp, lab, _) in enumerate(samples):
        inputs[i, : min(len(inp), local)] = inp[:local]
        labels[i, : min(len(lab), local)] = lab[:local]

    input_ids, label_ids = shift_inputs_and_labels(inputs, labels, tokenizer)
    label_ids[label_ids == pad_id] = ignore_index
    label_ids[(label_ids < 0) | (label_ids >= tokenizer.vocab_size)] = ignore_index
    return input_ids, label_ids, [data_id for _, _, data_id in samples]


def collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: Tokenizer,
    block_size: int,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Batch:
    """`collate_samples` followed by `pad_and_shift`: dataset rows straight to a padded micro-batch.

    The collate function of a padded loader (the validation loaders). A batch in which every row was dropped is a
    hard error — `block_size` too small for a whole validation batch is a configuration mistake, not something to
    swallow silently.
    """
    samples = collate_samples(batch, tokenizer, block_size, add_bos, add_eos)
    if not samples:
        raise ValueError(
            f"every row of the batch was dropped: none of the {len(batch)} rows keeps a supervised label within "
            f"block_size + 1 = {block_size + 1} tokens"
        )
    return pad_and_shift(samples, tokenizer, block_size, padding_multiple, ignore_index)
