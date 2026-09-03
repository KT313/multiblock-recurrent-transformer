# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Batch collation in two halves: `collate_samples` tokenizes rows into unpadded samples (in the dataloader workers)
and `pad_and_shift` turns a list of samples into one padded, shifted micro-batch (in the main process).

`collate_fn` composes the two for a padded loader (validation). The split lets `world_batch_micro_batches` group a
world batch into micro-batches BEFORE padding, so every micro-batch is padded once, to its own longest sample.
"""

from typing import Any, NamedTuple

import torch

from training.data.formats import apply_formatting
from training.data.tokenizer import Tokenizer

IGNORE_INDEX = -100  # label value of positions without a loss (padding, out-of-vocab); the model defaults to it too

Sample = tuple[torch.Tensor, torch.Tensor, str]  # one unpadded, unshifted row: (input_ids, labels, data_id)
Batch = tuple[torch.Tensor, torch.Tensor, list[str]]  # a padded, shifted micro-batch: (input_ids, labels, data_ids)


class WorkerBatch(NamedTuple):
    """
    What an unpadded (training) loader yields per worker batch: the samples that survived tokenization plus how
    many rows were READ to produce them, dropped rows included.

    The count travels with the batch from the worker, so `BatchStream` counts consumed rows in the unit a resume
    skips (`ParquetTextDataset.set_resume_offset`). Counting surviving samples would rewind a resume by one row per
    dropped row.
    """

    samples: list[Sample]
    rows_read: int


def find_multiple(value: int, multiple: int) -> int:
    """
    Smallest multiple of multiple that is >= value.
    """

    return value if value % multiple == 0 else value + multiple - (value % multiple)


def shift_inputs_and_labels(
    inputs: torch.Tensor, labels: torch.Tensor, tokenizer: Tokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Next-token shift: inputs drop the last position, labels drop the first.

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
    """
    Whether the shift of this unpadded labels row leaves a single position with a loss.

    The shift drops labels[0] and the collation masks pad ids and out-of-vocab ids, so the row is trainable iff
    some labels[1:] is a valid, non-pad id. False for a row that is one token long, for a row of pure padding
    (unknown tokens) and for an instruct row whose masked prompt alone fills block_size + 1.
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
    """
    Format and tokenize dataset rows into unpadded (input_ids, labels, data_id) samples.

    Rows are truncated to block_size + 1 tokens (the shift turns that into block_size positions); rows without
    a supervised label are DROPPED, never raised on: a `StopIteration` out of a collate function ends the worker or
    the epoch, so an unusable row must cost one row, not a loader.
    """

    max_tokens = block_size + 1
    samples: list[Sample] = []
    for row in batch:
        input_ids, labels = apply_formatting(row, tokenizer, add_bos, add_eos)
        input_ids, labels = input_ids[:max_tokens], labels[:max_tokens]
        if has_supervised_label(labels, tokenizer):
            samples.append((input_ids, labels, row["data_id"]))
    return samples


def collate_worker_batch(
    batch: list[dict[str, Any]],
    tokenizer: Tokenizer,
    block_size: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> WorkerBatch:
    """
    `collate_samples` plus the count of rows that went in: the collate function of the training loaders.
    rows_read advances by rows read from disk, dropped rows included, the unit a resume skips.
    """

    return WorkerBatch(collate_samples(batch, tokenizer, block_size, add_bos, add_eos), len(batch))


def pad_and_shift(
    samples: list[Sample],
    tokenizer: Tokenizer,
    block_size: int,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Batch:
    """
    Pad `samples` to one width and shift them into a (input_ids, labels, data_ids) micro-batch.

    The width is the longest sample of THIS micro-batch, rounded up to padding_multiple and capped at
    block_size + 1; the shift then drops one position from it. Pad positions become EOS in the inputs and
    ignore_index in the labels, as do labels outside the tokenizer's vocabulary.
    """

    if not samples:
        raise ValueError("pad_and_shift needs at least one sample; empty micro-batches are never assembled")
    max_tokens = block_size + 1
    longest = max(max(sample_inputs.shape[0], sample_labels.shape[0]) for sample_inputs, sample_labels, _ in samples)
    width = min(find_multiple(longest, padding_multiple) if padding_multiple else longest, max_tokens)

    pad_id = tokenizer.pad_id
    inputs = torch.full((len(samples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), width), pad_id, dtype=torch.long)
    for row, (sample_inputs, sample_labels, _) in enumerate(samples):
        inputs[row, : min(len(sample_inputs), width)] = sample_inputs[:width]
        labels[row, : min(len(sample_labels), width)] = sample_labels[:width]

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
    """
    `collate_samples` followed by `pad_and_shift`: the collate function of the validation loaders. A batch in which
    every row was dropped is an error (`block_size` too small for a validation batch is a configuration mistake).
    """

    samples = collate_samples(batch, tokenizer, block_size, add_bos, add_eos)
    if not samples:
        raise ValueError(
            f"every row of the batch was dropped: none of the {len(batch)} rows keeps a supervised label within "
            f"block_size + 1 = {block_size + 1} tokens"
        )
    return pad_and_shift(samples, tokenizer, block_size, padding_multiple, ignore_index)
