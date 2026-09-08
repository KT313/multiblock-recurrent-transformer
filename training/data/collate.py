# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Batch collation in two halves: `collate_samples` tokenizes rows into unpadded samples (in the dataloader workers)
and `pad_and_shift` turns a list of samples into one padded, shifted micro-batch (in the main process).

`collate_fn` composes the two for the padded validation loaders. The training loaders stop after the first half:
their workers hand out unpadded samples (`collate_worker_batch`) that `training.step.BatchStream` packs into one
row per micro-batch (`training.data.packing`).
"""

from typing import Any, NamedTuple

import torch

from training.data.formats import apply_formatting
from training.data.tokenizer import IGNORE_INDEX, Tokenizer

Sample = tuple[torch.Tensor, torch.Tensor, str]  # one unpadded, unshifted row: (input_ids, labels, data_id)


class Batch(NamedTuple):
    """
    A padded, shifted validation batch: `(input_ids, labels, data_ids)`, one data id per row. A tuple, so
    `input_ids, labels, data_ids = batch` keeps working; the first three fields of the packed
    `training.data.packing.PackedBatch` have the same names.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    data_ids: list[str]


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


def shift_inputs_and_labels(inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Next-token shift: inputs drop the last position, labels drop the first.
    """

    seq_len = inputs.shape[1]
    input_ids = inputs[:, : seq_len - 1].contiguous().long()
    label_ids = labels[:, 1:seq_len].contiguous().long()
    return input_ids, label_ids


def mask_label_ids(label_ids: torch.Tensor, tokenizer: Tokenizer, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """
    Shifted labels as the loss sees them: ids outside the tokenizer's vocabulary become `ignore_index`, in place;
    returns `label_ids`. `IGNORE_INDEX` (padding, masked prompts) is negative, so it passes through unchanged.
    """

    label_ids[(label_ids < 0) | (label_ids >= tokenizer.vocab_size)] = ignore_index
    return label_ids


def has_supervised_label(labels: torch.Tensor, tokenizer: Tokenizer) -> bool:
    """
    Whether the shift of this unpadded labels row leaves a single position with a loss.

    The shift drops labels[0] and the collation masks out-of-vocab ids, so the row is trainable iff some labels[1:]
    is a valid id (not `IGNORE_INDEX`). False for a row that is one token long and for an instruct row whose masked
    prompt alone fills training_max_sequence_length + 1.
    """

    tail = labels[1:]
    if tail.numel() == 0:
        return False
    valid = (tail != IGNORE_INDEX) & (tail >= 0) & (tail < tokenizer.vocab_size)
    return bool(valid.any())


def collate_samples(
    batch: list[dict[str, Any]],
    tokenizer: Tokenizer,
    training_max_sequence_length: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[Sample]:
    """
    Format and tokenize dataset rows into unpadded (input_ids, labels, data_id) samples.

    Rows are truncated to training_max_sequence_length + 1 tokens (the shift turns that into training_max_sequence_length positions); rows without
    a supervised label are DROPPED, never raised on: a `StopIteration` out of a collate function ends the worker or
    the epoch, so an unusable row must cost one row, not a loader.
    """

    max_tokens = training_max_sequence_length + 1
    samples: list[Sample] = []
    for row in batch:
        input_ids, labels = apply_formatting(row, tokenizer, add_bos, add_eos)
        if input_ids.shape[0] > max_tokens or labels.shape[0] > max_tokens:
            # cloned, not sliced: a slice is a view that keeps the WHOLE stored row alive (rows are stored cut at
            # dataset_max_sequence_length, trained cut at training_max_sequence_length), and `torch.save` writes a
            # view's whole storage - up to 8x the RAM of the pool and the buffers, the bytes the worker pickles
            # through its queue and the bytes of every checkpoint. The copy is free next to the tokenization above.
            input_ids, labels = input_ids[:max_tokens].clone(), labels[:max_tokens].clone()
        if has_supervised_label(labels, tokenizer):
            samples.append((input_ids, labels, row["data_id"]))
    return samples


def collate_worker_batch(
    batch: list[dict[str, Any]],
    tokenizer: Tokenizer,
    training_max_sequence_length: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> WorkerBatch:
    """
    `collate_samples` plus the count of rows that went in: the collate function of the training loaders.
    rows_read advances by rows read from disk, dropped rows included, the unit a resume skips.
    """

    return WorkerBatch(collate_samples(batch, tokenizer, training_max_sequence_length, add_bos, add_eos), len(batch))


def pad_and_shift(
    samples: list[Sample],
    tokenizer: Tokenizer,
    training_max_sequence_length: int,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Batch:
    """
    Pad `samples` to one width and shift them into a (input_ids, labels, data_ids) micro-batch.

    The width is the longest sample of THIS micro-batch, rounded up to padding_multiple and capped at
    training_max_sequence_length + 1; the shift then drops one position from it. Pad positions are EOS in the inputs and
    ignore_index in the labels, as are labels outside the tokenizer's vocabulary.

    The tensors are pageable on purpose: a pinned micro-batch that was copied to the device carries a CUDA event,
    and freeing it inside a forked DataLoader worker (an epoch restart forks one) aborts the worker with
    "CUDA error: initialization error". The driver stages a copy this small (a few hundred KiB) through its own
    pinned pool anyway, so the non-blocking copy does not wait for queued kernels.
    """

    if not samples:
        raise ValueError("pad_and_shift needs at least one sample; empty micro-batches are never assembled")
    max_tokens = training_max_sequence_length + 1
    longest = max(max(sample_inputs.shape[0], sample_labels.shape[0]) for sample_inputs, sample_labels, _ in samples)
    width = min(find_multiple(longest, padding_multiple) if padding_multiple else longest, max_tokens)

    inputs = torch.full((len(samples), width), tokenizer.eos_id, dtype=torch.long)
    labels = torch.full((len(samples), width), ignore_index, dtype=torch.long)
    for row, (sample_inputs, sample_labels, _) in enumerate(samples):
        inputs[row, : min(len(sample_inputs), width)] = sample_inputs[:width]
        labels[row, : min(len(sample_labels), width)] = sample_labels[:width]

    input_ids, label_ids = shift_inputs_and_labels(inputs, labels)
    mask_label_ids(label_ids, tokenizer, ignore_index)
    return Batch(input_ids, label_ids, [data_id for _, _, data_id in samples])


def collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: Tokenizer,
    training_max_sequence_length: int,
    padding_multiple: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Batch:
    """
    `collate_samples` followed by `pad_and_shift`: the collate function of the validation loaders. A batch in which
    every row was dropped is an error (`training_max_sequence_length` too small for a validation batch is a configuration mistake).
    """

    samples = collate_samples(batch, tokenizer, training_max_sequence_length, add_bos, add_eos)
    if not samples:
        raise ValueError(
            f"every row of the batch was dropped: none of the {len(batch)} rows keeps a supervised label within "
            f"training_max_sequence_length + 1 = {training_max_sequence_length + 1} tokens"
        )
    return pad_and_shift(samples, tokenizer, training_max_sequence_length, padding_multiple, ignore_index)
