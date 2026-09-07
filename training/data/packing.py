# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Sequence packing: documents laid end to end into one row of a fixed length, never split, each with its own
attention mask block and its own RoPE positions.

`PackPool` is the rolling pool of drawn documents `BatchStream` fills a pack from (first-fit from the front of the
pool), `pack_samples` turns the chosen documents into a `PackedBatch`. Validation still pads its rows
(`training.data.collate`); the two share `Sample`, `IGNORE_INDEX` and the label masking.

Per-document shift, then concatenation: every sample is shifted like a padded row (inputs drop the last token,
labels the first) BEFORE it is appended, so the last input of a document is labelled with that document's own EOS
and no label ever points into the next document. Per document, inputs and labels are identical to the padded row
`pad_and_shift` would make of it; the model sees the same tokens, only side by side instead of row by row.
"""

import logging
from typing import NamedTuple

import torch

from training.data.collate import Sample, mask_label_ids, shift_inputs_and_labels
from training.data.tokenizer import IGNORE_INDEX
from training.data.tokenizer import Tokenizer

log = logging.getLogger(__name__)

# The pool holds at least this many pack lengths of tokens before a pack is filled: the lookahead of the first-fit
# scan, and the window over which `BatchStream` balances the sources' token shares (a document waits up to sixteen
# packs between entering the pool and its pack). It decides which documents share a pack, so it is a fixed constant like
# `TRAIN_LOADER_BATCH_ROWS`, not a setting (a setting would have to be compared on resume).
POOL_TOKEN_FACTOR = 16


class PackedBatch(NamedTuple):
    """
    One packed micro-batch: a single row of `pack_length` tokens.

    `input_ids` / `labels` `(1, L)`: the shifted documents back to back, the tail filled with EOS in the inputs and
    `IGNORE_INDEX` in the labels. `position_ids` `(1, L)`: `0, 1, ...` restarting at every document and at the tail.
    `document_ids` `(1, L)` int32: `0, 1, ...` per document, the tail its own id (a self-attending "pad document",
    so no attention row is ever fully masked). `data_ids`: one per document in pack order (the tail has none).
    `padding_tokens`: the tail length, for the packing-efficiency metric. `data_tokens`: the slots each document
    occupies (`shifted_length`), parallel to `data_ids`; the composition metric counts these, never the tail.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    data_ids: list[str]
    position_ids: torch.Tensor
    document_ids: torch.Tensor
    padding_tokens: int
    data_tokens: list[int]


def shifted_length(sample: Sample) -> int:
    """
    The slots a sample occupies in a pack: its tokens minus one (the next-token shift drops one position).
    """

    return sample[0].shape[0] - 1


class PackPool:
    """
    The rolling pool of drawn documents a pack is filled from.

    `BatchStream` adds documents (in the order it picks their sources) until the pool holds `POOL_TOKEN_FACTOR`
    pack lengths of tokens (`needs_refill`), then `take_pack` walks the pool front to back and takes every document
    that still fits into the pack; the rest stay in the pool, in order, and lead the next pack. A leftover always
    fits an empty pack, so no document waits longer than one pack beyond the pool's lookahead of up to sixteen
    packs. A document longer than the pack can never be placed: `add` drops it
    with a warning (settings make this unreachable: `tokens_per_micro_batch >= training_max_sequence_length`, and documents are
    truncated to `training_max_sequence_length + 1` tokens, i.e. `training_max_sequence_length` slots).

    `state` / `restore`: the pool travels in the checkpoint (`BatchStream.state_dict`), so a resume fills the same
    packs from the same documents.
    """

    def __init__(self, pack_length: int) -> None:
        if pack_length <= 0:
            raise ValueError(f"pack_length must be positive, got {pack_length}")
        self.pack_length = pack_length
        self._samples: list[Sample] = []
        self.tokens = 0  # total shifted length of the pooled samples

    def __len__(self) -> int:
        return len(self._samples)

    def needs_refill(self) -> bool:
        """
        Whether the pool holds fewer than `POOL_TOKEN_FACTOR` pack lengths of tokens.
        """

        return self.tokens < POOL_TOKEN_FACTOR * self.pack_length

    def add(self, sample: Sample) -> bool:
        """
        Append a drawn document; False (and a warning) for one that is longer than the pack and can never be placed.
        """

        length = shifted_length(sample)
        if length > self.pack_length:
            log.warning(
                "Dropping a %d-token document of %r: longer than the pack length %d, it can never be packed. "
                "tokens_per_micro_batch must be >= training_max_sequence_length + 1 tokens per document.",
                length + 1,
                sample[2],
                self.pack_length,
            )
            return False
        self._samples.append(sample)
        self.tokens += length
        return True

    def take_pack(self) -> list[Sample]:
        """
        First-fit from the front: the documents of the next pack, in pool order, removed from the pool.
        """

        room = self.pack_length
        taken: list[Sample] = []
        kept: list[Sample] = []
        for sample in self._samples:
            length = shifted_length(sample)
            if length <= room:
                taken.append(sample)
                room -= length
            else:
                kept.append(sample)
        self._samples = kept
        self.tokens -= self.pack_length - room
        return taken

    def state(self) -> list[Sample]:
        """
        The pooled documents in order (what a checkpoint stores).
        """

        return list(self._samples)

    def restore(self, samples: list[Sample]) -> None:
        """
        Replace the pool's contents with `samples` (what `state` returned), through `add`: a resume with a smaller
        `tokens_per_micro_batch` (`allow_settings_change`) restores documents that no longer fit into a pack, and
        they would sit in front of every pack forever - dropped here with the same warning `add` gives.
        """

        self._samples = []
        self.tokens = 0
        for sample in samples:
            self.add(sample)


def pack_samples(
    samples: list[Sample], pack_length: int, tokenizer: Tokenizer, ignore_index: int = IGNORE_INDEX
) -> PackedBatch:
    """
    The `PackedBatch` of `samples` (in this order) for a pack of `pack_length` tokens.

    Every sample is shifted on its own (`shift_inputs_and_labels`) and appended; the tail is EOS in the inputs,
    `ignore_index` in the labels. Labels get the padded path's masking (`mask_label_ids`): out-of-vocab ids become
    `ignore_index`, masked prompts already are. The tensors are pageable on purpose, see `pad_and_shift`.
    """

    if not samples:
        raise ValueError("pack_samples needs at least one sample; empty packs are never assembled")
    total = sum(shifted_length(sample) for sample in samples)
    if total > pack_length:
        raise ValueError(f"the samples occupy {total} slots but the pack holds {pack_length}")

    input_ids = torch.full((1, pack_length), tokenizer.eos_id, dtype=torch.long)
    labels = torch.full((1, pack_length), ignore_index, dtype=torch.long)
    position_ids = torch.zeros((1, pack_length), dtype=torch.long)
    document_ids = torch.full((1, pack_length), len(samples), dtype=torch.int32)  # the tail's id unless overwritten

    offset = 0
    for document, (sample_inputs, sample_labels, _) in enumerate(samples):
        shifted_inputs, shifted_labels = shift_inputs_and_labels(sample_inputs[None], sample_labels[None])
        length = shifted_inputs.shape[1]
        input_ids[0, offset : offset + length] = shifted_inputs[0]
        labels[0, offset : offset + length] = shifted_labels[0]
        position_ids[0, offset : offset + length] = torch.arange(length)
        document_ids[0, offset : offset + length] = document
        offset += length
    position_ids[0, offset:] = torch.arange(pack_length - offset)
    mask_label_ids(labels, tokenizer, ignore_index)
    return PackedBatch(
        input_ids,
        labels,
        [data_id for _, _, data_id in samples],
        position_ids,
        document_ids,
        pack_length - offset,
        [shifted_length(sample) for sample in samples],
    )
