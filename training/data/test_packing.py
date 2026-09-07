# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `training.data.packing`: the pool (refill target, first-fit from the front, oversized documents, state)
and `pack_samples` (layout, per-document equality with the padded path, label masking, the tail).
"""

import itertools
import logging
from pathlib import Path

import pytest
import torch

from training.data.collate import Sample, collate_samples, pad_and_shift
from training.data.datasets import ParquetTextDataset
from training.data.packing import POOL_TOKEN_FACTOR, PackedBatch, PackPool, pack_samples, shifted_length
from training.data.tokenizer import IGNORE_INDEX, Tokenizer

INSTRUCT_SIGNATURE = {"keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output"}

# the synthetic tokenizer: <pad>=0, <bos>=1, <eos>=2, tok_i = 3 + i (vocab 259)
FIRST_WORD = 3


def _sample(slots: int, tag: str = "d", start: int = 0) -> Sample:
    """
    A supervised document occupying `slots` positions of a pack: `slots + 1` tokens (the shift drops one), bos first,
    eos last, distinct word ids in between.
    """

    ids = torch.tensor([1, *(FIRST_WORD + (start + i) % 250 for i in range(slots - 1)), 2], dtype=torch.long)
    return ids, ids.clone(), tag


def test_shifted_length_is_tokens_minus_one() -> None:
    assert shifted_length(_sample(7)) == 7
    assert _sample(7)[0].shape[0] == 8


# --- the pool ----------------------------------------------------------------------------------------------------------


def test_pool_first_fit_from_the_front_follows_the_worked_example() -> None:
    """
    The example the fill policy was decided on (pack 8000: A 2000, B 500, C 4000, D 100, E 1000, F 3000, G 300,
    H 2500, I 50, J 1700, K 3000), scaled by 1/25: A..E fill 7600 of 8000, F does not fit, G does, H does not, I
    does, J and K do not. The leftovers F H J K stay in draw order and lead the next pack.
    """

    pool = PackPool(pack_length=320)
    slots = {"A": 80, "B": 20, "C": 160, "D": 4, "E": 40, "F": 120, "G": 12, "H": 100, "I": 2, "J": 68, "K": 120}
    for tag, n in slots.items():
        assert pool.add(_sample(n, tag))
    assert pool.tokens == sum(slots.values()) == 726  # the refill target is not this test's concern

    taken = pool.take_pack()
    assert [tag for _, _, tag in taken] == ["A", "B", "C", "D", "E", "G", "I"]
    assert sum(shifted_length(s) for s in taken) == 318  # 2 slots of tail
    assert [tag for _, _, tag in pool.state()] == ["F", "H", "J", "K"]
    assert pool.tokens == 120 + 100 + 68 + 120 and len(pool) == 4

    second = pool.take_pack()  # the leftovers lead: F (120) H (100) fit, J (68) fits, K does not
    assert [tag for _, _, tag in second] == ["F", "H", "J"]
    assert [tag for _, _, tag in pool.state()] == ["K"]


def test_pool_refill_target_is_sixteen_pack_lengths() -> None:
    """
    The pool is refilled until it holds sixteen pack lengths: the lookahead of the first-fit scan and the window the
    stream balances the sources' token shares over.
    """

    pool = PackPool(pack_length=10)
    assert POOL_TOKEN_FACTOR == 16 and pool.needs_refill()
    for _ in range(15):
        pool.add(_sample(10))
    assert pool.tokens == 150 and pool.needs_refill()  # 150 < 160
    pool.add(_sample(9))
    assert pool.needs_refill()  # 159 < 160
    pool.add(_sample(1))
    assert not pool.needs_refill()  # 160


def test_a_leftover_always_fits_the_next_pack() -> None:
    """
    A document as long as the pack waits at most one pack: it cannot share a pack with anything, but it leads the
    next one on its own.
    """

    pool = PackPool(pack_length=10)
    for tag, n in (("small", 3), ("full", 10), ("tiny", 1)):
        pool.add(_sample(n, tag))
    assert [tag for _, _, tag in pool.take_pack()] == ["small", "tiny"]
    assert [tag for _, _, tag in pool.take_pack()] == ["full"]
    assert pool.take_pack() == [] and pool.tokens == 0


def test_pool_drops_an_oversized_document_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    pool = PackPool(pack_length=8)
    with caplog.at_level(logging.WARNING, logger="training.data.packing"):
        assert not pool.add(_sample(9, "too-long"))
    assert len(pool) == 0 and pool.tokens == 0
    assert "Dropping a 10-token document of 'too-long'" in caplog.text and "pack length 8" in caplog.text
    assert pool.add(_sample(8, "fits"))  # exactly the pack length is fine


def test_pool_state_round_trip() -> None:
    pool = PackPool(pack_length=50)
    for tag, n in (("a", 5), ("b", 7), ("c", 11)):
        pool.add(_sample(n, tag))
    state = pool.state()
    restored = PackPool(pack_length=50)
    restored.restore(state)
    assert restored.tokens == pool.tokens == 23 and len(restored) == 3
    assert [tag for _, _, tag in restored.state()] == ["a", "b", "c"]
    assert [tag for _, _, tag in restored.take_pack()] == ["a", "b", "c"]
    state.clear()  # `state` handed out a copy
    assert len(pool) == 3


def test_pool_rejects_a_non_positive_pack_length() -> None:
    with pytest.raises(ValueError, match="pack_length must be positive"):
        PackPool(0)


# --- pack_samples --------------------------------------------------------------------------------------------------


def test_pack_layout(tokenizer: Tokenizer) -> None:
    """
    Two documents of 4 and 3 slots in a pack of 10: their shifted tokens back to back, positions restarting per
    document and for the tail, document ids 0, 1 and 2 (the tail), EOS inputs and ignored labels in the tail.
    """

    a, b = _sample(4, "a", start=0), _sample(3, "b", start=100)
    pack = pack_samples([a, b], 10, tokenizer)
    assert isinstance(pack, PackedBatch)
    assert pack.input_ids.shape == pack.labels.shape == pack.position_ids.shape == pack.document_ids.shape == (1, 10)
    assert pack.input_ids.dtype == pack.labels.dtype == pack.position_ids.dtype == torch.long
    assert pack.document_ids.dtype == torch.int32
    assert pack.input_ids[0].tolist() == [*a[0][:-1].tolist(), *b[0][:-1].tolist(), 2, 2, 2]
    assert pack.labels[0].tolist() == [*a[1][1:].tolist(), *b[1][1:].tolist(), IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert pack.position_ids[0].tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
    assert pack.document_ids[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert pack.data_ids == ["a", "b"] and pack.padding_tokens == 3
    assert pack.data_tokens == [4, 3] == [shifted_length(a), shifted_length(b)]  # the slots per document, no tail
    assert pack.data_tokens == torch.bincount(pack.document_ids[0].long(), minlength=3)[:2].tolist()
    input_ids, labels, data_ids, *_ = pack  # the first three fields unpack like a padded `Batch`
    assert torch.equal(input_ids, pack.input_ids) and torch.equal(labels, pack.labels) and data_ids == ["a", "b"]


def test_a_full_pack_has_no_tail(tokenizer: Tokenizer) -> None:
    pack = pack_samples([_sample(6, "a"), _sample(4, "b")], 10, tokenizer)
    assert pack.padding_tokens == 0 and pack.document_ids[0].tolist() == [0] * 6 + [1] * 4
    assert pack.data_tokens == [6, 4]
    assert (pack.labels != IGNORE_INDEX).all()


def test_every_document_equals_its_padded_row(tokenizer: Tokenizer) -> None:
    """
    The packed path is the padded path per document: the slice of a pack that holds a document is exactly the
    (unpadded) row `pad_and_shift` makes of that document alone.
    """

    samples = [_sample(5, "a", 0), _sample(2, "b", 7), _sample(9, "c", 20)]
    pack = pack_samples(samples, 20, tokenizer)
    offset = 0
    for sample in samples:
        row_inputs, row_labels, _ = pad_and_shift([sample], tokenizer, training_max_sequence_length=64)
        n = row_inputs.shape[1]
        assert torch.equal(pack.input_ids[0, offset : offset + n], row_inputs[0])
        assert torch.equal(pack.labels[0, offset : offset + n], row_labels[0])
        offset += n
    assert offset == 16 and pack.padding_tokens == 4


def test_prompt_masks_and_out_of_vocab_labels_are_ignored(tokenizer: Tokenizer) -> None:
    """
    An instruct row masks its prompt with the ignore index; the collation turns a label outside the vocabulary into
    it too. Both survive the packing, at the document's own positions.
    """

    ids, labels, tag = _sample(6, "instruct")
    labels = labels.clone()
    labels[:3] = IGNORE_INDEX  # bos and two prompt tokens
    labels[5] = tokenizer.vocab_size  # out of range
    pack = pack_samples([_sample(2, "a"), (ids, labels, tag)], 10, tokenizer)
    doc = pack.labels[0, 2:8]  # the instruct document's six slots
    assert doc.tolist() == [IGNORE_INDEX, IGNORE_INDEX, *labels[3:5].tolist(), IGNORE_INDEX, labels[6].item()]
    assert (pack.input_ids >= 0).all(), "the sentinel never reaches the inputs"


def test_pack_samples_rejects_empty_and_overfull_packs(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        pack_samples([], 10, tokenizer)
    with pytest.raises(ValueError, match="occupy 11 slots but the pack holds 10"):
        pack_samples([_sample(6), _sample(5)], 10, tokenizer)


def test_custom_ignore_index(tokenizer: Tokenizer) -> None:
    pack = pack_samples([_sample(3)], 5, tokenizer, ignore_index=-7)
    assert pack.labels[0, 3:].tolist() == [-7, -7]


def test_real_instruct_rows_keep_every_supervised_label(tokenizer: Tokenizer, tiny_instruct_dir: Path) -> None:
    """
    Instruct rows carry their labels at the END of the row (the prompt is masked): packing keeps every supervised
    position of every document, whatever its prompt length.
    """

    rows = list(itertools.islice(iter(ParquetTextDataset(tiny_instruct_dir, "ft", INSTRUCT_SIGNATURE)), 8))
    samples = collate_samples(rows, tokenizer, training_max_sequence_length=255)
    expected = sorted(int((labels[1:] != IGNORE_INDEX).sum()) for _, labels, _ in samples)
    assert len(samples) == 8 and expected[0] > 0
    pack = pack_samples(samples, 8 * 255, tokenizer)
    supervised = [
        int((pack.labels[0, pack.document_ids[0] == document] != IGNORE_INDEX).sum()) for document in range(len(samples))
    ]
    assert sorted(supervised) == expected
