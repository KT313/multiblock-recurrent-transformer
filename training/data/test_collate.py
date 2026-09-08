# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import inspect
import itertools
from pathlib import Path
from typing import Any

import pytest
import torch

from model import RecurrentGPT
from training.data.collate import (
    collate_fn,
    collate_samples,
    collate_worker_batch,
    find_multiple,
    has_supervised_label,
    pad_and_shift,
    shift_inputs_and_labels,
)
from training.data.datasets import ParquetTextDataset
from training.data.tokenizer import IGNORE_INDEX, Tokenizer

SIG: dict[str, Any] = {"keys": ["text"], "format_fn": "pass_text"}
INSTR_SIG: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}


def _row(text: str, data_id: str = "d") -> dict[str, Any]:
    return {"text": text, "data_signature": SIG, "data_id": data_id}


def _words(n: int, start: int = 0) -> str:
    return " ".join(f"tok_{(start + i) % 256}" for i in range(n))


@pytest.mark.parametrize(("n", "k", "out"), [(0, 8, 0), (8, 8, 8), (9, 8, 16), (1, 4, 4), (17, 16, 32)])
def test_find_multiple(n: int, k: int, out: int) -> None:
    assert find_multiple(n, k) == out


def test_shift_inputs_and_labels() -> None:
    inputs = torch.tensor([[1, 4, 5, 6, 2, 2, 2]])
    labels = torch.tensor([[1, 4, 5, 6, 2, IGNORE_INDEX, IGNORE_INDEX]])
    inp, lab = shift_inputs_and_labels(inputs, labels)
    assert inp.tolist() == [[1, 4, 5, 6, 2, 2]]
    assert lab.tolist() == [[4, 5, 6, 2, IGNORE_INDEX, IGNORE_INDEX]]
    assert inp.dtype == torch.long and lab.dtype == torch.long
    assert inp.is_contiguous() and lab.is_contiguous()


def test_batch_shapes_and_label_shift(tokenizer: Tokenizer) -> None:
    batch = [_row(_words(5), "a"), _row(_words(9, 40), "b")]
    input_ids, labels, data_ids = collate_fn(batch, tokenizer, training_max_sequence_length=128)
    # longest row: bos + 9 + eos = 11 tokens -> shifted length 10
    assert input_ids.shape == labels.shape == (2, 10)
    assert data_ids == ["a", "b"]
    # row b has no padding: labels are exactly the next input token
    assert labels[1, :-1].tolist() == input_ids[1, 1:].tolist()
    assert labels[1, -1] == tokenizer.eos_id
    assert input_ids[1, 0] == tokenizer.bos_id


def test_padding_becomes_ignore_index_and_eos(tokenizer: Tokenizer) -> None:
    batch = [_row(_words(3)), _row(_words(8))]
    input_ids, labels, _ = collate_fn(batch, tokenizer, training_max_sequence_length=128, ignore_index=-100)
    # row 0: bos, 3 tokens, eos = 5 -> inputs [bos t t t eos] + 4 pad->eos ; labels [t t t eos] + 5 ignore
    assert input_ids[0].tolist() == [1, 3, 4, 5, 2, 2, 2, 2, 2]
    assert labels[0].tolist() == [3, 4, 5, 2, -100, -100, -100, -100, -100]
    assert (input_ids >= 0).all()


def test_ignore_index_is_the_default_and_matches_the_model(tokenizer: Tokenizer) -> None:
    """
    `IGNORE_INDEX` is defined once here; `collate_fn` defaults to it and it equals the model's own default.
    """

    assert IGNORE_INDEX == -100 == inspect.signature(RecurrentGPT.__init__).parameters["ignore_index"].default
    _, labels, _ = collate_fn([_row(_words(2)), _row(_words(6))], tokenizer, training_max_sequence_length=128)
    assert (labels == IGNORE_INDEX).sum() == 4


def test_custom_ignore_index(tokenizer: Tokenizer) -> None:
    _, labels, _ = collate_fn([_row(_words(2)), _row(_words(6))], tokenizer, training_max_sequence_length=128, ignore_index=-1)
    assert (labels == -1).sum() == 4
    assert (labels == -100).sum() == 0


def test_prompt_mask_survives_collation(tokenizer: Tokenizer) -> None:
    row = {
        "instruction": "tok_1 tok_2",
        "input": "",
        "output": "tok_20 tok_21",
        "data_signature": INSTR_SIG,
        "data_id": "ft",
    }
    input_ids, labels, _ = collate_fn([row], tokenizer, training_max_sequence_length=128)
    assert input_ids.tolist() == [[1, 4, 5, 23, 24]]
    assert labels.tolist() == [[-100, -100, 23, 24, 2]]


def test_out_of_vocab_labels_become_ignore_index(tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Labels >= vocab_size (or negative) are masked; the model's padded vocab must not train on them.
    """

    monkeypatch.setattr(type(tokenizer), "vocab_size", property(lambda self: 10))
    input_ids, labels, _ = collate_fn([_row("tok_1 tok_2 tok_100 tok_3")], tokenizer, training_max_sequence_length=128)
    # tokens: bos(1) 4 5 103 6 eos(2); labels: 4 5 103 6 2 -> 103 masked
    assert input_ids.tolist() == [[1, 4, 5, 103, 6]]
    assert labels.tolist() == [[4, 5, -100, 6, 2]]


@pytest.mark.parametrize(("n_words", "multiple", "expected_len"), [(5, 8, 8), (7, 8, 16), (7, 4, 12), (7, None, 9)])
def test_padding_multiple(tokenizer: Tokenizer, n_words: int, multiple: int | None, expected_len: int) -> None:
    # raw length = n_words + 2 (bos/eos); padded to multiple; then shifted (-1)
    input_ids, labels, _ = collate_fn([_row(_words(n_words))], tokenizer, training_max_sequence_length=128, padding_multiple=multiple)
    assert input_ids.shape == labels.shape == (1, expected_len - 1)


def test_padding_multiple_capped_at_sequence_length_plus_one(tokenizer: Tokenizer) -> None:
    input_ids, labels, _ = collate_fn([_row(_words(20))], tokenizer, training_max_sequence_length=16, padding_multiple=64)
    assert input_ids.shape == labels.shape == (1, 16)


def test_padding_multiple_not_dividing_cap_still_capped(tokenizer: Tokenizer) -> None:
    # 20 words + bos/eos = 22 -> multiple of 8 = 24 -> capped at 17 -> shifted 16
    input_ids, labels, _ = collate_fn([_row(_words(20))], tokenizer, training_max_sequence_length=16, padding_multiple=8)
    assert input_ids.shape == labels.shape == (1, 16)
    assert (labels == -100).sum() == 0


def test_truncation_at_sequence_length_plus_one(tokenizer: Tokenizer) -> None:
    length = 16
    input_ids, labels, _ = collate_fn([_row(_words(100))], tokenizer, training_max_sequence_length=length)
    assert input_ids.shape == (1, length)
    assert labels.shape == (1, length)
    full = tokenizer.encode(_words(100), bos=True, eos=True)
    assert input_ids[0].tolist() == full[:length]
    assert labels[0].tolist() == full[1 : length + 1]
    assert (labels == -100).sum() == 0


def test_short_and_long_rows_mixed(tokenizer: Tokenizer) -> None:
    length = 16
    input_ids, labels, _ = collate_fn([_row(_words(2)), _row(_words(100))], tokenizer, training_max_sequence_length=length)
    assert input_ids.shape == (2, length)
    assert (labels[0] == -100).sum() == length - 3  # 2 words + eos supervised
    assert (labels[1] == -100).sum() == 0


def test_unknown_tokens_are_supervised(tokenizer: Tokenizer) -> None:
    """
    Id 0 (the synthetic tokenizer's unk, once its pad id too) is a token like any other: it stays in the inputs and
    is a supervised label; only the sentinel marks "no loss".
    """

    batch = [_row("tok_1 zzz tok_2"), _row(_words(6))]
    samples = collate_samples(batch, tokenizer, training_max_sequence_length=128)
    assert samples[0][0].tolist() == [1, 4, 0, 5, 2]
    input_ids, labels, _ = pad_and_shift(samples, tokenizer, training_max_sequence_length=128)
    assert input_ids[0].tolist() == [1, 4, 0, 5, 2, 2, 2]
    assert labels[0].tolist() == [4, 0, 5, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_single_token_row_is_dropped(tokenizer: Tokenizer) -> None:
    """
    One token leaves nothing after the shift, so the row cannot be trained on.
    """

    assert collate_samples([_row("")], tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=True) == []


def test_dropped_rows_do_not_take_the_rest_of_the_batch_with_them(tokenizer: Tokenizer) -> None:
    """
    Regression: the unusable row used to raise `StopIteration` out of the collate function, which torch reads as
    'this worker is finished'. It now costs exactly that one row.
    """

    batch = [_row("tok_1", "bad"), _row(_words(5), "good")]  # a single token leaves nothing after the shift
    assert [s[2] for s in collate_samples(batch, tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=False)] == ["good"]
    _, _, data_ids = collate_fn(batch, tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=False)
    assert data_ids == ["good"]


def test_collate_worker_batch_counts_rows_read_including_dropped(tokenizer: Tokenizer) -> None:
    """
    `rows_read` counts every row that went in, the dropped ones too, while `samples` holds only
    the survivors. Rows read is the unit `BatchStream.consumed_rows` stores and a resume skips.
    """

    batch = [_row(_words(5), "a"), _row("tok_1", "a"), _row(_words(3), "b"), _row("tok_2", "b")]
    samples, rows_read = collate_worker_batch(batch, tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=False)
    assert rows_read == 4
    assert [s[2] for s in samples] == ["a", "b"]
    reference = collate_samples(batch, tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=False)
    assert len(samples) == len(reference)
    assert all(torch.equal(s[0], r[0]) and torch.equal(s[1], r[1]) and s[2] == r[2] for s, r in zip(samples, reference))


def test_collate_worker_batch_counts_a_fully_dropped_batch(tokenizer: Tokenizer) -> None:
    """
    A worker batch whose every row was dropped still reports its rows as read (no samples, no missing rows).
    """

    samples, rows_read = collate_worker_batch(
        [_row("tok_1", "a"), _row("tok_2", "a")], tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=False
    )
    assert samples == [] and rows_read == 2


def test_batch_of_only_dropped_rows_is_an_error(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="every row of the batch was dropped"):
        collate_fn([_row("tok_1")], tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=False)


def test_prompt_only_window_is_dropped(tokenizer: Tokenizer) -> None:
    """
    An instruction row whose prompt alone fills training_max_sequence_length+1 has no supervised label left after truncation.
    """

    row = {"instruction": _words(30), "input": "", "output": "tok_1", "data_signature": INSTR_SIG, "data_id": "ft"}
    assert collate_samples([row], tokenizer, training_max_sequence_length=16) == []
    # one more token of room and the first output token is supervised from the last prompt position
    input_ids, labels, _ = collate_fn([row], tokenizer, training_max_sequence_length=31)
    assert input_ids.shape == (1, 31)
    assert labels[0, :30].tolist() == [-100] * 30 and labels[0, 30].item() == 4


def test_has_supervised_label(tokenizer: Tokenizer) -> None:
    assert has_supervised_label(torch.tensor([IGNORE_INDEX, 4, 5]), tokenizer)
    assert has_supervised_label(torch.tensor([4, 0, 0]), tokenizer)  # id 0 is a token, not padding
    assert not has_supervised_label(torch.tensor([4, IGNORE_INDEX, IGNORE_INDEX]), tokenizer)
    assert not has_supervised_label(torch.tensor([4]), tokenizer)
    assert not has_supervised_label(torch.tensor([], dtype=torch.long), tokenizer)
    assert not has_supervised_label(torch.tensor([4, tokenizer.vocab_size]), tokenizer)


def test_collate_samples_are_unpadded_and_truncated(tokenizer: Tokenizer) -> None:
    samples = collate_samples([_row(_words(5), "a"), _row(_words(40), "b")], tokenizer, training_max_sequence_length=16)
    assert [(s[0].shape[0], s[2]) for s in samples] == [(7, "a"), (17, "b")]  # bos + words + eos, capped at 17
    for input_ids, labels, _ in samples:
        assert input_ids.shape == labels.shape and torch.equal(input_ids, labels)


def test_truncated_samples_carry_no_storage_of_the_untruncated_row(tokenizer: Tokenizer) -> None:
    """
    A truncated sample must be a copy, not a slice: a view keeps the whole stored row alive, and `torch.save`
    writes a view's entire storage. Rows are stored cut at `dataset_max_sequence_length` and trained cut at
    `training_max_sequence_length` (16384 vs 2048 for the shipped run), so views would carry 8x the bytes through
    the worker queue, the buffers, the packing pool and every checkpoint.
    """

    (input_ids, labels, _), = collate_samples([_row(_words(400), "long")], tokenizer, training_max_sequence_length=16)
    assert input_ids.shape == labels.shape == (17,)
    for tensor in (input_ids, labels):
        assert tensor.untyped_storage().nbytes() == 17 * tensor.element_size()
    short = collate_samples([_row(_words(5), "short")], tokenizer, training_max_sequence_length=16)
    assert short[0][0].shape == (7,), "a row that was not truncated is untouched"


def test_collate_fn_is_collate_samples_then_pad_and_shift(tokenizer: Tokenizer) -> None:
    rows = [_row(_words(5), "a"), _row(_words(9, 40), "b")]
    samples = collate_samples(rows, tokenizer, training_max_sequence_length=128)
    expected = pad_and_shift(samples, tokenizer, training_max_sequence_length=128, padding_multiple=16)
    actual = collate_fn(rows, tokenizer, training_max_sequence_length=128, padding_multiple=16)
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1]) and actual[2] == expected[2]


def test_pad_and_shift_width_is_this_micro_batch_only(tokenizer: Tokenizer) -> None:
    """
    The width comes from the samples handed in, never from how a loader grouped them earlier.
    """

    short = collate_samples([_row(_words(5))], tokenizer, training_max_sequence_length=128)
    long = collate_samples([_row(_words(60))], tokenizer, training_max_sequence_length=128)
    assert pad_and_shift(short, tokenizer, training_max_sequence_length=128, padding_multiple=16)[0].shape == (1, 15)
    assert pad_and_shift(short + long, tokenizer, training_max_sequence_length=128, padding_multiple=16)[0].shape == (2, 63)


def test_pad_and_shift_needs_samples(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        pad_and_shift([], tokenizer, training_max_sequence_length=128)


def test_bos_eos_flags(tokenizer: Tokenizer) -> None:
    input_ids, labels, _ = collate_fn([_row(_words(4))], tokenizer, training_max_sequence_length=128, add_bos=False, add_eos=False)
    assert input_ids.tolist() == [[3, 4, 5]]
    assert labels.tolist() == [[4, 5, 6]]


def test_collate_on_real_tiny_rows(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    ds = ParquetTextDataset(tiny_pretrain_dir, "pre")
    rows = list(itertools.islice(iter(ds), 4))
    input_ids, labels, data_ids = collate_fn(rows, tokenizer, training_max_sequence_length=128, padding_multiple=16)
    assert input_ids.shape == labels.shape == (4, 128)
    assert data_ids == ["pre"] * 4
    valid = labels != -100
    assert torch.equal(labels[:, :-1][valid[:, :-1]], input_ids[:, 1:][valid[:, :-1]])
    assert input_ids.min() >= 0 and input_ids.max() < tokenizer.vocab_size
