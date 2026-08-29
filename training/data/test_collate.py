# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import itertools
from pathlib import Path
from typing import Any

import pytest
import torch

from training.data.collate import collate_fn, find_multiple, shift_inputs_and_labels
from training.data.datasets import ParquetTextDataset
from training.data.tokenizer import Tokenizer

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


def test_shift_inputs_and_labels(tokenizer: Tokenizer) -> None:
    pad = tokenizer.pad_id
    inputs = torch.tensor([[1, 4, 5, 6, 2, pad, pad]])
    labels = inputs.clone()
    inp, lab = shift_inputs_and_labels(inputs, labels, tokenizer)
    assert inp.tolist() == [[1, 4, 5, 6, 2, tokenizer.eos_id]]
    assert lab.tolist() == [[4, 5, 6, 2, pad, pad]]
    assert inp.dtype == torch.long and lab.dtype == torch.long
    assert inp.is_contiguous() and lab.is_contiguous()


def test_batch_shapes_and_label_shift(tokenizer: Tokenizer) -> None:
    batch = [_row(_words(5), "a"), _row(_words(9, 40), "b")]
    input_ids, labels, data_ids = collate_fn(batch, tokenizer, block_size=128)
    # longest row: bos + 9 + eos = 11 tokens -> shifted length 10
    assert input_ids.shape == labels.shape == (2, 10)
    assert data_ids == ["a", "b"]
    # row b has no padding: labels are exactly the next input token
    assert labels[1, :-1].tolist() == input_ids[1, 1:].tolist()
    assert labels[1, -1] == tokenizer.eos_id
    assert input_ids[1, 0] == tokenizer.bos_id


def test_padding_becomes_ignore_index_and_eos(tokenizer: Tokenizer) -> None:
    batch = [_row(_words(3)), _row(_words(8))]
    input_ids, labels, _ = collate_fn(batch, tokenizer, block_size=128, ignore_index=-100)
    # row 0: bos, 3 tokens, eos = 5 -> inputs [bos t t t eos] + 4 pad->eos ; labels [t t t eos] + 5 ignore
    assert input_ids[0].tolist() == [1, 3, 4, 5, 2, 2, 2, 2, 2]
    assert labels[0].tolist() == [3, 4, 5, 2, -100, -100, -100, -100, -100]
    assert (input_ids == tokenizer.pad_id).sum() == 0
    assert (labels == tokenizer.pad_id).sum() == 0


def test_custom_ignore_index(tokenizer: Tokenizer) -> None:
    _, labels, _ = collate_fn([_row(_words(2)), _row(_words(6))], tokenizer, block_size=128, ignore_index=-1)
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
    input_ids, labels, _ = collate_fn([row], tokenizer, block_size=128)
    assert input_ids.tolist() == [[1, 4, 5, 23, 24]]
    assert labels.tolist() == [[-100, -100, 23, 24, 2]]


def test_out_of_vocab_labels_become_ignore_index(tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch) -> None:
    """Labels >= vocab_size (or negative) are masked; the model's padded vocab must not train on them."""
    monkeypatch.setattr(type(tokenizer), "vocab_size", property(lambda self: 10))
    input_ids, labels, _ = collate_fn([_row("tok_1 tok_2 tok_100 tok_3")], tokenizer, block_size=128)
    # tokens: bos(1) 4 5 103 6 eos(2); labels: 4 5 103 6 2 -> 103 masked
    assert input_ids.tolist() == [[1, 4, 5, 103, 6]]
    assert labels.tolist() == [[4, 5, -100, 6, 2]]


@pytest.mark.parametrize(("n_words", "multiple", "expected_len"), [(5, 8, 8), (7, 8, 16), (7, 4, 12), (7, None, 9)])
def test_padding_multiple(tokenizer: Tokenizer, n_words: int, multiple: int | None, expected_len: int) -> None:
    # raw length = n_words + 2 (bos/eos); padded to multiple; then shifted (-1)
    input_ids, labels, _ = collate_fn([_row(_words(n_words))], tokenizer, block_size=128, padding_multiple=multiple)
    assert input_ids.shape == labels.shape == (1, expected_len - 1)


def test_padding_multiple_capped_at_block_size_plus_one(tokenizer: Tokenizer) -> None:
    input_ids, labels, _ = collate_fn([_row(_words(20))], tokenizer, block_size=16, padding_multiple=64)
    assert input_ids.shape == labels.shape == (1, 16)


def test_padding_multiple_not_dividing_cap_still_capped(tokenizer: Tokenizer) -> None:
    # 20 words + bos/eos = 22 -> multiple of 8 = 24 -> capped at 17 -> shifted 16
    input_ids, labels, _ = collate_fn([_row(_words(20))], tokenizer, block_size=16, padding_multiple=8)
    assert input_ids.shape == labels.shape == (1, 16)
    assert (labels == -100).sum() == 0


def test_truncation_at_block_size_plus_one(tokenizer: Tokenizer) -> None:
    block = 16
    input_ids, labels, _ = collate_fn([_row(_words(100))], tokenizer, block_size=block)
    assert input_ids.shape == (1, block)
    assert labels.shape == (1, block)
    full = tokenizer.encode(_words(100), bos=True, eos=True)
    assert input_ids[0].tolist() == full[:block]
    assert labels[0].tolist() == full[1 : block + 1]
    assert (labels == -100).sum() == 0


def test_short_and_long_rows_mixed(tokenizer: Tokenizer) -> None:
    block = 16
    input_ids, labels, _ = collate_fn([_row(_words(2)), _row(_words(100))], tokenizer, block_size=block)
    assert input_ids.shape == (2, block)
    assert (labels[0] == -100).sum() == block - 3  # 2 words + eos supervised
    assert (labels[1] == -100).sum() == 0


def test_all_padding_batch_raises_stop_iteration(tokenizer: Tokenizer) -> None:
    # Every token unknown -> encoded as <pad> (the synthetic tokenizer's unk) -> labels are all pad.
    with pytest.raises(StopIteration, match="padding"):
        collate_fn([_row("zzz yyy")], tokenizer, block_size=128, add_bos=False, add_eos=False)


def test_all_eos_batch_raises_stop_iteration(tokenizer: Tokenizer) -> None:
    with pytest.raises(StopIteration, match="padding"):
        collate_fn([_row("")], tokenizer, block_size=128, add_bos=False, add_eos=True)


def test_prompt_only_window_raises_stop_iteration(tokenizer: Tokenizer) -> None:
    """An instruction row whose prompt alone fills block_size+1 has no supervised label left after truncation."""
    row = {"instruction": _words(30), "input": "", "output": "tok_1", "data_signature": INSTR_SIG, "data_id": "ft"}
    with pytest.raises(StopIteration, match="padding"):
        collate_fn([row], tokenizer, block_size=16)
    # one more token of room and the first output token is supervised from the last prompt position
    input_ids, labels, _ = collate_fn([row], tokenizer, block_size=31)
    assert input_ids.shape == (1, 31)
    assert labels[0, :30].tolist() == [-100] * 30 and labels[0, 30].item() == 4


def test_shift_keeps_pads_when_tokenizer_has_no_eos(tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokenizer, "eos_id", None)
    pad = tokenizer.pad_id
    inp, lab = shift_inputs_and_labels(torch.tensor([[1, 4, pad, pad]]), torch.tensor([[1, 4, pad, pad]]), tokenizer)
    assert inp.tolist() == [[1, 4, pad]] and lab.tolist() == [[4, pad, pad]]


def test_bos_eos_flags(tokenizer: Tokenizer) -> None:
    input_ids, labels, _ = collate_fn([_row(_words(4))], tokenizer, block_size=128, add_bos=False, add_eos=False)
    assert input_ids.tolist() == [[3, 4, 5]]
    assert labels.tolist() == [[4, 5, 6]]


def test_collate_on_real_tiny_rows(tokenizer: Tokenizer, tiny_validation_dir: Path) -> None:
    ds = ParquetTextDataset(tiny_validation_dir, "pre")
    rows = list(itertools.islice(iter(ds), 4))
    input_ids, labels, data_ids = collate_fn(rows, tokenizer, block_size=128, padding_multiple=16)
    assert input_ids.shape == labels.shape == (4, 128)
    assert data_ids == ["pre"] * 4
    valid = labels != -100
    assert torch.equal(labels[:, :-1][valid[:, :-1]], input_ids[:, 1:][valid[:, :-1]])
    assert input_ids.min() >= 0 and input_ids.max() < tokenizer.vocab_size
