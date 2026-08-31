# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from typing import Any

import pytest
import torch

from training.data.formats import (
    FORMAT_FNS,
    apply_formatting,
    concatenate_instruction_input_output,
    pass_text,
)
from training.data.tokenizer import Tokenizer


def _sig(fmt: str, keys: list[str] | None = None) -> dict[str, Any]:
    return {"keys": keys or ["text"], "format_fn": fmt}


def test_pass_text_supervises_everything(tokenizer: Tokenizer) -> None:
    inp, lab = pass_text({"text": "tok_1 tok_2 tok_3"}, tokenizer, add_bos=True, add_eos=True)
    assert inp.dtype == torch.long and lab.dtype == torch.long
    assert inp.tolist() == [1, 4, 5, 6, 2]
    assert torch.equal(inp, lab)
    assert inp.data_ptr() != lab.data_ptr()


def test_pass_text_without_specials(tokenizer: Tokenizer) -> None:
    inp, _ = pass_text({"text": "tok_1 tok_2"}, tokenizer, add_bos=False, add_eos=False)
    assert inp.tolist() == [4, 5]


def test_pass_text_missing_text_raises(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="'text'"):
        pass_text({"other": "x"}, tokenizer, True, True)


@pytest.mark.parametrize("with_input", [False, True])
def test_instruction_format_masks_prompt(tokenizer: Tokenizer, with_input: bool) -> None:
    row: dict[str, Any] = {"instruction": "tok_1 tok_2", "output": "tok_20 tok_21 tok_22"}
    if with_input:
        row["input"] = "tok_5"
    inp, lab = concatenate_instruction_input_output(row, tokenizer, add_bos=True, add_eos=True)
    prompt_ids = [1, 4, 5] + ([8] if with_input else [])
    output_ids = [23, 24, 25, 2]
    assert inp.tolist() == prompt_ids + output_ids
    assert lab.tolist() == [tokenizer.pad_id] * len(prompt_ids) + output_ids
    assert inp.shape == lab.shape


def test_instruction_format_empty_input_is_like_no_input(tokenizer: Tokenizer) -> None:
    base = {"instruction": "tok_1", "output": "tok_2"}
    a = concatenate_instruction_input_output(base, tokenizer, True, True)
    b = concatenate_instruction_input_output({**base, "input": "  "}, tokenizer, True, True)
    c = concatenate_instruction_input_output({**base, "input": None}, tokenizer, True, True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])


def test_instruction_format_without_bos_eos(tokenizer: Tokenizer) -> None:
    inp, lab = concatenate_instruction_input_output(
        {"instruction": "tok_1", "output": "tok_2"}, tokenizer, add_bos=False, add_eos=False
    )
    assert inp.tolist() == [4, 5]
    assert lab.tolist() == [tokenizer.pad_id, 5]


def test_instruction_format_missing_fields_raise(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError):
        concatenate_instruction_input_output({"instruction": "x"}, tokenizer, True, True)
    with pytest.raises(ValueError):
        concatenate_instruction_input_output({"output": "x"}, tokenizer, True, True)


# --- registry / dispatch ---------------------------------------------------------------------------------------------


def test_registry_contents() -> None:
    """Only the two formats of the thesis run; the upstream chat-template formats are gone."""
    assert set(FORMAT_FNS) == {"pass_text", "concatenate_instruction_input_output"}
    assert FORMAT_FNS["pass_text"] is pass_text
    assert FORMAT_FNS["concatenate_instruction_input_output"] is concatenate_instruction_input_output


def test_apply_formatting_dispatches(tokenizer: Tokenizer) -> None:
    row = {"text": "tok_1", "data_signature": _sig("pass_text")}
    inp, lab = apply_formatting(row, tokenizer, add_bos=True, add_eos=False)
    assert inp.tolist() == [1, 4]
    row = {"instruction": "tok_1", "output": "tok_2", "data_signature": _sig("concatenate_instruction_input_output")}
    inp, lab = apply_formatting(row, tokenizer, add_bos=False, add_eos=False)
    assert lab.tolist() == [tokenizer.pad_id, 5]


def test_apply_formatting_unknown_format_raises(tokenizer: Tokenizer) -> None:
    with pytest.raises(KeyError):
        apply_formatting({"text": "tok_1", "data_signature": _sig("nope")}, tokenizer, True, True)
