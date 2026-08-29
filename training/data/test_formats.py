# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from typing import Any

import pytest
import torch

from training.data.formats import (
    FORMAT_FNS,
    apply_chat_template_supervise_all,
    apply_chat_template_supervise_assistant,
    _single_chat_key,
    apply_formatting,
    concatenate_instruction_input_output,
    fix_chat_template_for_masking,
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


# --- chat templates -------------------------------------------------------------------------------------------------

CHAT_ROW: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "tok_10 tok_11"},
        {"role": "assistant", "content": "tok_20 tok_21 tok_22"},
        {"role": "user", "content": "tok_12"},
        {"role": "assistant", "content": "tok_23"},
    ],
    "data_signature": _sig("apply_chat_template_supervise_assistant", ["messages"]),
}


def test_fix_chat_template_wraps_assistant_branch(chat_tokenizer: Tokenizer) -> None:
    proc = chat_tokenizer.processor
    assert "{% generation %}" not in proc.chat_template
    fix_chat_template_for_masking(proc)
    fixed = proc.chat_template
    assert fixed.count("{% generation %}") == 1 and fixed.count("{% endgeneration %}") == 1
    assert fixed.index("'assistant'") < fixed.index("{% generation %}") < fixed.index("{% endgeneration %}")
    fix_chat_template_for_masking(proc)
    assert proc.chat_template == fixed, "must be idempotent"


def test_fix_chat_template_leaves_unrelated_templates_alone(chat_tokenizer: Tokenizer) -> None:
    proc = chat_tokenizer.processor
    proc.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    fix_chat_template_for_masking(proc)
    assert "generation" not in proc.chat_template
    proc.chat_template = None
    fix_chat_template_for_masking(proc)
    assert proc.chat_template is None


def test_supervise_assistant_masks_only_non_assistant_tokens(chat_tokenizer: Tokenizer) -> None:
    inp, lab = apply_chat_template_supervise_assistant(CHAT_ROW, chat_tokenizer, add_bos=True, add_eos=True)
    # user turns render as "tok_1 <content>", assistant turns as "<content> tok_2"
    expected_inp = [4, 13, 14, 23, 24, 25, 5, 4, 15, 26, 5]
    assert inp.tolist() == expected_inp
    pad = chat_tokenizer.pad_id
    assert lab.tolist() == [pad, pad, pad, 23, 24, 25, 5, pad, pad, 26, 5]
    assert inp.shape == lab.shape and inp.dtype == torch.long


def test_supervise_assistant_trains_on_something(chat_tokenizer: Tokenizer) -> None:
    """The upstream bug: without generation tags the mask was all zeros and nothing was supervised."""
    _, lab = apply_chat_template_supervise_assistant(CHAT_ROW, chat_tokenizer, True, True)
    assert (lab != chat_tokenizer.pad_id).sum() == 6


def test_supervise_assistant_ignores_bos_eos_flags(chat_tokenizer: Tokenizer) -> None:
    """The template-tokenised path never adds BOS/EOS (unlike supervise_all, which does); pin that so a change
    in either direction is deliberate."""
    a = apply_chat_template_supervise_assistant(CHAT_ROW, chat_tokenizer, add_bos=True, add_eos=True)
    b = apply_chat_template_supervise_assistant(CHAT_ROW, chat_tokenizer, add_bos=False, add_eos=False)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert chat_tokenizer.bos_id not in a[0].tolist() and chat_tokenizer.eos_id not in a[0].tolist()


def test_supervise_assistant_rejects_non_list(chat_tokenizer: Tokenizer) -> None:
    row = {"messages": "not a list", "data_signature": _sig("apply_chat_template_supervise_assistant", ["messages"])}
    with pytest.raises(ValueError, match="list"):
        apply_chat_template_supervise_assistant(row, chat_tokenizer, True, True)


def test_single_chat_key_returns_the_key() -> None:
    assert _single_chat_key(CHAT_ROW) == "messages"


def test_chat_formats_need_exactly_one_key(chat_tokenizer: Tokenizer) -> None:
    row = {**CHAT_ROW, "data_signature": _sig("apply_chat_template_supervise_all", ["messages", "extra"])}
    with pytest.raises(ValueError, match="exactly one key"):
        apply_chat_template_supervise_all(row, chat_tokenizer, True, True)


def test_supervise_all_supervises_everything(chat_tokenizer: Tokenizer) -> None:
    inp, lab = apply_chat_template_supervise_all(CHAT_ROW, chat_tokenizer, add_bos=True, add_eos=True)
    assert inp.tolist() == [1, 4, 13, 14, 23, 24, 25, 5, 4, 15, 26, 5, 2]
    assert torch.equal(inp, lab)


# --- registry / dispatch ---------------------------------------------------------------------------------------------


def test_registry_contents() -> None:
    assert set(FORMAT_FNS) == {
        "pass_text",
        "concatenate_instruction_input_output",
        "apply_chat_template_supervise_all",
        "apply_chat_template_supervise_assistant",
    }
    assert FORMAT_FNS["pass_text"] is pass_text
    assert FORMAT_FNS["concatenate_instruction_input_output"] is concatenate_instruction_input_output
    assert FORMAT_FNS["apply_chat_template_supervise_assistant"] is apply_chat_template_supervise_assistant


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
