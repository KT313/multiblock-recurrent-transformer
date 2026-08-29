# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Row -> (input_ids, labels) formatting functions, selected by ``data_signature["format_fn"]``.

Every function returns two equal-length ``torch.long`` tensors. Positions that must not be supervised are set to
``tokenizer.pad_id`` in ``labels``; the collate function turns those into the ignore index.
"""

import re
from typing import Any, Callable, cast

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from training.data.tokenizer import Tokenizer

Row = dict[str, Any]
FormatFn = Callable[[Row, Tokenizer, bool, bool], tuple[torch.Tensor, torch.Tensor]]


def pass_text(row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain language modelling on the ``text`` column; every token is supervised."""
    text = row.get("text")
    if text is None:
        raise ValueError(f"Dataset row has no 'text' field. Row keys: {list(row.keys())}")
    input_ids = torch.tensor(tokenizer.encode(text, bos=add_bos, eos=add_eos), dtype=torch.long)
    return input_ids, input_ids.clone()


def concatenate_instruction_input_output(
    row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """``instruction`` [+ ``input``] form the prompt (masked); only ``output`` is supervised."""
    instruction = row.get("instruction")
    output = row.get("output")
    if instruction is None or output is None:
        raise ValueError(f"Dataset row missing 'instruction' or 'output'. Row keys: {list(row.keys())}")
    input_text = row.get("input") or ""
    prompt = instruction.strip()
    if input_text.strip():
        prompt = prompt + "\n\n" + input_text.strip()
    full = prompt + "\n\n" + output.strip()

    prompt_len = len(tokenizer.encode(prompt, bos=add_bos, eos=False))
    input_ids = torch.tensor(tokenizer.encode(full, bos=add_bos, eos=add_eos), dtype=torch.long)
    labels = input_ids.clone()
    labels[:prompt_len] = tokenizer.pad_id
    return input_ids, labels


def _single_chat_key(row: Row) -> str:
    keys: list[str] = row["data_signature"]["keys"]
    if len(keys) != 1:
        raise ValueError("Chat-template formats need exactly one key in data_signature['keys'].")
    return keys[0]


def apply_chat_template_supervise_all(
    row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render the conversation with the tokenizer's chat template and supervise every token."""
    # tokenize=False always renders a single str; the stub's return union covers the tokenized overloads.
    text = cast(str, tokenizer.processor.apply_chat_template(row[_single_chat_key(row)], tokenize=False))
    input_ids = torch.tensor(tokenizer.encode(text, bos=add_bos, eos=add_eos), dtype=torch.long)
    return input_ids, input_ids.clone()


def fix_chat_template_for_masking(processor: PreTrainedTokenizerBase) -> None:
    """Wrap the assistant branch of a Llama-2 style chat template in ``{% generation %}`` tags.

    ``return_assistant_tokens_mask`` only works with those tags; upstream templates lack them, so the mask was all
    zeros and nothing was trained. Idempotent.
    """
    template: str | None = getattr(processor, "chat_template", None)
    if not template or "{% generation %}" in template or "message['role'] == 'assistant'" not in template:
        return
    pattern = r"(\{%\s*elif\s+message\['role'\]\s*==\s*'assistant'\s*%\})(.*?)(\{%\s*endif\s*%\})"
    processor.chat_template = re.sub(
        pattern,
        lambda m: f"{m.group(1)}{{% generation %}}{m.group(2)}{{% endgeneration %}}{m.group(3)}",
        template,
        flags=re.DOTALL,
    )


def apply_chat_template_supervise_assistant(
    row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render the conversation with the chat template and supervise only assistant turns."""
    fix_chat_template_for_masking(tokenizer.processor)
    messages = row[_single_chat_key(row)]
    if not isinstance(messages, list):
        raise ValueError("Chat-template format expects a list of messages.")
    # return_dict=True with tokenize=True always yields a BatchEncoding; the stub's union covers other overloads.
    encoded = cast(
        BatchEncoding,
        tokenizer.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
            return_tensors="pt",
        ),
    )
    input_ids = encoded["input_ids"][0].to(torch.long)
    labels = input_ids.clone()
    assistant_mask = torch.as_tensor(encoded["assistant_masks"][0]).to(torch.bool)
    labels[~assistant_mask] = tokenizer.pad_id
    return input_ids, labels


FORMAT_FNS: dict[str, FormatFn] = {
    "pass_text": pass_text,
    "concatenate_instruction_input_output": concatenate_instruction_input_output,
    "apply_chat_template_supervise_all": apply_chat_template_supervise_all,
    "apply_chat_template_supervise_assistant": apply_chat_template_supervise_assistant,
}


def apply_formatting(row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch a dataset row to the format function named in its ``data_signature``."""
    return FORMAT_FNS[row["data_signature"]["format_fn"]](row, tokenizer, add_bos, add_eos)
