# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Row -> (input_ids, labels) formatting functions, selected by data_signature["format_fn"].

Every function returns two equal-length torch.long tensors. Positions that must not be supervised are
`IGNORE_INDEX` in labels, never a token id, so a real `<unk>` or `<pad>` token stays supervised. Two formats exist:
pass_text (pretrain sources) and concatenate_instruction_input_output (instruct sources,
`INSTRUCT_DATA_SIGNATURE` of the dataset resolver).
"""

from typing import Any, Callable

import torch

from data_preparation.lib.stages.row_pipeline import instruct_text
from training.data.tokenizer import IGNORE_INDEX, Tokenizer

Row = dict[str, Any]
FormatFn = Callable[[Row, Tokenizer, bool, bool], tuple[torch.Tensor, torch.Tensor]]


def pass_text(row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Plain language modelling on the text column; every token is supervised.
    """

    text = row.get("text")
    if text is None:
        raise ValueError(f"Dataset row has no 'text' field. Row keys: {list(row.keys())}")
    input_ids = torch.tensor(tokenizer.encode(text, bos=add_bos, eos=add_eos), dtype=torch.long)
    return input_ids, input_ids.clone()


def concatenate_instruction_input_output(
    row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    instruction [+ input] form the prompt (masked); only output is supervised. The full text is
    data_preparation's instruct_text, so the token counts stored at download time are the lengths seen here.
    """

    instruction = row.get("instruction")
    output = row.get("output")
    if instruction is None or output is None:
        raise ValueError(f"Dataset row missing 'instruction' or 'output'. Row keys: {list(row.keys())}")
    input_text = row.get("input") or ""
    prompt = instruction.strip()
    if input_text.strip():
        prompt = prompt + "\n\n" + input_text.strip()
    full = instruct_text(row)

    prompt_len = len(tokenizer.encode(prompt, bos=add_bos, eos=False))
    input_ids = torch.tensor(tokenizer.encode(full, bos=add_bos, eos=add_eos), dtype=torch.long)
    labels = input_ids.clone()
    labels[:prompt_len] = IGNORE_INDEX
    return input_ids, labels


FORMAT_FNS: dict[str, FormatFn] = {
    "pass_text": pass_text,
    "concatenate_instruction_input_output": concatenate_instruction_input_output,
}


def apply_formatting(row: Row, tokenizer: Tokenizer, add_bos: bool, add_eos: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch a dataset row to the format function named in its data_signature.
    """

    return FORMAT_FNS[row["data_signature"]["format_fn"]](row, tokenizer, add_bos, add_eos)
