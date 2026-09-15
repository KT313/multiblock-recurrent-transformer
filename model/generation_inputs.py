# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Prepare positions, recurrence counts and masks for stateful generation."""

import torch
from torch import Tensor

from .blocks.recurrence import NumSteps, normalize_num_steps
from .generation import GenerationState
from .layers.attention import AttentionMask


def prepare_generation_inputs(
    input_ids: Tensor, attention_mask: AttentionMask, position_ids: Tensor | None,
    num_steps: NumSteps, state: GenerationState, use_cache: bool,
    num_core_blocks: int, mean_recurrence: int | list[int],
) -> tuple[tuple[int, ...], int, Tensor, Tensor]:
    specs = normalize_num_steps(num_steps, num_core_blocks)
    means = mean_recurrence
    assert isinstance(means, list)
    steps = tuple(means[i] if spec is None else sum(spec) for i, spec in enumerate(specs))
    start = state.get_seq_length() if use_cache else 0
    total = start + input_ids.shape[1]
    if attention_mask is None:
        padding = torch.ones((input_ids.shape[0], total), device=input_ids.device, dtype=torch.bool)
    elif isinstance(attention_mask, Tensor) and attention_mask.dim() == 2:
        padding = attention_mask.to(torch.bool)
    else:
        raise ValueError("generation state supports only a (B, S) padding mask, not packed attention")
    if position_ids is None:
        position_ids = (padding.long().cumsum(-1) - 1).clamp(min=0)[:, start:]
    elif position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
    elif position_ids.dim() == 2 and position_ids.shape[0] == 1:
        position_ids = position_ids.expand(input_ids.shape[0], -1)
    return steps, total, padding, position_ids


def build_generation_mask(input_ids: Tensor, padding: Tensor, start: int, total: int) -> Tensor:
    """Let each appended query attend to valid earlier keys and its own storage column."""
    query_columns = torch.arange(start, total, device=input_ids.device)[:, None]
    key_columns = torch.arange(total, device=input_ids.device)[None, :]
    mask = (padding[:, None, None, :] & (key_columns <= query_columns)) | (key_columns == query_columns)
    return mask
