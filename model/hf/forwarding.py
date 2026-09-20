# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Validate HF forward cache requests and resolve evaluation recurrence defaults."""

import os

import torch
from torch import Tensor

from ..generation import GenerationState
from ..blocks.recurrence import StepsPair, StepsSpec, canon_steps


def prepare_forward_state(
    past_key_values: GenerationState | None, use_cache: bool, training: bool,
    labels: Tensor | None, logits_to_keep: int,
) -> GenerationState | None:
    if past_key_values is not None and not isinstance(past_key_values, GenerationState):
        raise ValueError("past_key_values must be this model's GenerationState")
    if past_key_values is not None and not use_cache:
        raise ValueError("past_key_values requires use_cache=True")
    if use_cache and (training or torch.is_grad_enabled() or labels is not None):
        if past_key_values is not None:
            past_key_values.invalidate()
        raise ValueError("cached generation requires eval, no_grad/inference_mode, and no labels")
    if labels is not None and logits_to_keep:
        raise ValueError("logits_to_keep requires inference without labels")
    state = (past_key_values or GenerationState()) if use_cache else None

    return state


def resolve_evaluation_steps(num_recurrent_blocks: int, mean_recurrences: list[int]) -> StepsPair | list[StepsSpec] | None:
    env_steps = os.environ.get("EVAL_RECURRENCE_STEPS", "").strip()
    if env_steps:
        num_steps = parse_recurrence_steps(env_steps, num_recurrent_blocks)
    else:
        per_block: list[StepsSpec] = []
        for mean_recurrence in mean_recurrences:
            per_block.append((mean_recurrence, 0))
        num_steps = per_block

    return num_steps


def parse_recurrence_steps(steps_str: str, num_blocks: int) -> StepsPair | list[StepsSpec] | None:
    """
    Parse "12" into (12, 0) for all blocks, or "4,12,4" into one (n, 0) pair per block.
    """

    steps_str = steps_str.strip()
    if not steps_str:
        return None
    if "," not in steps_str:
        return canon_steps(int(steps_str))

    per_block: list[StepsSpec] = []
    for value in steps_str.split(","):
        per_block.append(canon_steps(int(value.strip())))
    if len(per_block) != num_blocks:
        raise ValueError(f"got {len(per_block)} recurrence values but the model has {num_blocks} recurrent blocks")
    return per_block
