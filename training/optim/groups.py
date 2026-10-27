# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Build the checkpoint-stable parameter groups and pin embedding state precision."""

from typing import Any, Iterable, cast

from torch import Tensor
from torch.nn import Module

# Index of the embedding group in `get_param_groups`' output; ELLISAdam8bit keeps its moments in fp32.
EMBEDDING_GROUP = 1


def get_param_groups(
    model: Module, weight_decay: float, no_wd_for_bias_and_norm: bool = True
) -> list[dict[str, Any]]:
    """
    Split parameters into weights / embeddings / scale-and-norm groups, as upstream did.

    Group order matters for checkpoints: 0 = matrices, 1 = embeddings (+ tied lm_head, `EMBEDDING_GROUP`),
    2 = norms, biases and the trainable initial states (`use_trainable_initial_state`; no weight decay pulling
    them toward zero under `no_wd_for_bias_and_norm`).
    """

    weights_group: list[Tensor] = []
    embedding_group: list[Tensor] = []
    scale_and_norm_group: list[Tensor] = []
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if "wte" in name_lower or "embedding" in name_lower or "lm_head" in name_lower:
            embedding_group.append(param)
        elif "ln_f" in name_lower or "norm" in name_lower or "bias" in name_lower or "initial_state" in name_lower:
            scale_and_norm_group.append(param)
        elif "proj" in name_lower or "qkv" in name_lower or "fc" in name_lower or param.ndim == 2:
            weights_group.append(param)
        else:
            raise ValueError(f"param {name} could not be matched to an optim group")

    param_groups = [
        {"params": weights_group, "weight_decay": weight_decay},
        {"params": embedding_group, "weight_decay": weight_decay},
        {"params": scale_and_norm_group, "weight_decay": weight_decay},
    ]
    if no_wd_for_bias_and_norm:
        param_groups[-1]["weight_decay"] = 0.0
    return param_groups


def _pin_embedding_group_to_fp32(params: Iterable[Tensor] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The `get_param_groups` list with `state_bits: 32` set on the embedding group (copies of the group dicts, the
    caller's are left alone). Anything but that list is refused: ELLISAdam8bit needs to know which group holds the
    embeddings.
    """

    groups = list(params)
    if len(groups) != 3 or not all(isinstance(group, dict) for group in groups):
        raise ValueError(
            "ELLISAdam8bit needs the three parameter groups of get_param_groups (the embedding group keeps fp32 "
            "moments), not a plain parameter iterable"
        )
    pinned = [dict(cast(dict[str, Any], group)) for group in groups]
    pinned[EMBEDDING_GROUP]["state_bits"] = 32
    return pinned
