# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Input guards shared by the native forward pipeline."""

from torch import Tensor

from .generation import GenerationState


def validate_forward_options(
    labels: Tensor | None, return_logits: bool, return_token_losses_chunked_nograd: bool,
    logits_to_keep: int, generation_state: GenerationState | None, use_cache: bool, return_loss_statistics: bool,
) -> None:
    if return_loss_statistics and (labels is None or return_logits or return_token_losses_chunked_nograd):
        raise ValueError("return_loss_statistics requires labels and the loss-only training path")
    if logits_to_keep < 0:
        raise ValueError("logits_to_keep must be nonnegative")
    if logits_to_keep and (labels is not None or return_token_losses_chunked_nograd):
        raise ValueError("logits_to_keep requires inference without labels/token losses")
    if use_cache and generation_state is None:
        raise ValueError("use_cache requires an explicit generation_state")


def validate_sequence_length(input_ids: Tensor, position_ids: Tensor | None, max_sequence_length: int) -> None:
    """Guard the implicit RoPE prefix using a shape-only check under torch.compile.

    Explicit positions can describe packed sequences longer than the table. Checking their tensor values here
    would break the graph; an out-of-range position still fails in the gather.
    """
    sequence_length = input_ids.shape[1]
    if position_ids is None and sequence_length > max_sequence_length:
        raise ValueError(
            f"sequence length {sequence_length} is longer than model_max_sequence_length "
            f"{max_sequence_length} (the RoPE table covers that many positions)"
        )
