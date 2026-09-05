# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Evaluation between optimizer steps: the validation loss at every `partial_depth_eval` depth and at the model's
mean recurrence, and the rule that says when it runs.

Numerics: every forward consumes the global torch RNG, so the number and order of validation forwards is part of
the training numerics. One pass over the loader, every depth scored on each batch before the next is fetched
(depths in `partial_depth_eval` order, the mean recurrence last), at most `eval_iters` batches. The golden run in
`test_run.py` fails on any change.
"""

from collections.abc import Iterable
from itertools import islice
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module

from model.model import RecurrentGPT
from training.backend.base import Backend, plain_model
from training.data.collate import Batch
from training.settings import Settings
from training.stage_manager import StageManager


@torch.no_grad()
def evaluate(settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]) -> dict[str, Tensor]:
    """
    Validation loss at every depth in `partial_depth_eval` and at the model's mean recurrence.

    Returns `val_loss` / `val_ppl` (mean recurrence) plus `val_loss_<depth>` / `val_ppl_<depth>` per depth. The mean
    is over the batches actually seen (at most `eval_iters`), all-reduced; a loader that yields no batch is an error.
    `val_loss/<data id>`: the per-token loss at the mean recurrence per validation source (the batch's data ids),
    computed from the same forward's per-token losses (`token_losses`), so `val_loss` itself is unchanged. Those
    per-token losses come from the model's chunked loss (`return_token_losses_chunked_nograd`), which never holds
    the full logits: that is what keeps the validation peak small.
    """

    model.eval()
    config = plain_model(model).config
    mean_recurrence = cast(list[int], config.mean_recurrence)  # broadcast to a list in RecurrentConfig.__post_init__
    depths: list[int | list[int]] = [*settings.partial_depth_eval, mean_recurrence]
    steps_per_depth = [
        [(block_depth, 0) for block_depth in depth] if isinstance(depth, list) else [(depth, 0)] * len(mean_recurrence)
        for depth in depths
    ]
    loss_sums = torch.zeros(len(depths), device=backend.device)
    source_token_losses: dict[str, Tensor] = {}  # data id -> (summed token loss, token count) at the mean recurrence
    number_of_batches_seen = 0
    for input_ids, labels, data_ids in islice(val_loader, settings.eval_iters):
        input_ids, labels = input_ids.to(backend.device), labels.to(backend.device)
        token_losses: Tensor | None = None  # the last depth's (the mean recurrence)
        for depth_idx, steps in enumerate(steps_per_depth):
            with backend.autocast():
                # token losses at every depth: the chunked loss path, which never builds the full logits
                output = model(input_ids, labels=labels, num_steps=steps, return_token_losses_chunked_nograd=True)
            loss_sums[depth_idx] += output["loss"]
            token_losses = output["token_losses"]
        assert token_losses is not None
        _add_source_token_losses(source_token_losses, plain_model(model), token_losses, labels, data_ids)
        number_of_batches_seen += 1
    if number_of_batches_seen == 0:
        model.train()  # leave the model as it was found, whichever way this returns
        raise RuntimeError(
            "the validation loader yielded no batch: there is nothing to compute a validation loss from. Its "
            "stage has no validation rows left after the split (or fewer than one micro-batch per rank); give the "
            "stage a larger validation source, raise validation_fraction in the dataset config, or lower "
            f"micro_batch_size ({settings.micro_batch_size})"
        )
    losses = backend.all_reduce(loss_sums / number_of_batches_seen)
    metrics = {"val_loss": losses[-1], "val_ppl": losses[-1].exp()}
    for depth_idx, depth in enumerate(depths):
        metrics[f"val_loss_{depth}"] = losses[depth_idx]
        metrics[f"val_ppl_{depth}"] = losses[depth_idx].exp()
    # every rank must have seen the same data ids: the stacked sums are reduced by position
    per_source = backend.all_reduce(
        torch.stack([source_token_losses[data_id] for data_id in sorted(source_token_losses)])
    )
    for data_id, (loss_sum, token_count) in zip(sorted(source_token_losses), per_source):
        metrics[f"val_loss/{data_id}"] = loss_sum / token_count
    model.train()
    return metrics


def _add_source_token_losses(
    sums: dict[str, Tensor], model: RecurrentGPT, token_losses: Tensor, labels: Tensor, data_ids: list[str]
) -> None:
    """
    Add each row's summed token loss (`token_losses`: the model's `(B, S)` per-token losses, zero where ignored)
    and token count (the model's label masking) to its data id's entry.
    """

    counted = model.mask_labels(labels) != model.ignore_index
    for row, data_id in enumerate(data_ids):
        entry = torch.stack([token_losses[row].sum(), counted[row].sum().to(token_losses.dtype)])
        sums[data_id] = sums[data_id] + entry if data_id in sums else entry


def is_evaluation_step(settings: Settings, completed_steps: int, stage_manager: StageManager) -> bool:
    """
    Whether to evaluate after `completed_steps` completed optimizer steps: every `eval_step_interval` steps and after the
    last step.
    """

    return completed_steps % settings.eval_step_interval == 0 or completed_steps >= stage_manager.total_steps
