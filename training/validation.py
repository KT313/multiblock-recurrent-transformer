# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Supporting depth setup and supervised-token metrics for training validation."""

import logging
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module

from evaluation.wrapper import recurrence_label
from model.model import RecurrentGPT
from training.backend.base import Backend
from training.settings import Settings


def prepare_validation_depths(settings: Settings, model: RecurrentGPT) -> tuple[list[int | list[int]], list[list[tuple[int, int]]]]:
    """Keep configured depths in order, followed by the model's per-core mean recurrence."""

    mean_recurrence = cast(list[int], model.config.mean_recurrence)  # broadcast by RecurrentConfig.__post_init__
    depths: list[int | list[int]] = [*settings.partial_depth_eval, mean_recurrence]
    steps_per_depth = [
        [(block_depth, 0) for block_depth in depth] if isinstance(depth, list) else [(depth, 0)] * len(mean_recurrence)
        for depth in depths
    ]
    return depths, steps_per_depth


def evaluate_batch_depths(
    backend: Backend, model: Module, input_ids: Tensor, labels: Tensor, counted: Tensor,
    steps_per_depth: list[list[tuple[int, int]]], loss_sums: Tensor,
) -> Tensor:
    """Accumulate every depth on this batch and return the mean recurrence's token losses."""

    token_losses: Tensor | None = None
    for depth_idx, steps in enumerate(steps_per_depth):
        with backend.autocast():
            output = model(input_ids, labels=labels, num_steps=steps, return_token_losses_chunked_nograd=True)
        token_losses = output["token_losses"]
        assert token_losses is not None
        loss_sums[depth_idx] += token_losses.masked_fill(~counted, 0).sum()
    assert token_losses is not None
    return token_losses


def check_validation_batches_seen(number_of_batches_seen: int, settings: Settings) -> None:
    """Refuse an empty local validation pass before entering metric collectives."""

    if number_of_batches_seen == 0:
        raise RuntimeError(
            "the validation loader yielded no batch: there is nothing to compute a validation loss from. Its "
            "stage has no validation rows left after the split (or fewer than one micro-batch per rank); give the "
            "stage a larger validation source, raise validation_fraction in the dataset config, or lower "
            f"validation_batch_size ({settings.validation_batch_size})"
        )


def reduce_validation_losses(backend: Backend, loss_sums: Tensor, supervised_count: Tensor) -> Tensor:
    """Reduce token sums, then int64 counts, and validate the global supervised-token means."""

    loss_sums = backend.all_reduce(loss_sums, op="sum")
    supervised_count = backend.all_reduce(supervised_count, op="sum")
    if supervised_count.item() == 0:
        raise RuntimeError("Validation contains no supervised target tokens across all ranks")
    losses = loss_sums / supervised_count
    if not torch.isfinite(losses).all():
        raise RuntimeError("Validation supervised-token loss is non-finite")
    return losses


def build_depth_metrics(losses: Tensor, depths: list[int | list[int]]) -> dict[str, Tensor]:
    """Name the default and depth-specific losses and perplexities."""

    metrics = {"val_loss": losses[-1], "val_ppl": losses[-1].exp()}
    for depth_idx, depth in enumerate(depths):
        label = format_depth_label(depth)
        metrics[f"val_loss_{label}"] = losses[depth_idx]
        metrics[f"val_ppl_{label}"] = losses[depth_idx].exp()
    return metrics


def add_source_metrics(
    metrics: dict[str, Tensor], backend: Backend, source_token_losses: dict[str, tuple[Tensor, Tensor]], *, log: logging.Logger,
) -> None:
    """Gather source totals as CPU objects; different ranks may have seen different sources."""

    totals: dict[str, tuple[Tensor, Tensor]] = {}
    payload = {data_id: (entry[0].cpu(), entry[1].cpu()) for data_id, entry in source_token_losses.items()}
    for per_rank in backend.all_gather_object(payload):
        for data_id, (numerator, count) in per_rank.items():
            if data_id in totals:
                previous_sum, previous_count = totals[data_id]
                numerator, count = previous_sum + numerator, previous_count + count
            totals[data_id] = numerator, count
    for data_id in sorted(totals):
        loss_sum, token_count = totals[data_id]
        if token_count.item() == 0:
            log.warning("Validation source %s contains no supervised target tokens; omitting its undefined mean", data_id)
            continue
        metrics[f"val_loss/{data_id}"] = loss_sum / token_count


def format_depth_label(depth: int | list[int]) -> str:
    """
    The metric-key suffix of an evaluated depth: the number itself for a `partial_depth_eval` entry (every block
    runs that many iterations), `recurrence_label` for the per-block mean recurrence ("12-12-12", not "[12, 12, 12]":
    wandb keys with brackets, commas and spaces are unusable).
    """

    return str(depth) if isinstance(depth, int) else recurrence_label(depth)


def add_source_token_losses(
    sums: dict[str, tuple[Tensor, Tensor]], model: RecurrentGPT, token_losses: Tensor, labels: Tensor, data_ids: list[str]
) -> None:
    """
    Add each row's summed token loss (`token_losses`: the model's `(B, S)` per-token losses, zero where ignored)
    and token count (the model's label masking) to its data id's entry.
    """

    counted = model.mask_labels(labels) != model.ignore_index
    for row, data_id in enumerate(data_ids):
        numerator, count = token_losses[row].masked_fill(~counted[row], 0).sum(), counted[row].sum()
        if data_id in sums:
            previous_sum, previous_count = sums[data_id]
            numerator, count = previous_sum + numerator, previous_count + count
        sums[data_id] = numerator, count
