# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Evaluation between optimizer steps: the validation loss at every `partial_depth_eval` depth and at the model's
mean recurrence, and the rule that says when it runs.

Numerics: every forward consumes the global torch RNG, but `evaluate` runs under `torch.random.fork_rng`, so the
generators are restored afterwards and the training stream continues as if no validation had run: how often and how
much validation runs does not change the training numbers. What is fixed is the shape of one evaluation: one pass
over the loader, every depth scored on each batch before the next is fetched (depths in `partial_depth_eval` order,
the mean recurrence last), at most `eval_iters_per_rank` batches on each rank. The per-depth losses are the mean of
the per-rank means (exact: every rank scores the same number of batches), the per-source losses are summed over the
ranks as gathered objects, so the ranks need not have seen the same sources. The golden run in `test_run.py` pins
the reported losses.
"""

from collections.abc import Iterable
from itertools import islice
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module

from evaluation.wrapper import recurrence_label
from model.model import RecurrentGPT
from training.backend.base import Backend
from training.data.collate import Batch
from training.settings import Settings
from training.stage_manager import StageManager


@torch.no_grad()
def evaluate(settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]) -> dict[str, Tensor]:
    """
    Validation loss at every depth in `partial_depth_eval` and at the model's mean recurrence.

    Returns `val_loss` / `val_ppl` (mean recurrence) plus `val_loss_<label>` / `val_ppl_<label>` per depth, the label
    being the depth for a `partial_depth_eval` entry and `recurrence_label` for the mean recurrence ("12-12-12"). The
    mean is over the batches actually seen (at most `eval_iters_per_rank`), all-reduced; a loader that yields no
    batch is an error.
    `val_loss/<data id>`: the per-token loss at the mean recurrence per validation source (the batch's data ids),
    computed from the same forward's per-token losses (`token_losses`), so `val_loss` itself is unchanged. Those
    per-token losses come from the model's chunked loss (`return_token_losses_chunked_nograd`), which never holds
    the full logits: that is what keeps the validation peak small. Every rank's summed token losses and token counts
    per source are gathered (`all_gather_object`) and added, then divided once.
    """

    # the recurrent blocks draw their initial state from the global RNG: validation must not shift the training draws
    devices = [backend.device.index or 0] if backend.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        return _evaluate(settings, backend, model, val_loader)


def _evaluate(settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]) -> dict[str, Tensor]:
    model.eval()
    try:
        return _evaluate_in_eval_mode(settings, backend, model, val_loader)
    finally:
        model.train()  # leave the model as it was found, whichever way this returns


def _evaluate_in_eval_mode(
    settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]
) -> dict[str, Tensor]:
    plain = backend.plain_model(model)
    config = plain.config
    mean_recurrence = cast(list[int], config.mean_recurrence)  # broadcast to a list in RecurrentConfig.__post_init__
    depths: list[int | list[int]] = [*settings.partial_depth_eval, mean_recurrence]
    steps_per_depth = [
        [(block_depth, 0) for block_depth in depth] if isinstance(depth, list) else [(depth, 0)] * len(mean_recurrence)
        for depth in depths
    ]
    loss_sums = torch.zeros(len(depths), device=backend.device)
    source_token_losses: dict[str, Tensor] = {}  # data id -> (summed token loss, token count) at the mean recurrence
    number_of_batches_seen = 0
    for input_ids, labels, data_ids in islice(val_loader, settings.eval_iters_per_rank(backend.world_size)):
        input_ids, labels = input_ids.to(backend.device), labels.to(backend.device)
        token_losses: Tensor | None = None  # the last depth's (the mean recurrence)
        for depth_idx, steps in enumerate(steps_per_depth):
            with backend.autocast():
                # token losses at every depth: the chunked loss path, which never builds the full logits
                output = model(input_ids, labels=labels, num_steps=steps, return_token_losses_chunked_nograd=True)
            loss_sums[depth_idx] += output["loss"]
            token_losses = output["token_losses"]
        assert token_losses is not None
        _add_source_token_losses(source_token_losses, plain, token_losses, labels, data_ids)
        number_of_batches_seen += 1
    if number_of_batches_seen == 0:
        raise RuntimeError(
            "the validation loader yielded no batch: there is nothing to compute a validation loss from. Its "
            "stage has no validation rows left after the split (or fewer than one micro-batch per rank); give the "
            "stage a larger validation source, raise validation_fraction in the dataset config, or lower "
            f"validation_batch_size ({settings.validation_batch_size})"
        )
    losses = backend.all_reduce(loss_sums / number_of_batches_seen)
    metrics = {"val_loss": losses[-1], "val_ppl": losses[-1].exp()}
    for depth_idx, depth in enumerate(depths):
        label = _depth_label(depth)
        metrics[f"val_loss_{label}"] = losses[depth_idx]
        metrics[f"val_ppl_{label}"] = losses[depth_idx].exp()
    # per source: every rank's (summed token loss, token count), gathered as CPU objects and added per data id. A
    # source a rank never saw is simply absent from its dict, so the ranks' key sets need not agree.
    totals: dict[str, Tensor] = {}
    for per_rank in backend.all_gather_object({data_id: entry.cpu() for data_id, entry in source_token_losses.items()}):
        for data_id, entry in per_rank.items():
            totals[data_id] = totals[data_id] + entry if data_id in totals else entry
    for data_id in sorted(totals):
        loss_sum, token_count = totals[data_id]
        metrics[f"val_loss/{data_id}"] = loss_sum / token_count
    return metrics


def _depth_label(depth: int | list[int]) -> str:
    """
    The metric-key suffix of an evaluated depth: the number itself for a `partial_depth_eval` entry (every block
    runs that many iterations), `recurrence_label` for the per-block mean recurrence ("12-12-12", not "[12, 12, 12]":
    wandb keys with brackets, commas and spaces are unusable).
    """

    return str(depth) if isinstance(depth, int) else recurrence_label(depth)


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
