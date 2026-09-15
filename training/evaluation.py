# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Evaluation between optimizer steps: the validation loss at every `partial_depth_eval` depth and at the model's
mean recurrence, and the rule that says when it runs.

Numerics: every forward consumes the global torch RNG, but `evaluate` runs under `torch.random.fork_rng`, so the
generators are restored afterwards and the training stream continues as if no validation had run: how often and how
much validation runs does not change the training numbers. What is fixed is the shape of one evaluation: one pass
over the loader, every depth scored on each batch before the next is fetched (depths in `partial_depth_eval` order,
the mean recurrence last), at most `eval_iters_per_rank` batches on each rank. Per-depth and per-source losses
are global supervised-token means; source sums/counts travel as gathered objects, so ranks need not see the same sources. The golden run in `test_run.py` pins
the reported losses.
"""

import logging
from collections.abc import Iterable
from itertools import islice

import torch
from torch import Tensor
from torch.nn import Module

from evaluation.mode import evaluation_mode
from training.backend.base import Backend
from training.data.collate import Batch
from training.settings import Settings
from training.stage_manager import StageManager
from training.validation import (
    add_source_metrics, add_source_token_losses as _add_source_token_losses, build_depth_metrics,
    check_validation_batches_seen, evaluate_batch_depths, prepare_validation_depths, reduce_validation_losses,
)
from training.validation import format_depth_label as _depth_label  # preserve the existing helper import

__all__ = ["evaluate", "is_evaluation_step", "_add_source_token_losses", "_depth_label"]

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]) -> dict[str, Tensor]:
    """
    Validation loss at every depth in `partial_depth_eval` and at the model's mean recurrence.

    Returns `val_loss` / `val_ppl` (mean recurrence) plus `val_loss_<label>` / `val_ppl_<label>` per depth, the label
    being the depth for a `partial_depth_eval` entry and `recurrence_label` for the mean recurrence ("12-12-12"). The
    mean is over supervised targets actually seen (at most `eval_iters_per_rank` batches), globally summed; an
    empty loader or globally empty supervision is an error.
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
    with evaluation_mode(model):
        return _evaluate_in_eval_mode(settings, backend, model, val_loader)


def _evaluate_in_eval_mode(
    settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]
) -> dict[str, Tensor]:
    # prepare depth order and supervised-token accumulators
    plain = backend.plain_model(model)
    depths, steps_per_depth = prepare_validation_depths(settings, plain)
    loss_sums = torch.zeros(len(depths), device=backend.device)
    supervised_count = torch.zeros((), device=backend.device, dtype=torch.int64)
    source_token_losses: dict[str, tuple[Tensor, Tensor]] = {}
    number_of_batches_seen = 0

    # score every depth on each batch before fetching the next one
    for input_ids, labels, data_ids in islice(val_loader, settings.eval_iters_per_rank(backend.world_size)):
        input_ids, labels = input_ids.to(backend.device), labels.to(backend.device)
        counted = plain.mask_labels(labels) != plain.ignore_index
        supervised_count += counted.sum()
        token_losses = evaluate_batch_depths(backend, model, input_ids, labels, counted, steps_per_depth, loss_sums)
        _add_source_token_losses(source_token_losses, plain, token_losses, labels, data_ids)
        number_of_batches_seen += 1

    # validate local progress and publish globally weighted metrics
    check_validation_batches_seen(number_of_batches_seen, settings)
    losses = reduce_validation_losses(backend, loss_sums, supervised_count)
    metrics = build_depth_metrics(losses, depths)
    add_source_metrics(metrics, backend, source_token_losses, log=log)
    return metrics


def is_evaluation_step(settings: Settings, completed_steps: int, stage_manager: StageManager) -> bool:
    """
    Whether to evaluate after `completed_steps` completed optimizer steps: every `eval_step_interval` steps and after the
    last step.
    """

    return completed_steps % settings.eval_step_interval == 0 or completed_steps >= stage_manager.total_steps
