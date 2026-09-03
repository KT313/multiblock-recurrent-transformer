# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Evaluation between optimizer steps: the validation loss at every `partial_depth_eval` depth and at the model's
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

from training.backend.base import Backend, plain_model
from training.data.collate import Batch
from training.settings import Settings
from training.stage_manager import StageManager


@torch.no_grad()
def evaluate(settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]) -> dict[str, Tensor]:
    """Validation loss at every depth in `partial_depth_eval` and at the model's mean recurrence.

    Returns `val_loss` / `val_ppl` (mean recurrence) plus `val_loss_<depth>` / `val_ppl_<depth>` per depth. The mean
    is over the batches actually seen (at most `eval_iters`), all-reduced; a loader that yields no batch is an error.
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
    number_of_batches_seen = 0
    for input_ids, labels, _ in islice(val_loader, settings.eval_iters):
        input_ids, labels = input_ids.to(backend.device), labels.to(backend.device)
        for depth_idx, steps in enumerate(steps_per_depth):
            with backend.autocast():
                loss_sums[depth_idx] += model(input_ids, labels=labels, num_steps=steps)["loss"]
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
    model.train()
    return metrics


def is_evaluation_step(settings: Settings, completed_steps: int, stage_manager: StageManager) -> bool:
    """Whether to evaluate after `completed_steps` completed optimizer steps: every `eval_step_interval` steps and after the
    last step."""
    return completed_steps % settings.eval_step_interval == 0 or completed_steps >= stage_manager.total_steps
