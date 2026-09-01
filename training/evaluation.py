# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Evaluation between optimizer steps: the validation loss at every `partial_depth_eval` depth and at the model's
mean recurrence, and the rule that says when it runs.

Numerics: every forward consumes the global torch RNG (the latent-state draw), so the number and order of validation
forwards between training steps is part of the training numerics. The loop is batch-major: one pass over the
validation loader (one `iter()`, and each `iter(DataLoader)` draws a base seed from the global torch RNG), scoring
every depth on the batch in hand before the next batch is fetched — depths in `partial_depth_eval` order first, the
mean recurrence last, at most `eval_iters` batches, `model.eval()` / `model.train()` around it, `torch.no_grad()`.
Every depth therefore sees exactly the same batches (a paired comparison) and the mean is over the batches actually
delivered, not over the planned `eval_iters` (which a short validation split cannot fill). This departs from the
thesis loop, which iterated the loader once per depth and divided by `eval_iters`; both changes are pinned by
`training/golden_tiny_run.json`.
"""

from collections.abc import Iterable
from itertools import islice
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module

from model import RecurrentGPT
from training.backend import Backend
from training.checkpoint import unwrap_compiled
from training.data.collate import Batch
from training.settings import Settings
from training.stage_manager import StageManager
from training.step import TrainingProgress


@torch.no_grad()
def evaluate(settings: Settings, backend: Backend, model: Module, val_loader: Iterable[Batch]) -> dict[str, Tensor]:
    """Validation loss at every depth in `partial_depth_eval` and at the model's mean recurrence.

    Returns `val_loss` / `val_ppl` (mean recurrence) plus `val_loss_<depth>` / `val_ppl_<depth>` per depth; a depth
    is one `(depth, 0)` pair for every core block (a list of per-block depths for the mean recurrence).

    At most `eval_iters` batches are taken from `val_loader`, each of them scored at every depth: the per-depth
    running sums are divided by the number of batches actually seen (a validation split shorter than
    `eval_iters × micro_batch_size` rows reports the honest mean of the batches it has, not a loss scaled down by
    the missing ones) and all-reduced (identity on one device). A loader that yields no batch at all is an error.
    """
    model.eval()
    config = cast(RecurrentGPT, unwrap_compiled(model)).config
    mean_recurrence = cast(list[int], config.mean_recurrence)  # broadcast to a list in RecurrentConfig.__post_init__
    depths: list[int | list[int]] = [*settings.partial_depth_eval, mean_recurrence]
    steps_per_depth = [
        [(d, 0) for d in depth] if isinstance(depth, list) else [(depth, 0)] * len(mean_recurrence) for depth in depths
    ]
    loss_sums = torch.zeros(len(depths), device=backend.device)
    number_of_batches_seen = 0
    for input_ids, labels, _ in islice(val_loader, settings.eval_iters):
        input_ids, labels = input_ids.to(backend.device), labels.to(backend.device)
        for depth_idx, steps in enumerate(steps_per_depth):
            with backend.autocast():
                loss_sums[depth_idx] += model(input_ids, labels=labels, num_steps_pair=steps)["loss"]
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


def is_evaluation_step(settings: Settings, progress: TrainingProgress, stage_manager: StageManager) -> bool:
    """Whether to evaluate after `progress.done` completed optimizer steps: every `eval_step_interval` steps and
    after the last step."""
    return progress.done % settings.eval_step_interval == 0 or progress.done >= stage_manager.total_steps
