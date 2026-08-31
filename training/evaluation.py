# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Evaluation between optimizer steps: the validation loss at every `partial_depth_eval` depth and at the model's
mean recurrence, and the rule that says when it runs.

Numerics: every forward consumes the global torch RNG (the latent-state draw), so the number and order of validation
forwards between training steps is part of the training numerics. `evaluate` is the thesis `validate` moved unchanged:
depths in `partial_depth_eval` order first, the mean recurrence last, `eval_iters` batches per depth, the validation
loader re-iterated from its start for every depth (each `iter(DataLoader)` draws a base seed from the global torch
RNG — keep it), `model.eval()` / `model.train()` around it, `torch.no_grad()`.
"""

from collections.abc import Iterable
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
    is one `(depth, 0)` pair for every core block (a list of per-block depths for the mean recurrence). The losses
    matrix is `eval_iters × depths`, averaged over the batches and all-reduced (identity on one device).
    """
    model.eval()
    config = cast(RecurrentGPT, unwrap_compiled(model)).config
    mean_recurrence = cast(list[int], config.mean_recurrence)  # broadcast to a list in RecurrentConfig.__post_init__
    depths: list[int | list[int]] = [*settings.partial_depth_eval, mean_recurrence]
    losses = torch.zeros(settings.eval_iters, len(depths), device=backend.device)
    for depth_idx, depth in enumerate(depths):
        steps = [(d, 0) for d in depth] if isinstance(depth, list) else [(depth, 0)] * len(mean_recurrence)
        for k, (input_ids, labels, _) in enumerate(val_loader):
            if k >= settings.eval_iters:
                break
            input_ids, labels = input_ids.to(backend.device), labels.to(backend.device)
            with backend.autocast():
                losses[k, depth_idx] = model(input_ids, labels=labels, num_steps_pair=steps)["loss"]
    losses = backend.all_reduce(losses.mean(dim=0))
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
