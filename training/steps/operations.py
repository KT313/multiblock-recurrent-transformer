# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Numerical phases of an optimizer step, preserving accumulation and synchronization order."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from model.layers.attention import document_attention_mask
from model.kernels.runtime import CustomKernelError, DISABLE_HINT
from training.backend.base import Backend
from training.data.packing import PackedBatch
from training.logger import track_gradient_metrics
from training.lr_schedule import get_lr_multistage
from training.settings import Settings
from training.stage_manager import StageManager
from training.steps.state import AccumulatedGradients, NonFiniteLossError, TrainingProgress


def get_scheduled_learning_rate(settings: Settings, stage_manager: StageManager, progress: TrainingProgress) -> float:
    """
    The LR of optimizer step `progress.step`: trapezoid warmup / cooldown over the whole run, the per-stage base
    LR in between (linearly interpolated inside a transition).
    """

    return get_lr_multistage(
        progress.step,
        stage_manager.total_steps,
        stage_manager,
        min_lr=settings.min_lr,
        warmup_steps=settings.warmup_steps,
        cooldown_steps=settings.cooldown_steps,
        schedule=settings.lr_schedule,
    )



def build_model_inputs(batch: PackedBatch, backend: Backend) -> dict[str, Any]:
    """
    The keyword arguments of the model's forward for one packed micro-batch, on the device: `input_ids`, `labels`,
    the per-document `position_ids` and the document attention mask (`document_attention_mask`, built here, outside
    the model's forward and any compiled region).
    """

    return {
        "input_ids": backend.to_device(batch.input_ids),
        "labels": backend.to_device(batch.labels),
        "position_ids": backend.to_device(batch.position_ids),
        "attention_mask": document_attention_mask(backend.to_device(batch.document_ids)),
    }



def run_microbatch_backward(
    settings: Settings, backend: Backend, model: Module, inputs: dict[str, Any], local_capacity: int,
) -> dict[str, Any]:
    """Run forward/backward with the existing error guidance; never retry a partially executed microbatch."""

    try:
        with backend.autocast():
            outputs: dict[str, Any] = model(**inputs, return_loss_statistics=True)
        backend.backward(outputs["loss_sum"] / local_capacity)
    except CustomKernelError:
        raise
    except Exception as error:
        if settings.use_custom_kernels:
            raise CustomKernelError(
                f"Training forward/backward failed with custom kernels enabled: {error}. {DISABLE_HINT}"
            ) from error
        raise
    return outputs


def create_accumulated_gradients(settings: Settings, backend: Backend, accumulation_steps: int) -> AccumulatedGradients:
    """Initialize local loss/count totals and the data composition before reading any batches."""

    loss_sum = torch.zeros((), device=backend.device)
    supervised_count = torch.zeros((), device=backend.device, dtype=torch.int64)
    local_capacity = accumulation_steps * settings.tokens_per_micro_batch
    return AccumulatedGradients(loss_sum, supervised_count, local_capacity, [], {}, 0)


def record_batch_composition(accumulated: AccumulatedGradients, batch: PackedBatch) -> None:
    """Count document slots by source and keep pack tails separate."""

    accumulated.data_ids.extend(batch.data_ids)
    for data_id, tokens in zip(batch.data_ids, batch.data_tokens):
        accumulated.data_tokens[data_id] = accumulated.data_tokens.get(data_id, 0) + tokens
    accumulated.padding_tokens += batch.padding_tokens


def select_gradient_sync_context(
    backend: Backend, model: Module, micro_batch_index: int, accumulation_steps: int,
) -> AbstractContextManager[None]:
    """Suppress DDP synchronization before the final microbatch, preserving its default synchronization."""

    return backend.no_sync(model) if micro_batch_index < accumulation_steps - 1 else nullcontext()


def reduce_supervised_loss(backend: Backend, accumulated: AccumulatedGradients, step: int) -> tuple[Tensor, Tensor]:
    """Reduce loss and int64 token count, rejecting an empty update or non-finite global loss."""

    loss_sum, supervised_count = accumulated.loss_sum, accumulated.supervised_count
    loss_sum = backend.all_reduce(loss_sum, op="sum")
    supervised_count = backend.all_reduce(supervised_count, op="sum")
    if supervised_count.item() == 0:
        raise RuntimeError(f"No supervised target tokens across the optimizer update at step {step}")
    loss = loss_sum / supervised_count
    if not torch.isfinite(loss):
        raise NonFiniteLossError(f"Loss is {loss.item()} at step {step}")

    return loss, supervised_count


def normalize_and_clip_gradients(
    settings: Settings, backend: Backend, model: Module, local_capacity: int, supervised_count: Tensor, step: int,
) -> Tensor:
    """Apply the global supervised-token correction before clipping and checking the gradient norm."""

    correction = (backend.world_size * local_capacity) / supervised_count
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    torch._foreach_mul_(gradients, correction)  # pyright: ignore[reportPrivateImportUsage]  # in-place multi-tensor scaling
    grad_norm = backend.clip_grad_norm(model, settings.grad_clip)
    if not torch.isfinite(grad_norm):
        raise NonFiniteLossError(f"Gradient norm is non-finite at step {step}")
    return grad_norm


def collect_step_metrics(
    settings: Settings, backend: Backend, model: Module, optimizer: Optimizer, step: int, padding_tokens: int,
) -> dict[str, Tensor]:
    """Collect diagnostics at their original intervals, after the update and before clearing gradients."""

    metrics: dict[str, Tensor] = {}
    gradient_interval = settings.log_gradient_metrics_interval
    if gradient_interval > 0 and (step + 1) % gradient_interval == 0:
        metrics = track_gradient_metrics(backend.plain_model(model), optimizer)  # the DDP wrapper hides `.transformer`
    if (step + 1) % settings.log_step_interval == 0:
        metrics["packing/padding_fraction"] = torch.tensor(padding_tokens / settings.tokens_per_optimizer_step)  # pack tails
    return metrics
