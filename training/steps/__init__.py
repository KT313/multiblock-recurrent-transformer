# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Stable imports for optimizer-step state, microbatch streams, and numerical helpers."""

from training.steps.batches import MAX_CONSECUTIVE_REJECTS, PACK_TENSORS, BatchStream, RankBatches
from training.steps.operations import (
    build_model_inputs, collect_step_metrics, create_accumulated_gradients, get_scheduled_learning_rate,
    record_batch_composition, select_gradient_sync_context,
    normalize_and_clip_gradients, reduce_supervised_loss, run_microbatch_backward,
)
from training.steps.state import AccumulatedGradients, NonFiniteLossError, StepResult, TrainingProgress

__all__ = [
    "AccumulatedGradients", "BatchStream", "MAX_CONSECUTIVE_REJECTS", "NonFiniteLossError", "PACK_TENSORS",
    "RankBatches", "StepResult", "TrainingProgress", "build_model_inputs",
    "collect_step_metrics", "get_scheduled_learning_rate", "normalize_and_clip_gradients", "reduce_supervised_loss",
    "run_microbatch_backward", "create_accumulated_gradients", "record_batch_composition", "select_gradient_sync_context",
]
