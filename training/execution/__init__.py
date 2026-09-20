# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Stable imports for training setup, resource lifetimes, and artifact helpers."""

from training.execution.artifacts import export_if_requested, run_benchmarks, write_samples
from training.execution.checkpoints import (
    checkpoint_before_failed_step, resolve_resume_checkpoint, restore_checkpoint,
    restore_checkpoint_if_resuming, save_run_checkpoint,
)
from training.execution.lifecycle import (
    build_run_triggers, close_backend_on_exit, close_loaders_on_exit, is_stop_requested,
    open_run_logger, open_training_dataset, prepare_run_stream, select_resume_for_run, validate_resolved_schedule,
)
from training.execution.loop import finish_training_loop, run_and_log_training_step, run_scheduled_inference, save_checkpoint_if_due
from training.execution.setup import (
    build_run_model, build_run_optimizer, build_stage_manager, check_evaluation_recurrences,
    check_sequence_lengths, check_tokenizer_vocabulary, create_backend, get_run_directory,
    prepare_run_directory, record_run_config,
)
from training.execution.state import ResumePoint, RunState

__all__ = [
    "finish_training_loop", "run_and_log_training_step", "run_scheduled_inference", "save_checkpoint_if_due",
    "ResumePoint", "RunState", "build_run_model", "build_run_optimizer", "build_run_triggers", "build_stage_manager",
    "check_evaluation_recurrences", "check_sequence_lengths", "check_tokenizer_vocabulary", "checkpoint_before_failed_step",
    "close_backend_on_exit", "close_loaders_on_exit", "create_backend", "export_if_requested", "get_run_directory",
    "is_stop_requested", "open_run_logger", "open_training_dataset", "prepare_run_directory", "prepare_run_stream",
    "record_run_config", "resolve_resume_checkpoint", "restore_checkpoint", "restore_checkpoint_if_resuming",
    "run_benchmarks", "save_run_checkpoint", "select_resume_for_run", "validate_resolved_schedule", "write_samples",
]
