# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Sample generations and benchmark scores for a `RecurrentGPT`, during training or on a saved checkpoint.

`generate_and_save_samples` writes what the model produces for a fixed set of prompts, `evaluate_on_benchmarks`
scores it with lm-eval-harness (the `eval` extra). Both leave the training numerics alone (`wrapper.isolated_inference`).
The CLI `evaluation/evaluate.py` runs them on a checkpoint.
"""

from evaluation.benchmarks import DEFAULT_TASKS, benchmarks_path, evaluate_on_benchmarks
from evaluation.prompts import DEFAULT_PROMPTS, Prompt, load_prompts
from evaluation.samples import GeneratedSample, generate_and_save_samples, generate_samples, samples_path

__all__ = [
    "DEFAULT_PROMPTS",
    "DEFAULT_TASKS",
    "GeneratedSample",
    "Prompt",
    "benchmarks_path",
    "evaluate_on_benchmarks",
    "generate_and_save_samples",
    "generate_samples",
    "load_prompts",
    "samples_path",
]
