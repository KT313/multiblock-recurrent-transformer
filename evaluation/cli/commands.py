# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Run and report the sample and benchmark operations selected by the evaluation CLI."""

import argparse

from evaluation.benchmarks import benchmarks_path, evaluate_on_benchmarks
from evaluation.cli.checkpoint import CheckpointEvaluation
from evaluation.prompts import load_prompts
from evaluation.samples import generate_and_save_samples, samples_path


def run_checkpoint_samples(arguments: argparse.Namespace, evaluation: CheckpointEvaluation) -> None:
    # generate and save completions for the requested prompts
    path = samples_path(evaluation.out_dir, evaluation.step)
    samples = generate_and_save_samples(
        evaluation.model, evaluation.tokenizer, path, step=evaluation.step,
        prompts=load_prompts(file=arguments.prompts_file), max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature, recurrences=evaluation.recurrences, batch_size=arguments.batch_size,
        seed=arguments.seed, execution_policy=evaluation.execution_policy, use_cache=not arguments.no_sample_cache,
    )

    # print the saved samples in their original order
    print(f"{len(samples)} samples written to {path}")
    for sample in samples:
        print(f"--- [{sample.kind}, recurrence {sample.recurrence or 'mean'}] {sample.prompt!r}\n{sample.completion}")


def run_checkpoint_benchmarks(arguments: argparse.Namespace, evaluation: CheckpointEvaluation, tasks: list[str]) -> None:
    # score the requested tasks and save their complete results
    path = benchmarks_path(evaluation.out_dir, evaluation.step)
    metrics = evaluate_on_benchmarks(
        evaluation.model, evaluation.tokenizer, tasks, num_fewshot=arguments.num_fewshot, limit=arguments.limit,
        batch_size=arguments.batch_size, recurrences=evaluation.recurrences, out_path=path, step=evaluation.step,
        seed=arguments.seed, execution_policy=evaluation.execution_policy,
    )

    # print the flattened metrics
    print(f"benchmark results written to {path}")
    for name, value in metrics.items():
        print(f"  {name:40s} {value:.4f}")
