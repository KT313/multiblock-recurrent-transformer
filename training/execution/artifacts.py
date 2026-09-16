# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Optional sample, benchmark, and export artifacts for a training run."""

from __future__ import annotations

from pathlib import Path

from data_preparation.lib.log import get_logger
from evaluation.benchmarks import benchmarks_path, evaluate_on_benchmarks
from evaluation.prompts import load_prompts
from evaluation.samples import GeneratedSample, generate_and_save_samples, samples_path
from training.data.tokenizer import Tokenizer
from training.execution.state import RunState
from training.logger import RunLogger

log = get_logger("training.run")


def write_samples(state: RunState, logger: RunLogger, tokenizer: Tokenizer) -> list[GeneratedSample]:
    """
    Sample generations of the model as it is now, written to `samples/step-XXXXXXXX.jsonl` of the run directory
    and noted by the logger. RNG-isolated: the training numerics do not change. An empty list when generation
    failed (a warning, the run goes on), the policy `run_benchmarks` has: neither is worth a run.
    """

    settings, step = state.settings, state.progress.step
    path = samples_path(state.run_directory, step)
    try:
        with logger.working("generating samples"):
            samples = generate_and_save_samples(
                state.backend.plain_model(state.model),
                tokenizer,
                path,
                step=step,
                prompts=load_prompts(),
                max_new_tokens=settings.sample_max_new_tokens,
                temperature=settings.sample_temperature,
                use_cache=settings.sample_use_cache,
                recurrences=settings.sample_recurrences or [None],
                execution_policy=state.backend.execution_policy,
            )
    except Exception as error:  # a prompt the model cannot take, an OOM in generation: the run must not end on it
        log.warning("sample generation failed, the run continues: %s", error)
        return []
    logger.log_samples(path, samples)
    return samples



def run_benchmarks(state: RunState, logger: RunLogger, tokenizer: Tokenizer) -> dict[str, float] | None:
    """
    lm-eval scores of the model as it is now, written to `benchmarks/step-XXXXXXXX.json` of the run directory and
    logged as `benchmark/<recurrence>/<task>/<metric>`; None when the harness failed (logged as a warning, the run goes on).
    """

    settings, step = state.settings, state.progress.step
    path = benchmarks_path(state.run_directory, step)
    try:
        with logger.working("benchmarking"):
            metrics = evaluate_on_benchmarks(
                state.backend.plain_model(state.model),
                tokenizer,
                settings.benchmark_tasks,
                num_fewshot=settings.benchmark_num_fewshot,
                apply_chat_template=settings.benchmark_apply_chat_template,
                limit=settings.benchmark_limit,
                batch_size=settings.benchmark_batch_size,
                recurrences=settings.benchmark_recurrences or [None],
                out_path=path,
                step=step,
                seed=settings.seed,
                execution_policy=state.backend.execution_policy,
            )
    except Exception as error:  # the harness needs the extra and the network; the run must not end on it
        logger.log_benchmark_failure(error)
        return None
    logger.log_benchmarks(metrics, path, step)
    return metrics



def export_if_requested(state: RunState, logger: RunLogger) -> Path | None:
    """
    With `export_to_hf`: write the HuggingFace folder (`export_hf_path`, default `run_directory / hf_export`), tell
    the logger and return the folder; None otherwise.
    """

    settings = state.settings
    if not settings.export_to_hf:
        return None
    export_dir = Path(settings.export_hf_path) if settings.export_hf_path else state.run_directory / "hf_export"
    logger.status("exporting")
    trained_model = state.backend.plain_model(state.model)
    from model.hf import export_to_hf  # transformers behind it: imported when a run exports, not at start-up

    export_to_hf(
        trained_model, trained_model.config, export_dir, tokenizer_dir=state.dataset.tokenizer_dir,
        execution_policy=state.backend.execution_policy,
    )
    logger.log_export(export_dir)
    return export_dir
