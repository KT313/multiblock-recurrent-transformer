# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
`train()`: one training run as a readable entry function, plus the setup helpers it is made of.

    create_backend                    device, precision, torch flags; then `seed_everything`
    prepare_run_directory             out_dir/<run_name>/checkpoints, run_config.json
    run_lock                          one training run per out_dir (`data_preparation/lib/build/lock.py`)
    resolve_dataset                   verify / auto-prepare the dataset config, validation split, tokenizer dir
    build_stage_manager               token budgets -> optimizer-step boundaries, transitions, per-stage base LR/weights
    build_run_dataloaders             one train loader per SOURCE (whole run), one validation loader per stage
    build_run_model                   architecture yaml + overrides, sequence-length check, model_config.json, to device
    build_run_optimizer               parameter groups, optimizer, backend wrap
    RunState                          the objects above in one place for the helpers below
    restore_checkpoint_if_resuming    latest / explicit checkpoint -> model, optimizer, RNG state, progress
    loop                              run_one_optimizer_step -> advance -> evaluate -> log -> checkpoint (or stop)
    export_if_requested               the HuggingFace folder, once training finished

Steps are OPTIMIZER steps (one world batch each). `train()` touches no tensor, device, clock or `print`: numerics
live in `step.py` and `evaluation.py`, device code in `backend/`, console lines and the dashboard in `logger.py`.
The setup order is itself numerics: seed, dataset and loaders (no torch RNG draw), model (its init is the first RNG
consumer), optimizer, resume (restores the stored RNG state). The golden run in `test_run.py` fails on any change.

A resume continues the run exactly (the checkpoint carries `BatchStream.state_dict()`, the RNG states and the
optimizer state; the loaders seed their iterators from a private generator): on a deterministic backend the resumed
steps reproduce the uninterrupted run's numbers, on a GPU the usual nondeterminism applies.

The CLI around this is `training/train.py`; `TrainingReport` is defined next to `RunLogger` in `logger.py`.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast, Any

from torch.nn import Module
from torch.optim import Optimizer

from data_preparation.dataset_config import DatasetConfig
from data_preparation.lib.log import get_logger
from data_preparation.lib.abort import StopCheck
from model import RecurrentConfig, RecurrentGPT
from training.backend import get_backend
from training.backend.base import Backend
from training.checkpoint import (
    CheckpointMetadata,
    checkpoint_dir,
    check_settings_unchanged,
    checkpoint_path,
    find_latest_checkpoint,
    is_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
)
from training.data.tokenizer import IGNORE_INDEX
from evaluation.benchmarks import benchmarks_path, evaluate_on_benchmarks
from evaluation.prompts import load_prompts
from evaluation.samples import GeneratedSample, generate_and_save_samples, samples_path
from training.data.tokenizer import Tokenizer
from training.data.loader import build_run_dataloaders
from training.data.dataset_resolver import ResolvedDataset, check_dataset_unchanged, resolve_dataset
from training.evaluation import evaluate, is_evaluation_step
from training.logger import RunLogger, TrainingReport
from training.optim import build_optimizer, get_param_groups
from data_preparation.lib.build.lock import TRAIN_LOCK_NAME, run_lock
from training.settings import Settings
from training.stage_manager import StageManager
from training.triggers import StepTriggers
from training.step import BatchStream, NonFiniteLossError, TrainingProgress, run_one_optimizer_step


log = get_logger(__name__)


@dataclass(frozen=True)
class RunState:
    """
    The run once it is set up, handed to `restore_checkpoint_if_resuming`, `save_run_checkpoint` and
    `export_if_requested` as one argument. Every member keeps its identity for the whole run; only `progress` moves.
    """

    settings: Settings
    run_directory: Path
    backend: Backend
    model: Module
    optimizer: Optimizer
    dataset: ResolvedDataset
    stage_manager: StageManager
    progress: TrainingProgress


@dataclass(frozen=True)
class ResumePoint:
    """
    Where a resumed run continues from: the checkpoint it was restored from and the data-stream state stored in
    it (None in a checkpoint written before the stream existed), for `BatchStream.load_state_dict`.
    """

    checkpoint: Path
    data_stream: dict[str, Any] | None


def train(
    settings: Settings,
    *,
    backend: Backend | None = None,
    should_stop: StopCheck | None = None,
    started_at: float | None = None,
    keep_history: bool = False,
) -> TrainingReport:
    """
    Run the training run described by `settings` and return its report.

    `backend`: created from the settings unless given (tests inject the CPU backend); seeded here either way.
    `should_stop`: the run's stop request (the CLI's Ctrl-C), polled between build shards and after every optimizer
    step; the loop then saves a checkpoint, skips the export and returns with `report.stopped`.
    `started_at`: the caller's clock reading at the start of the run (`report.setup_seconds`).
    `keep_history`: a test knob; `report.history` then holds every log step's metric dict.
    `out_dir` is locked for the whole run: a second run on the same `out_dir` fails with `RunLocked`.

    Numerics: the setup order (module docstring) and the loop body (step, evaluation, then the checkpoint) are the
    thesis loop's; `test_golden_tiny_run` fails on any change. Evaluation runs under `torch.random.fork_rng`
    (`training/evaluation.py`), so it draws nothing the training stream would miss.
    """

    check_evaluation_recurrences(settings)  # before anything is created or built
    backend = backend or create_backend(settings)
    if backend.world_size != 1:
        raise NotImplementedError(f"world_size {backend.world_size}: no multi-rank backend exists yet, the loop runs on one device")
    backend.seed_everything(settings.seed)
    run_directory = prepare_run_directory(settings)
    # the main rank holds the run lock (released on every way out, exception included); other ranks share its run
    lock = run_lock(Path(settings.out_dir) / TRAIN_LOCK_NAME, "training") if backend.is_main else nullcontext()
    with lock:
        dataset = resolve_dataset(settings, backend, should_stop=should_stop)
        stage_manager = build_stage_manager(settings, dataset, backend.world_size)
        sample_triggers = StepTriggers.from_settings(
            settings.sample_step_interval, settings.sample_at_training_progress, stage_manager.total_steps
        )
        benchmark_triggers = StepTriggers.from_settings(
            settings.benchmark_step_interval, settings.benchmark_at_training_progress, stage_manager.total_steps
        )
        loaders = build_run_dataloaders(settings, dataset, backend)
        try:
            model = build_run_model(settings, dataset, backend, run_directory)
            check_tokenizer_vocabulary(loaders.tokenizer, backend.plain_model(model).config)
            optimizer = build_run_optimizer(settings, model, backend)
            state = RunState(settings, run_directory, backend, model, optimizer, dataset, stage_manager, TrainingProgress())
            resume = restore_checkpoint_if_resuming(state)
            progress = state.progress

            with RunLogger.open(
                settings,
                run_directory,
                dataset,
                model,
                stage_manager,
                progress,
                backend,
                setup_started=started_at,
                keep_history=keep_history,
            ) as logger:
                if backend.is_main and (resume is None or not (run_directory / "run_config.json").exists()):
                    record_run_config(settings, run_directory)
                if resume is None:
                    logger.log_fresh_start()
                else:
                    logger.log_resume(resume.checkpoint, progress.step)
                logger.log_triggers("samples", sample_triggers.listed(stage_manager.total_steps))
                logger.log_triggers("benchmarks", benchmark_triggers.listed(stage_manager.total_steps))
                batches = BatchStream(settings, loaders, stage_manager, progress)
                if resume is not None and resume.data_stream is not None:
                    batches.load_state_dict(resume.data_stream)
                logger.status("training")
                stopped = False
                while progress.step < stage_manager.total_steps and not stopped:
                    try:
                        result = run_one_optimizer_step(
                            settings, backend, model, optimizer, stage_manager, batches, progress
                        )
                    except NonFiniteLossError as error:
                        raise RuntimeError(f"{error}. Terminating; {_checkpoint_before_failed_step(state, logger, batches)}") from None
                    progress.advance()
                    if is_evaluation_step(settings, progress.step, stage_manager):
                        validation_loader = loaders.val_loaders[stage_manager.entering_stage_at(progress.step)]
                        with logger.evaluating():
                            result.validation = evaluate(settings, backend, model, validation_loader)
                    logger.log_step(result, progress, data_wait=loaders.take_wait_seconds())
                    # a request arriving during the last step changes nothing: the run is finished, not stopped;
                    # decided over all ranks, so every rank saves and stops after the same step
                    stopped = backend.any_flag(stop_requested(should_stop)) and progress.step < stage_manager.total_steps
                    if stopped:
                        logger.status("stopping after this step, saving a checkpoint")
                    if is_checkpoint_step(settings, progress.step, stage_manager) or stopped:
                        save_run_checkpoint(state, logger, batches)
                    # after the checkpoint: a failing benchmark (network, the missing extra) never costs one.
                    # Samples, benchmarks and the export are the main rank's work (RNG-isolated inference, files)
                    if backend.is_main and not stopped and sample_triggers.due(progress.step):
                        write_samples(state, logger, loaders.tokenizer)
                    if backend.is_main and not stopped and benchmark_triggers.due(progress.step):
                        run_benchmarks(state, logger, loaders.tokenizer)
                export_dir = None if stopped or not backend.is_main else export_if_requested(state, logger)
                return logger.close(progress, export_dir, stopped=stopped)
        finally:
            loaders.close()  # the loader workers stop now, not when the GC finds the iterators
            backend.shutdown()


# --- setup -----------------------------------------------------------------------------------------------------------


def create_backend(settings: Settings) -> Backend:
    """
    The run's backend (`settings.backend` at `settings.precision`); its constructor picks the device and sets the
    torch flags. `train()` seeds it right after.
    """

    return get_backend(settings.backend, precision=settings.precision)


def run_directory_of(settings: Settings) -> Path:
    """
    The run directory: `out_dir/<run_name>`, where the checkpoints, logs and wandb files of the run go.
    """

    return Path(settings.out_dir) / settings.run_name


def prepare_run_directory(settings: Settings) -> Path:
    """
    Create the run directory (`run_directory_of`) with its `checkpoints/` folder. Returns the run directory.
    """

    run_directory = run_directory_of(settings)
    checkpoint_dir(run_directory).mkdir(parents=True, exist_ok=True)
    return run_directory


def record_run_config(settings: Settings, run_directory: Path) -> None:
    """
    Write `run_config.json` (the settings as parsed). A fresh run always writes it; a resume only into a run
    directory without one (an explicit `resume_checkpoint_path` into a new directory) and keeps it otherwise.
    """

    with open(run_directory / "run_config.json", "w") as file:
        json.dump(asdict(settings), file, indent=4)


def build_stage_manager(settings: Settings, dataset: ResolvedDataset, world_size: int) -> StageManager:
    """
    The run's `StageManager`: the dataset's stage budgets turned into optimizer-step boundaries of
    `micro_batches_per_step x tokens_per_micro_batch` tokens; the packs of a step must split evenly over the devices.
    """

    settings.micro_batches_per_rank(world_size)  # refuses packs that do not split evenly over the ranks
    return StageManager(
        dataset.stages,
        tokens_per_step=settings.tokens_per_optimizer_step,
        world_size=world_size,
        warmup_steps=settings.warmup_steps,
        cooldown_steps=settings.cooldown_steps,
    )


def check_evaluation_recurrences(settings: Settings) -> None:
    """
    Every `sample_recurrences` / `benchmark_recurrences` setting must name one step count per core block of the
    architecture config (with `model_overwrite` applied).
    """

    model_config = RecurrentConfig.from_yaml(settings.model_architecture_config, **settings.model_overwrite)
    blocks = len(cast(list[int], model_config.n_layers_in_recurrent_block))  # a list after __post_init__
    for name in ("sample_recurrences", "benchmark_recurrences"):
        for index, setting in enumerate(getattr(settings, name)):
            if len(setting) != blocks:
                raise ValueError(
                    f"{name}[{index}] = {setting} has {len(setting)} entries but the model architecture "
                    f"{settings.model_architecture_config} has {blocks} recurrent blocks"
                )


def check_sequence_lengths(settings: Settings, dataset_config: DatasetConfig, model_config: RecurrentConfig) -> None:
    """
    Training cuts rows at `training_max_sequence_length`, which must fit both the model's RoPE table
    (`model_max_sequence_length` positions) and the stored rows (cut at `dataset_max_sequence_length` when
    downloaded): longer than the model's table is impossible, longer than the data was cut means every row is
    shorter than the training window, never what was intended. The two upper bounds are independent (a dataset
    may store 16k-token rows for a model whose table covers 2k). A run cutting rows at another length than the
    dataset config planned its downloads for (`training_target_sequence_length`) is warned about: the rows on
    disk serve fewer tokens than budgeted when the run cuts shorter (the sampler cycles the source), more when it
    cuts longer.
    """

    model, dataset, training = (
        model_config.model_max_sequence_length, dataset_config.dataset_max_sequence_length, settings.training_max_sequence_length
    )
    if training > model or training > dataset:
        raise ValueError(
            f"training_max_sequence_length {training} (the run config) must be at most model_max_sequence_length {model} "
            f"({settings.model_architecture_config}, with model_overwrite applied) and dataset_max_sequence_length {dataset} "
            f"({settings.dataset_config})"
        )
    target = dataset_config.training_target_sequence_length
    if training != target:
        log.warning(
            "training_max_sequence_length %d differs from training_target_sequence_length %d of %s: the downloads were sized "
            "for rows cut at %d tokens", training, target, settings.dataset_config, target
        )


def check_tokenizer_vocabulary(tokenizer: Tokenizer, model_config: RecurrentConfig) -> None:
    """
    The dataset's tokenizer and the model's vocabulary must agree: every id the tokenizer produces (added tokens
    included, `len(tokenizer)`) has to be below `vocab_size`. A dataset config that swaps in a larger tokenizer
    would otherwise fail on an index error at the first batch holding an id beyond the padded table, and for ids
    between `vocab_size` and `padded_vocab_size` train nothing at all: those are the padding rows of the embedding
    table, whose labels the loss ignores (`model.mask_labels`) and whose logits the HF wrapper sets to -inf. The
    padded size therefore does not enter the check (`padded_vocab_size >= vocab_size` always holds).

    A tokenizer smaller than the architecture's declared `vocab_size` only trains rows that never occur, so it is
    a warning: the number is worth seeing when a run reports its parameter count.
    """

    tokens, vocab = len(tokenizer), model_config.vocab_size
    if tokens > vocab:
        raise ValueError(
            f"The tokenizer of the dataset has {tokens} tokens but the model's vocabulary holds {vocab} ids "
            f"(vocab_size of the model architecture config, padded to {model_config.padded_vocab_size} embedding "
            "rows): ids from vocab_size on are never trained. Raise vocab_size or use the tokenizer the "
            "architecture was sized for."
        )
    if vocab != tokens:
        log.warning(
            "The model architecture declares vocab_size %d but the dataset's tokenizer has %d tokens: the "
            "difference is embedding rows that never occur in the data",
            model_config.vocab_size, tokens
        )


def build_run_model(settings: Settings, dataset: ResolvedDataset, backend: Backend, run_directory: Path) -> Module:
    """
    The run's model: architecture yaml + `model_overwrite`, the sequence-length check against the dataset config,
    `RecurrentGPT`, `model_config.json`, then `backend.setup_model` (device, optional compile).

    Numerics: the parameter init is the first consumer of the global torch RNG after `seed_everything`; nothing that
    draws may run before it.
    """

    model_config = RecurrentConfig.from_yaml(settings.model_architecture_config, **settings.model_overwrite)
    check_sequence_lengths(settings, dataset.config, model_config)
    model = RecurrentGPT(
        model_config, ignore_index=IGNORE_INDEX, gradient_checkpointing=settings.gradient_checkpointing
    )
    if backend.is_main:
        model_config.to_json(run_directory / "model_config.json")
    return backend.setup_model(model, compile_model=settings.compile_model)


def build_run_optimizer(settings: Settings, model: Module, backend: Backend) -> Optimizer:
    """
    The run's optimizer: the three parameter groups of `get_param_groups`, `settings.optimizer` with
    `settings.optim_config`, wrapped by `backend.setup_optimizer`.
    """

    param_groups = get_param_groups(
        model, settings.optim_config.weight_decay, settings.no_weight_decay_for_bias_and_norm_params
    )
    return backend.setup_optimizer(build_optimizer(settings.optimizer, param_groups, settings.optim_config))


def restore_checkpoint_if_resuming(state: RunState) -> ResumePoint | None:
    """
    Restore the run from its checkpoint when it resumes and say where from; None for a fresh run.

    With `settings.resume`: `resume_checkpoint_path` if set, else the most recently written checkpoint of `run_name`
    in the run directory; none found means a fresh start. Restores model and optimizer state, verifies dataset and
    settings against the checkpoint, refuses a checkpoint written with another number of ranks (its RNG states are
    per rank), restores this rank's RNG state and sets `progress.step = progress.resume_step = checkpoint step`. The
    returned data-stream state goes into `BatchStream.load_state_dict` once the stream exists.
    """

    settings = state.settings
    if not settings.resume:
        return None
    if settings.resume_checkpoint_path:
        resume_path: Path | None = Path(settings.resume_checkpoint_path)
    else:
        resume_path = find_latest_checkpoint(state.run_directory, settings.run_name)
    if resume_path is None:
        return None
    metadata = load_training_checkpoint(state.backend, resume_path, state.model, state.optimizer)
    check_dataset_unchanged(metadata, state.dataset, settings.allow_dataset_change)
    model_config = state.backend.plain_model(state.model).config.to_dict()
    check_settings_unchanged(metadata, settings, model_config, settings.allow_settings_change)
    if metadata.world_size != state.backend.world_size:
        raise ValueError(
            f"{resume_path} was written by a run with {metadata.world_size} rank(s), this run has "
            f"{state.backend.world_size}: resume with the same number of ranks (its RNG states are per rank; "
            "continuing on another number of GPUs is not supported)"
        )
    state.progress.step = state.progress.resume_step = metadata.step
    state.backend.set_rng_state(metadata.rng_states[state.backend.rank])
    return ResumePoint(resume_path, metadata.data_stream)


# --- inside the loop -------------------------------------------------------------------------------------------------


def stop_requested(should_stop: StopCheck | None) -> bool:
    """
    Whether the caller asked the run to stop (None: never).
    """

    return should_stop is not None and should_stop()


def _checkpoint_before_failed_step(state: RunState, logger: RunLogger, batches: BatchStream) -> str:
    """
    A step that produced a non-finite loss or gradient norm did not update the model (`optimizer.step` never ran),
    so the model and optimizer state are those of the completed steps: save them, unless no step completed yet.
    Returns the note for the error message.

    Written as `...-failed.pth`, beside (never over) the regular checkpoint of the same step, and skipped by
    `find_latest_checkpoint` so a plain `resume: true` continues from the last regular checkpoint. The data stream
    is the one AFTER the failed step: the step read all its micro-batches before the loss was checked, so a resume
    from this file re-runs the step on the NEXT documents and the failed step's documents are skipped. To train
    them, resume from a regular checkpoint instead.
    """

    if state.progress.step == 0:
        return "no checkpoint written (the first step failed)"
    state.optimizer.zero_grad(set_to_none=True)
    path = save_run_checkpoint(state, logger, batches, failed=True)
    return (
        f"the model before this step is checkpointed as {path} (resuming from it continues AFTER this step's "
        "documents, which are skipped; the regular checkpoints are untouched)"
    )


def save_run_checkpoint(state: RunState, logger: RunLogger, batches: BatchStream, failed: bool = False) -> Path:
    """
    Write the checkpoint of `state.progress.step` completed optimizer steps and tell the logger.

    Named `step-{step:08d}-{run_name}.pth`, plus `-stage-{i}_end` after the last plain step of stage i; `stage` is
    the stage the run is heading for (`StageManager.entering_stage_at`). Called after evaluation and logging; the
    stored RNG states (one per rank, gathered) are those after the step, evaluation having drawn under
    `torch.random.fork_rng`. `failed`
    appends `-failed` to the name, for the checkpoint of a step that ended the run on a non-finite loss
    (`_checkpoint_before_failed_step`).
    """

    settings, progress, stage_manager = state.settings, state.progress, state.stage_manager
    stage_end = stage_manager.stage_ending_at(progress.step - 1)
    path = checkpoint_path(state.run_directory, settings.run_name, progress.step, stage_end, failed=failed)
    metadata = CheckpointMetadata(
        step=progress.step,
        stage=stage_manager.entering_stage_at(progress.step),
        world_size=state.backend.world_size,
        rng_states=state.backend.all_gather_object(state.backend.rng_state()),
        settings=asdict(settings),
        model_config=state.backend.plain_model(state.model).config.to_dict(),
        dataset_config_hash=state.dataset.config_hash,
        validation_rows=state.dataset.validation_rows,
        source_rows=state.dataset.source_rows,
        data_stream=batches.state_dict(),
    )
    with logger.saving_checkpoint():
        save_training_checkpoint(state.backend, path, state.model, state.optimizer, metadata)
    logger.log_checkpoint(path)
    return path


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
                recurrences=settings.sample_recurrences or [None],
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
                limit=settings.benchmark_limit,
                batch_size=settings.benchmark_batch_size,
                recurrences=settings.benchmark_recurrences or [None],
                out_path=path,
                step=step,
                seed=settings.seed,
            )
    except Exception as error:  # the harness needs the extra and the network; the run must not end on it
        logger.log_benchmark_failure(error)
        return None
    logger.log_benchmarks(metrics, path, step)
    return metrics


# --- after the loop --------------------------------------------------------------------------------------------------


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

    export_to_hf(trained_model, trained_model.config, export_dir, tokenizer_dir=state.dataset.tokenizer_dir)
    logger.log_export(export_dir)
    return export_dir
