# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
`train()` and `run_training_loop()`: the core training pipelines; supporting helpers live in `execution/`.

    build_stage_manager               validate the configured schedule from YAML before backend/data setup
    create_backend                    device, precision, torch flags; then `seed_everything`
    prepare_run_directory             out_dir/<run_name>/checkpoints
    run_lock                          one training run per out_dir (`data_preparation/lib/build/lock.py`)
    resolve_resume_checkpoint         select checkpoint once; reject fresh reuse of established run evidence
    resolve_dataset                   verify / auto-prepare the dataset config, validation split, tokenizer dir
    build_stage_manager               token budgets -> optimizer-step boundaries, transitions, per-stage base LR/weights
    build_run_dataloaders             one train loader per SOURCE on the main rank (whole run), one validation loader per stage
    build_run_model                   architecture yaml + overrides, sequence-length check, to device
    build_run_optimizer               parameter groups, optimizer, backend wrap
    RunState                          the objects above in one place for the helpers below
    restore_checkpoint                selected checkpoint -> model, optimizer, RNG state, progress
    loop                              run_one_optimizer_step -> advance -> evaluate -> log -> checkpoint (or stop)
    export_if_requested               the HuggingFace folder, once training finished

Steps are OPTIMIZER steps (one world batch each). `train()` touches no tensor, device, clock or `print`: numerics
live in `step.py` and `evaluation.py`, device code in `backend/`, console lines and the dashboard in `logger.py`.
Every rank of a multi-rank run (`backend: ddp`) runs this same function; what is the main rank's alone is the data
stream (`RankBatches` scatters its packs), the run lock, the files it writes and the logging (the other ranks' logger
is silent).
The setup order is itself numerics: seed, dataset and loaders (no torch RNG draw), model (its init is the first RNG
consumer), optimizer, resume (restores the stored RNG state). The golden run in `test_run.py` fails on any change.

A resume continues the run exactly (the checkpoint carries `BatchStream.state_dict()`, the RNG states and the
optimizer state; the loaders seed their iterators from a private generator): on a deterministic backend the resumed
steps reproduce the uninterrupted run's numbers, on a GPU the usual nondeterminism applies.

The CLI around this is `training/train.py`; `TrainingReport` is defined next to `RunLogger` in `logger.py`.
"""

from __future__ import annotations

from data_preparation import load_dataset_config
from data_preparation.lib.abort import StopCheck
from training.backend.base import Backend
from training.data.dataset_resolver import resolve_dataset
from training.data.loader import RunDataloaders, build_run_dataloaders
from training.execution import (
    RunState, build_run_model, build_run_optimizer, build_run_triggers, build_stage_manager,
    check_evaluation_recurrences, check_tokenizer_vocabulary,
    close_backend_on_exit, close_loaders_on_exit, open_run_logger,
    open_training_dataset, prepare_run_directory, prepare_run_stream, restore_checkpoint,
    select_resume_for_run, validate_resolved_schedule,
    create_backend, finish_training_loop, run_and_log_training_step, run_scheduled_inference, save_checkpoint_if_due,
)
from training.failure import FatalHandler
from training.logger import RunLogger, TrainingReport
from training.provenance import publish_configuration
from training.settings import Settings
from training.steps import RankBatches, TrainingProgress
from training.stopping import StopController, complete_main_phase
from training.tokenizer_contract import check_profile_config, prepare_run_tokenizer
from tokenization.validation import check_model_vocabulary
from training.triggers import StepTriggers


def train(
    settings: Settings,
    *,
    backend: Backend | None = None,
    should_stop: StopCheck | None = None,
    started_at: float | None = None,
    keep_history: bool = False,
    on_fatal_error: FatalHandler | None = None,
) -> TrainingReport:
    """
    Run the training run described by `settings` and return its report.

    `backend`: created from the settings unless given (tests inject the CPU backend); seeded here either way.
    `should_stop`: the run's stop request (the CLI's Ctrl-C), polled between build shards and at completed training phase
    boundaries; the loop saves the completed state and skips optional work. Incomplete runs set `report.stopped`.
    `started_at`: the caller's clock reading at the start of the run (`report.setup_seconds`).
    `keep_history`: a test knob; `report.history` then holds every log step's metric dict.
    `on_fatal_error`: launcher-owned CLI policy invoked before run-level cleanup; library callers leave it unset.
    `out_dir/run_name` is locked for the whole run: a second run on the same directory fails with `RunLocked`.

    Numerics: the setup order (module docstring) and the loop body (step, evaluation, then the checkpoint) are the
    thesis loop's; `test_golden_tiny_run` fails on any change. Evaluation runs under `torch.random.fork_rng`
    (`training/evaluation.py`), so it draws nothing the training stream would miss.
    """

    # validate the pure plan before creating resources
    model_config = check_evaluation_recurrences(settings)
    dataset_config = load_dataset_config(settings.dataset_config)
    check_profile_config(dataset_config, model_config)
    configured_schedule = build_stage_manager(settings, dataset_config, world_size=1)
    backend = backend or create_backend(settings)

    # acquire the run and dataset, preserving the seed and setup order
    with close_backend_on_exit(backend, on_fatal_error):
        settings.validate_world_size(backend.world_size)
        backend.seed_everything(settings.seed)
        run_directory = prepare_run_directory(settings)
        with open_training_dataset(settings, backend, on_fatal_error) as dataset_lease:
            resume_path = select_resume_for_run(settings, run_directory, backend)
            dataset = resolve_dataset(settings, backend, should_stop=should_stop, dataset_lease=dataset_lease)
            stage_manager = validate_resolved_schedule(settings, dataset, backend, configured_schedule)
            sample_triggers, benchmark_triggers = build_run_triggers(settings, stage_manager.total_steps)
            dataset, tokenizer_contract = prepare_run_tokenizer(dataset, model_config, run_directory, backend)
            loaders = build_run_dataloaders(settings, dataset, backend)

            # build model and optimizer, then restore the selected checkpoint
            with close_loaders_on_exit(loaders, on_fatal_error):
                check_tokenizer_vocabulary(loaders.tokenizer, model_config)
                model = build_run_model(settings, dataset, backend, run_directory, resume_checkpoint=resume_path, model_config=model_config)
                check_model_vocabulary(model_config, tokenizer_contract, backend.plain_model(model))
                optimizer = build_run_optimizer(settings, model, backend)
                state = RunState(settings, run_directory, backend, model, optimizer, dataset, stage_manager, TrainingProgress(), tokenizer_contract)
                resume = restore_checkpoint(state, resume_path) if resume_path is not None else None

                # initialize logging and the stream before publishing configuration and training
                with open_run_logger(state, started_at, keep_history, on_fatal_error) as logger:
                    batches = prepare_run_stream(state, loaders, logger, resume, sample_triggers, benchmark_triggers)
                    complete_main_phase(backend, "configuration publication", lambda: publish_configuration(state, resume))
                    return run_training_loop(state, loaders, logger, batches, sample_triggers, benchmark_triggers, should_stop, on_fatal_error)


def run_training_loop(
    state: RunState, loaders: RunDataloaders, logger: RunLogger, batches: RankBatches,
    sample_triggers: StepTriggers, benchmark_triggers: StepTriggers, should_stop: StopCheck | None,
    on_fatal_error: FatalHandler | None,
) -> TrainingReport:
    """Run optimizer steps in their original order, then stop/checkpoint or export and close the report."""

    # initialize stopping and checkpoint state
    progress, stage_manager = state.progress, state.stage_manager
    stop = StopController(state.backend, should_stop)
    checkpoint_fresh = False  # restored metadata may differ from the source checkpoint

    # process each step, checking for stops between completed phases
    while progress.step < stage_manager.total_steps:
        if stop.poll("before optimizer step"):
            break
        checkpoint_fresh = False
        run_and_log_training_step(state, loaders, logger, batches, stop, on_fatal_error)
        if bool(stop.requested):  # helpers can change the stop flag
            break

        # publish scheduled checkpoints before optional inference
        checkpoint_fresh = save_checkpoint_if_due(state, logger, batches, stop)
        if bool(stop.requested):
            break
        checkpoint_fresh = run_scheduled_inference(state, logger, loaders.tokenizer, sample_triggers, benchmark_triggers, stop, checkpoint_fresh, on_fatal_error)
        if bool(stop.requested):
            break

    # save or export the final state and close the report
    return finish_training_loop(state, logger, batches, stop, checkpoint_fresh)
