# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Model, optimizer, schedule, and path setup for a training run."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import cast

from torch.nn import Module
from torch.optim import Optimizer

from data_preparation import DatasetConfig
from data_preparation.lib.log import get_logger
from model import RecurrentConfig, RecurrentGPT
from model.layers.init import checkpoint_initialization
from training.backend import get_backend
from training.backend.base import Backend
from training.checkpoint import checkpoint_dir
from training.data.dataset_resolver import ResolvedDataset, resolve_stage_plan
from training.data.tokenizer import IGNORE_INDEX, Tokenizer
from training.logger import num_parameters
from training.optim import build_optimizer, get_param_groups
from training.provenance import record_run_config as _record_run_config
from training.settings import Settings
from training.stage_manager import StageManager

log = get_logger("training.run")


def create_backend(settings: Settings) -> Backend:
    """
    The run's backend (`settings.backend` at `settings.precision`); its constructor picks the device and sets the
    torch flags. `train()` seeds it right after.
    """

    return get_backend(settings.backend, precision=settings.precision)



def get_run_directory(settings: Settings) -> Path:
    """
    The run directory: `out_dir/<run_name>`, where the checkpoints, logs and wandb files of the run go.
    """

    return Path(settings.out_dir) / settings.run_name



def prepare_run_directory(settings: Settings) -> Path:
    """
    Create the run directory (`get_run_directory`) with its `checkpoints/` folder. Returns the run directory.
    """

    run_directory = get_run_directory(settings)
    checkpoint_dir(run_directory).mkdir(parents=True, exist_ok=True)
    return run_directory



def record_run_config(settings: Settings, run_directory: Path) -> None:
    """Publish fresh settings only when absent; retained as a public setup helper."""
    _record_run_config(settings, run_directory)



def build_stage_manager(settings: Settings, dataset: DatasetConfig | ResolvedDataset, world_size: int) -> StageManager:
    """
    The run's `StageManager`: the dataset's stage budgets turned into optimizer-step boundaries of
    `micro_batches_per_step x tokens_per_micro_batch` tokens; the packs of a step must split evenly over the devices.
    A `DatasetConfig` provides the same schedule before dataset I/O; a `ResolvedDataset` includes validation entries.
    """

    settings.micro_batches_per_rank(world_size)  # refuses packs that do not split evenly over the ranks
    return StageManager(
        resolve_stage_plan(settings, dataset) if isinstance(dataset, DatasetConfig) else dataset.stages,
        tokens_per_step=settings.tokens_per_optimizer_step,
        world_size=world_size,
        warmup_steps=settings.warmup_steps,
        cooldown_steps=settings.cooldown_steps,
    )



def check_evaluation_recurrences(settings: Settings) -> RecurrentConfig:
    """
    Every `sample_recurrences` / `benchmark_recurrences` setting must name one step count per core block of the
    architecture config (with `model_overwrite` applied). Return that validated config for weight construction.
    """

    model_config = RecurrentConfig.from_yaml(
        settings.model_architecture_config, **(settings.model_overwrite | {"use_custom_kernels": settings.use_custom_kernels})
    )
    blocks = len(cast(list[int], model_config.n_layers_in_recurrent_block))  # a list after __post_init__
    for name in ("sample_recurrences", "benchmark_recurrences"):
        for index, setting in enumerate(getattr(settings, name)):
            if len(setting) != blocks:
                raise ValueError(
                    f"{name}[{index}] = {setting} has {len(setting)} entries but the model architecture "
                    f"{settings.model_architecture_config} has {blocks} recurrent blocks"
                )

    return model_config



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



def build_run_model(
    settings: Settings, dataset: ResolvedDataset, backend: Backend, run_directory: Path,
    *, resume_checkpoint: Path | None = None, model_config: RecurrentConfig | None = None,
) -> Module:
    """
    The run's model: architecture yaml + `model_overwrite`, the sequence-length check against the dataset config,
    `RecurrentGPT`, then `backend.setup_model` (device, optional compile).

    Fresh runs keep their original initialization/RNG order. A selected checkpoint uses cheap weight placeholders;
    the caller must restore the complete checkpoint (including RNG) before training.
    """

    if model_config is None:
        model_config = check_evaluation_recurrences(settings)
    check_sequence_lengths(settings, dataset.config, model_config)
    if resume_checkpoint is None:
        log.info(
            "building the model of %s: initialising the parameters on the CPU, which takes a while for a large model",
            settings.model_architecture_config,
        )
    else:
        log.info("building the model of %s: skipping orthogonal weight initialization; restoring %s",
                 settings.model_architecture_config, resume_checkpoint)
    started = time.monotonic()
    with checkpoint_initialization() if resume_checkpoint is not None else nullcontext():
        model = RecurrentGPT(
            model_config, ignore_index=IGNORE_INDEX, gradient_checkpointing=settings.gradient_checkpointing
        )
    log.info("model built: %s parameters in %.1fs, moving it to %s%s", f"{num_parameters(model):,}",
             time.monotonic() - started, backend.device, ", compiled on the first step" if settings.compile_model else "")
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
