# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Training run settings: the YAML/CLI schema consumed by `training/train.py`.

Framework-neutral (no torch imports). Every `*_steps` / `*_interval` value counts OPTIMIZER steps, i.e. world
batches of `world_batch_size × training_max_sequence_length` tokens. Defaults make a run on one local GPU work out of the box.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

# re-exported from jsonargparse._actions at runtime but missing from the package's typed public surface
from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]

# The value rules of `Settings`, one loop each in `__post_init__`: fields that must be set, be > 0, be >= 0. Rules
# relating two fields stay explicit below the loops.
REQUIRED_SETTINGS: dict[str, str] = {
    "dataset_config": "path to config/datasets/<name>.yaml",
    "model_architecture_config": "path to config/model_architecture/<name>.yaml",
    "stage_base_lrs": "one base LR per stage of the dataset config",
}
POSITIVE_SETTINGS: dict[str, str] = {
    "log_step_interval": "save_step_interval is the only interval 0 disables",
    "eval_step_interval": "save_step_interval is the only interval 0 disables",
    "eval_iters": "validation micro-batches per depth",
    "grad_clip": "0 would zero every gradient",
    "micro_batch_size": "sequences per forward/backward; 0 or less makes the micro-batch loop of a step run zero times",
    "world_batch_size": "sequences per optimizer step",
    "prepare_pass_workers": "process pool size of each cleaning pass of the in-process dataset build",
    "sample_max_new_tokens": "tokens generated per sample prompt",
    "benchmark_batch_size": "sequences per lm-eval forward",
}
NON_NEGATIVE_SETTINGS: tuple[str, ...] = (
    "save_step_interval",
    "warmup_steps",
    "cooldown_steps",
    "sample_step_interval",
    "benchmark_step_interval",
)
DEFAULT_BENCHMARK_TASKS: tuple[str, ...] = ("arc_challenge", "hellaswag", "mmlu", "winogrande")  # the thesis benchmarks


@dataclass
class OptimizerConfig:
    """
    The `optim_config:` mapping of a run config: the optimizer's constructor options, typed.

    A dataclass so jsonargparse merges per-field CLI overrides (`--optim_config.lr 3e-4`) and rejects unknown names.
    `lr` is NOT the schedule's LR (`stage_base_lrs` is); ELLISAdam keeps it as `init_lr`, the weight-decay reference.
    The last four flags exist only on ELLISAdam; `build_optimizer` rejects non-default values for other optimizers.
    """

    lr: float = 1e-4  # constructor LR; for ELLISAdam the weight-decay reference (decay = lr / init_lr × weight_decay)
    weight_decay: float = 4e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: Optional[float] = None  # None: the optimizer's own default (ELLISAdam 1e-6, torch AdamW 1e-8)
    update_clipping: bool = False  # ELLISAdam only: clip the update by its gradient RMS
    atan_adam: bool = False  # ELLISAdam only: atan2 update instead of the eps-guarded division
    running_init: bool = False  # ELLISAdam only: initialise the moments from the first gradient
    decouple_wd: bool = True  # ELLISAdam only: weight decay relative to init_lr instead of multiplied by lr


@dataclass
class Settings:
    # Config references (required): everything about the data (sources, stages, token budgets, weights, tokenizer)
    # lives in the dataset config and everything about the model architecture (sizes, depth, recurrence) in the
    # model architecture config; the run config only references both.
    dataset_config: str  # path to config/datasets/<name>.yaml
    model_architecture_config: str  # path to config/model_architecture/<name>.yaml

    # Data: `train()` (`training/run.py`) verifies the prepared data and, with `auto_prepare`, builds what is missing
    # (`python data_preparation/prepare.py prepare --dataset_config ...`; auto-prepare never deletes raw folders).
    dataset_dir: str = "dataset"  # root of the prepared data (sources/, processed/, tokenizers/)
    auto_prepare: bool = True  # build missing data in-process before training; False: fail with the build command
    prepare_num_workers: int = 2  # sources processed at a time by the in-process build (= prepare.py --num_workers)
    prepare_pass_workers: int = 4  # worker processes of each cleaning pass of the in-process build (--pass_workers)
    prepare_max_parallel_downloads: int = 2  # sources downloading at a time during the in-process build
    stage_base_lrs: list[float] = field(default_factory=list)  # base LR per dataset-config stage, positional
    allow_dataset_change: bool = False  # resume from a checkpoint written with a different dataset config
    allow_settings_change: bool = False  # resume although settings / model config differ (see training/checkpoint.py)

    # Run
    run_name: str = "crow-300m"
    out_dir: str = "outputs"  # the run directory is {out_dir}/{run_name}: checkpoints/, wandb/, train.log, run_config.json
    resume: bool = True  # resume from the most recently written checkpoint of `run_name` in its run directory if one exists
    resume_checkpoint_path: Optional[str] = None  # explicit checkpoint to resume from (overrides the search)
    seed: int = 1337

    # Model
    model_overwrite: dict[str, Any] = field(default_factory=dict)  # RecurrentConfig keys overriding the architecture
    training_max_sequence_length: int = 2048  # documents are cut to this many tokens at training time; packs and padded rows are sized by it; at most the dataset's and the model's length

    # Data loading (train loaders always run one worker per source; there is no worker-count knob)
    sort_batches_by_length: bool = True  # regroup each world batch into length-sorted micro-batches
    sequence_padding_multiple: Optional[int] = 128  # pad micro-batches to a multiple of this (None: max length)

    # Backend
    backend: str = "single_device"
    precision: str = "bf16-mixed"
    compile_model: bool = False
    gradient_checkpointing: bool = False

    # Batching in padded rows: validation always batches `micro_batch_size` rows, and with `pack_sequences: false`
    # training does too (one optimizer step = world_batch_size sequences).
    micro_batch_size: int = 4
    world_batch_size: int = 1024

    # Sequence packing, the default (training only; validation stays padded). Documents are laid end to end into ONE
    # row of `tokens_per_micro_batch` tokens per micro-batch, never split, attention masked per document, RoPE
    # positions restarting per document. One optimizer step is `micro_batches_per_step` such rows, i.e.
    # `micro_batches_per_step x tokens_per_micro_batch` tokens. Both left unset are the padded equivalents,
    # `micro_batch_size x training_max_sequence_length` and `world_batch_size / micro_batch_size`, so a config written in rows keeps its
    # token arithmetic and only the batch layout changes. With packing, `micro_batch_size` / `world_batch_size` only
    # size the validation batches, and `sort_batches_by_length` / `sequence_padding_multiple` apply to validation only.
    pack_sequences: bool = True  # true strongly recommended: false trains on padded rows, which waste the padding
    tokens_per_micro_batch: Optional[int] = None  # pack length; >= training_max_sequence_length (the longest document after truncation)
    micro_batches_per_step: Optional[int] = None  # packed micro-batches per optimizer step (a multiple of the number of devices)

    # Optimizer + LR schedule
    optimizer: str = "ELLISAdam"
    optim_config: OptimizerConfig = field(default_factory=OptimizerConfig)  # typed; CLI overrides merge per field
    no_weight_decay_for_bias_and_norm_params: bool = True
    grad_clip: float = 1.0
    lr_schedule: str = "trapezoid"
    warmup_steps: int = 0
    cooldown_steps: int = 0
    min_lr: float = 0.0

    # Evaluation / logging / checkpoints
    log_step_interval: int = 1
    log_gradient_metrics: bool = True  # per-parameter-group gradient/update statistics at every log step
    eval_step_interval: int = 100
    eval_iters: int = 50  # validation micro-batches per depth
    partial_depth_eval: list[int] = field(default_factory=list)  # extra recurrence depths evaluated at validation
    save_step_interval: int = 1000
    save_last_step: bool = True
    logger_project: str = "multiblock-recurrent"
    wandb_offline: bool = True
    wandb_enabled: bool = True

    # Export
    export_to_hf: bool = False  # write a HuggingFace trust_remote_code folder at the end of training
    export_hf_path: Optional[str] = None  # default: {out_dir}/{run_name}/hf_export

    # Samples and benchmarks (`evaluation/`): text the model writes for fixed prompts, lm-eval-harness scores. Both
    # run RNG-isolated (`evaluation/wrapper.py`), so they never change the training numerics. The percentages are
    # turned into step numbers once the stage plan is known (`training/triggers.py`). Files go to
    # {run dir}/samples/ and {run dir}/benchmarks/, named by step.
    sample_step_interval: int = 0  # write samples every this many steps (0: never)
    sample_at_training_progress: list[float] = field(default_factory=lambda: [100.0])  # ... and after the steps at these percentages of the run (0: after the first step, 100: after the last); combined with the interval
    sample_max_new_tokens: int = 64
    sample_temperature: float = 0.0  # 0: greedy decoding
    sample_recurrences: list[list[int]] = field(default_factory=list)  # recurrent steps per block per sampling pass, e.g. [[4, 4, 4], [12, 12, 12]]; empty: the mean recurrence once
    benchmark_step_interval: int = 0  # run the benchmarks every this many steps (0: never)
    benchmark_at_training_progress: list[float] = field(default_factory=list)  # ... and at these percentages of the run, like sample_at_training_progress (needs the eval extra: uv sync --extra eval)
    benchmark_tasks: list[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARK_TASKS))  # lm-eval task names
    benchmark_limit: Optional[int] = None  # examples per task (None: all); a few hundred keeps in-training runs short
    benchmark_num_fewshot: int = 0
    benchmark_batch_size: int = 8
    benchmark_recurrences: list[list[int]] = field(default_factory=list)  # like sample_recurrences, for the benchmarks

    def __post_init__(self) -> None:
        # dataclasses check no types at runtime, and this setting used to be a free-form dict: fail here, by name
        if not isinstance(self.optim_config, OptimizerConfig):
            raise ValueError(
                f"optim_config must be an OptimizerConfig, got {type(self.optim_config).__name__} "
                f"({self.optim_config!r}); construct OptimizerConfig(**mapping) instead of passing the mapping"
            )
        for name, why in REQUIRED_SETTINGS.items():
            if not getattr(self, name):
                raise ValueError(f"{name} is required ({why})")
        for name, why in POSITIVE_SETTINGS.items():
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive ({why})")
        for name in NON_NEGATIVE_SETTINGS:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if any(lr < 0 for lr in self.stage_base_lrs):
            raise ValueError("stage_base_lrs must be non-negative")
        if self.world_batch_size < self.micro_batch_size:
            raise ValueError(
                f"world_batch_size ({self.world_batch_size}) must be >= micro_batch_size ({self.micro_batch_size}): "
                "one optimizer step is at least one micro-batch"
            )
        if self.world_batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"world_batch_size ({self.world_batch_size}) must be a multiple of micro_batch_size "
                f"({self.micro_batch_size}): gradient_accumulation_steps is their integer quotient, so anything else "
                "silently trains on fewer sequences per step than configured"
            )
        self._check_packing()
        if self.eval_step_interval % self.log_step_interval != 0:  # both are POSITIVE_SETTINGS, no 0-disables case
            raise ValueError(
                f"eval_step_interval ({self.eval_step_interval}) must be a multiple of log_step_interval "
                f"({self.log_step_interval}): validation results would be computed and never logged"
            )
        if self.resume_checkpoint_path and not self.resume:
            raise ValueError("resume_checkpoint_path is set but resume is false; set resume: true to use it")
        if self.sample_temperature < 0:
            raise ValueError("sample_temperature must be >= 0 (0: greedy)")
        if self.benchmark_limit is not None and self.benchmark_limit <= 0:
            raise ValueError("benchmark_limit must be positive or null (all examples)")
        for name in ("sample_at_training_progress", "benchmark_at_training_progress"):
            if any(not 0 <= percentage <= 100 for percentage in getattr(self, name)):
                raise ValueError(f"{name} must list percentages between 0 and 100, got {getattr(self, name)}")
        for name in ("sample_recurrences", "benchmark_recurrences"):
            for setting in getattr(self, name):
                if not setting or any(steps <= 0 for steps in setting):
                    raise ValueError(f"{name}: every setting needs one positive step count per recurrent block, got {setting}")
        if (self.benchmark_at_training_progress or self.benchmark_step_interval) and not self.benchmark_tasks:
            raise ValueError(
                "benchmarks are requested (benchmark_at_training_progress / benchmark_step_interval) but benchmark_tasks is empty"
            )

    def _check_packing(self) -> None:
        """
        The packing fields: none without `pack_sequences`; with it, a field left unset becomes its padded
        equivalent (checked like a given one, and recorded that way in run_config.json and the checkpoints), the
        pack at least one full document long, the step at least one micro-batch. Whether the micro-batches split
        evenly over the devices is the stage manager's check (it knows the world size).
        """

        if not self.pack_sequences:
            for name in ("tokens_per_micro_batch", "micro_batches_per_step"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} is set but pack_sequences is false; set pack_sequences: true to use it")
            return
        if self.tokens_per_micro_batch is None:
            self.tokens_per_micro_batch = self.micro_batch_size * self.training_max_sequence_length
        if self.micro_batches_per_step is None:
            self.micro_batches_per_step = self.world_batch_size // self.micro_batch_size
        if self.tokens_per_micro_batch < self.training_max_sequence_length:
            raise ValueError(
                f"tokens_per_micro_batch ({self.tokens_per_micro_batch}) must be >= training_max_sequence_length ({self.training_max_sequence_length}): a "
                "document is up to training_max_sequence_length tokens after truncation and is never split across packs"
            )
        if self.micro_batches_per_step <= 0:
            raise ValueError(f"micro_batches_per_step must be positive, got {self.micro_batches_per_step}")

    @property
    def gradient_accumulation_steps(self) -> int:
        """
        Micro-batches per optimizer step on one device (divide by world_size once distributed training exists).
        """

        if self.pack_sequences:
            assert self.micro_batches_per_step is not None  # `_check_packing`
            return self.micro_batches_per_step
        return self.world_batch_size // self.micro_batch_size

    @property
    def tokens_per_optimizer_step(self) -> int:
        """
        Tokens per optimizer step, the unit of the stage budgets and the throughput metrics:
        `micro_batches_per_step x tokens_per_micro_batch` when packing, else `world_batch_size x training_max_sequence_length` (the
        padded rows counted at full length, as the thesis did).
        """

        if self.pack_sequences:
            assert self.micro_batches_per_step is not None and self.tokens_per_micro_batch is not None  # `_check_packing`
            return self.micro_batches_per_step * self.tokens_per_micro_batch
        return self.world_batch_size * self.training_max_sequence_length



def parse_settings(args: Optional[list[str]] = None) -> Settings:
    """
    `--config file.yaml` plus `--key value` overrides for any field (nested keys with dots).
    """

    parser = ArgumentParser(description="Train a multi-block recurrent transformer.")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML settings file")
    parser.add_class_arguments(Settings, nested_key=None)
    namespace = parser.parse_args(args)
    namespace.pop("config", None)
    return Settings(**parser.instantiate(namespace).as_dict())
