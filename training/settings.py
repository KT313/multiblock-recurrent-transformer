# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Training run settings: the YAML/CLI schema consumed by `training/train.py`.

Framework-neutral (no torch imports). Every `*_steps` / `*_interval` value counts OPTIMIZER steps, i.e. world
batches of `world_batch_size × block_size` tokens. Defaults make a run on one local GPU work out of the box.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# re-exported from jsonargparse._actions at runtime but missing from the package's typed public surface
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace  # type: ignore[attr-defined]

# The value rules of `Settings`, as three tables read by one loop each in `Settings.__post_init__` (the same idea as
# `SOURCE_FIELD_SCOPES` in `data_preparation/dataset_config.py`, one size smaller): a field that must be set, one
# that must be > 0, one that must be >= 0. Rules relating two fields stay explicit below the loops.
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
}
NON_NEGATIVE_SETTINGS: tuple[str, ...] = (
    "save_step_interval",
    "warmup_steps",
    "cooldown_steps",
    "resume_warmup_steps",
)


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
    prepare_num_workers: int = 2  # sources processed at a time by the in-process build (= prepare.py --num_workers); also the
    # pool size of each decontamination / minhash pass, so the product is what runs with those toggles on
    prepare_max_parallel_downloads: int = 2  # sources downloading at a time during the in-process build
    stage_base_lrs: list[float] = field(default_factory=list)  # base LR per dataset-config stage, positional
    allow_dataset_change: bool = False  # resume from a checkpoint written with a different dataset config
    allow_settings_change: bool = False  # resume although numerics-relevant settings / model config differ from the checkpoint

    # Run
    run_name: str = "crow-300m"
    out_dir: str = "outputs"  # checkpoints go to {out_dir}/checkpoints, wandb files to {out_dir}/wandb
    resume: bool = True  # resume from the latest checkpoint of `run_name` in `out_dir` if one exists
    resume_checkpoint_path: Optional[str] = None  # explicit checkpoint to resume from (overrides the search)
    seed: int = 1337

    # Model
    model_overwrite: dict[str, Any] = field(default_factory=dict)  # RecurrentConfig keys overriding the architecture
    # config, e.g. `--model_overwrite '{"n_embd": 512}'` for a CLI sweep; {} = the file as is
    block_size: int = 2048  # sequence length; must equal the block_size of the architecture config and of the
    # dataset config (the planner sized the data in sequences of it; <= max_seq_length follows from the schema)

    # Data loading
    dataloader_num_workers: int = 4
    sort_batches_by_length: bool = True  # regroup each world batch into length-sorted micro-batches
    sequence_padding_multiple: Optional[int] = 128  # pad micro-batches to a multiple of this (None: max length)

    # Backend
    backend: str = "single_device"
    precision: str = "bf16-mixed"
    compile_model: bool = False
    gradient_checkpointing: bool = False

    # Batching (one optimizer step = world_batch_size sequences)
    micro_batch_size: int = 4
    world_batch_size: int = 1024

    # Optimizer + LR schedule
    optimizer: str = "ELLISAdam"
    optim_config: dict[str, Any] = field(default_factory=lambda: dict(lr=1e-4, weight_decay=4e-5, betas=(0.9, 0.95)))
    no_weight_decay_for_bias_and_norm_params: bool = True
    grad_clip: float = 1.0
    lr_schedule: str = "trapezoid"
    warmup_steps: int = 0
    cooldown_steps: int = 0
    min_lr: float = 0.0
    resume_warmup_steps: int = 0  # LR ramps from min_lr back to schedule over this many steps after a resume

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
    export_hf_path: Optional[str] = None  # default: {out_dir}/hf_export

    def __post_init__(self) -> None:
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
        if self.resume_checkpoint_path and not self.resume:
            raise ValueError("resume_checkpoint_path is set but resume is false; set resume: true to use it")

    @property
    def gradient_accumulation_steps(self) -> int:
        """Micro-batches per optimizer step on one device (divide by world_size once distributed training exists)."""
        return self.world_batch_size // self.micro_batch_size



def parse_settings(args: Optional[list[str]] = None) -> Settings:
    """`--config file.yaml` plus `--key value` overrides for any field (nested keys with dots)."""
    parser = ArgumentParser(description="Train a multi-block recurrent transformer.")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML settings file")
    parser.add_class_arguments(Settings, nested_key=None)
    ns = parser.parse_args(args)
    ns.pop("config", None)
    instantiate: Callable[[Namespace], Namespace | dict[str, Any]] = (
        getattr(parser, "instantiate", None) or parser.instantiate_classes  # jsonargparse >=4.49 / older
    )
    instantiated = instantiate(ns)
    values = instantiated.as_dict() if isinstance(instantiated, Namespace) else instantiated
    return Settings(**values)
