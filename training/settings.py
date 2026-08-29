# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Training run settings: the YAML/CLI schema consumed by `training/train.py`.

Framework-neutral (no torch imports). Every `*_steps` / `*_interval` value counts OPTIMIZER steps, i.e. world
batches of `world_batch_size × block_size` tokens. Defaults make a run on one local GPU work out of the box.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# re-exported from jsonargparse._actions at runtime but missing from the package's typed public surface
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace  # type: ignore[attr-defined]

from training.stage_manager import TrainingStage


@dataclass
class DataEntry:
    """One parquet dataset directory inside a stage mixture."""

    prefix: str  # unique name, used for logging
    data_dir: str  # directory with *.parquet files
    weight: float = 1.0  # sampling weight relative to the other entries of the same stage
    data_signature: Optional[dict[str, Any]] = None  # {"keys": [...], "format_fn": "..."}; default: text column


@dataclass
class StageConfig:
    """One training stage; see docs/multistage_training.md."""

    name: str
    tokens: int  # global token budget of this stage
    base_lr: float
    train_data: list[DataEntry]
    val_data: list[DataEntry]
    transition_pct: float = 0.0  # fraction of this stage (at its end) blending into the next stage's data/LR


@dataclass
class Settings:
    # Run
    run_name: str = "crow-300m"
    out_dir: str = "outputs"  # checkpoints go to {out_dir}/checkpoints, wandb files to {out_dir}/wandb
    resume: bool = True  # resume from the latest checkpoint of `run_name` in `out_dir` if one exists
    resume_checkpoint_path: Optional[str] = None  # explicit checkpoint to resume from (overrides the search)
    seed: int = 1337

    # Model
    model_name: str = "crow-300m-final"  # preset name from model/presets.py
    model_overwrite: dict[str, Any] = field(default_factory=dict)  # overrides passed to the preset
    block_size: int = 2048  # sequence length; must match the model preset
    tokenizer_path: str = "dataset/tokenizer"

    # Data / curriculum
    training_stages: list[StageConfig] = field(default_factory=list)
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
        if not self.training_stages:
            raise ValueError("training_stages must contain at least one stage")
        if self.world_batch_size % self.micro_batch_size != 0:
            raise ValueError("world_batch_size must be a multiple of micro_batch_size")
        if not Path(self.tokenizer_path).exists():
            raise FileNotFoundError(f"tokenizer_path {self.tokenizer_path!r} does not exist")

    @property
    def gradient_accumulation_steps(self) -> int:
        """Micro-batches per optimizer step on one device (divide by world_size once distributed training exists)."""
        return self.world_batch_size // self.micro_batch_size

    def stage_manager_stages(self) -> list[TrainingStage]:
        """The stages in the plain-dict form `training.stage_manager.StageManager` expects."""
        return [
            TrainingStage(
                name=s.name,
                tokens=s.tokens,
                base_lr=s.base_lr,
                transition_pct=s.transition_pct,
                train_data=[vars(d) for d in s.train_data],
                val_data=[vars(d) for d in s.val_data],
            )
            for s in self.training_stages
        ]


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
