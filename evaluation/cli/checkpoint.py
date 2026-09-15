# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Load the checkpoint, tokenizer and execution settings used by the evaluation CLI."""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from data_preparation import DatasetLayout, load_dataset_config
from evaluation.cli.arguments import parse_recurrences
from model.config import RecurrentConfig
from model.execution import ExecutionPolicy
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer


@dataclass(frozen=True)
class CheckpointEvaluation:
    checkpoint: Path
    step: int
    out_dir: Path
    model: RecurrentGPT
    tokenizer: Tokenizer
    recurrences: list[list[int] | None]
    execution_policy: ExecutionPolicy
    state: dict[str, Any]  # retain loaded checkpoint tensors for the command lifetime, as the original CLI did


def load_checkpoint_evaluation(arguments: argparse.Namespace) -> CheckpointEvaluation:
    # read the checkpoint and resolve its execution settings
    checkpoint = Path(arguments.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    execution_policy = ExecutionPolicy(arguments.precision if arguments.precision is not None
                                       else state.get("settings", {}).get("precision"))
    step = int(state["step"])
    out_dir = Path(arguments.out_dir) if arguments.out_dir else checkpoint.parent.parent

    # restore the model and tokenizer before parsing recurrence overrides
    model = load_checkpoint_model(state, arguments.device)
    tokenizer = Tokenizer(arguments.tokenizer_dir or tokenizer_dir_of(state))
    recurrences = parse_recurrences(arguments.recurrence)
    return CheckpointEvaluation(checkpoint, step, out_dir, model, tokenizer, recurrences, execution_policy, state)


def load_checkpoint_model(state: dict[str, Any], device: str) -> RecurrentGPT:
    """
    The checkpoint's model on device (its stored `model_config` and `model` state dict).
    """

    known = {field.name for field in fields(RecurrentConfig)}
    config = RecurrentConfig(**{key: value for key, value in state["model_config"].items() if key in known})
    model = RecurrentGPT(config)
    model.load_state_dict(state["model"])
    return model.to(device)


def tokenizer_dir_of(state: dict[str, Any]) -> Path:
    """
    The tokenizer directory the checkpoint's run used: `dataset_dir/tokenizers/<name>` of its dataset config.
    """

    settings = state["settings"]
    dataset_config = load_dataset_config(settings["dataset_config"])
    return DatasetLayout(Path(settings["dataset_dir"])).tokenizer_dir(dataset_config.tokenizer.name)
