# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Run-owned tokenizer publication, preflight and checkpoint compatibility."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from data_preparation.lib.dataset_config import DatasetConfig
from evaluation.rng import preserve_rng
from model.config import RecurrentConfig
from tokenization.chat import VOCAB_SIZE
from tokenization.profile import copy_profile, validate_profile
from tokenization.validation import check_model_vocabulary
from training.tokenizer_parity import check_template_parity
from training.backend.base import Backend
from training.data.dataset_resolver import ResolvedDataset
from training.data.tokenizer import Tokenizer
from training.stopping import complete_main_phase


def check_profile_config(dataset: DatasetConfig, model: RecurrentConfig) -> None:
    if dataset.tokenizer.profile and model.vocab_size != VOCAB_SIZE:
        raise ValueError(f"tokenizer profile {dataset.tokenizer.profile} requires model vocab_size={VOCAB_SIZE}, got {model.vocab_size}")


def prepare_run_tokenizer(dataset: ResolvedDataset, config: RecurrentConfig, run_directory: Path, backend: Backend) -> tuple[ResolvedDataset, dict[str, Any] | None]:
    # Validate on every rank before the digest collective; propagate local failures through the existing backend vote.
    error = None
    contract = None
    try:
        tokenizer = Tokenizer(dataset.tokenizer_dir)
        contract = tokenizer.contract
        check_model_vocabulary(config, contract)
        if contract:
            with preserve_rng(torch.device("cpu")):
                check_template_parity(tokenizer)
    except Exception as caught:
        error = caught
    if backend.any_flag(error is not None):
        if error is not None:
            raise error
        raise RuntimeError("another rank failed tokenizer preflight")
    contracts = backend.all_gather_object(contract)
    if any(value != contract for value in contracts):
        raise ValueError("ranks selected different tokenizer contracts")
    if contract is None:
        return dataset, None

    # Publish one portable immutable copy; all dataloader workers subsequently reload this path.
    destination = run_directory / "tokenizer"
    def publish() -> None:
        if destination.exists():
            if validate_profile(destination) != contract:
                raise ValueError(f"existing run tokenizer at {destination} is incompatible")
            return
        with TemporaryDirectory(prefix=".tokenizer-", dir=run_directory) as staging:
            temporary = Path(staging) / "payload"
            copy_profile(Path(dataset.tokenizer_dir), temporary)
            os.replace(temporary, destination)
    complete_main_phase(backend, "run tokenizer publication", publish)
    error = None
    try:
        if validate_profile(destination) != contract:
            raise ValueError("published tokenizer differs from selected profile")
    except Exception as caught:
        error = caught
    if backend.any_flag(error is not None):
        if error is not None:
            raise error
        raise RuntimeError("another rank cannot validate the published run tokenizer")
    return replace(dataset, tokenizer_dir=str(destination)), contract


def check_checkpoint_tokenizer(saved: dict[str, Any] | None, selected: dict[str, Any] | None) -> None:
    if saved != selected:
        raise ValueError("checkpoint tokenizer identity differs from the selected tokenizer; settings/data overrides cannot migrate token IDs or chat formatting")


def resolve_checkpoint_tokenizer(state: dict[str, Any], checkpoint: Path, override: str | None) -> Path | None:
    contract = state.get("tokenizer_contract")
    if contract is None:
        if override is not None and validate_profile(override) is not None:
            raise ValueError("legacy checkpoint has no identity for the new chat tokenizer")
        return Path(override) if override else None
    if contract.get("relative_path") != "tokenizer":
        raise ValueError("invalid checkpoint tokenizer reference")
    path = Path(override) if override else checkpoint.parent.parent / "tokenizer"
    if not path.is_dir():
        raise ValueError(f"checkpoint tokenizer missing at {path}; copy the run tokenizer or supply --tokenizer_dir with the matching artifact")
    check_checkpoint_tokenizer(contract, validate_profile(path))
    return path
