# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded synchronous inference communication on the training-owned backend."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from importlib.metadata import version
from typing import Any, TypeVar

import torch

from model.model import RecurrentGPT
from tokenization.validation import check_model_vocabulary
from training.backend.base import Backend
from training.data.tokenizer import Tokenizer
from training.stopping import StopController

T = TypeVar("T")
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
MAX_JOB_BYTES = 1024 * 1024


class EvaluationCancelled(BaseException):
    """Internal, collectively agreed cancellation; never an arbitrary worker failure."""


def check_payload(value: object, limit: int = MAX_PAYLOAD_BYTES) -> None:
    size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    if size > limit:
        raise ValueError(f"evaluation payload is {size} bytes (limit {limit}); reduce job or prompt size")


def exchange(backend: Backend, value: T) -> list[T]:
    check_payload(value)
    with torch.inference_mode(False):
        return backend.all_gather_object(value)


def share_from_main(backend: Backend, value: T | None) -> T:
    values = exchange(backend, value)
    if values[0] is None or any(item is not None for item in values[1:]):
        raise RuntimeError("evaluation requires exactly one rank-zero command")
    return values[0]


def poll_stop(stop: StopController, boundary: str) -> bool:
    with torch.inference_mode(False):
        return stop.poll(boundary)


def finish_publication(backend: Backend) -> None:
    with torch.inference_mode(False):
        backend.barrier()


def agree_on_phase(
    backend: Backend, model: RecurrentGPT, tokenizer: Tokenizer, settings: dict[str, Any],
) -> None:
    check_model_vocabulary(model.config, tokenizer.contract, model)
    if min(model.config.vocab_size, model.transformer.wte.num_embeddings, model.lm_head.out_features) < len(tokenizer):
        raise ValueError("model embedding table is smaller than the selected tokenizer")
    digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        path = tokenizer.path / name
        digest.update(name.encode())
        if path.exists():
            digest.update(path.read_bytes())
    record = {
        "settings": settings, "model": asdict(model.config), "tokenizer": tokenizer.contract,
        "tokenizer_files": digest.hexdigest(), "special_ids": [tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id],
        "precision": backend.execution_policy.precision,
        "versions": {name: version(name) for name in ("torch", "transformers", "lm_eval")},
    }
    fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    fingerprints = exchange(backend, fingerprint)
    if len(set(fingerprints)) != 1:
        raise ValueError("evaluation configuration/tokenizer/dependencies differ across ranks")
