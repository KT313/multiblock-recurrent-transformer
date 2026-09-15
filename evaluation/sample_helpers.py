# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Prompt batching, completion decoding and JSONL output for sample generation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from data_preparation.lib.log import get_logger
from evaluation.prompts import Prompt
from evaluation.wrapper import Recurrence
from model.execution import ExecutionPolicy
from training.data.tokenizer import Tokenizer

log = get_logger("evaluation.samples")  # preserve the logger identity of the sample-generation entry point


@dataclass
class GeneratedSample:
    prompt: str
    kind: str
    completion: str
    new_tokens: int  # generated tokens before the EOS (or the cap)
    stopped_at_eos: bool
    recurrence: list[int] | None = None  # recurrent steps per block the sample was generated with; None: the mean


def select_fitting_prompts(
    prompts: Sequence[Prompt], tokenizer: Tokenizer, max_new_tokens: int, model_max_sequence_length: int
) -> list[tuple[Prompt, list[int]]]:
    """
    The prompts that can be generated from, with their token ids: prompt tokens plus max_new_tokens must fit the
    model's position table (`model_max_sequence_length`). A longer prompt is left out with a warning naming it
    (a training run must not die on one entry of a prompts file).
    """

    fitting: list[tuple[Prompt, list[int]]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt.text, bos=True)
        if len(ids) + max_new_tokens > model_max_sequence_length:
            log.warning(
                "prompt %r skipped: its %d tokens plus max_new_tokens %d exceed the model's %d positions",
                prompt.text[:60], len(ids), max_new_tokens, model_max_sequence_length
            )
            continue
        fitting.append((prompt, ids))
    return fitting


def build_prompt_batch(batch: list[tuple[Prompt, list[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    width = max(len(ids) for _, ids in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (_, ids) in enumerate(batch):  # left-padded: the wrapper derives positions from the mask
        input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, width - len(ids) :] = 1
    return input_ids, attention_mask, width


def decode_generated_sample(
    prompt: Prompt, generated_ids: list[int], tokenizer: Tokenizer, recurrence: Recurrence = None
) -> GeneratedSample:
    """Cut at the first EOS to drop trailing filler; without EOS, even pad IDs are generated output."""
    eos_id = tokenizer.eos_id
    stopped_at_eos = eos_id in generated_ids
    if stopped_at_eos:
        generated_ids = generated_ids[: generated_ids.index(eos_id)]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    steps = None if recurrence is None else [int(value) for value in recurrence]
    return GeneratedSample(prompt.text, prompt.kind, completion, len(generated_ids), stopped_at_eos, steps)


def save_generated_samples(
    samples: list[GeneratedSample], out_path: Path, *, step: int, temperature: float, max_new_tokens: int,
    seed: int, batch_size: int, use_cache: bool, execution_policy: ExecutionPolicy | None,
) -> None:
    decoding: dict[str, float | str | int | None] = {
        "temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed, "batch_size": batch_size,
        "use_cache": use_cache, "latent_policy": "fixed_per_token" if use_cache else "resample_prefix",
        "latent_rng": "per_core_token_columns_v1" if use_cache else "global_prefix_v1",
        "logits_to_keep": 1 if use_cache else 0,
    }
    if execution_policy is not None:
        decoding["execution_precision"] = execution_policy.precision
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps({"step": step, **asdict(sample), "decoding": decoding}, ensure_ascii=False) + "\n")
