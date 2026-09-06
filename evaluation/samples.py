# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Sample generations: what the model writes after each prompt, saved as JSON lines.

Greedy by default (`temperature 0`); prompts are left-padded into batches, so every batch is one `generate` call
of the HuggingFace wrapper (no KV cache: the whole sequence is recomputed per token).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from evaluation.prompts import DEFAULT_PROMPTS, Prompt
from evaluation.wrapper import Recurrence, check_recurrence, hf_wrapper_around, isolated_inference
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer

SAMPLES_DIR = "samples"  # under the run directory


@dataclass
class GeneratedSample:
    prompt: str
    kind: str
    completion: str
    new_tokens: int  # generated tokens before the EOS (or the cap)
    stopped_at_eos: bool
    recurrence: list[int] | None = None  # recurrent steps per block the sample was generated with; None: the mean


def samples_path(run_directory: Path, step: int) -> Path:
    return run_directory / SAMPLES_DIR / f"step-{step:08d}.jsonl"


def generate_samples(
    model: RecurrentGPT,
    tokenizer: Tokenizer,
    prompts: Sequence[Prompt] = DEFAULT_PROMPTS,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    recurrence: Recurrence = None,
    batch_size: int = 8,
    seed: int = 0,
) -> list[GeneratedSample]:
    """
    One completion per prompt: greedy when temperature is 0, sampled at that temperature otherwise; at most
    max_new_tokens tokens, cut at the first EOS. recurrence (steps per core block, e.g. [4, 4, 4]) overrides the
    model's mean recurrence. seed seeds the isolated RNG (the initial latent state, and the sampling), so the
    output is reproducible whatever the global RNG state.
    """

    check_recurrence(recurrence, model)
    samples: list[GeneratedSample] = []
    with isolated_inference(model, recurrence, seed=seed):
        wrapper = hf_wrapper_around(model, tokenizer)
        generate = cast(Any, wrapper).generate  # set dynamically by transformers, invisible to the type checkers
        device = next(model.parameters()).device
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = [tokenizer.encode(prompt.text, bos=True) for prompt in batch]
            width = max(len(ids) for ids in encoded)
            input_ids = torch.full((len(batch), width), tokenizer.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, ids in enumerate(encoded):  # left-padded: the wrapper derives the positions from the mask
                input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, width - len(ids) :] = 1
            sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
            output = generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_id,
                eos_token_id=tokenizer.eos_id,
                **sampling,
            )
            for row, prompt in enumerate(batch):
                samples.append(_sample_from(prompt, output[row, width:].tolist(), tokenizer, recurrence))
    return samples


def _sample_from(
    prompt: Prompt, generated_ids: list[int], tokenizer: Tokenizer, recurrence: Recurrence = None
) -> GeneratedSample:
    eos_id = tokenizer.eos_id
    stopped_at_eos = eos_id is not None and eos_id in generated_ids
    if eos_id is not None and stopped_at_eos:
        generated_ids = generated_ids[: generated_ids.index(eos_id)]
    else:  # a row that finished early is padded by `generate`
        while generated_ids and generated_ids[-1] == tokenizer.pad_id:
            generated_ids.pop()
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    steps = None if recurrence is None else [int(value) for value in recurrence]
    return GeneratedSample(prompt.text, prompt.kind, completion, len(generated_ids), stopped_at_eos, steps)


def generate_and_save_samples(
    model: RecurrentGPT,
    tokenizer: Tokenizer,
    out_path: Path,
    *,
    step: int,
    prompts: Sequence[Prompt] = DEFAULT_PROMPTS,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    recurrences: Sequence[Recurrence] = (None,),
    batch_size: int = 8,
    seed: int = 0,
) -> list[GeneratedSample]:
    """
    `generate_samples` once per recurrence setting, then one JSON line per sample in out_path (parents created):
    the sample's fields (its `recurrence` included) plus `step` and the decoding settings.
    """

    samples: list[GeneratedSample] = []
    for recurrence in recurrences:
        samples.extend(
            generate_samples(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                recurrence=recurrence,
                batch_size=batch_size,
                seed=seed,
            )
        )
    decoding = {"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps({"step": step, **asdict(sample), "decoding": decoding}, ensure_ascii=False) + "\n")
    return samples
