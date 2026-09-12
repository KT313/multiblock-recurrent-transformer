# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Sample generations: what the model writes after each prompt, saved as JSON lines.

Greedy by default (`temperature 0`); prompts are left-padded into batches, so every batch is one `generate` call
of the HuggingFace wrapper. Cached decoding retains each token's random initial latent per core;
`use_cache=False` selects the legacy full-prefix resampling semantics.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from data_preparation.lib.log import get_logger
from evaluation.prompts import DEFAULT_PROMPTS, Prompt
from evaluation.wrapper import Recurrence, check_recurrence
from evaluation.session import inference_session
from model.execution import ExecutionPolicy
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer

SAMPLES_DIR = "samples"  # under the run directory

log = get_logger(__name__)


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
    execution_policy: ExecutionPolicy | None = None,
    use_cache: bool = True,
) -> list[GeneratedSample]:
    """
    One completion per prompt: greedy when temperature is 0, sampled at that temperature otherwise; at most
    max_new_tokens tokens, cut at the first EOS. recurrence (steps per core block, e.g. [4, 4, 4]) overrides the
    model's mean recurrence. A prompt whose tokens plus max_new_tokens do not fit the model's position table is
    skipped with a warning (`_fitting_prompts`) instead of crashing the run.

    use_cache=True retains per-token/core latents and distinct per-recurrence K/V for this call only.
    It changes historical seeded samples; False restores legacy prefix resampling and full-head projection.

    seed seeds the isolated RNG, which is reseeded to `seed + <index of the batch's first prompt>` before each
    batch (the initial latent state and the sampling are drawn from it). So: the same prompts in the same order,
    with the same batch size, model and recurrence, give the same samples, whatever the global RNG state; and a
    prompt appended to the list leaves the batches before it unchanged. Inside a batch the latent state is drawn
    for all rows at once, so a row's noise depends on its neighbours and on the padded width: the same prompt under
    a different batch size, or twice in one batch, may complete differently.
    """

    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch_size and max_new_tokens must be positive")
    check_recurrence(recurrence, model)
    fitting = _fitting_prompts(prompts, tokenizer, max_new_tokens, model.config.model_max_sequence_length)
    samples: list[GeneratedSample] = []
    with inference_session(model, recurrence, seed=seed, execution_policy=execution_policy) as session:
        wrapper = session.hf_wrapper(tokenizer)
        generate = cast(Any, wrapper).generate  # set dynamically by transformers, invisible to the type checkers
        device = session.device
        for start in range(0, len(fitting), batch_size):
            batch = fitting[start : start + batch_size]
            width = max(len(ids) for _, ids in batch)
            input_ids = torch.full((len(batch), width), tokenizer.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, (_, ids) in enumerate(batch):  # left-padded: the wrapper derives the positions from the mask
                input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, width - len(ids) :] = 1
            sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
            # Reseed each batch so prior batches cannot affect its latent seed or token sampling stream.
            # Cached generation uses private normal generators; legacy forwards draw full-prefix latent states.
            torch.manual_seed(seed + start)
            output = generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_id,
                eos_token_id=tokenizer.eos_id,
                use_cache=use_cache,
                logits_to_keep=1 if use_cache else 0,
                **sampling,
            )
            for row, (prompt, _) in enumerate(batch):
                samples.append(_sample_from(prompt, output[row, width:].tolist(), tokenizer, recurrence))
    return samples


def _fitting_prompts(
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


def _sample_from(
    prompt: Prompt, generated_ids: list[int], tokenizer: Tokenizer, recurrence: Recurrence = None
) -> GeneratedSample:
    # `generate` pads a row only after its EOS, so the cut at the first EOS drops the filler too; a row without EOS
    # ran to max_new_tokens and every id in it, a pad id included, is model output (an untrained model emits them)
    eos_id = tokenizer.eos_id
    stopped_at_eos = eos_id in generated_ids
    if stopped_at_eos:
        generated_ids = generated_ids[: generated_ids.index(eos_id)]
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
    execution_policy: ExecutionPolicy | None = None,
    use_cache: bool = True,
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
                execution_policy=execution_policy,
                use_cache=use_cache,
            )
        )
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
    return samples
