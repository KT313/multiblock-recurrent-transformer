# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Sample generations: what the model writes after each prompt, saved as JSON lines.

Greedy by default (`temperature 0`); prompts are left-padded into batches, so every batch is one `generate` call
of the HuggingFace wrapper. Cached decoding retains each token's random initial latent per core;
`use_cache=False` selects the legacy full-prefix resampling semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from evaluation.prompts import DEFAULT_PROMPTS, Prompt
from evaluation.rng import seed_model_rng
from evaluation.sample_helpers import (
    GeneratedSample,
    build_prompt_batch,
    decode_generated_sample,
    save_generated_samples,
    select_fitting_prompts,
)
from evaluation.session import inference_session
from evaluation.wrapper import Recurrence, check_recurrence
from model.execution import ExecutionPolicy
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer

_sample_from = decode_generated_sample  # preserve existing helper imports
_fitting_prompts = select_fitting_prompts

SAMPLES_DIR = "samples"  # under the run directory


__all__ = [
    "GeneratedSample",
    "SAMPLES_DIR",
    "generate_and_save_samples",
    "generate_samples",
    "samples_path",
]


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
    skipped with a warning (`select_fitting_prompts`) instead of crashing the run.

    use_cache=True retains per-token/core latents and distinct per-recurrence K/V for this call only.
    It changes historical seeded samples; False restores legacy prefix resampling and full-head projection.

    seed seeds the isolated RNG, which is reseeded to `seed + <index of the batch's first prompt>` before each
    batch (the initial latent state and the sampling are drawn from it). So: the same prompts in the same order,
    with the same batch size, model and recurrence, give the same samples, whatever the global RNG state; and a
    prompt appended to the list leaves the batches before it unchanged. Inside a batch the latent state is drawn
    for all rows at once, so a row's noise depends on its neighbours and on the padded width: the same prompt under
    a different batch size, or twice in one batch, may complete differently.
    """

    # validate generation limits and select prompts that fit the position table
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch_size and max_new_tokens must be positive")
    check_recurrence(recurrence, model)
    fitting = select_fitting_prompts(prompts, tokenizer, max_new_tokens, model.config.model_max_sequence_length)
    samples: list[GeneratedSample] = []

    # generate each prompt batch inside one isolated inference session
    with inference_session(model, recurrence, seed=seed, execution_policy=execution_policy) as session:
        wrapper = session.hf_wrapper(tokenizer)
        generate = cast(Any, wrapper).generate  # set dynamically by transformers, invisible to the type checkers
        device = session.device
        for start in range(0, len(fitting), batch_size):
            # prepare the left-padded inputs and sampling options
            batch = fitting[start : start + batch_size]
            input_ids, attention_mask, width = build_prompt_batch(batch, tokenizer.pad_id)
            sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}

            # generate from this batch's independent seed
            seed_model_rng(seed + start, device)  # prior batches cannot affect latent or token sampling streams
            output = generate(
                input_ids.to(device), attention_mask=attention_mask.to(device), max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_id, eos_token_id=tokenizer.eos_id, use_cache=use_cache,
                logits_to_keep=1 if use_cache else 0, **sampling,
            )

            # Return only the generated suffix, retaining unpadded prompt context for chat decoding.
            for row, (prompt, prompt_ids) in enumerate(batch):
                samples.append(decode_generated_sample(
                    prompt, output[row, width:].tolist(), tokenizer, recurrence, prompt_ids=prompt_ids
                ))
    return samples


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

    # collect each recurrence setting's samples in the requested order
    samples: list[GeneratedSample] = []
    for recurrence in recurrences:
        samples.extend(generate_samples(
            model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
            recurrence=recurrence, batch_size=batch_size, seed=seed, execution_policy=execution_policy,
            use_cache=use_cache,
        ))

    # write completions together with the decoding settings that produced them
    save_generated_samples(
        samples, out_path, step=step, temperature=temperature, max_new_tokens=max_new_tokens,
        seed=seed, batch_size=batch_size, use_cache=use_cache, execution_policy=execution_policy,
    )
    return samples
