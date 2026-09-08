# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The HuggingFace wrapper around a live `RecurrentGPT` and the inference context both evaluation entry points use.

Every forward draws the latent state from the global torch RNG, so inference in the middle of a training run
happens under `torch.random.fork_rng`: afterwards the training stream continues as if nothing had run, and the
golden tests stay valid.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer

if TYPE_CHECKING:
    from model.hf.modeling import RecurrentGPTForCausalLM

RECURRENCE_ENV = "EVAL_RECURRENCE_STEPS"  # read by the wrapper's forward in eval mode ("12" or "4,12,4")
Recurrence = Sequence[int] | None  # recurrent steps per core block; None: the architecture's mean recurrence
MEAN_LABEL = "mean"


def recurrence_label(recurrence: Recurrence) -> str:
    """
    The name of a recurrence setting in file and metric names: "4-8-12", or "mean" for None.
    """

    return MEAN_LABEL if recurrence is None else "-".join(str(steps) for steps in recurrence)


def recurrence_env(recurrence: Recurrence) -> str | None:
    """
    The `EVAL_RECURRENCE_STEPS` value of a setting ("4,8,12"), None for the mean recurrence.
    """

    return None if recurrence is None else ",".join(str(steps) for steps in recurrence)


def hf_wrapper_around(model: RecurrentGPT, tokenizer: Tokenizer) -> RecurrentGPTForCausalLM:
    """
    `RecurrentGPTForCausalLM` over model's own tensors (nothing is copied), in eval mode, with the tokenizer's
    special ids in its generation config.
    """

    # `model.hf` imports transformers (seconds, hundreds of MB): only a run that samples or benchmarks pays for it
    from model.hf.modeling import RecurrentGPTConfig, RecurrentGPTForCausalLM

    with torch.device("meta"):
        wrapper = RecurrentGPTForCausalLM(RecurrentGPTConfig.from_recurrent_config(model.config))
    tensors = {f"model.{name}": tensor for name, tensor in model.state_dict().items()}
    tensors["model.freqs_cis"] = model.freqs_cis  # persistent in the wrapper only, so it is expected here
    wrapper.load_state_dict(tensors, assign=True)
    generation = wrapper.generation_config
    generation.pad_token_id = tokenizer.pad_id
    generation.eos_token_id = tokenizer.eos_id
    generation.bos_token_id = tokenizer.bos_id
    wrapper.train(False)
    return wrapper


def check_recurrence(recurrence: Recurrence, model: RecurrentGPT) -> None:
    """
    Fail on a recurrence setting that does not give one positive step count per core block of the model.
    """

    if recurrence is None:
        return
    blocks = len(model.transformer.core_blocks)
    if len(recurrence) != blocks or any(steps <= 0 for steps in recurrence):
        raise ValueError(
            f"recurrence {list(recurrence)} must give one positive step count per core block ({blocks} blocks)"
        )


@contextmanager
def isolated_inference(model: torch.nn.Module, recurrence: Recurrence = None, *, seed: int = 0) -> Iterator[None]:
    """
    Eval mode and no autograd for the block, the global RNGs (CPU and the model's CUDA device) seeded with seed
    inside and restored on exit (the recurrent blocks draw their initial state from them, so even greedy output
    depends on the RNG), the model back in the mode it had, and `EVAL_RECURRENCE_STEPS` set to recurrence for the
    duration, unset for the mean recurrence (a value left over from elsewhere must not win).
    """

    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    was_training = model.training
    previous = os.environ.get(RECURRENCE_ENV)
    env_value = recurrence_env(recurrence)
    if env_value is None:
        os.environ.pop(RECURRENCE_ENV, None)
    else:
        os.environ[RECURRENCE_ENV] = env_value
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            model.eval()
            yield
    finally:
        model.train(was_training)
        if previous is None:
            os.environ.pop(RECURRENCE_ENV, None)
        else:
            os.environ[RECURRENCE_ENV] = previous
