# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One operation's precision, RNG, recurrence and model-mode lifetime."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from evaluation.wrapper import RECURRENCE_ENV, Recurrence, check_recurrence, hf_wrapper_around, recurrence_env
from model.execution import ExecutionPolicy
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer

if TYPE_CHECKING:
    from model.hf.modeling import RecurrentGPTForCausalLM


@dataclass(frozen=True)
class InferenceSession:
    model: torch.nn.Module
    device: torch.device
    execution_policy: ExecutionPolicy

    def hf_wrapper(self, tokenizer: Tokenizer) -> "RecurrentGPTForCausalLM":
        if not isinstance(self.model, RecurrentGPT):
            raise TypeError("a live HF wrapper requires RecurrentGPT")
        return hf_wrapper_around(self.model, tokenizer)

    @property
    def mixed_precision_dtype(self) -> torch.dtype | None:
        # HFLM opens its own autocast context; an outer context alone cannot enable BF16 there.
        return self.execution_policy.autocast_dtype


@contextmanager
def inference_session(
    model: torch.nn.Module, recurrence: Recurrence = None, *, seed: int = 0,
    execution_policy: ExecutionPolicy | None = None,
) -> Iterator[InferenceSession]:
    """Preserve the operation seed and CPU/model-CUDA RNG, including exceptional exits.

    Python/NumPy and other CUDA-device RNG isolation retain their historical limitations. No parameter is
    constructed, copied, moved or cast here; the live wrapper shares the original versioned tensors.
    """
    policy = execution_policy or ExecutionPolicy()
    device = next(model.parameters()).device
    if isinstance(model, RecurrentGPT):
        check_recurrence(recurrence, model)
    devices = [device.index or 0] if device.type == "cuda" else []
    was_training = model.training
    previous = os.environ.get(RECURRENCE_ENV)
    env_value = recurrence_env(recurrence)
    if env_value is None:
        os.environ.pop(RECURRENCE_ENV, None)
    else:
        os.environ[RECURRENCE_ENV] = env_value
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), policy.autocast(device):
            torch.manual_seed(seed)
            model.eval()
            if isinstance(model, RecurrentGPT) and policy.precision is not None:
                # Legacy callers can enter their own autocast inside this context; kernels still enforce support.
                policy.check_custom_kernels(device, enabled=model.config.use_custom_kernels)
            yield InferenceSession(model, device, policy)
    finally:
        model.train(was_training)
        if previous is None:
            os.environ.pop(RECURRENCE_ENV, None)
        else:
            os.environ[RECURRENCE_ENV] = previous
