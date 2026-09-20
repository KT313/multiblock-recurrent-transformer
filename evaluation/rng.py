# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Global RNG ownership for evaluation on CPU and a single model CUDA device.

Private generators (loader workers, few-shot samplers, cached latents) keep their own lifetimes. Native
sampling and the harness adapter seed only the generators they consume; in particular they never enqueue
an all-device seed that could overwrite an unrelated CUDA generator during later lazy initialization.
These process-global contexts support nesting, but must not overlap across threads.
"""

import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import torch

from training.failure import FatalHandler, handle_fatal_error


def seed_model_rng(seed: int, device: torch.device) -> None:
    """The CPU/model-device streams of torch.manual_seed, without its other-device side effects."""
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
    if device.type == "cuda":
        # A CUDA model already initialized this device. CPU evaluation never calls a CUDA seeding API.
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


@contextmanager
def preserve_rng(device: torch.device, on_fatal_error: FatalHandler | None = None) -> Iterator[None]:
    """Restore Python, NumPy legacy global RNG, Torch CPU and the model's CUDA generator.

    Callers must use scoped seeding, including third-party adapters: torch.manual_seed seeds all devices
    (even lazily). This context intentionally does not initialize or reseed unrelated GPUs.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    restore: list[Callable[[], object]] = [
        lambda: random.setstate(python_state),
        lambda: np.random.set_state(numpy_state),
        lambda: torch.set_rng_state(torch_state),
    ]
    if cuda_state is not None:
        restore.append(lambda: torch.cuda.set_rng_state(cuda_state, device))
    original: BaseException | None = None
    try:
        yield
    except BaseException as error:
        original = error
        handle_fatal_error(on_fatal_error, error)
        raise
    finally:
        # Attempt every restoration even if one fails, and retain the original evaluation exception.
        failures: list[BaseException] = []
        for reset in restore:
            try:
                reset()
            except BaseException as error:
                handle_fatal_error(on_fatal_error, error)
                failures.append(error)
        if failures:
            primary = original if original is not None else failures[0]
            for failure in failures:
                primary.add_note(f"Evaluation RNG restoration failed: {failure!r}")
            if original is None:
                raise primary
