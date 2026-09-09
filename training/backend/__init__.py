# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Backend registry: `single_device` (one GPU or the CPU) and `ddp` (one process per GPU under torchrun); the run
config's `backend` key picks one (`training.run.create_backend`).
"""

from collections.abc import Callable
from typing import Any

from training.backend.base import Backend
from training.backend.ddp import DDPBackend
from training.backend.single_device import SingleDeviceBackend

BACKENDS: dict[str, Callable[..., Backend]] = {"single_device": SingleDeviceBackend, "ddp": DDPBackend}


def get_backend(name: str = "single_device", **kwargs: Any) -> Backend:
    """
    Instantiate the backend registered under `name` with the given keyword arguments.
    """

    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
