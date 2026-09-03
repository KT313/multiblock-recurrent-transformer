# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Backend registry. Only the single-device backend exists; distributed backends are added here later.
"""

from collections.abc import Callable
from typing import Any

from training.backend.base import Backend
from training.backend.single_device import SingleDeviceBackend

BACKENDS: dict[str, Callable[..., Backend]] = {"single_device": SingleDeviceBackend}


def get_backend(name: str = "single_device", **kwargs: Any) -> Backend:
    """
    Instantiate the backend registered under `name` with the given keyword arguments.
    """

    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
