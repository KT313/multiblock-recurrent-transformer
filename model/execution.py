# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Shared, caller-controlled precision contexts, also usable from flat HF exports."""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

import torch

PRECISIONS = ("bf16-mixed", "32")


@dataclass(frozen=True)
class ExecutionPolicy:
    """None is historical caller-controlled precision; 32 also preserves ambient autocast.

    This policy never casts parameters or changes global Torch precision flags. Constructing it does not
    initialize a device or load kernels. Forward contexts stay separate from backward/optimizer contexts.
    """

    precision: str | None = None

    def __post_init__(self) -> None:
        if self.precision is not None and self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return torch.bfloat16 if self.precision == "bf16-mixed" else None

    def autocast(self, device: torch.device | str) -> AbstractContextManager[None]:
        dtype = self.autocast_dtype
        if dtype is None:
            return nullcontext()
        return cast(AbstractContextManager[None], torch.autocast(device_type=torch.device(device).type, dtype=dtype))

    def check_custom_kernels(self, device: torch.device, *, enabled: bool) -> None:
        """Check the effective context before inference starts; detailed tensor checks remain in the kernels."""
        if enabled:
            from .kernels.runtime import require_supported

            require_supported(
                device.type == "cuda" and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16,
                "Inference", "Requires CUDA tensors under BF16 autocast; select precision='bf16-mixed'.",
            )
