# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Framework-neutral checks shared by run settings and optimizer entry points."""

from math import isfinite
from numbers import Real


def validate_scalar(value: object, name: str, *, positive: bool = False) -> None:
    """Reject coercions (especially bool), nonfinite values and invalid signs."""

    rule = "positive and finite" if positive else "non-negative and finite"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be {rule}, got {value!r}")
    if not isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be {rule}, got {value!r}")


def validate_adam_hyperparameters(*, betas: object, eps: object, weight_decay: object, prefix: str) -> None:
    """The shared Adam ranges; callers resolve optimizer-specific epsilon defaults."""

    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError(f"{prefix}.betas must contain exactly two coefficients in [0, 1), got {betas!r}")
    for index, beta in enumerate(betas):
        validate_scalar(beta, f"{prefix}.betas[{index}]")
        if beta >= 1:
            raise ValueError(f"{prefix}.betas[{index}] must be in [0, 1), got {beta!r}")
    validate_scalar(eps, f"{prefix}.eps")
    validate_scalar(weight_decay, f"{prefix}.weight_decay")
