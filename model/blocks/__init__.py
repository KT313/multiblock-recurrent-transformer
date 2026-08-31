# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""How layers form a transformer block (`sandwich`) and how a core block is iterated (`recurrence`)."""

from .recurrence import (
    NumSteps,
    StepsPair,
    StepsSpec,
    canon_steps,
    core_block_forward,
    initialize_state,
    iterate_core_block,
    normalize_num_steps,
    sample_recurrence_steps,
)
from .sandwich import SandwichBlock

__all__ = [
    "NumSteps",
    "SandwichBlock",
    "StepsPair",
    "StepsSpec",
    "canon_steps",
    "core_block_forward",
    "initialize_state",
    "iterate_core_block",
    "normalize_num_steps",
    "sample_recurrence_steps",
]
