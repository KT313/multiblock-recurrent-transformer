# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Temporary eval mode, preserving deliberately mixed child training flags."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch.nn import Module


@contextmanager
def evaluation_mode(model: Module) -> Iterator[None]:
    """Restore exact flags on exit without recursively overwriting a child's original mode."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
