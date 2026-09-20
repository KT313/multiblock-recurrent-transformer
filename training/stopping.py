# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Healthy-rank cooperative stopping at completed training phases."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from data_preparation.lib.abort import StopCheck
from training.backend.base import Backend

T = TypeVar("T")


def complete_main_phase(backend: Backend, phase: str, operation: Callable[[], T]) -> T | None:
    """Finish rank-zero work before proceeding; propagate ordinary publication/phase errors on every rank.

    The operation must contain no collectives. Shared prerequisites (checkpoint RNG gathering, for example)
    belong before this function. The status vote orders completion before callers sample their local stop flag.
    Forced aborts and failures inside a collective are intentionally outside this healthy-rank protocol.
    """
    result = None
    error = None
    if backend.is_main:
        try:
            result = operation()
        except Exception as caught:
            error = caught
    if backend.any_flag(error is not None):
        if error is not None:
            raise error
        raise RuntimeError(f"rank zero failed during {phase}")
    return result


@dataclass
class StopController:
    """Only globally agreed requests affect control flow; phase names make the last boundary inspectable."""

    backend: Backend
    should_stop: StopCheck | None = None
    requested: bool = False
    boundary: str | None = None

    def poll(self, boundary: str) -> bool:
        self.boundary = boundary
        local = self.should_stop is not None and self.should_stop()
        self.requested = self.backend.any_flag(self.requested or local)
        return self.requested
