# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
When the sample generations and the benchmarks run: every `interval` steps and after the steps that correspond to
listed training-progress percentages, both resolved to step numbers once the total step count is known.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


def progress_steps(percentages: Iterable[float], total_steps: int) -> list[int]:
    """
    The completed-step numbers matching training-progress percentages (0 is after the first step, 100 after the
    last, in between rounded to the nearest step), sorted and unique.
    """

    steps = {max(1, min(total_steps, int(total_steps * percentage / 100 + 0.5))) for percentage in percentages}
    return sorted(steps)


@dataclass(frozen=True)
class StepTriggers:
    """
    The steps an action runs after: every `interval` steps (0: none) and the fixed `steps`.
    """

    interval: int
    steps: frozenset[int]

    @classmethod
    def from_settings(cls, interval: int, percentages: Iterable[float], total_steps: int) -> StepTriggers:
        return cls(interval, frozenset(progress_steps(percentages, total_steps)))

    def due(self, completed_steps: int) -> bool:
        """
        Whether the action runs after `completed_steps` completed steps.
        """

        return (self.interval > 0 and completed_steps % self.interval == 0) or completed_steps in self.steps

    def listed(self, total_steps: int) -> list[int]:
        """
        Every step the action runs after within a run of `total_steps` steps, sorted.
        """

        return sorted(step for step in range(1, total_steps + 1) if self.due(step))
