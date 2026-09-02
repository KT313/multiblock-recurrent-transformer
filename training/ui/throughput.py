# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The smoothed steps-per-second estimate behind the ETA of the bars, the metrics table and the fallback lines."""

from __future__ import annotations

import time

from training.ui.common import Clock

RATE_SMOOTHING = 0.1  # weight of the newest seconds/step sample in the exponential moving average behind the ETA


class Throughput:
    """Smoothed seconds per optimizer step from the times :meth:`record` is called with, and the ETA derived from it.

    The first interval sets the estimate, the second one replaces it (the first interval of a run holds the
    ``torch.compile`` and the loader start-up, an outlier that would otherwise dominate the ETA for dozens of steps),
    later ones move it by :data:`RATE_SMOOTHING` (an exponential moving average — a stall or one slow evaluation step
    does not swing the ETA). ``start_step`` is the step the run (re)starts at, so a resumed run does not count the
    checkpointed steps as done in zero seconds.
    """

    def __init__(self, total_steps: int, *, start_step: int = 0, clock: Clock = time.monotonic) -> None:
        self._total_steps = total_steps
        self._clock = clock
        self._started = clock()
        self._last_time = self._started
        self._last_step = start_step
        self.seconds_per_step: float | None = None
        self._samples = 0

    def record(self, step: int) -> None:
        """Note that ``step`` optimizer steps are done now (a step not beyond the last recorded one is ignored)."""
        now = self._clock()
        advanced = step - self._last_step
        if advanced <= 0:
            return
        sample = (now - self._last_time) / advanced
        self._samples += 1
        previous = self.seconds_per_step  # None for the first sample only; the check below narrows the type
        if previous is None or self._samples <= 2:
            self.seconds_per_step = sample  # the first sample is provisional, the second replaces it
        else:
            self.seconds_per_step = (1 - RATE_SMOOTHING) * previous + RATE_SMOOTHING * sample
        self._last_time = now
        self._last_step = step

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    @property
    def steps_per_second(self) -> float | None:
        if not self.seconds_per_step:
            return None
        return 1.0 / self.seconds_per_step

    def remaining(self, step: int) -> float | None:
        """Estimated seconds until ``total_steps`` (None before the first interval)."""
        if self.seconds_per_step is None:
            return None
        return self.seconds_per_step * max(self._total_steps - step, 0)
