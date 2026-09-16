# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The smoothed steps-per-second estimate behind the ETA of the bars, the metrics table and the fallback lines.
"""

from __future__ import annotations

import time

from training.ui.common import Clock

RATE_SMOOTHING = 0.1  # weight of the newest seconds/step sample in the EMA behind the ETA


class Throughput:
    """
    Smoothed seconds per optimizer step from the times :meth:`record` is called with, and the ETA derived from it.

    Ignore startup through the first actual optimizer update, including its compilation. Fresh training skips
    optimizer.step at step index 0, so completed steps 1 and 2 are warmup; a resumed run skips its first completed
    interval. The next complete interval initializes the estimate; subsequent intervals use RATE_SMOOTHING.
    If the caller reports several steps at once, discard the whole interval that contains warmup rather than
    guessing which portion was compilation. Elapsed wall time is unchanged. discount() excludes non-training
    blocks (evaluation, checkpointing, etc.) from the next rate sample.
    """

    def __init__(self, total_steps: int, *, start_step: int = 0, clock: Clock = time.monotonic) -> None:
        self._total_steps = total_steps
        self._clock = clock
        self._started = clock()
        self._last_time = self._started
        self._last_step = start_step
        self.seconds_per_step: float | None = None
        self._warmup_until_step = max(start_step, 1) + 1
        self._warming_up = True

    def record(self, step: int) -> None:
        """
        Note that step optimizer steps are done now (a step not beyond the last recorded one is ignored).
        """

        now = self._clock()
        advanced = step - self._last_step
        if advanced <= 0:
            return
        sample = max(now - self._last_time, 0.0) / advanced  # `discount` can have moved the interval start past now
        self._last_time = now
        self._last_step = step
        if self._warming_up:
            self._warming_up = step < self._warmup_until_step
            return
        previous = self.seconds_per_step  # None for the first sample only; the check below narrows the type
        if previous is None:
            self.seconds_per_step = sample
        else:
            self.seconds_per_step = (1 - RATE_SMOOTHING) * previous + RATE_SMOOTHING * sample

    def discount(self, seconds: float) -> None:
        """
        `seconds` of the running interval were not training (an evaluation, a checkpoint write, samples,
        benchmarks): the next :meth:`record` measures the interval without them. `elapsed` stays wall time.
        """

        self._last_time += seconds

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    @property
    def steps_per_second(self) -> float | None:
        if not self.seconds_per_step:
            return None
        return 1.0 / self.seconds_per_step

    def remaining(self, step: int) -> float | None:
        """
        Estimated seconds until total_steps (None until a post-warmup interval is available).
        """

        if self.seconds_per_step is None:
            return None
        return self.seconds_per_step * max(self._total_steps - step, 0)
