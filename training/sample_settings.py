# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Validate sample temperatures without importing the model or inference runtime."""
import math


def normalize_sample_temperatures(value: float | list[float]) -> list[float]:
    temperatures = value if isinstance(value, list) else [value]
    if not temperatures or any(isinstance(item, bool) or not isinstance(item, (int, float))
                               or not math.isfinite(item) or item < 0 for item in temperatures):
        raise ValueError("sample_temperature must be a finite number >= 0 or a nonempty list of such numbers (0: greedy)")
    return [float(item) for item in temperatures]
