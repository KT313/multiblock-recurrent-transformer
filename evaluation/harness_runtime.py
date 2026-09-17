# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bound harness CPU workers without changing official metric aggregation."""
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
import math


@contextmanager
def use_serial_bootstrap() -> Iterator[None]:
    previous = os.environ.get("DISABLE_MULTIPROC")
    os.environ["DISABLE_MULTIPROC"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DISABLE_MULTIPROC", None)
        else:
            os.environ["DISABLE_MULTIPROC"] = previous


def check_benchmark_results(results: Any) -> None:
    if not isinstance(results, Mapping) or not results.get("results"):
        raise ValueError("benchmark evaluator returned no task results")
    for task, metrics in results["results"].items():
        primary = [(key, value) for key, value in metrics.items() if "," in key and "_stderr," not in key]
        if not primary:
            raise ValueError(f"benchmark {task} returned no primary metrics")
        for key, value in primary:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"benchmark {task}/{key} returned invalid primary score {value!r}")
