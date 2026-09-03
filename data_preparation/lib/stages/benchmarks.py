# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Benchmark test sets used for decontamination and their n-gram sets.

BENCHMARKS maps the names used in decontamination.benchmarks to (hf_id, config, split).
A benchmark that cannot be loaded is an error (a build must never silently decontaminate against less).
"""

from __future__ import annotations

from typing import Any

from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.row_pipeline import get_ngram_set

log = get_logger(__name__)

# name -> (hf dataset id, config name or None, split)
BENCHMARKS: dict[str, tuple[str, str | None, str]] = {
    "gsm8k_test": ("openai/gsm8k", "main", "test"),
    "math_test": ("EleutherAI/hendrycks_math", "all", "test"),
    "humaneval": ("openai/openai_humaneval", None, "test"),
    "mbpp_test": ("google-research-datasets/mbpp", None, "test"),
    "arc_challenge_test": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "hellaswag_test": ("Rowan/hellaswag", None, "test"),
    "mmlu_test": ("cais/mmlu", "all", "test"),
    "winogrande_test": ("allenai/winogrande", "winogrande_xl", "test"),
}


def example_text(example: dict[str, Any]) -> str:
    """
    All string fields (and strings inside list fields) of a benchmark example, joined.
    """

    parts: list[str] = []
    for value in example.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(item for item in value if isinstance(item, str))
    return " ".join(parts)


def _benchmark_ngrams(name: str, n: int, cache_dir: str | None) -> set[str]:
    """
    Download one benchmark test set and collect the normalized n-grams of all its examples.
    """

    from datasets import load_dataset

    hf_id, config, split = BENCHMARKS[name]
    log.info("loading benchmark %s (%s)", name, hf_id)
    dataset: Any = load_dataset(hf_id, config, split=split, cache_dir=cache_dir)  # iterable of example dicts
    ngrams: set[str] = set()
    for example in dataset:
        ngrams.update(get_ngram_set(example_text(example), n))
    return ngrams


def load_benchmark_ngrams(names: list[str], n: int = 13, cache_dir: str | None = None) -> dict[str, set[str]]:
    """
    Download every named benchmark test set and collect its normalized n-grams.

    Raises KeyError for an unknown name and re-raises whatever datasets.load_dataset raises.
    """

    unknown = [name for name in names if name not in BENCHMARKS]
    if unknown:
        raise KeyError(f"unknown benchmark(s) {unknown}; known: {sorted(BENCHMARKS)}")

    all_ngrams: dict[str, set[str]] = {}
    for name in names:
        all_ngrams[name] = _benchmark_ngrams(name, n, cache_dir)
        log.info("  %s: %d %d-grams", name, len(all_ngrams[name]), n)
    total = sum(len(ngrams) for ngrams in all_ngrams.values())
    log.info("benchmarks: %d unique %d-grams in total", total, n)
    return all_ngrams
