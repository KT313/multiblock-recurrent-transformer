# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Benchmark test sets used for decontamination and their n-gram sets.

BENCHMARKS maps the names used in decontamination.benchmarks to (hf_id, config, split, revision). The
revision pins the Hub repo at a commit: what a decontaminated build was filtered against must not move with the
repo's main branch, so it enters the processed hash (:meth:`DatasetConfig.processed_hash`) and a re-pin rebuilds.
A benchmark that cannot be loaded is an error (a build must never silently decontaminate against less).
"""

from __future__ import annotations

from typing import Any, NamedTuple

from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.row_pipeline import get_ngram_set

log = get_logger(__name__)


class Benchmark(NamedTuple):
    hf_id: str
    config: str | None
    split: str
    revision: str  # Hub commit sha, resolved once with `HfApi().dataset_info(hf_id).sha` and pasted here


BENCHMARKS: dict[str, Benchmark] = {
    "gsm8k_test": Benchmark("openai/gsm8k", "main", "test", "740312add88f781978c0658806c59bc2815b9866"),
    "math_test": Benchmark("EleutherAI/hendrycks_math", "all", "test", "21a5633873b6a120296cce3e2df9d5550074f4a3"),
    "humaneval": Benchmark("openai/openai_humaneval", None, "test", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"),
    "mbpp_test": Benchmark("google-research-datasets/mbpp", None, "test", "4bb6404fdc6cacfda99d4ac4205087b89d32030c"),
    "arc_challenge_test": Benchmark("allenai/ai2_arc", "ARC-Challenge", "test", "210d026faf9955653af8916fad021475a3f00453"),
    "hellaswag_test": Benchmark("Rowan/hellaswag", None, "test", "218ec52e09a7e7462a5400043bb9a69a41d06b76"),
    "mmlu_test": Benchmark("cais/mmlu", "all", "test", "c30699e8356da336a370243923dbaf21066bb9fe"),
    "winogrande_test": Benchmark("allenai/winogrande", "winogrande_xl", "test", "01e74176c63542e6b0bcb004dcdea22d94fb67b5"),
}


def benchmark_revisions(names: list[str]) -> dict[str, str]:
    """
    name -> pinned revision of every named benchmark; KeyError for an unknown name.
    """

    unknown = [name for name in names if name not in BENCHMARKS]
    if unknown:
        raise KeyError(f"unknown benchmark(s) {unknown}; known: {sorted(BENCHMARKS)}")
    return {name: BENCHMARKS[name].revision for name in names}


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

    hf_id, config, split, revision = BENCHMARKS[name]
    log.info("loading benchmark %s (%s@%s)", name, hf_id, revision[:8])
    dataset: Any = load_dataset(hf_id, config, split=split, revision=revision, cache_dir=cache_dir)  # iterable of example dicts
    ngrams: set[str] = set()
    for example in dataset:
        ngrams.update(get_ngram_set(example_text(example), n))
    return ngrams


def load_benchmark_ngrams(names: list[str], n: int = 13, cache_dir: str | None = None) -> dict[str, set[str]]:
    """
    Download every named benchmark test set and collect its normalized n-grams.

    Raises KeyError for an unknown name and re-raises whatever datasets.load_dataset raises.
    """

    benchmark_revisions(names)  # the unknown-name check
    all_ngrams: dict[str, set[str]] = {}
    for name in names:
        all_ngrams[name] = _benchmark_ngrams(name, n, cache_dir)
        log.info("  %s: %d %d-grams", name, len(all_ngrams[name]), n)
    total = sum(len(ngrams) for ngrams in all_ngrams.values())
    log.info("benchmarks: %d unique %d-grams in total", total, n)
    return all_ngrams
