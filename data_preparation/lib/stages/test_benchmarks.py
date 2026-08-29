# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.stages.benchmarks with a stubbed `datasets` module."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from data_preparation.lib.stages import benchmarks as bm
from data_preparation.lib.stages.row_pipeline import get_ngram_set

WORDS = " ".join(f"q{i}" for i in range(15))


class FakeHub:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.failing = failing or set()

    def load_dataset(self, path: str, config: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((path, config, kwargs.get("split"), kwargs.get("cache_dir")))
        if path in self.failing:
            raise OSError(f"offline: {path}")
        return [{"question": WORDS, "choices": ["c1 " * 20, "c2"], "answer": 3}]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()
    module = types.ModuleType("datasets")
    module.load_dataset = fake.load_dataset  # type: ignore[attr-defined]  # fake module attribute
    monkeypatch.setitem(sys.modules, "datasets", module)
    return fake


def test_benchmark_ids_are_the_migrated_ones() -> None:
    assert bm.BENCHMARKS["math_test"] == ("EleutherAI/hendrycks_math", "all", "test")
    assert bm.BENCHMARKS["gsm8k_test"] == ("openai/gsm8k", "main", "test")
    assert all(split == "test" for _, _, split in bm.BENCHMARKS.values())


def test_example_text_joins_strings_and_string_lists() -> None:
    assert bm.example_text({"a": "x", "b": ["y", 2, "z"], "c": 3, "d": None}) == "x y z"


def test_load_benchmark_ngrams(hub: FakeHub) -> None:
    grams = bm.load_benchmark_ngrams(["gsm8k_test", "humaneval"], n=13, cache_dir="/cache")
    assert set(grams) == {"gsm8k_test", "humaneval"}
    expected = get_ngram_set(WORDS + " " + "c1 " * 20 + " c2", 13)
    assert grams["gsm8k_test"] == expected and len(expected) > 0
    assert hub.calls == [
        ("openai/gsm8k", "main", "test", "/cache"),
        ("openai/openai_humaneval", None, "test", "/cache"),
    ]


def test_load_failure_is_an_error(hub: FakeHub) -> None:
    hub.failing.add("Rowan/hellaswag")
    with pytest.raises(OSError, match="offline"):
        bm.load_benchmark_ngrams(["gsm8k_test", "hellaswag_test"])


def test_unknown_benchmark_name(hub: FakeHub) -> None:
    with pytest.raises(KeyError, match="unknown benchmark"):
        bm.load_benchmark_ngrams(["gsm8k_test", "nope"])
    assert hub.calls == []
