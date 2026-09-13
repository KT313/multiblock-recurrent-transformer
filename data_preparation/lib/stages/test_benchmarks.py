# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.stages.benchmarks with a stubbed `datasets` module.
"""

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
        self.calls.append((path, config, kwargs.get("split"), kwargs.get("cache_dir"), kwargs.get("revision")))
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


def test_benchmark_ids_are_the_migrated_ones_pinned_to_a_commit() -> None:
    assert bm.BENCHMARKS["math_test"][:3] == ("EleutherAI/hendrycks_math", "all", "test")
    assert bm.BENCHMARKS["gsm8k_test"][:3] == ("openai/gsm8k", "main", "test")
    assert all(benchmark.split == "test" for benchmark in bm.BENCHMARKS.values())
    assert all(len(benchmark.revision) == 40 and int(benchmark.revision, 16) >= 0 for benchmark in bm.BENCHMARKS.values()), "a full commit sha each"


def test_benchmark_revisions_are_the_pins_and_reject_unknown_names() -> None:
    assert bm.benchmark_revisions(["humaneval", "gsm8k_test"]) == {
        "humaneval": bm.BENCHMARKS["humaneval"].revision, "gsm8k_test": bm.BENCHMARKS["gsm8k_test"].revision,
    }  # fmt: skip
    assert bm.benchmark_revisions([]) == {}
    with pytest.raises(KeyError, match="unknown benchmark"):
        bm.benchmark_revisions(["gsm8k_test", "nope"])


def test_example_text_joins_strings_and_string_lists() -> None:
    assert bm.example_text({"a": "x", "b": ["y", 2, "z"], "c": 3, "d": None}) == "x y z"


def test_load_benchmark_ngrams(hub: FakeHub) -> None:
    grams = bm.load_benchmark_ngrams(["gsm8k_test", "humaneval"], n=13, cache_dir="/cache")
    assert set(grams) == {"gsm8k_test", "humaneval"}
    expected = get_ngram_set(WORDS + " " + "c1 " * 20 + " c2", 13)
    assert grams["gsm8k_test"] == expected and len(expected) > 0
    assert hub.calls == [
        ("openai/gsm8k", "main", "test", "/cache", bm.BENCHMARKS["gsm8k_test"].revision),
        ("openai/openai_humaneval", None, "test", "/cache", bm.BENCHMARKS["humaneval"].revision),
    ], "every load is pinned to the benchmark's commit"


def test_load_failure_is_an_error(hub: FakeHub) -> None:
    hub.failing.add("Rowan/hellaswag")
    with pytest.raises(OSError, match="offline"):
        bm.load_benchmark_ngrams(["gsm8k_test", "hellaswag_test"])


def test_unknown_benchmark_name(hub: FakeHub) -> None:
    with pytest.raises(KeyError, match="unknown benchmark"):
        bm.load_benchmark_ngrams(["gsm8k_test", "nope"])
    assert hub.calls == []


EXACT_EXAMPLES: dict[str, dict[str, Any]] = {
    "arc_challenge": {"question": "Which planet?", "choices": {"text": ["Earth", "Mars"], "label": ["A", "B"]}, "answerKey": "B", "id": "ignored"},
    "hellaswag": {"ctx": "Someone walks.", "ctx_a": "Someone", "ctx_b": "walks.", "activity_label": "Walking", "endings": ["North", "South", "East", "West"], "label": "0"},
    "mmlu": {"subject": "astronomy", "question": "Which planet?", "choices": ["Earth", "Mars", "Venus", "Jupiter"], "answer": 1},
    "winogrande": {"sentence": "The _ is red.", "option1": "apple", "option2": "sky", "answer": "1"},
}
EXACT_RENDERINGS = {
    "arc_challenge": ("Which planet?\nA. Earth\nB. Mars", "B. Mars"),
    "hellaswag": ("Activity: Walking\nSomeone walks.\nA. North\nB. South\nC. East\nD. West", "A. North"),
    "mmlu": ("Subject: astronomy\nWhich planet?\nA. Earth\nB. Mars\nC. Venus\nD. Jupiter", "B. Mars"),
    "winogrande": ("The _ is red.\n1. apple\n2. sky", "1. apple"),
}


@pytest.mark.parametrize("name", EXACT_EXAMPLES)
def test_complete_example_adapters_and_matching_boundary(name: str) -> None:
    from copy import deepcopy

    from data_preparation.lib.stages.benchmark_seeds import complete_example, example_keys
    from data_preparation.lib.stages.global_dedup import GlobalAdmission, GlobalFrontier, global_key

    original = deepcopy(EXACT_EXAMPLES[name])
    prompt, answer = EXACT_RENDERINGS[name]
    assert complete_example(name, original) == (prompt, answer)
    keys = example_keys(name, original)
    assert keys == (global_key("instruct", {"instruction": prompt, "input": "", "output": answer}),
                    global_key("pretrain", {"text": f"{prompt}\nAnswer: {answer}"}))
    admission = GlobalAdmission(("a", "b"), memory_mb=1, preseed_keys=[*keys, *keys])
    kept: list[dict[str, Any]] = []

    def publish(rows: list[dict[str, Any]], frontier: GlobalFrontier) -> None:
        kept.extend(rows)

    admission.commit_batch("a", "instruct", [
        {"instruction": prompt.upper().replace("\n", "  "), "input": None, "output": answer.upper()},
        {"instruction": prompt, "input": "", "output": "changed answer"},
        {"instruction": "Wrapper: " + prompt, "input": "", "output": answer},
    ], publish)
    admission.finish_source("a", publish)
    admission.commit_batch("b", "pretrain", [
        {"text": f"{prompt}\nAnswer: {answer}"}, {"text": prompt}, {"text": answer},
    ], publish)
    assert len(kept) == 4
    assert admission.frontier.bloom_positive == 2
    assert admission.frontier.preseed_count == 4  # duplicates count conservatively
    assert original == EXACT_EXAMPLES[name]


def test_seed_loader_is_pinned_streamed_and_keeps_all_representations(monkeypatch: pytest.MonkeyPatch) -> None:
    from copy import deepcopy

    from data_preparation.lib.stages.benchmark_seeds import load_benchmark_seeds

    calls: list[tuple[str, str | None, dict[str, Any]]] = []
    originals = deepcopy(EXACT_EXAMPLES)

    def load(path: str, config: str | None, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((path, config, kwargs))
        name = next(name for name, (registry, _) in bm.BLOOM_BENCHMARKS.items() if bm.BENCHMARKS[registry].hf_id == path)
        records = [originals[name], originals[name]]
        if name == "mmlu":
            records.append({**originals[name], "subject": "physics"})
        return records

    module = types.ModuleType("datasets")
    module.load_dataset = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    with load_benchmark_seeds([*EXACT_EXAMPLES, "mmlu"], memory_mb=1, hf_token="fixture") as seeds:
        keys = list(seeds.keys())
        assert keys == list(seeds.keys())
        assert len(keys) == seeds.count == 18
        assert len(set(keys)) == 10
    assert originals == EXACT_EXAMPLES
    assert len(calls) == 4
    for (path, config, kwargs), name in zip(calls, sorted(EXACT_EXAMPLES), strict=True):
        registry, split = bm.BLOOM_BENCHMARKS[name]
        definition = bm.BENCHMARKS[registry]
        assert (path, config) == (definition.hf_id, definition.config)
        assert kwargs == {"split": split, "revision": definition.revision, "streaming": True, "token": "fixture"}
    assert next(config for path, config, _ in calls if path == "cais/mmlu") == "all"
    assert {name: split for name, (_, split) in bm.BLOOM_BENCHMARKS.items()} == {
        "arc_challenge": "test", "hellaswag": "validation", "mmlu": "test", "winogrande": "validation",
    }


def test_seed_empty_does_not_import_optional_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    from data_preparation.lib.stages.benchmark_seeds import load_benchmark_seeds

    monkeypatch.setitem(sys.modules, "datasets", None)
    with load_benchmark_seeds([], memory_mb=1) as seeds:
        assert list(seeds.keys()) == [] and seeds.count == 0


@pytest.mark.parametrize("failure", ["empty", "load", "bad", "capacity"])
def test_seed_failures_are_explicit(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    from data_preparation.lib.stages import benchmark_seeds as seeds_module

    def records(name: str, **kwargs: Any) -> list[dict[str, Any]]:
        if failure == "load":
            raise OSError("fixture loader failed")
        return [] if failure == "empty" else [{}] if failure == "bad" else [EXACT_EXAMPLES[name]]

    monkeypatch.setattr(seeds_module, "benchmark_records", records)
    if failure == "capacity":
        monkeypatch.setattr(seeds_module, "expected_items", lambda memory_mb: 0)
    with pytest.raises((ValueError, OSError)) as error, seeds_module.load_benchmark_seeds(["mmlu"], memory_mb=1):
        pytest.fail("invalid seed material was accepted")
    assert str(error.value)


def test_seed_and_dataset_capacity_are_combined() -> None:
    from data_preparation.lib.stages.global_dedup import GlobalAdmission

    admission = GlobalAdmission(("a",), memory_mb=1, preseed_keys=[1, 2])
    admission.seen.expected_items = 2
    admission.check_capacity(2)
    with pytest.raises(ValueError, match="5 keys"):
        admission.check_capacity(3)
