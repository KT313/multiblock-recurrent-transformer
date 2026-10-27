# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The contamination check on hand-written shards and docs: no Hub, no checkpoint, no shipped config."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluation.benchmark_helpers import load_benchmark_samples
from evaluation.contamination_check import (
    build_examples, build_index, example_ngrams, load_lm_eval_docs, markdown_report, plan_units, sample_lists, scan,
    summarize, trained_ranges, write_sample_lists,
)

QUESTION = ("which of the following best explains why the moon shows phases to an observer standing on the earth "
            "during one month")
SOLUTION = ("def add_numbers(first, second): return the sum of both arguments after converting each of them to an "
            "integer value first")
FILLER = "plain filler words that never appear in any benchmark doc at all, repeated to make a long enough row"


def write_shards(directory: Path, column: str, rows: list[Any], per_file: int) -> None:
    directory.mkdir(parents=True)
    for number, start in enumerate(range(0, len(rows), per_file)):
        pq.write_table(pa.table({column: rows[start:start + per_file]}), directory / f"data-{number:05d}.parquet")


def chat(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": FILLER}, {"role": "assistant", "content": text}]


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Any]:
    web = [FILLER] * 10
    web[1] = f"leaked into a validation row: {QUESTION}"  # rows [0, 2) are validation, never trained
    web[5] = f"some text before. {QUESTION} answer: the moon orbits the earth"
    web[9] = f"beyond the rows the stream read: {SOLUTION}"
    code_words = SOLUTION.split()
    messages = [chat(FILLER)] * 4
    messages[2] = chat(" ".join(code_words[:14]))  # 2 of the solution's n-grams, under half
    write_shards(tmp_path / "web", "text", web, per_file=3)
    write_shards(tmp_path / "chat", "messages", messages, per_file=4)
    docs = {  # task -> leaf task -> docs; "qa" is a group of two leaves like mmlu
        "qa": {"qa_one": [{"question": QUESTION, "choices": {"text": ["a", "b"]}}], "qa_two": [{"question": "too short to check"}]},
        "code": {"code": [{"prompt": SOLUTION, "entry_point": "add_numbers"}]},
    }
    ranges = trained_ranges({"web": 2, "chat": 0}, {"web": 10, "chat": 4}, {"web": 6, "chat": 9, "unused": 0},
                            scope="trained")
    return {"directories": {"web": tmp_path / "web", "chat": tmp_path / "chat"}, "docs": docs, "ranges": ranges}


def test_trained_ranges_follow_the_stream() -> None:
    validation, total = {"a": 2, "b": 0}, {"a": 10, "b": 4, "c": 5}
    consumed = {"a": 6, "b": 9, "c": 0}
    assert trained_ranges(validation, total, consumed, scope="trained") == {"a": (2, 8), "b": (0, 4)}  # b wrapped
    assert trained_ranges(validation, total, consumed, scope="all") == {"a": (2, 10), "b": (0, 4)}
    assert trained_ranges(validation, total, consumed, scope="trained", sources=["b"]) == {"b": (0, 4)}
    with pytest.raises(KeyError, match="not trained"):
        trained_ranges(validation, total, consumed, scope="trained", sources=["c"])


def test_plan_units_cuts_the_range_at_shard_boundaries(setup: dict[str, Any]) -> None:
    units = plan_units(setup["directories"], {"web": (2, 8)})
    assert [(Path(unit.path).name, unit.start, unit.stop) for unit in units] == [
        ("data-00000.parquet", 2, 3), ("data-00001.parquet", 0, 3), ("data-00002.parquet", 0, 2),
    ]


def test_example_ngrams_use_fields_alone_then_joined() -> None:
    assert example_ngrams({"a": QUESTION, "b": {"c": ["x"]}}) == example_ngrams({"a": QUESTION})
    short = {"a": "seven words in this first field here", "b": "and seven more words in the second"}
    assert len(example_ngrams(short)) == 2  # 14 words joined: two 13-grams


@pytest.mark.parametrize("workers", [1, 2])
def test_scan_counts_trained_rows_only(setup: dict[str, Any], workers: int) -> None:
    examples, unchecked, _ = build_examples(setup["docs"])
    tasks = ["qa", "code"]
    units = plan_units(setup["directories"], setup["ranges"])
    results = scan(units, build_index(examples, tasks), workers)
    report = summarize(examples, unchecked, tasks, results, setup["ranges"])

    qa, code = report["tasks"]["qa"], report["tasks"]["code"]
    assert (qa["examples"], qa["unchecked"], qa["any_hit"], qa["likely"], qa["complete"]) == (2, 1, 1, 1, 1)
    assert qa["likely_by_source"] == {"web": 1} and qa["worst"][0]["sources"] == {"web": 1.0}
    # the solution's full text sits only in row 9, outside the trained rows; the chat row holds 2 of its n-grams
    assert (code["any_hit"], code["likely"], code["complete"]) == (1, 0, 0)
    assert code["worst"][0]["sources"] == {"chat": pytest.approx(2 / len(example_ngrams({"p": SOLUTION})), abs=1e-4)}
    web, chat_source = report["sources"]["web"], report["sources"]["chat"]
    assert (web["rows_scanned"], web["rows_with_hit"], web["rows_over_share"]) == (6, 1, {"qa": 1})
    assert (chat_source["rows_scanned"], chat_source["rows_with_hit"]) == (4, 1)
    text = markdown_report(report, {"step": 1})
    assert "| qa | 2 | 1 | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | web 1 |" in text
    assert qa["worst"][0]["leaf"] == "qa_one"


def test_sample_lists_round_trip_into_benchmark_selection(setup: dict[str, Any], tmp_path: Path) -> None:
    """
    Flagged examples leave the clean list of their leaf task; a short unchecked example stays clean; the files load
    as the selection `benchmark_checkpoint.py --samples` reads.
    """

    examples, _, sizes = build_examples(setup["docs"])
    results = scan(plan_units(setup["directories"], setup["ranges"]), build_index(examples, ["qa", "code"]), 1)
    clean, contaminated, shares = sample_lists(examples, sizes, results, flag_share=0.5)
    assert contaminated == {"qa_one": [0]}  # the code example holds under half of its n-grams
    assert clean == {"qa_two": [0], "code": [0]}  # qa_one has nothing clean left, so it has no clean entry
    assert shares["qa_one"] == {0: 1.0} and 0 < shares["code"][0] < 0.5
    assert sample_lists(examples, sizes, results, flag_share=0.2)[1] == {"qa_one": [0], "code": [0]}

    write_sample_lists(tmp_path, clean, contaminated, shares, {"flag_share": 0.5})
    selection = load_benchmark_samples(tmp_path / "clean_samples.json")
    assert selection.indices == clean and selection.criterion == {"flag_share": 0.5, "selection": "clean"}
    assert load_benchmark_samples(tmp_path / "contaminated_samples.json").indices == contaminated


def test_tasks_that_cannot_load_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    import lm_eval.tasks

    class Task:
        def __init__(self, count: int) -> None:
            self.eval_docs = [{"text": f"doc {index}"} for index in range(count)]

    def get_task_dict(names: list[str], manager: Any) -> dict[str, Any]:
        if names == ["gated"]:
            raise PermissionError("dataset is gated")
        return {"group": {"group_a": Task(2), "group_b": Task(1)}} if names == ["group"] else {names[0]: Task(3)}

    monkeypatch.setattr(lm_eval.tasks, "TaskManager", lambda: None)
    monkeypatch.setattr(lm_eval.tasks, "get_task_dict", get_task_dict)
    docs, skipped = load_lm_eval_docs(["group", "gated", "plain"])
    assert {task: {leaf: len(items) for leaf, items in leaves.items()} for task, leaves in docs.items()} == {
        "group": {"group_a": 2, "group_b": 1}, "plain": {"plain": 3},
    }
    assert skipped == {"gated": "PermissionError: dataset is gated"}
