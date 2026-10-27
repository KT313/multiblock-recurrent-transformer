# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The contamination check on hand-written shards and docs: no Hub, no checkpoint, no shipped config."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluation.benchmark_helpers import load_benchmark_samples
from evaluation.contamination_check import (
    DEFAULT_TASKS, QUESTION_FIELDS, build_examples, build_index, example_sequences, fields, load_lm_eval_docs,
    markdown_report, plan_units, race_questions, sample_lists, scan, summarize, trained_ranges, write_sample_lists,
)

QUESTION = ("which of the following best explains why the moon shows phases to an observer standing on the earth "
            "during one month")  # 21 words: 16-word windows from words 0 and 5
SHORT_QUESTION = "what is the boiling point of water at sea level"  # 10 words: one sequence as a whole
ANSWER = "the moon reflects sunlight and its lit part changes as it orbits the planet earth"  # 16 words
SOLUTION = ("def add_numbers(first, second): return the sum of both arguments after converting each of them to an "
            "integer value first and then check that the result is not negative before returning it")  # 31 words: 3 windows
FILLER = "plain filler words that never appear in any benchmark doc at all, repeated to make a long enough row"


TEST_FIELDS = {"qa": fields("question"), "code": fields("prompt")}  # the answer-side fields stay unchecked


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
    web[6] = f"the teacher asked: {SHORT_QUESTION}"  # a short field matches as a whole, here at the row's very end
    web[9] = f"beyond the rows the stream read: {SOLUTION}"
    web[7] = f"an answer option is no question: {ANSWER}"  # in the trained rows, but only in an answer field
    code_words = SOLUTION.split()
    messages = [chat(FILLER)] * 4
    messages[2] = chat(" ".join(code_words[:22]))  # 1 of the solution's 3 windows (words 0-15), under half
    write_shards(tmp_path / "web", "text", web, per_file=3)
    write_shards(tmp_path / "chat", "messages", messages, per_file=4)
    docs = {  # task -> leaf task -> docs; "qa" is a group of two leaves like mmlu
        "qa": {"qa_one": [{"question": QUESTION, "choices": {"text": [ANSWER, "b"]}}],
               "qa_two": [{"question": "too short to check"}, {"question": SHORT_QUESTION, "choices": ["100", "90"]}]},
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


def test_example_sequences_are_windows_of_long_fields_and_whole_short_ones() -> None:
    assert len(example_sequences([QUESTION])) == 2

    def windows(count: int) -> set[tuple[int, int]]:
        words = [f"w{index}" for index in range(count)]
        found = example_sequences([" ".join(words)])
        return {(words.index(sequence.split()[0]), words.index(sequence.split()[-1])) for sequence in found}

    assert windows(25) == {(0, 15), (8, 23), (9, 24)}  # every 8 words, the last one ends at the last word
    assert windows(24) == {(0, 15), (8, 23)}  # the stride reaches the end: no extra window
    assert windows(33) == {(0, 15), (8, 23), (16, 31), (17, 32)}
    assert windows(16) == {(0, 15)}
    assert windows(15) == {(0, 14)}  # 8 to 15 words: the whole field
    assert windows(8) == {(0, 7)}
    words = [f"w{index}" for index in range(20)]
    assert example_sequences([" ".join(words[:7]), "short"]) == frozenset()  # fields are never joined
    assert example_sequences(["The Moon  SHOWS\nphases to an observer on earth"]) == {
        "the moon shows phases to an observer on earth"}  # normalized like the training side


def test_question_fields_leave_out_answers_and_passages() -> None:
    assert set(DEFAULT_TASKS) <= set(QUESTION_FIELDS)
    gsm8k = {"question": "how many eggs", "answer": "16 - 3 = <<16-3=13>>13 #### 13"}
    assert QUESTION_FIELDS["gsm8k"](gsm8k) == ["how many eggs"]
    boolq = {"question": "is the sky blue", "passage": "Sky -- the sky is blue because ...", "label": 1}
    assert QUESTION_FIELDS["boolq"](boolq) == ["is the sky blue"]
    assert QUESTION_FIELDS["anli"]({"premise": "p", "hypothesis": "h", "reason": "r", "uid": "x"}) == ["p", "h"]
    race = {"article": "a long article", "problems": "[{'answer': 'A', 'options': ['x'], 'question': 'Why?'}, "
                                                    "{'answer': 'B', 'options': ['y'], 'question': 'Who?'}]"}
    assert race_questions(race) == ["Why?", "Who?"]


@pytest.mark.parametrize("workers", [1, 2])
def test_scan_counts_trained_rows_only(setup: dict[str, Any], workers: int) -> None:
    examples, unchecked, _ = build_examples(setup["docs"], TEST_FIELDS)
    tasks = ["qa", "code"]
    units = plan_units(setup["directories"], setup["ranges"])
    results = scan(units, build_index(examples, tasks), workers)
    report = summarize(examples, unchecked, tasks, results, setup["ranges"])

    qa, code = report["tasks"]["qa"], report["tasks"]["code"]
    assert (qa["examples"], qa["unchecked"], qa["any_hit"], qa["likely"], qa["complete"]) == (3, 1, 2, 2, 2)
    assert qa["likely_by_source"] == {"web": 2} and qa["worst"][0]["sources"] == {"web": 1.0}
    assert {(entry["leaf"], entry["index"]) for entry in qa["worst"]} == {("qa_one", 0), ("qa_two", 1)}
    # the solution's full text sits only in row 9, outside the trained rows; the chat row holds 1 of its 3 windows
    assert (code["any_hit"], code["likely"], code["complete"]) == (1, 0, 0)
    assert code["worst"][0]["sources"] == {"chat": 0.3333}
    web, chat_source = report["sources"]["web"], report["sources"]["chat"]
    assert (web["rows_scanned"], web["rows_with_hit"], web["rows_over_share"]) == (6, 2, {"qa": 2})
    assert (chat_source["rows_scanned"], chat_source["rows_with_hit"]) == (4, 1)
    text = markdown_report(report, {"step": 1})
    assert "| qa | 3 | 1 | 2 (100.0%) | 2 (100.0%) | 2 (100.0%) | 2 (100.0%) | web 2 |" in text


def test_sample_lists_round_trip_into_benchmark_selection(setup: dict[str, Any], tmp_path: Path) -> None:
    """
    Flagged examples leave the clean list of their leaf task; a short unchecked example stays clean; the files load
    as the selection `benchmark_checkpoint.py --samples` reads.
    """

    examples, _, sizes = build_examples(setup["docs"], TEST_FIELDS)
    results = scan(plan_units(setup["directories"], setup["ranges"]), build_index(examples, ["qa", "code"]), 1)
    clean, contaminated, shares = sample_lists(examples, sizes, results, flag_share=0.5)
    assert contaminated == {"qa_one": [0], "qa_two": [1]}  # the code example holds under half of its windows
    assert clean == {"qa_two": [0], "code": [0]}  # qa_one has nothing clean left, so it has no clean entry
    assert shares == {"qa_one": {0: 1.0}, "qa_two": {1: 1.0}, "code": {0: 0.3333}}
    assert sample_lists(examples, sizes, results, flag_share=0.2)[1] == {"qa_one": [0], "qa_two": [1], "code": [0]}

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
