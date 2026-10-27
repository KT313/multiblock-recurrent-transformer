# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The contamination check on hand-written shards and docs: no Hub, no checkpoint, no shipped config."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluation.contamination_check import (
    build_examples, build_index, example_ngrams, markdown_report, plan_units, scan, summarize, trained_ranges,
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
    docs = {
        "qa": [{"question": QUESTION, "choices": {"text": ["a", "b"]}}, {"question": "too short to check"}],
        "code": [{"prompt": SOLUTION, "entry_point": "add_numbers"}],
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
    examples, unchecked = build_examples(setup["docs"])
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
    assert "| qa | 2 | 1 | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | web 1 |" in text
