# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Row converters and filters: `get_converter(source)` maps a raw row to the standard shape (pretrain `{"text"}`,
instruct `{"instruction", "input", "output"}`), `get_filter(name)` a row predicate (`sharegpt_quality`),
`expected_format(source)` says in words what the converter of source accepts (for the error that fails a download
whose rows keep coming out malformed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from data_preparation.lib.dataset_config import SourceConfig
from data_preparation.lib.sources.conversations import opening_exchange

Row = dict[str, Any]
Converter = Callable[[Row], Row]
Filter = Callable[[Row], bool]

INSTRUCT_FIELDS = ("instruction", "input", "output")


def _require(row: Row, *keys: str) -> None:
    """
    Raise ValueError naming the missing columns if row lacks any of keys.
    """

    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"row is missing column(s) {missing}; available columns: {sorted(row)}")


def text_or_empty(value: Any) -> str:
    """
    str(value), with None becoming the empty string.
    """

    return "" if value is None else str(value)


# --- converters --------------------------------------------------------------------------------------------------------


def gsm8k_question_answer(row: Row) -> Row:
    """
    GSM8K `question` + `answer` -> one pretraining document.
    """

    _require(row, "question", "answer")
    return {"text": f"Question: {text_or_empty(row['question'])}\n\nAnswer: {text_or_empty(row['answer'])}"}


def _conversations(row: Row) -> list[Any]:
    """
    The `conversations` list of a row (an error if the column is missing or not a list).
    """

    _require(row, "conversations")
    conversations = row["conversations"]
    if not isinstance(conversations, list):
        raise ValueError(f"'conversations' must be a list, got {type(conversations).__name__}; columns: {sorted(row)}")
    return conversations


def sharegpt_conversations(row: Row) -> Row:
    """
    Keep the first complete opening [system,] human/GPT exchange; ignore all subsequent turns.

    System context becomes input, the opening human instruction, and its GPT reply output. Malformed
    openings raise ValueError; no later exchange is searched for and later context cannot leak backward.
    """

    selected = opening_exchange(row)
    return {"instruction": selected.question, "input": selected.system, "output": selected.answer}


def first_two_turns(row: Row) -> Row:
    """
    `conversations` without role tags (WizardLM): first `value` = instruction, second = output.
    """

    conversations = _conversations(row)
    if len(conversations) < 2:
        raise ValueError(f"first_two_turns: need at least two turns, got {len(conversations)}; columns: {sorted(row)}")
    first, second = conversations[0], conversations[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError(f"first_two_turns: turns must be dicts with a 'value' key, got {conversations[:2]!r}")
    return {"instruction": text_or_empty(first.get("value")), "input": "", "output": text_or_empty(second.get("value"))}


def fields_converter(fields: dict[str, str]) -> Converter:
    """
    Converter mapping `{instruction: <col>, input: <col>?, output: <col>}` to the standard instruct row.
    """

    if not {"instruction", "output"} <= set(fields):
        raise ValueError(f"fields must map at least instruction and output, got {sorted(fields)}")
    unknown = set(fields) - set(INSTRUCT_FIELDS)
    if unknown:
        raise ValueError(f"fields has unknown keys {sorted(unknown)}; allowed: instruction, input, output")
    instruction_col = fields["instruction"]
    output_col = fields["output"]
    input_col = fields.get("input")  # optional; missing column or None value -> ""

    def convert(row: Row) -> Row:
        _require(row, instruction_col, output_col)
        input_value = row.get(input_col, "") if input_col is not None else ""
        return {
            "instruction": text_or_empty(row[instruction_col]),
            "input": text_or_empty(input_value),
            "output": text_or_empty(row[output_col]),
        }

    return convert


_IDENTITY_FIELDS_CONVERTER = fields_converter({name: name for name in INSTRUCT_FIELDS})


def instruction_input_output(row: Row) -> Row:
    """
    Rows that already carry `instruction`/`output` (and optionally `input`); missing input -> "".
    """

    return _IDENTITY_FIELDS_CONVERTER(row)


CONVERTERS: dict[str, Converter] = {
    "gsm8k_question_answer": gsm8k_question_answer,
    "sharegpt_conversations": sharegpt_conversations,
    "first_two_turns": first_two_turns,
    "instruction_input_output": instruction_input_output,
}


IDENTITY_FORMAT = "columns instruction, output[, input]"

# What each named converter expects a source row to look like, in the words of the error that fails a download
# after too many malformed rows in a row (the row's own column names and value types are printed next to it).
EXPECTED_FORMATS: dict[str, str] = {
    "gsm8k_question_answer": "columns question, answer",
    "sharegpt_conversations": "a `conversations` list opening with [system,] human, gpt turns, each with from/value keys",
    "first_two_turns": "a `conversations` list of at least two `{value}` turns (instruction, then output)",
    "instruction_input_output": IDENTITY_FORMAT,
}


def expected_format(source: SourceConfig) -> str:
    """
    The input format of source's converter in words: a `fields` mapping as "columns instruction=<col>,
    output=<col>[, input=<col>]", a named converter's entry of :data:`EXPECTED_FORMATS`, the identity case
    (no converter: the rows carry instruction / output already) as "columns instruction, output[, input]".
    """

    if source.fields is not None:
        columns = ", ".join(f"{name}={source.fields[name]}" for name in ("instruction", "output", "input") if name in source.fields)
        return f"columns {columns}"
    if source.converter is None:
        return IDENTITY_FORMAT
    return EXPECTED_FORMATS.get(source.converter, f"whatever converter {source.converter!r} accepts (no description registered)")


def get_converter(source: SourceConfig) -> Converter | None:
    """
    `fields` mapping first, then the named `converter`, else None (row used as is; `text_field` applied later).
    """

    if source.fields is not None:
        return fields_converter(source.fields)
    if source.converter is None:
        return None
    if source.converter not in CONVERTERS:
        raise ValueError(f"unknown converter {source.converter!r}; known converters: {sorted(CONVERTERS)}")
    return CONVERTERS[source.converter]


# --- filters ----------------------------------------------------------------------------------------------------------

SHAREGPT_MIN_CHARS = 50
SHAREGPT_MAX_CHARS = 2000
SHAREGPT_CODE_BLOCK_MARKERS = ("```python", "```java", "```cpp", "```javascript")


def sharegpt_quality(row: Row) -> bool:
    """
    Check the selected opening human/GPT exchange: 50-2000 chars per side, no listed code markers.

    A valid system-prefixed opening remains quality-ineligible, as before. Malformed openings raise
    ValueError so the download's malformed-row diagnostics apply instead of counting them as low quality.
    """

    selected = opening_exchange(row)
    if selected.question_index != 0:
        return False
    for text in (selected.question, selected.answer):
        if not SHAREGPT_MIN_CHARS <= len(text) <= SHAREGPT_MAX_CHARS:
            return False
    answer = selected.answer.lower()
    return not any(marker in answer for marker in SHAREGPT_CODE_BLOCK_MARKERS)


FILTERS: dict[str, Filter] = {"sharegpt_quality": sharegpt_quality}


def get_filter(name: str) -> Filter:
    """
    The registered filter called name (ValueError for an unknown name).
    """

    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}; known filters: {sorted(FILTERS)}")
    return FILTERS[name]
