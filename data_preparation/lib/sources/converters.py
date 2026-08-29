# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Row converters and filters: `get_converter(source)` maps a raw row to the standard shape (pretrain `{"text"}`,
instruct `{"instruction", "input", "output"}`), `get_filter(name)` a row predicate (`sharegpt_quality`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from data_preparation.lib.schema.dataset_config import SourceConfig

Row = dict[str, Any]
Converter = Callable[[Row], Row]
Filter = Callable[[Row], bool]


def _require(row: Row, *keys: str) -> None:
    missing = [k for k in keys if k not in row]
    if missing:
        raise ValueError(f"row is missing column(s) {missing}; available columns: {sorted(row)}")


def gsm8k_question_answer(row: Row) -> Row:
    """GSM8K `question` + `answer` -> one pretraining document."""
    _require(row, "question", "answer")
    return {"text": f"Question: {row['question']}\n\nAnswer: {row['answer']}"}


def _conversations(row: Row) -> list[Any]:
    _require(row, "conversations")
    convs = row["conversations"]
    if not isinstance(convs, list):
        raise ValueError(f"'conversations' must be a list, got {type(convs).__name__}; columns: {sorted(row)}")
    return convs


def sharegpt_conversations(row: Row) -> Row:
    """SlimOrca / ShareGPT `conversations` with `from`/`value` turns: system->input, human->instruction, gpt->output.

    Later turns of the same role overwrite earlier ones (as in the thesis pipeline: only one exchange is kept).
    """
    system_msg = human_msg = gpt_msg = ""
    for turn in _conversations(row):
        if not isinstance(turn, dict) or "from" not in turn:
            raise ValueError(f"sharegpt_conversations: turns need 'from'/'value' keys, got {turn!r}")
        role, value = turn["from"], str(turn.get("value", ""))
        if role == "system":
            system_msg = value
        elif role == "human":
            human_msg = value
        elif role == "gpt":
            gpt_msg = value
    return {"instruction": human_msg, "input": system_msg, "output": gpt_msg}


def first_two_turns(row: Row) -> Row:
    """`conversations` without role tags (WizardLM): first `value` = instruction, second = output."""
    convs = _conversations(row)
    if len(convs) < 2:
        raise ValueError(f"first_two_turns: need at least two turns, got {len(convs)}; columns: {sorted(row)}")
    first, second = convs[0], convs[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError(f"first_two_turns: turns must be dicts with a 'value' key, got {convs[:2]!r}")
    return {"instruction": str(first.get("value", "")), "input": "", "output": str(second.get("value", ""))}


def instruction_input_output(row: Row) -> Row:
    """Rows that already carry `instruction`/`output` (and optionally `input`); missing input -> ""."""
    return fields_converter({"instruction": "instruction", "input": "input", "output": "output"})(row)


def fields_converter(fields: dict[str, str]) -> Converter:
    """Converter mapping `{instruction: <col>, input: <col>?, output: <col>}` to the standard instruct row."""
    if not {"instruction", "output"} <= set(fields):
        raise ValueError(f"fields must map at least instruction and output, got {sorted(fields)}")
    unknown = set(fields) - {"instruction", "input", "output"}
    if unknown:
        raise ValueError(f"fields has unknown keys {sorted(unknown)}; allowed: instruction, input, output")
    instruction_col, output_col = fields["instruction"], fields["output"]
    input_col = fields.get("input")

    def convert(row: Row) -> Row:
        _require(row, instruction_col, output_col)
        input_value = row.get(input_col, "") if input_col is not None else ""
        return {
            "instruction": str(row[instruction_col]),
            "input": "" if input_value is None else str(input_value),
            "output": str(row[output_col]),
        }

    return convert


CONVERTERS: dict[str, Converter] = {
    "gsm8k_question_answer": gsm8k_question_answer,
    "sharegpt_conversations": sharegpt_conversations,
    "first_two_turns": first_two_turns,
    "instruction_input_output": instruction_input_output,
}


def get_converter(source: SourceConfig) -> Converter | None:
    """`fields` mapping first, then the named `converter`, else None (row used as is; `text_field` applied later)."""
    if source.fields is not None:
        return fields_converter(source.fields)
    if source.converter is None:
        return None
    if source.converter not in CONVERTERS:
        raise ValueError(f"unknown converter {source.converter!r}; known converters: {sorted(CONVERTERS)}")
    return CONVERTERS[source.converter]


# --- filters ----------------------------------------------------------------------------------------------------------


def sharegpt_quality(row: Row) -> bool:
    """ShareGPT quality filter: human->gpt opening, 50-2000 chars per side, no code blocks in the answer."""
    convs = row.get("conversations")
    if not isinstance(convs, list) or len(convs) < 2:
        return False
    first, second = convs[0], convs[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if first.get("from") != "human" or second.get("from") != "gpt":
        return False
    human_text, gpt_text = str(first.get("value", "")), str(second.get("value", ""))
    if not 50 <= len(human_text) <= 2000 or not 50 <= len(gpt_text) <= 2000:
        return False
    return not any(p in gpt_text.lower() for p in ("```python", "```java", "```cpp", "```javascript"))


FILTERS: dict[str, Filter] = {"sharegpt_quality": sharegpt_quality}


def get_filter(name: str) -> Filter:
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}; known filters: {sorted(FILTERS)}")
    return FILTERS[name]
