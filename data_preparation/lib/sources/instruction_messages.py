# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Explicit source adapters; excluded records are different from malformed schemas."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from data_preparation.lib.conversation_format import validate_messages

MESSAGE_CONVERTERS = frozenset({"opencode_messages", "webinstruct_messages", "nemotron_messages"})


class ExcludedConversation(Exception):
    """An intentional, counted source-policy exclusion."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def check_opencode_score(row: dict[str, Any]) -> bool:
    value = row.get("average_test_score")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("average_test_score must be a numeric string or number")
    try:
        score = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("invalid average_test_score") from error
    if not score.is_finite() or not 0 <= score <= 1:
        raise ValueError("average_test_score must be finite and within [0, 1]")
    return score == 1


def convert_pair(row: dict[str, Any], question: str, answer: str) -> dict[str, Any]:
    return {"messages": validate_messages([{"role": "user", "content": row.get(question)},
                                          {"role": "assistant", "content": row.get(answer)}])}


def convert_opencode_messages(row: dict[str, Any]) -> dict[str, Any]:
    return convert_pair(row, "input", "output")


def convert_webinstruct_messages(row: dict[str, Any]) -> dict[str, Any]:
    return convert_pair(row, "question", "answer")


def convert_nemotron_messages(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("reasoning") != "off":
        raise ValueError("Nemotron messages require reasoning == 'off'")
    value = row.get("messages")
    if not isinstance(value, list) or not value:
        raise ValueError("Nemotron messages must be a nonempty list")
    messages = value
    system = None
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        system = messages[0].get("content")
        if not isinstance(system, str):
            raise ValueError("system content must be a string")
        messages = messages[1:]
    canonical = validate_messages(messages)  # fail malformed later turns even when the system is excluded
    if system is not None and system.strip():
        raise ExcludedConversation("nonempty_system")
    return {"messages": canonical}
