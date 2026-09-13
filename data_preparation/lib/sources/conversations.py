# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The versioned ShareGPT opening-exchange policy shared by conversion, filtering and raw identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Changing selection or accepted opening shapes requires a new identity: raw rows discard conversation history.
SHAREGPT_EXCHANGE_POLICY = "first_opening_exchange_v2"


class OrphanAssistantOpening(ValueError):
    """A recognized GPT opening with no preceding question; skip the row without declaring schema failure."""


@dataclass(frozen=True)
class OpeningExchange:
    question: str
    answer: str
    system: str
    question_index: int
    answer_index: int


def opening_exchange(row: dict[str, Any]) -> OpeningExchange:
    """
    Select [system,] human, gpt at the start, never search forward past a malformed opening.

    At most one initial system turn applies. Once this complete pair is selected, all subsequent turns are
    ignored (including unanswered questions and later system context). Text scalars retain their existing
    str conversion and None becomes empty; containers and missing values are malformed, not text.
    """

    turns = row.get("conversations")
    if not isinstance(turns, list):
        raise ValueError("sharegpt_conversations: 'conversations' must be a list of from/value turns")

    def turn(index: int) -> tuple[str, str]:
        if index >= len(turns):
            raise ValueError(f"sharegpt_conversations: incomplete opening exchange; missing turn {index}")
        item = turns[index]
        if not isinstance(item, dict) or "from" not in item or "value" not in item:
            raise ValueError(f"sharegpt_conversations: turn {index} needs 'from'/'value' keys")
        role, value = item["from"], item["value"]
        if not isinstance(role, str) or role not in ("system", "human", "gpt"):
            raise ValueError(f"sharegpt_conversations: turn {index} has unsupported role {role!r}")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"sharegpt_conversations: turn {index} value must be text, a scalar or null, got {type(value).__name__}")
        return role, "" if value is None else str(value)

    index = 0
    system = ""
    role, question = turn(index)
    if role == "system":
        system = question
        index = 1
        role, question = turn(index)
    if role != "human":
        error = OrphanAssistantOpening if role == "gpt" else ValueError
        raise error(f"sharegpt_conversations: opening turn {index} must be human, got {role!r}")
    answer_role, answer = turn(index + 1)
    if answer_role != "gpt":
        raise ValueError(f"sharegpt_conversations: opening turn {index + 1} must be gpt, got {answer_role!r}")
    return OpeningExchange(question, answer, system, index, index + 1)
