# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Versioned chat tokens, assistant masks, and complete-exchange fitting shared by preparation and training."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

CHAT_POLICY = "messages-complete-exchanges-v1"
CHAT_COLUMNS = ("messages", "exchange_ends", "tokens")


class Message(TypedDict):
    role: str
    content: str


class ChatTokenizer(Protocol):
    @property
    def bos_id(self) -> int | None: ...
    @property
    def eos_id(self) -> int | None: ...
    def encode_literal(self, text: str) -> list[int]: ...


@dataclass
class EncodedConversation:
    ids: list[int]
    supervised: list[bool]
    exchange_ends: list[int]  # exclusive token ends, with BOS/EOS included
    messages: list[Message]
    removed_exchanges: int
    trimmed_user: bool


def validate_messages(value: Any, *, allow_trailing_user: bool = True) -> list[Message]:
    """Validate canonical roles without stripping content or searching past malformed turns."""
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a nonempty list")
    messages: list[Message] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise ValueError(f"message {index} must have role {expected!r}, got {message.get('role')!r}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be a nonempty string")
        messages.append({"role": expected, "content": content})
    if not allow_trailing_user and len(messages) % 2:
        raise ValueError("conversation must end with an assistant response")
    return messages


def encode_exchange(user: Message, assistant: Message, tokenizer: ChatTokenizer, *, first: bool, eos: bool) -> tuple[list[int], list[bool]]:
    prefix = "" if first else "\n\n"
    context = (tokenizer.encode_literal(prefix + "User:\n") + tokenizer.encode_literal(user["content"])
               + tokenizer.encode_literal("\n\nAssistant:\n"))
    answer = tokenizer.encode_literal(assistant["content"])
    if eos:
        if tokenizer.eos_id is None:
            raise ValueError("conversation formatting requires an EOS token")
        answer.append(tokenizer.eos_id)
    return context + answer, [False] * len(context) + [True] * len(answer)


def fit_conversation(value: Any, tokenizer: ChatTokenizer, max_tokens: int | None = None, *, bos: bool = True, eos: bool = True) -> EncodedConversation:
    """Keep only complete leading exchanges; max_tokens bounds unshifted serialized IDs."""
    messages = validate_messages(value)
    if getattr(tokenizer, "profile", None):
        return fit_literal_conversation(messages, tokenizer, max_tokens, bos=bos, eos=eos)
    if max_tokens is not None and max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    if bos and tokenizer.bos_id is None:
        raise ValueError("conversation formatting requires a BOS token")
    ids = [tokenizer.bos_id] if bos and tokenizer.bos_id is not None else []
    supervised = [False] * len(ids)
    ends: list[int] = []
    for index in range(0, len(messages) - 1, 2):
        turn_ids, mask = encode_exchange(messages[index], messages[index + 1], tokenizer, first=index == 0, eos=eos)
        if max_tokens is not None and len(ids) + len(turn_ids) > max_tokens:
            break
        ids.extend(turn_ids)
        supervised.extend(mask)
        ends.append(len(ids))
    kept = messages[:2 * len(ends)]
    return EncodedConversation(ids if ends else [], supervised if ends else [], ends, kept,
                               len(messages) // 2 - len(ends), bool(len(messages) % 2))


def encode_chat_prompt(value: Any, tokenizer: ChatTokenizer, *, max_tokens: int | None = None, bos: bool = True) -> list[int]:
    """Encode completed history and a final user turn, ready to generate an assistant answer."""
    messages = validate_messages(value)
    if len(messages) % 2 != 1:
        raise ValueError("generation messages must end with a user query")
    if getattr(tokenizer, "profile", None):
        from tokenization.chat import encode_chat

        if not bos:
            raise ValueError("literal chat profile requires BOS")
        ids = encode_chat(messages, tokenizer.encode_literal, generation_prompt=True).ids
        if max_tokens is not None and len(ids) > max_tokens:
            raise ValueError("chat generation prompt exceeds the context budget")
        return ids
    if len(messages) > 1:
        ids = fit_conversation(messages[:-1], tokenizer, bos=bos).ids
    else:
        if bos and tokenizer.bos_id is None:
            raise ValueError("conversation formatting requires a BOS token")
        ids = [tokenizer.bos_id] if bos and tokenizer.bos_id is not None else []
    prefix = "\n\n" if len(messages) > 1 else ""
    ids += tokenizer.encode_literal(prefix + "User:\n") + tokenizer.encode_literal(messages[-1]["content"])
    ids += tokenizer.encode_literal("\n\nAssistant:\n")
    if max_tokens is not None and len(ids) > max_tokens:
        raise ValueError("chat generation prompt exceeds the context budget; shorten the conversation explicitly")
    return ids


def hash_conversation(value: Any) -> int:
    """Hash exact ordered content and roles; code case/whitespace remain significant."""
    messages = validate_messages(value, allow_trailing_user=False)
    payload = json.dumps([CHAT_POLICY, [[m["role"], m["content"]] for m in messages]], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big", signed=True)


def count_fitted_positions(ends: list[int], target: int) -> int:
    """Count shifted positions usable at target from persisted complete-exchange token ends."""
    if not ends or any(type(end) is not int or end < 2 for end in ends) or any(b <= a for a, b in zip(ends, ends[1:])):
        raise ValueError("invalid conversation exchange_ends")
    return max((end - 1 for end in ends if end <= target + 1), default=0)


def fit_literal_conversation(messages: list[Message], tokenizer: ChatTokenizer, max_tokens: int | None, *, bos: bool, eos: bool) -> EncodedConversation:
    """Keep complete exchanges with masks computed from message provenance, never token-string matching."""
    from tokenization.chat import encode_chat

    if not bos or not eos or (max_tokens is not None and max_tokens < 0):
        raise ValueError("literal chat requires BOS/EOS and a nonnegative token budget")
    complete = messages[:len(messages) // 2 * 2]
    if not complete:
        return EncodedConversation([], [], [], [], 0, True)
    encoded = encode_chat(complete, tokenizer.encode_literal)
    ends = [end for end in encoded.exchange_ends if max_tokens is None or end <= max_tokens]
    limit = ends[-1] if ends else 0
    return EncodedConversation(encoded.ids[:limit], encoded.supervised[:limit], ends, complete[:2 * len(ends)],
                               len(complete) // 2 - len(ends), bool(len(messages) % 2))
