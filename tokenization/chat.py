# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Framework-neutral chat encoding. Only this formatter inserts structural token IDs."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer as RustTokenizer, pre_tokenizers

PROFILE = "llama32k_chat_v1"
BASE_REPO = "hf-internal-testing/llama-tokenizer"
BASE_REVISION = "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
BASE_SIZE, VOCAB_SIZE = 32000, 32002
BOS, EOS, USER, ASSISTANT = 1, 2, 32000, 32001
FORMAT_VERSION = "literal-messages-v1"
# This template is the readable rendering. Literal-aware tokenization uses message provenance, not a rescan of it.
CHAT_TEMPLATE = "{{ bos_token }}{% for m in messages %}{{ '<user>' if m['role'] == 'user' else '<assistant>' }}{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}{{ eos_token }}{% endgeneration %}{% else %}{{ m['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"


@dataclass
class ChatEncoding:
    ids: list[int]
    supervised: list[bool]
    exchange_ends: list[int]
    text: str


def validate_chat(messages: Any) -> list[dict[str, str]]:
    """Validate roles and preserve every content character, including literal special-token spellings."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat messages must be a nonempty list")
    result = []
    for index, message in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError(f"chat message {index} must have role {role!r}; system/tool roles are unsupported")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"chat message {index} content must be a nonempty string")
        result.append({"role": role, "content": content})
    return result


def encode_chat(messages: Any, encode_literal: Callable[[str], list[int]], *, generation_prompt: bool = False) -> ChatEncoding:
    """Encode each literal body independently; retain boundaries without guessing from the rendered string."""
    validated = validate_chat(messages)
    if generation_prompt and validated[-1]["role"] != "user":
        raise ValueError("add_generation_prompt requires a final user message")
    ids, supervised, ends = [BOS], [False], []
    parts = ["<s>"]
    for message in validated:
        assistant = message["role"] == "assistant"
        body = encode_literal(message["content"])
        if any(token < 0 or token >= BASE_SIZE or token in (BOS, EOS) for token in body):
            raise ValueError("literal content encoder produced a structural or invalid token ID")
        ids.extend([ASSISTANT if assistant else USER, *body, EOS])
        supervised.extend([False, *([assistant] * len(body)), assistant])
        parts.extend(["<assistant>" if assistant else "<user>", message["content"], "</s>"])
        if assistant:
            ends.append(len(ids))
    if generation_prompt:
        ids.append(ASSISTANT)
        supervised.append(False)
        parts.append("<assistant>")
    return ChatEncoding(ids, supervised, ends, "".join(parts))


def build_literal_encoder(backend: RustTokenizer, *, chat_body: bool = False) -> RustTokenizer:
    """Disable special-string recognition; chat bodies start after a header, without an invented prefix space."""
    literal = RustTokenizer.from_str(backend.to_str())
    literal.encode_special_tokens = True
    if chat_body:
        literal.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=False)
    return literal
