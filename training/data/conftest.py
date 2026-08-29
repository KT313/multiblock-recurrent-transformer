# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Fixtures shared by the training/data tests."""

import json
import shutil
from pathlib import Path

import pytest

from training.data.tokenizer import Tokenizer

# Chat template built only from words the synthetic WordLevel tokenizer knows. Deliberately without
# {% generation %} tags so fix_chat_template_for_masking has something to fix. The separator space after an
# assistant turn sits outside the elif branch: transformers computes the assistant mask from character offsets,
# and with a Whitespace pre-tokenizer a trailing space *inside* the generation span bleeds the mask into the
# following user turn.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}tok_1 {{ message['content'] }} "
    "{% elif message['role'] == 'assistant' %}{{ message['content'] }} tok_2{% endif %} "
    "{% endfor %}"
)


@pytest.fixture(scope="session")
def tokenizer(tiny_tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer(tiny_tokenizer_dir)


@pytest.fixture
def chat_tokenizer(tiny_tokenizer_dir: Path, tmp_path: Path) -> Tokenizer:
    """Copy of the tiny tokenizer carrying a Llama-2-style chat template (fresh per test: the template is mutated)."""
    dst = tmp_path / "chat_tokenizer"
    shutil.copytree(tiny_tokenizer_dir, dst)
    cfg_path = dst / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["chat_template"] = CHAT_TEMPLATE
    cfg_path.write_text(json.dumps(cfg))
    return Tokenizer(dst)
