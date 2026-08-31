# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Fixtures shared by the training/data tests."""

from pathlib import Path

import pytest

from training.data.tokenizer import Tokenizer


@pytest.fixture(scope="session")
def tokenizer(tiny_tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer(tiny_tokenizer_dir)
