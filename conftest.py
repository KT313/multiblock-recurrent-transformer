# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Shared pytest fixtures: the synthetic tiny dataset + tokenizer and the tiny model config."""

from pathlib import Path

import pytest
import torch

from data_preparation.lib.make_tiny_dataset import make_tiny_dataset
from model import RecurrentGPT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA device")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`dataset/tiny` layout (tokenizer/, pretrain/{train,val}, finetune/{train,val}) in a session temp dir."""
    out: Path = tmp_path_factory.mktemp("tiny_dataset")
    make_tiny_dataset(out)
    return out


@pytest.fixture(scope="session")
def tiny_tokenizer_path(tiny_dataset_dir: Path) -> Path:
    return tiny_dataset_dir / "tokenizer"


@pytest.fixture
def tiny_model() -> RecurrentGPT:
    """Fresh, seeded `tiny` preset model on the CPU (~256K parameters)."""
    from model import build_model

    torch.manual_seed(0)
    return build_model("tiny")
