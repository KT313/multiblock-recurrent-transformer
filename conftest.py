# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Shared pytest fixtures: the tiny dataset config built into a session temp dir, its tokenizer, the tiny model."""

from pathlib import Path

import pytest
import torch

from data_preparation.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build import build
from model import RecurrentGPT

REPO_ROOT = Path(__file__).resolve().parent
TINY_DATASET_CONFIG = REPO_ROOT / "config" / "datasets" / "tiny.yaml"
TINY_MODEL_ARCHITECTURE = REPO_ROOT / "config" / "model_architecture" / "tiny.yaml"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA device")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_dataset_config() -> DatasetConfig:
    return load_dataset_config(TINY_DATASET_CONFIG)


@pytest.fixture(scope="session")
def tiny_dataset_dir(tmp_path_factory: pytest.TempPathFactory, tiny_dataset_config: DatasetConfig) -> Path:
    """`config/datasets/tiny.yaml` built into a session temp root: the `dataset/` layout (sources/, processed/,
    tokenizers/) that `config/tiny.yaml` expects under `dataset/`."""
    root: Path = tmp_path_factory.mktemp("tiny_dataset")
    build(tiny_dataset_config, DatasetLayout(root))
    return root


@pytest.fixture(scope="session")
def tiny_layout(tiny_dataset_dir: Path) -> DatasetLayout:
    return DatasetLayout(tiny_dataset_dir)


@pytest.fixture(scope="session")
def tiny_pretrain_dir(tiny_layout: DatasetLayout) -> Path:
    """`processed/synthetic_pretrain`: text rows."""
    return tiny_layout.processed_dir("synthetic_pretrain")


@pytest.fixture(scope="session")
def tiny_instruct_dir(tiny_layout: DatasetLayout) -> Path:
    """`processed/synthetic_instruct`: instruction / input / output rows."""
    return tiny_layout.processed_dir("synthetic_instruct")


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tiny_layout: DatasetLayout) -> Path:
    return tiny_layout.tokenizer_dir("synthetic")


@pytest.fixture
def tiny_model() -> RecurrentGPT:
    """Fresh, seeded `tiny` model (config/model_architecture/tiny.yaml) on the CPU (~256K parameters)."""
    from model import build_model

    torch.manual_seed(0)
    return build_model(TINY_MODEL_ARCHITECTURE)
