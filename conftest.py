# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Shared pytest fixtures: the tiny dataset config built into a session temp dir, its tokenizer, the tiny model.

Under pytest-xdist every worker process imports torch; the intra-op thread count is capped to the machine's
share per worker so the four workers do not oversubscribe the cores. GPU tests share one worker (`xdist_group`).
"""

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(os.environ["PYTEST_XDIST_WORKER_COUNT"])))

from data_preparation.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build import prepare
from model import RecurrentGPT

REPO_ROOT = Path(__file__).resolve().parent
TINY_DATASET_CONFIG = REPO_ROOT / "config" / "datasets" / "tiny.yaml"
TINY_MODEL_ARCHITECTURE = REPO_ROOT / "config" / "model_architecture" / "tiny.yaml"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """GPU tests share one xdist worker (one 8 GB device) and are skipped without a CUDA device."""
    cuda = torch.cuda.is_available()
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(pytest.mark.xdist_group("gpu"))
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="no CUDA device"))


@pytest.fixture
def short_tmp_path() -> Iterator[Path]:
    """A short per-test directory (`/tmp/pytest-short-*`) for tests that show a path on a fixed-width screen: pytest's
    `tmp_path` grows under xdist (`popen-gwN/`) and with the session counter (`pytest-NNN`), and pushes such lines past
    the width they are asserted at."""
    path = Path(tempfile.mkdtemp(prefix="pytest-short-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_dataset_config() -> DatasetConfig:
    return load_dataset_config(TINY_DATASET_CONFIG)


@pytest.fixture(scope="session")
def tiny_dataset_dir(tmp_path_factory: pytest.TempPathFactory, tiny_dataset_config: DatasetConfig) -> Path:
    """`config/datasets/tiny.yaml` built into a session temp root: the `dataset/` layout (sources/, processed/,
    tokenizers/) that `config/tiny.yaml` expects under `dataset/`."""
    root: Path = tmp_path_factory.mktemp("tiny_dataset")
    prepare(TINY_DATASET_CONFIG, root, assume_yes=False)
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
