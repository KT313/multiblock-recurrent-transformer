# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Shared pytest fixtures: the tiny dataset config built into a session temp dir, its tokenizer, the tiny model.

Under pytest-xdist every worker gets the machine's share of the cores: torch's intra-op threads, the OpenMP / MKL
pools of every process the tests start (the PTY runs of train.py and prepare.py, spawned dedup workers) and the
inductor compile worker pool, so the workers do not oversubscribe the cores. GPU tests share one worker (`xdist_group`).
"""

import fcntl
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# No test may reach the Hub: every Hub request raises `OfflineModeIsEnabled` at once instead of downloading a
# real source (a wrong config or a failed refusal check would otherwise start a multi-GB download). Set before
# huggingface_hub / datasets are imported (they read the env at import time); subprocess tests inherit it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# The env caps every child process and the pools torch, numpy and inductor size at import; set before torch is imported.
if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
    _core_share = str(max(1, (os.cpu_count() or 1) // int(os.environ["PYTEST_XDIST_WORKER_COUNT"])))
    for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
        os.environ.setdefault(_name, _core_share)

import torch

if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

from data_preparation.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.runner import prepare
from model import RecurrentGPT

REPO_ROOT = Path(__file__).resolve().parent
TINY_DATASET_CONFIG = REPO_ROOT / "config" / "datasets" / "tiny.yaml"
TINY_MODEL_ARCHITECTURE = REPO_ROOT / "config" / "model_architecture" / "tiny.yaml"


def _uses_a_module_fixture(item: pytest.Item) -> bool:
    manager = item.session._fixturemanager
    for name in getattr(item, "fixturenames", ()):
        for definition in manager.getfixturedefs(name, item) or ():
            if definition.scope == "module":
                return True
    return False


@pytest.hookimpl(tryfirst=True)  # before xdist's worker hook, which writes the group names into the node ids
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    GPU tests share one xdist worker (one 8 GB device) and are skipped without a CUDA device. The consumers of a
    module-scoped fixture share one worker too: `--dist loadgroup` would otherwise scatter them, and every worker
    that got one would build the fixture again (the full tiny run of `training/test_run.py`, several seconds each).
    """

    cuda = torch.cuda.is_available()
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(pytest.mark.xdist_group("gpu"))
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        elif _uses_a_module_fixture(item):
            item.add_marker(pytest.mark.xdist_group(f"module:{item.nodeid.split('::')[0]}"))


@pytest.fixture
def short_tmp_path() -> Iterator[Path]:
    """
    A short per-test directory (`/tmp/pytest-short-*`) for tests that show a path on a fixed-width screen: pytest's
    `tmp_path` grows under xdist (`popen-gwN/`) and with the session counter (`pytest-NNN`), and pushes such lines past
    the width they are asserted at.
    """

    path = Path(tempfile.mkdtemp(prefix="pytest-short-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_dataset_config() -> DatasetConfig:
    return load_dataset_config(TINY_DATASET_CONFIG)


@pytest.fixture(scope="session")
def session_shared_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    The directory a session-wide build goes to: the session's base temp, which under xdist is the parent of the
    workers' base temps, so every worker sees one copy (`tiny_dataset_dir`, `reference_run` in training/conftest.py).
    """

    base = tmp_path_factory.getbasetemp()
    if "PYTEST_XDIST_WORKER" in os.environ:  # the workers' base temps sit under one session directory: build there once
        base = base.parent
    return base


@pytest.fixture(scope="session")
def tiny_dataset_dir(session_shared_base: Path, tiny_dataset_config: DatasetConfig) -> Path:
    """
    `config/datasets/tiny.yaml` built into a session temp root: the `dataset/` layout (sources/, processed/,
    tokenizers/) that `config/tiny.yaml` expects under `dataset/`. Built once per session, under xdist by the first
    worker that needs it (the others wait on the lock and reuse the build).
    """

    base = session_shared_base
    root = base / "tiny_dataset"
    ready = base / "tiny_dataset.ready"
    with open(base / "tiny_dataset.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not ready.exists():
            prepare(TINY_DATASET_CONFIG, root, assume_yes=False)
            ready.touch()
    return root


@pytest.fixture(scope="session")
def tiny_layout(tiny_dataset_dir: Path) -> DatasetLayout:
    return DatasetLayout(tiny_dataset_dir)


@pytest.fixture(scope="session")
def tiny_pretrain_dir(tiny_layout: DatasetLayout) -> Path:
    """
    `processed/synthetic_pretrain`: text rows.
    """

    return tiny_layout.processed_dir("synthetic_pretrain")


@pytest.fixture(scope="session")
def tiny_instruct_dir(tiny_layout: DatasetLayout) -> Path:
    """
    `processed/synthetic_instruct`: instruction / input / output rows.
    """

    return tiny_layout.processed_dir("synthetic_instruct")


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tiny_layout: DatasetLayout) -> Path:
    return tiny_layout.tokenizer_dir("synthetic")


@pytest.fixture
def tiny_model() -> RecurrentGPT:
    """
    Fresh, seeded `tiny` model (config/model_architecture/tiny.yaml) on the CPU (~256K parameters).
    """

    from model import build_model

    torch.manual_seed(0)
    return build_model(TINY_MODEL_ARCHITECTURE)
