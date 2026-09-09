# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the DDP backend on a ONE-rank gloo process group in this process: construction from the torchrun
environment, the wrapper layering, the collectives at world size 1, the main-only checkpoint write, and the golden
tiny run through `backend: ddp` (the numerics oracle: DDP at world size 1 must be the single device). The two-rank
behaviour is tested through real torchrun launches in `training/test_distributed.py`.

The process group is process-global, so every test here shares one xdist worker (`xdist_group`) and the fixture
destroys the group after each test.
"""

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model import RecurrentGPT
from training.backend import get_backend
from training.backend.base import Backend
from training.backend.ddp import DDP_TIMEOUT, TORCHRUN_VARIABLES, DDPBackend
from training.run import run_directory_of, train
from training.settings import parse_settings
from training.testing.network import free_port
from training.testing.golden import (
    GOLDEN_RUN_PATH,
    ReferenceRun,
    golden_exact_requested,
    golden_mismatches,
    golden_run_json,
    reference_metrics,
    single_thread_deterministic,
    write_tiny_yaml,
)

pytestmark = pytest.mark.xdist_group("ddp")


@pytest.fixture
def torchrun_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    The variables torchrun sets, for a one-rank group on a free port; the group is destroyed afterwards.
    """

    values = {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port())}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture
def backend(torchrun_env: None) -> DDPBackend:
    return DDPBackend(device="cpu", precision="32")


def test_construction_needs_the_torchrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in TORCHRUN_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=r"needs the torchrun environment but \['RANK', 'WORLD_SIZE'.*torchrun --standalone"):
        DDPBackend(device="cpu", precision="32")
    assert not dist.is_initialized()


def test_one_rank_group_on_the_cpu_uses_gloo(backend: DDPBackend) -> None:
    assert dist.is_initialized() and dist.get_backend() == "gloo"
    assert (backend.world_size, backend.rank, backend.is_main) == (1, 0, True)
    assert backend.device == torch.device("cpu") and backend.pin_memory is False
    assert DDP_TIMEOUT.total_seconds() >= 3600  # the other ranks wait through a dataset build or a benchmark


def test_registry_builds_the_ddp_backend(torchrun_env: None) -> None:
    backend = get_backend("ddp", device="cpu", precision="32")
    assert isinstance(backend, DDPBackend) and dist.is_initialized()


def test_cpu_fallback_warns_without_cuda(torchrun_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="falling back to CPU"):
        backend = DDPBackend(precision="32")
    assert backend.device == torch.device("cpu")


def test_explicit_cpu_does_not_warn(torchrun_env: None) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DDPBackend(device="cpu", precision="32")


def test_ddp_backend_implements_the_protocol(backend: DDPBackend) -> None:
    protocol_methods = {name for name in vars(Backend) if not name.startswith("_")}
    assert protocol_methods <= set(dir(DDPBackend))
    typed: Backend = backend  # static check: satisfies the Protocol
    assert typed.world_size == 1


def test_setup_model_wraps_in_ddp_then_compiles_and_records_the_layering(backend: DDPBackend, tiny_model: RecurrentGPT) -> None:
    """
    DDP is the inner wrapper, compile the outer (compiling only wraps here, no forward runs); `plain_model` follows
    the recorded layering and the state dict of the plain model carries no `module.` prefix.
    """

    wrapped = backend.setup_model(tiny_model)
    assert isinstance(wrapped, DistributedDataParallel) and backend.wrappers == ("ddp",)
    assert backend.plain_model(wrapped) is tiny_model
    assert set(backend.plain_model(wrapped).state_dict()) == set(tiny_model.state_dict())
    with pytest.raises(TypeError, match="expected a ddp wrapper"):
        backend.plain_model(tiny_model)
    assert not isinstance(backend.no_sync(wrapped), type(None))

    compiled = backend.setup_model(tiny_model, compile_model=True)
    assert list(backend.wrappers) == ["compile", "ddp"]
    assert not isinstance(compiled, DistributedDataParallel)
    assert backend.plain_model(compiled) is tiny_model
    backend.no_sync(compiled)  # reaches the DDP module behind the compile wrapper


def test_no_sync_refuses_a_model_without_the_ddp_wrapper(backend: DDPBackend, tiny_model: RecurrentGPT) -> None:
    backend.setup_model(tiny_model)
    with pytest.raises(TypeError, match="expected the DDP wrapper, found RecurrentGPT"):
        backend.no_sync(tiny_model)


def test_collectives_at_world_size_one(backend: DDPBackend) -> None:
    values = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(backend.all_reduce(values.clone()), values)
    assert torch.equal(backend.all_reduce(values.clone(), op="sum"), values)
    with pytest.raises(ValueError, match="op must be"):
        backend.all_reduce(values.clone(), op="max")
    backend.barrier()
    state: dict[str, Any] = {"a": 1, "tensor": torch.arange(3)}
    gathered = backend.all_gather_object(state)
    assert len(gathered) == 1 and gathered[0]["a"] == 1 and torch.equal(gathered[0]["tensor"], torch.arange(3))
    assert backend.any_flag(True) is True and backend.any_flag(False) is False


def test_scatter_packs_hands_the_main_rank_its_slice(backend: DDPBackend) -> None:
    packs = torch.arange(2 * 4 * 8, dtype=torch.int64).view(2, 4, 8)[:1]  # world size 1: one slice
    received = backend.scatter_packs(packs, (4, 8))
    assert torch.equal(received, packs[0]) and received.dtype == torch.int64
    with pytest.raises(ValueError, match="expected \\(1, 4, 8\\)"):
        backend.scatter_packs(torch.zeros((2, 4, 8), dtype=torch.int64), (4, 8))
    with pytest.raises(ValueError, match="main rank must pass the packs"):
        backend.scatter_packs(None, (4, 8))


def test_shutdown_destroys_the_group_and_is_idempotent(backend: DDPBackend) -> None:
    backend.shutdown()
    assert not dist.is_initialized()
    backend.shutdown()


def test_save_checkpoint_writes_on_the_main_rank_and_round_trips(backend: DDPBackend, tmp_path: Path) -> None:
    state = {"step": 3, "tensor": torch.arange(4)}
    path = tmp_path / "ckpt.pth"
    backend.save_checkpoint(path, state)
    assert path.exists()
    loaded = backend.load_checkpoint(path)
    assert loaded["step"] == 3 and torch.equal(loaded["tensor"], torch.arange(4))
    backend.is_main = False
    backend.save_checkpoint(tmp_path / "other.pth", state)
    assert not (tmp_path / "other.pth").exists()


@pytest.mark.slow
def test_golden_tiny_run_through_the_ddp_backend(torchrun_env: None, tiny_dataset_dir: Path, tmp_path: Path) -> None:
    """
    The DDP backend at world size 1 on gloo IS the single device: the 20-step tiny run in the golden configuration
    (fp32, CPU, one thread, deterministic algorithms) through `backend: ddp` reproduces `golden_tiny_run.json`
    exactly as `test_golden_tiny_run` does for the single-device backend. The DDP wrapper, the one-rank all-reduces
    of the loss and the gradients, the gathered RNG state and the scatter of every pack to rank 0 itself must not
    move a number.
    """

    with single_thread_deterministic():
        out_dir = tmp_path / "out"
        yaml_path = write_tiny_yaml(
            tmp_path, tiny_dataset_dir, out_dir, backend="ddp", precision="32", wandb_enabled=False, export_to_hf=False, resume=False
        )
        settings = parse_settings(["--config", str(yaml_path)])
        assert settings.backend == "ddp"
        report = train(settings, backend=DDPBackend(device="cpu", precision="32"), keep_history=True)
    reference = ReferenceRun(report.history, yaml_path, run_directory_of(settings))
    expected = json.loads(GOLDEN_RUN_PATH.read_text())
    actual = reference_metrics(reference)
    assert actual["optimizer_steps"] == 19
    stored = torch.load(reference.run_directory / "checkpoints" / actual["checkpoints"][-1], map_location="cpu", weights_only=False)
    assert stored["world_size"] == 1 and len(stored["rng_states"]) == 1
    assert not set(stored["model"]).intersection(key for key in stored["model"] if key.startswith("module."))
    mismatches = golden_mismatches(expected, json.loads(golden_run_json(actual)), exact=golden_exact_requested())
    assert not mismatches, "golden run through the DDP backend differs:\n" + "\n".join(mismatches)
    assert not dist.is_initialized(), "train() shuts the backend down on its way out"
