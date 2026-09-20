# (c) 2026 Tobias Kerner. Apache-2.0.
"""Bounded two-rank CPU parity checks; invoked by test_optimizer_sharding, never on import."""

import copy
import io
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from training.backend.ddp import DDPBackend
from training.backend import ddp as ddp_module
from training.logger import track_gradient_metrics
from training.optim import OptimState8bit, build_optimizer, get_param_groups, set_lr
from training.optim.sharding import local_optimizer, prepare_optimizer_state, release_optimizer_state
from training.settings import OptimizerConfig
from training.stopping import complete_main_phase


class MetricModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_embd=64, head_size=16, num_attention_heads=4)
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(128, 64)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"qkv": nn.Linear(64, 192, bias=False),
                           "mlp": nn.ModuleDict({"proj": nn.Linear(64, 64, bias=False)}),
                           "norm": nn.LayerNorm(64)}) for _ in range(3)
        ])
        self.lm_head = nn.Linear(64, 128, bias=False)
        self.lm_head.weight = self.transformer.wte.weight


def compare(left: object, right: object) -> None:
    if isinstance(left, OptimState8bit):
        assert isinstance(right, OptimState8bit)
        assert left.signed == right.signed
        for attr in ("codes", "scale", "qmap"):
            torch.testing.assert_close(getattr(left, attr), getattr(right, attr), rtol=0, atol=0)
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            compare(a, b)
    else:
        assert left == right, (left, right)


def snapshot(value: object) -> object:
    # The vendored quantized tensor deliberately does not implement clone/deepcopy.
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def check_optimizer(name: str, backend: DDPBackend) -> None:
    torch.manual_seed(117)
    model = MetricModel()
    reference = copy.deepcopy(model)
    options = OptimizerConfig(lr=1e-3, weight_decay=0.02)
    if name != "AdamW":
        options.running_init = options.update_clipping = options.atan_adam = True

    def construct(target: nn.Module, sharding: str = "none") -> torch.optim.Optimizer:
        return build_optimizer(name, get_param_groups(target, options.weight_decay), options, sharding=sharding)

    optimizer, ordinary = construct(model, "zero1"), construct(reference)
    generator = torch.Generator().manual_seed(81)
    for step in range(6):
        # Initially missing gradients/state, then later initialization; tied weight appears exactly once.
        for index, (param, ref) in enumerate(zip(model.parameters(), reference.parameters(), strict=True)):
            grad = torch.randn(param.shape, generator=generator) * 0.03
            param.grad = None if step < 2 and index % 4 == 0 else grad.clone()
            ref.grad = None if param.grad is None else grad.clone()
        set_lr(optimizer, 0.0 if step < 2 else 0.001 / step)
        set_lr(ordinary, 0.0 if step < 2 else 0.001 / step)
        if step:
            optimizer.step()
            ordinary.step()
        compare(model.state_dict(), reference.state_dict())
        inner = local_optimizer(optimizer)
        owners = backend.all_gather_object([
            param_name for param_name, param in model.named_parameters() if inner.state.get(param)
        ])
        assigned = [param_name for rank_names in owners for param_name in rank_names]
        expected_names = [param_name for param_name, param in reference.named_parameters() if ordinary.state.get(param)]
        assert len(assigned) == len(set(assigned)) and set(assigned) == set(expected_names)
        reference_parameters = dict(reference.named_parameters())
        for param_name, param in model.named_parameters():
            if param in inner.state:
                compare(inner.state[param], ordinary.state[reference_parameters[param_name]])

        # Metrics must match and must not modify state, grads, modes or RNG.
        before = snapshot(inner.state_dict())
        rng = torch.get_rng_state().clone()
        actual = track_gradient_metrics(model, optimizer, backend=backend)
        expected = track_gradient_metrics(reference, ordinary)
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=1e-7)
        compare(before, inner.state_dict())
        compare(rng, torch.get_rng_state())
        for param, ref in zip(model.parameters(), reference.parameters(), strict=True):
            compare(param.grad, ref.grad)

        # Save empty, partially initialized and complete states; deserialize CPU tensors on each rank.
        if step in (0, 1, 3):
            prepare_optimizer_state(optimizer)
            if backend.is_main:
                consolidated = optimizer.state_dict()
                compare(consolidated, ordinary.state_dict())
                stream = io.BytesIO()
                torch.save(consolidated, stream)
                payload = stream.getvalue()
            else:
                payload = b""
            payload = backend.all_gather_object(payload)[0]
            release_optimizer_state(optimizer)
            try:
                optimizer.state_dict()
            except RuntimeError:
                pass
            else:
                raise AssertionError("stale consolidated snapshot survived publication")
            optimizer = construct(model, "zero1")
            saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            before_load = snapshot(saved)
            if step == 3:
                # Precision/schema rejection must leave the fresh optimizer empty.
                index = next(iter(saved["state"]))
                original = saved["state"][index]["exp_avg"]
                saved["state"][index]["exp_avg"] = torch.zeros(original.shape, dtype=torch.bfloat16)
                try:
                    optimizer.load_state_dict(saved)
                except ValueError as error:
                    assert "FP32" in str(error)
                else:
                    raise AssertionError("non-FP32 checkpoint state accepted")
                assert not local_optimizer(optimizer).state
                saved["state"][index]["exp_avg"] = original
            optimizer.load_state_dict(saved)
            compare(saved, before_load)
            assert not optimizer.state, "outer optimizer retained redundant moment storage"
        optimizer.zero_grad(set_to_none=True)
        ordinary.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in model.parameters())
    # Every healthy rank observes a writer failure; consolidation snapshots are still released.
    prepare_optimizer_state(optimizer)

    def fail_publication() -> None:
        raise OSError("intentional publication failure")

    try:
        complete_main_phase(backend, "test checkpoint", fail_publication)
    except (OSError, RuntimeError):
        pass
    else:
        raise AssertionError("publication failure was not propagated")
    finally:
        release_optimizer_state(optimizer)
    if backend.is_main:
        print(f"PASS {name}: updates, owner moments, metrics, empty/partial/full resume", flush=True)


def check_empty_owner(backend: DDPBackend) -> None:
    """A whole rank owns no parameters, but still updates replicas and joins metric collectives."""
    model = nn.Linear(64, 64, bias=False)
    dist.broadcast(model.weight.data, src=0)
    reference = copy.deepcopy(model)
    options = OptimizerConfig(lr=1e-3)
    optimizer = build_optimizer("ELLISAdam8bit", get_param_groups(model, 0.01), options, sharding="zero1")
    ordinary = build_optimizer("ELLISAdam8bit", get_param_groups(reference, 0.01), options)
    model.weight.grad = torch.ones_like(model.weight)
    reference.weight.grad = torch.ones_like(reference.weight)
    optimizer.step()
    ordinary.step()
    counts = backend.all_gather_object(len(local_optimizer(optimizer).state))
    assert sorted(counts) == [0, 1]
    compare(model.state_dict(), reference.state_dict())
    actual = track_gradient_metrics(model, optimizer, backend=backend)
    expected = track_gradient_metrics(reference, ordinary)
    compare(actual, expected)

    try:
        build_optimizer("ELLISAdam", [nn.Parameter(torch.ones(2, dtype=torch.bfloat16))], options, sharding="zero1")
    except ValueError as error:
        assert "FP32 parameter" in str(error)
    else:
        raise AssertionError("BF16 parameter storage accepted")


def main() -> None:
    torch.set_num_threads(1)
    ddp_module.DDP_TIMEOUT = timedelta(seconds=45)
    backend = DDPBackend(device="cpu", precision="32")
    try:
        for name in ("ELLISAdam", "ELLISAdam8bit", "AdamW"):
            check_optimizer(name, backend)
        check_empty_owner(backend)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
