# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the parameter-group split, `build_optimizer`/`set_lr` and hand-checked ELLISAdam steps.
"""

import copy
import math
from typing import Any

import pytest
import torch

from model import RecurrentGPT
from training.optim import ELLISAdam, _single_tensor_modded_adamw, build_optimizer, get_param_groups, set_lr
from training.settings import OptimizerConfig


def _param_ids(groups: list[dict[str, Any]]) -> list[int]:
    return [id(p) for g in groups for p in g["params"]]


def test_param_groups_cover_every_parameter_exactly_once(tiny_model: RecurrentGPT) -> None:
    groups = get_param_groups(tiny_model, weight_decay=0.1)
    ids = _param_ids(groups)
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in tiny_model.parameters()}
    assert len(groups) == 3


def test_no_wd_group_is_exactly_bias_and_norm_params(tiny_model: RecurrentGPT) -> None:
    groups = get_param_groups(tiny_model, weight_decay=0.1, no_wd_for_bias_and_norm=True)
    no_wd = {id(p) for p in groups[2]["params"]}
    expected = {
        id(p) for n, p in tiny_model.named_parameters() if any(k in n.lower() for k in ("norm", "bias", "ln_f"))
    }
    assert no_wd == expected
    assert groups[2]["weight_decay"] == 0.0
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.1
    # every no-WD parameter is 1-D or a qk_bias tensor; every WD parameter is a matrix
    assert all(p.ndim == 1 or p.shape[0] == 2 for p in groups[2]["params"])
    assert all(p.ndim == 2 for p in groups[0]["params"] + groups[1]["params"])


def test_embedding_group_holds_wte_only(tiny_model: RecurrentGPT) -> None:
    groups = get_param_groups(tiny_model, weight_decay=0.1)
    assert [id(p) for p in groups[1]["params"]] == [id(tiny_model.transformer.wte.weight)]


def test_no_wd_flag_off_keeps_weight_decay(tiny_model: RecurrentGPT) -> None:
    groups = get_param_groups(tiny_model, weight_decay=0.05, no_wd_for_bias_and_norm=False)
    assert all(g["weight_decay"] == 0.05 for g in groups)


def test_unmatched_parameter_raises() -> None:
    class Odd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mystery = torch.nn.Parameter(torch.zeros(3))

    with pytest.raises(ValueError, match="could not be matched"):
        get_param_groups(Odd(), weight_decay=0.1)


def test_plain_2d_parameter_goes_to_the_weights_group() -> None:
    class Mat(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.table = torch.nn.Parameter(torch.zeros(3, 3))

    groups = get_param_groups(Mat(), weight_decay=0.1)
    assert len(groups[0]["params"]) == 1 and not groups[1]["params"] and not groups[2]["params"]


@pytest.mark.parametrize("name,cls", [("AdamW", torch.optim.AdamW), ("ELLISAdam", ELLISAdam)])
def test_build_optimizer(name: str, cls: type[torch.optim.Optimizer]) -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    opt = build_optimizer(name, [p], OptimizerConfig(lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95)))
    assert isinstance(opt, cls)
    with pytest.raises(ValueError, match="Invalid optimizer"):
        build_optimizer("SGD", [p], OptimizerConfig(lr=1e-3))


def _group_options(opt: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """
    Every per-group option except the parameter list, tensors compared by value.
    """

    return [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]


def test_build_optimizer_matches_the_old_dict_path() -> None:
    """
    The dataclass path hands the optimizer exactly the keyword arguments the free-form dict did: the shipped
    ELLISAdam options give the same defaults, and an unset `eps` keeps each optimizer's own default.
    """

    p = torch.nn.Parameter(torch.zeros(2))
    config = OptimizerConfig(
        lr=1e-4, weight_decay=4e-5, betas=(0.9, 0.95), update_clipping=True, atan_adam=True, running_init=True
    )
    built = build_optimizer("ELLISAdam", [p], config)
    q = torch.nn.Parameter(torch.zeros(2))
    reference = ELLISAdam(
        [q], lr=1e-4, weight_decay=4e-5, betas=(0.9, 0.95), update_clipping=True, atan_adam=True, running_init=True
    )
    for group, ref_group in zip(_group_options(built), _group_options(reference), strict=True):
        assert set(group) == set(ref_group)
        for key, value in group.items():
            ref = ref_group[key]
            assert torch.equal(value, ref) if isinstance(value, torch.Tensor) else value == ref, key
    assert built.param_groups[0]["eps"] == 1e-6  # ELLISAdam's default
    adamw = build_optimizer("AdamW", [p], OptimizerConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)))
    assert adamw.param_groups[0]["eps"] == 1e-8  # torch AdamW's default
    assert build_optimizer("AdamW", [p], OptimizerConfig(eps=1e-5)).param_groups[0]["eps"] == 1e-5


def test_build_optimizer_rejects_ellis_only_options_for_adamw() -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match=r"\['atan_adam'\] apply only to 'ELLISAdam'"):
        build_optimizer("AdamW", [p], OptimizerConfig(atan_adam=True))
    with pytest.raises(ValueError, match=r"\['decouple_wd'\] apply only to 'ELLISAdam'"):
        build_optimizer("AdamW", [p], OptimizerConfig(decouple_wd=False))


def test_set_lr_sets_a_tensor_lr_on_every_group() -> None:
    p1, p2 = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p1]}, {"params": [p2]}], lr=1.0)
    set_lr(opt, 2e-4)
    assert all(torch.is_tensor(g["lr"]) for g in opt.param_groups)
    assert [float(g["lr"]) for g in opt.param_groups] == pytest.approx([2e-4, 2e-4])


def _reference_ellis_adam_step(
    p: float,
    g: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    m: float,
    v: float,
    step: int,
    update_clipping: bool,
    atan_adam: bool,
) -> float:
    """
    Plain-python re-implementation of `_single_tensor_modded_adamw` for a scalar at `init_lr == lr`.
    """

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    step_size = lr
    if update_clipping:
        rms = math.sqrt(g * g / max(v, eps**2))
        step_size = step_size / max(rms, 1.0)
    step_size = step_size / (1 - beta1**step)
    denom = math.sqrt(v) / math.sqrt(1 - beta2**step)
    p = p * (1 - lr / lr * wd)
    if atan_adam:
        return p - step_size * math.atan2(m, denom)
    return p - step_size * m / (denom + eps)


@pytest.mark.parametrize(
    "flags,expected",
    [
        # m=0.05, v=0.0025, bc1=0.1, bc2_sqrt=0.1, step_size=1, denom=0.5 -> 0.99 - 0.05/(0.5+1e-6)
        ({}, 0.99 - 0.05 / (0.5 + 1e-6)),
        # rms = sqrt(0.25/0.0025) = 10 -> step_size 0.1/10/0.1 = 0.1
        ({"update_clipping": True}, 0.99 - 0.1 * 0.05 / (0.5 + 1e-6)),
        # atan2(0.05, 0.5) with step_size 1
        ({"atan_adam": True}, 0.99 - math.atan2(0.05, 0.5)),
    ],
)
def test_ellis_adam_one_step_matches_hand_calculation(flags: dict[str, bool], expected: float) -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([0.5])
    opt = ELLISAdam([p], lr=0.1, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.01, **flags)
    opt.step()
    assert p.item() == pytest.approx(expected, abs=1e-7)
    ref = _reference_ellis_adam_step(
        1.0,
        0.5,
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        eps=1e-6,
        wd=0.01,
        m=0.0,
        v=0.0,
        step=1,
        update_clipping=flags.get("update_clipping", False),
        atan_adam=flags.get("atan_adam", False),
    )
    assert p.item() == pytest.approx(ref, abs=1e-7)
    state = opt.state[p]
    assert state["step"].item() == 1 and state["step"].device.type == "cpu"
    assert state["exp_avg"].item() == pytest.approx(0.05)
    assert state["exp_avg_sq"].item() == pytest.approx(0.0025)


def test_ellis_adam_two_steps_match_reference() -> None:
    p = torch.nn.Parameter(torch.tensor([0.3]))
    opt = ELLISAdam([p], lr=0.05, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.1, update_clipping=True)
    ref_p, m, v = 0.3, 0.0, 0.0
    for step, g in enumerate([0.7, -0.2], start=1):
        p.grad = torch.tensor([g])
        opt.step()
        m = 0.9 * m + 0.1 * g
        v = 0.95 * v + 0.05 * g * g
        rms = math.sqrt(g * g / max(v, 1e-12))
        step_size = 0.05 / max(rms, 1.0) / (1 - 0.9**step)
        ref_p = ref_p * (1 - 0.1) - step_size * m / (math.sqrt(v) / math.sqrt(1 - 0.95**step) + 1e-6)
        assert p.item() == pytest.approx(ref_p, abs=1e-6)


def test_ellis_adam_final_config_one_step_hand_value() -> None:
    """
    All four switches of `config/crow_300m_final.yaml` at once (update_clipping, atan_adam, running_init,
    decouple_wd). Hand calculation for p=1, g=0.5, lr=0.1, betas=(0.9, 0.95), wd=0.01:
    running_init -> m = v = g, g^2 after the (idempotent) first moment update -> m=0.5, v=0.25;
    rms = sqrt(g^2 / v) = 1 -> no clipping; step_size = 0.1 / bc1(0.1) = 1.0;
    denom = sqrt(v) / sqrt(1 - 0.95) = 0.5 / sqrt(0.05); decay 1 - 0.01; update = atan2(0.5, denom).
    """

    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([0.5])
    opt = ELLISAdam(
        [p],
        lr=0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.01,
        update_clipping=True,
        atan_adam=True,
        running_init=True,
        decouple_wd=True,
    )
    opt.step()
    assert p.item() == pytest.approx(0.99 - math.atan2(0.5, 0.5 / math.sqrt(0.05)), abs=1e-7)
    assert opt.state[p]["exp_avg"].item() == pytest.approx(0.5)
    assert opt.state[p]["exp_avg_sq"].item() == pytest.approx(0.25)
    # second step with a sign flip, against the python reference (running-init moments carried over)
    p.grad = torch.tensor([-0.3])
    set_lr(opt, 0.05)  # scheduled LR is half the constructor LR: decoupled decay is halved too
    opt.step()
    m = 0.9 * 0.5 + 0.1 * -0.3
    v = 0.95 * 0.25 + 0.05 * 0.09
    rms = math.sqrt(0.09 / v)
    step_size = 0.05 / max(rms, 1.0) / (1 - 0.9**2)
    prev = 0.99 - math.atan2(0.5, 0.5 / math.sqrt(0.05))
    expected = prev * (1 - 0.05 / 0.1 * 0.01) - step_size * math.atan2(m, math.sqrt(v) / math.sqrt(1 - 0.95**2))
    assert p.item() == pytest.approx(expected, abs=1e-6)  # two fp32 steps


def test_update_clipping_eps_floor_is_applied_in_place() -> None:
    """
    When `exp_avg_sq < eps^2` the RMS uses the floor `eps^2` -- and, as upstream, the `clamp_` is in place, so
    the stored second moment is raised to `eps^2` as a side effect. Pinned so a rewrite notices the change.
    """

    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([1e-9])
    opt = ELLISAdam([p], lr=0.1, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0, update_clipping=True)
    opt.step()
    assert opt.state[p]["exp_avg_sq"].item() == pytest.approx(1e-12)  # clamped in place, was 1e-20
    # rms = sqrt(1e-18 / 1e-12) = 1e-3 < 1 -> no clipping; update = lr / bc1 * m / (sqrt(1e-12) / 0.1 + eps)
    expected = 1.0 - 0.1 / 0.1 * 1e-10 / (1e-6 / 0.1 + 1e-6)
    assert p.item() == pytest.approx(expected, abs=1e-7)  # fp32 parameter: ulp ~6e-8


def test_decoupled_weight_decay_scales_with_scheduled_lr() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.zeros(1)
    opt = ELLISAdam([p], lr=0.1, weight_decay=0.5, eps=1e-6)
    set_lr(opt, 0.05)  # half the constructor LR -> half the decay
    opt.step()
    assert p.item() == pytest.approx(1.0 - 0.05 / 0.1 * 0.5)


def test_coupled_weight_decay_multiplies_by_lr() -> None:
    """
    `decouple_wd=False`: decay per step is `lr * weight_decay` (called directly on the kernel).
    """

    p = torch.tensor([1.0])
    _single_tensor_modded_adamw(
        [p],
        [torch.zeros(1)],
        [torch.zeros(1)],
        [torch.zeros(1)],
        [torch.tensor(0)],
        beta1=0.9,
        beta2=0.99,
        lr=0.1,
        init_lr=0.1,
        weight_decay=0.5,
        eps=1e-6,
        decouple_wd=False,
    )
    assert p.item() == pytest.approx(1.0 - 0.1 * 0.5)


def test_running_init_seeds_moments_with_first_gradient() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([0.5])
    opt = ELLISAdam([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0, running_init=True)
    opt.step()
    state = opt.state[p]
    assert state["exp_avg"].item() == pytest.approx(0.5)  # lerp(0.5 -> 0.5) stays 0.5
    assert state["exp_avg_sq"].item() == pytest.approx(0.25)


def test_params_without_gradient_are_skipped() -> None:
    p, q = torch.nn.Parameter(torch.tensor([1.0])), torch.nn.Parameter(torch.tensor([2.0]))
    p.grad = torch.tensor([0.5])
    opt = ELLISAdam([p, q], lr=0.1, weight_decay=0.0)
    opt.step()
    assert q.item() == 2.0 and q not in opt.state and p in opt.state


def test_step_with_closure_returns_its_loss() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = ELLISAdam([p], lr=0.1, weight_decay=0.0)

    def closure() -> float:
        opt.zero_grad()
        loss = (p * 3).sum()
        loss.backward()  # type: ignore[no-untyped-call]  # Tensor.backward is unannotated in torch
        return loss.item()

    assert opt.step(closure) == pytest.approx(3.0)
    assert p.grad is not None and p.grad.item() == pytest.approx(3.0)
    assert p.item() < 1.0
    assert opt.step() is None


def test_ellis_adam_state_dict_round_trip() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    p.grad = torch.tensor([0.5, 0.25])
    opt = ELLISAdam([p], lr=0.1)
    opt.step()
    q = torch.nn.Parameter(p.detach().clone())
    opt2 = ELLISAdam([q], lr=0.1)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))  # torch shallow-copies: avoid aliasing live state
    assert torch.is_tensor(opt2.state[q]["step"]) and opt2.state[q]["step"].item() == 1
    q.grad = p.grad = torch.tensor([-0.1, 0.3])
    opt.step()
    opt2.step()
    assert torch.equal(p, q)


def test_ellis_adam_on_tiny_model_reduces_loss(tiny_model: RecurrentGPT) -> None:
    """
    `weight_decay` is a per-step fraction under `decouple_wd` (the final run used 4e-5); with 0.1 the parameters
    shrink 10 % every step and the loss rises, so use the real setting here.
    """

    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 32))
    groups = get_param_groups(tiny_model, weight_decay=4e-5)
    opt = ELLISAdam(groups, lr=1e-3, betas=(0.9, 0.95), update_clipping=True)
    losses = []
    for _ in range(5):
        loss = tiny_model(x, labels=x, num_steps=(0, 2))["loss"]
        assert loss is not None
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_cpu_update_stays_eager_and_scalars_come_as_tensors() -> None:
    """
    On the CPU the group update runs eagerly (`torch.compile` is built for CUDA parameters only), and the scalar
    coefficients handed to it are 0-d float32 tensors, the form that keeps the compiled path free of recompiles.
    """

    from training import optim

    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    p.grad = torch.tensor([0.5, 0.25])
    opt = ELLISAdam([p], lr=0.1, update_clipping=True, atan_adam=True)
    seen: dict[str, Any] = {}
    original = optim._adamw_group_update

    def spy(*args: Any, **kwargs: Any) -> None:
        seen["step_sizes"], seen["bc1"], seen["bc2_sqrt"], seen["decays"] = args[4:8]
        original(*args, **kwargs)

    optim._adamw_group_update = spy
    try:
        opt.step()
    finally:
        optim._adamw_group_update = original
    assert optim._compiled_group_update is None
    for key in ("step_sizes", "bc1", "bc2_sqrt", "decays"):
        assert all(t.device.type == "cpu" and t.dtype == torch.float32 and t.ndim == 0 for t in seen[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled update is built for CUDA only")
def test_compiled_cuda_update_matches_eager_at_rounding_level() -> None:
    """
    The `torch.compile`d group update on CUDA reproduces the eager kernel to fp32 rounding (Inductor fuses and may
    reorder the elementwise chain), across LR changes without a recompile, for every option combination.
    """

    from training import optim

    for flags in (
        dict(update_clipping=True, atan_adam=True, running_init=True),
        dict(update_clipping=False, atan_adam=False, running_init=False, decouple_wd=False),
    ):
        results = []
        for compiled in (False, True):
            torch.manual_seed(0)
            p = torch.nn.Parameter(torch.randn(64, 32, device="cuda"))
            opt = ELLISAdam([p], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, **flags)
            optim._compiled_group_update = None
            original = optim._compiled_adamw_group_update
            if not compiled:
                optim._compiled_adamw_group_update = lambda: optim._adamw_group_update
            try:
                for step in range(4):
                    p.grad = torch.randn(64, 32, device="cuda") * 1e-2
                    set_lr(opt, 1e-3 * (step + 1))
                    opt.step()
            finally:
                optim._compiled_adamw_group_update = original
            results.append((p.detach().clone(), opt.state[p]["exp_avg"].clone(), opt.state[p]["exp_avg_sq"].clone()))
        for eager, fused in zip(*results, strict=True):
            torch.testing.assert_close(fused, eager, rtol=1e-6, atol=1e-7)  # a few fp32 ulps
