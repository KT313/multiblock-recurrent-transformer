# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `model.blocks.recurrence`: step canonicalisation and broadcasting, the latent state init, the
poisson-lognormal-filling depth sampler and the (no-grad, backprop) iteration loop.
"""

from typing import Any

import pytest
import torch
from torch import Tensor

from model.blocks import recurrence
from model.blocks.recurrence import (
    canon_steps,
    core_block_forward,
    initialize_state,
    iterate_core_block,
    normalize_num_steps,
    sample_recurrence_steps,
)
from model.model import RecurrentGPT


def block_parts(model: RecurrentGPT, idx: int) -> tuple[torch.nn.Module, torch.nn.ModuleList]:
    layers = model.transformer.core_blocks[idx]
    assert isinstance(layers, torch.nn.ModuleList)
    return model.transformer.adapters[idx], layers


# --- steps -----------------------------------------------------------------------------------------------------------


def test_canon_steps_all_input_forms() -> None:
    assert canon_steps((3, 2)) == (3, 2)
    assert canon_steps(4) == (4, 0)
    assert canon_steps(torch.tensor(4)) == (4, 0)
    assert canon_steps(torch.tensor([4])) == (4, 0)
    assert canon_steps(torch.tensor([[3], [2]])) == (3, 2)
    assert canon_steps((0, 1)) == (0, 1)
    for depthless in (0, (0, 0), torch.tensor([0]), (-1, 2)):  # zero steps would hand the random initial state on as the output
        with pytest.raises(ValueError, match="at least one recurrent step"):
            canon_steps(depthless)
    n, k = canon_steps(torch.tensor([3, 2], dtype=torch.long))
    assert isinstance(n, int) and isinstance(k, int)


def test_normalize_num_steps_broadcasts_and_validates() -> None:
    assert normalize_num_steps(None, 3) == [None, None, None]
    assert normalize_num_steps((1, 2), 3) == [(1, 2)] * 3
    assert normalize_num_steps(3, 2) == [(3, 0), (3, 0)]
    assert normalize_num_steps(torch.tensor([1, 2]), 2) == [(1, 2), (1, 2)]
    assert normalize_num_steps([(1, 2), torch.tensor([4, 1]), 5], 3) == [(1, 2), (4, 1), (5, 0)]
    with pytest.raises(ValueError, match="num_steps has 3 entries but there are 2 blocks"):
        normalize_num_steps([(1, 1), (1, 1), (1, 1)], 2)


# --- latent state ----------------------------------------------------------------------------------------------------


def test_initialize_state_is_a_seeded_standard_normal() -> None:
    x = torch.zeros(4, 64, 64)
    torch.manual_seed(0)
    a = initialize_state(x)
    torch.manual_seed(0)
    b = initialize_state(x)
    assert a.shape == x.shape and a.dtype == x.dtype
    assert torch.equal(a, b)
    assert abs(a.mean().item()) < 0.05 and a.std().item() == pytest.approx(1.0, abs=0.05)


# --- sampler ---------------------------------------------------------------------------------------------------------


def test_sampler_eval_returns_mean_recurrence_and_zero_grad_steps() -> None:
    for step in (0, 17):
        n, k = sample_recurrence_steps(12, 8, step=step, block_idx=0, training=False)
        assert n.dtype == torch.long and k.dtype == torch.long
        assert (n.item(), k.item()) == (12, 0)


def test_sampler_respects_backprop_bound_and_is_positive() -> None:
    for mean, bound in ((12, 8), (6, 3)):
        ks, ns = set(), set()
        for step in range(300):
            n, k = sample_recurrence_steps(mean, bound, step=step, block_idx=0, training=True)
            assert 1 <= k.item() <= bound
            assert n.item() >= 0
            assert n.item() + k.item() >= 1
            ks.add(k.item())
            ns.add(n.item())
        assert len(ks) > 1 and len(ns) > 1  # actually random
        assert bound in ks  # the bound is attained when p >= s


def test_sampler_total_depth_mean_is_mean_recurrence() -> None:
    """
    n + k == p == Poisson(LogNormal(log(mean - 1) - sigma^2/2, sigma)) + 1, whose mean is `mean_recurrence` (the
    same depth eval mode and the init scaling use) and whose minimum is the one guaranteed pass. Also:
    k == min(s, p), n == p - k.
    """

    for mean, s in ((12, 8), (4, 3)):
        totals = []
        for step in range(2000):
            n, k = sample_recurrence_steps(mean, s, step=step, block_idx=0, training=True)
            p = n.item() + k.item()
            assert p >= 1
            assert k.item() == min(s, p) and n.item() == p - k.item()
            totals.append(p)
        avg = sum(totals) / len(totals)
        assert avg == pytest.approx(mean, abs=0.5), avg


def test_sampler_mean_recurrence_one_always_draws_a_single_pass() -> None:
    for step in range(50):
        n, k = sample_recurrence_steps(1, 1, step=step, block_idx=0, training=True)
        assert (n.item(), k.item()) == (0, 1)


def test_sampler_deterministic_in_step_and_block_independent_of_global_rng() -> None:
    for block_idx in (0, 1):
        torch.manual_seed(0)
        a = sample_recurrence_steps(2, 2, step=42, block_idx=block_idx, training=True)
        torch.manual_seed(999)
        b = sample_recurrence_steps(2, 2, step=42, block_idx=block_idx, training=True)
        assert (a[0].item(), a[1].item()) == (b[0].item(), b[1].item())
    draws = {tuple(v.item() for v in sample_recurrence_steps(2, 2, step=step, block_idx=0, training=True)) for step in range(50)}
    assert len(draws) > 1


def test_sampler_blocks_with_equal_means_draw_independently() -> None:
    def sequence(block_idx: int) -> list[tuple[int, int]]:
        draws = []
        for step in range(200):
            n, k = sample_recurrence_steps(12, 8, step=step, block_idx=block_idx, training=True)
            draws.append((int(n.item()), int(k.item())))
        return draws

    assert sequence(0) != sequence(1)


def test_sampler_advances_global_rng() -> None:
    """
    The (meta-check) `torch.rand` draw advances the global RNG; pinned for bit-identity with the thesis code.
    """

    for training in (True, False):
        torch.manual_seed(0)
        sample_recurrence_steps(2, 2, step=0, block_idx=0, training=training)
        after = torch.rand(())
        torch.manual_seed(0)
        torch.rand((1,))
        assert torch.equal(after, torch.rand(()))


def test_sampler_on_meta_device_returns_the_expected_depths() -> None:
    with torch.device("meta"):
        n, k = sample_recurrence_steps(12, 8, step=3, block_idx=0, training=True)
    assert (n, k) == (4, 8)


# --- iteration -------------------------------------------------------------------------------------------------------


def test_core_block_forward_matches_hand_composition(tiny_model: RecurrentGPT) -> None:
    freqs = tiny_model.freqs_cis[:, :6]
    x_latent, x_base = torch.randn(1, 6, 64), torch.randn(1, 6, 64)
    for idx in range(2):
        adapter, layers = block_parts(tiny_model, idx)
        got = core_block_forward(x_latent, x_base, freqs, None, adapter, layers)
        expected = adapter(torch.cat([x_latent, x_base], dim=-1))
        for layer in layers:
            expected = layer(expected, freqs, None)
        torch.testing.assert_close(got, expected)


def test_iterate_core_block_matches_manual_loop(tiny_model: RecurrentGPT) -> None:
    freqs = tiny_model.freqs_cis[:, :6]
    x_latent, x_base = torch.randn(1, 6, 64), torch.randn(1, 6, 64)
    adapter, layers = block_parts(tiny_model, 1)
    got = iterate_core_block(
        x_latent, x_base, freqs, None, 2, 1, adapter=adapter, layers=layers, gradient_checkpointing=False
    )
    expected = x_latent
    for _ in range(3):
        expected = core_block_forward(expected, x_base, freqs, None, adapter, layers)
    torch.testing.assert_close(got, expected)
    assert got.requires_grad
    # tensor step counts (the sampler's output) work like ints
    two, one = torch.tensor(2), torch.tensor(1)
    same = iterate_core_block(
        x_latent, x_base, freqs, None, two, one, adapter=adapter, layers=layers, gradient_checkpointing=False
    )
    torch.testing.assert_close(same, expected)
    no_grad = iterate_core_block(
        x_latent, x_base, freqs, None, 2, 0, adapter=adapter, layers=layers, gradient_checkpointing=False
    )
    assert not no_grad.requires_grad  # all steps under no_grad


@pytest.mark.parametrize(("n", "k"), [(2, 1), (1, 2), (0, 3), (3, 0)])
def test_first_n_iterations_run_without_grad_and_last_k_with_grad(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch, n: int, k: int
) -> None:
    """
    (n, k) is not symmetric: exactly the first n core-block applications happen under no_grad.
    """

    grad_modes: list[bool] = []
    orig = recurrence.core_block_forward

    def spy(*args: Any, **kwargs: Any) -> Tensor:
        grad_modes.append(torch.is_grad_enabled())
        return orig(*args, **kwargs)

    monkeypatch.setattr(recurrence, "core_block_forward", spy)
    adapter, layers = block_parts(tiny_model, 0)
    x = torch.randn(1, 4, 64)
    iterate_core_block(
        x, x, tiny_model.freqs_cis[:, :4], None, n, k, adapter=adapter, layers=layers, gradient_checkpointing=False
    )
    assert grad_modes == [False] * n + [True] * k


def test_gradient_checkpointing_wraps_each_backprop_iteration(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    orig_checkpoint = recurrence._checkpoint

    def counting_checkpoint(*args: Any, **kwargs: Any) -> Tensor:
        calls.append(1)
        return orig_checkpoint(*args, **kwargs)  # type: ignore[no-any-return]  # functools.partial of an untyped checkpoint

    monkeypatch.setattr(recurrence, "_checkpoint", counting_checkpoint)
    adapter, layers = block_parts(tiny_model, 0)
    x = torch.randn(1, 4, 64)
    freqs = tiny_model.freqs_cis[:, :4]
    plain = iterate_core_block(x, x, freqs, None, 1, 3, adapter=adapter, layers=layers, gradient_checkpointing=False)
    assert calls == []
    ckpt = iterate_core_block(x, x, freqs, None, 1, 3, adapter=adapter, layers=layers, gradient_checkpointing=True)
    assert len(calls) == 3
    torch.testing.assert_close(plain, ckpt)
