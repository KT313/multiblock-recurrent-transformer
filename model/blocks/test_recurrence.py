# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `model.blocks.recurrence`: step canonicalisation and broadcasting, the latent state init, the
poisson-lognormal-filling depth sampler and the (no-grad, backprop) iteration loop.
"""

import copy
from typing import Any, cast

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
    recurrence_seed,
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


def test_sampler_microbatch_schedule_replays_across_rank_rngs() -> None:
    schedules: list[list[tuple[int, int]]] = []
    latents: list[Tensor] = []
    for rank in range(8):
        torch.manual_seed(42 + rank)
        latents.append(initialize_state(torch.zeros(2, 4, 8)))
        schedule = []
        for step in (0, 1):
            for micro in range(8):
                for block in range(3):
                    n, k = sample_recurrence_steps(
                        12, 8, step=step, micro_batch_index=micro, block_idx=block, training=True,
                    )
                    schedule.append((int(n.item()), int(k.item())))
        schedules.append(schedule)
    assert all(schedule == schedules[0] for schedule in schedules)
    assert all(not torch.equal(latent, latents[0]) for latent in latents[1:])
    # Fresh draws can coincide; compare schedules, never require every pair of counts to differ.
    assert len(set(schedules[0][:24])) > 1
    assert schedules[0][:24] != schedules[0][24:]
    assert schedules[0][0::3] != schedules[0][1::3]


def test_recurrence_seed_mixes_each_coordinate_and_checks_negative_inputs() -> None:
    keys = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2**24, 0, 0), (0, 2**32, 0)]
    seeds = [recurrence_seed(step=s, micro_batch_index=m, block_idx=b) for s, m, b in keys]
    assert seeds == [2530474308, 1331452466, 2370498770, 574480444, 1441205481, 1494029112]
    assert len(set(seeds)) == len(keys)
    assert all(0 <= seed < 2**32 for seed in seeds)
    for step, micro, block in [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]:
        with pytest.raises(ValueError, match="must be nonnegative"):
            recurrence_seed(step=step, micro_batch_index=micro, block_idx=block)


@pytest.mark.parametrize("training", [False, True])
def test_microbatch_context_preserves_sampler_global_rng_consumption(training: bool) -> None:
    torch.manual_seed(72)
    sample_recurrence_steps(12, 8, step=13, micro_batch_index=7, block_idx=2, training=training)
    actual = torch.get_rng_state()
    torch.manual_seed(72)
    torch.rand((1,))
    assert torch.equal(actual, torch.get_rng_state())


def test_microbatch_context_does_not_change_eval_or_explicit_depths(tiny_model: RecurrentGPT) -> None:
    x = torch.tensor([[1, 2, 3, 4]])
    for training, explicit in [(False, None), (True, (1, 1))]:
        tiny_model.train(training)
        tiny_model.step, tiny_model.micro_batch_index = 0, 0
        torch.manual_seed(19)
        first = tiny_model(x, num_steps=explicit, return_logits=True)["logits"]
        rng = torch.get_rng_state()
        tiny_model.step, tiny_model.micro_batch_index = 42, 7
        torch.manual_seed(19)
        second = tiny_model(x, num_steps=explicit, return_logits=True)["logits"]
        assert first is not None and second is not None
        assert torch.equal(first, second) and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_sampled_checkpoint_backward_does_not_reread_microbatch_context(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    tiny_model.step, tiny_model.micro_batch_index = 9, 3
    reference = copy.deepcopy(tiny_model)
    tiny_model.gradient_checkpointing = recurrence.check_checkpoint_mode(mode)
    calls: list[tuple[int, int, int]] = []
    original = tiny_model.sample_block_depths

    def record(block_idx: int = 0) -> tuple[Tensor, Tensor]:
        calls.append((tiny_model.step, tiny_model.micro_batch_index, block_idx))
        steps: tuple[Tensor, Tensor] = original(block_idx)
        return steps

    monkeypatch.setattr(tiny_model, "sample_block_depths", record)
    x = torch.tensor([[1, 2, 3, 4]])
    for model in (reference, tiny_model):
        torch.manual_seed(83)
        loss = model(x, labels=x)["loss"]
        assert loss is not None
        # The checkpointed iterations must only use their captured tensor inputs.
        model.step, model.micro_batch_index = 123, 7
        loss.backward()
    assert calls == [(9, 3, 0), (9, 3, 1)]
    for expected, actual in zip(reference.parameters(), tiny_model.parameters()):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


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
        x_latent, x_base, freqs, None, 2, 1, adapter=adapter, layers=layers, gradient_checkpointing="none"
    )
    expected = x_latent
    for _ in range(3):
        expected = core_block_forward(expected, x_base, freqs, None, adapter, layers)
    torch.testing.assert_close(got, expected)
    assert got.requires_grad
    # tensor step counts (the sampler's output) work like ints
    two, one = torch.tensor(2), torch.tensor(1)
    same = iterate_core_block(
        x_latent, x_base, freqs, None, two, one, adapter=adapter, layers=layers, gradient_checkpointing="none"
    )
    torch.testing.assert_close(same, expected)
    no_grad = iterate_core_block(
        x_latent, x_base, freqs, None, 2, 0, adapter=adapter, layers=layers, gradient_checkpointing="none"
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
        x, x, tiny_model.freqs_cis[:, :4], None, n, k, adapter=adapter, layers=layers, gradient_checkpointing="none"
    )
    assert grad_modes == [False] * n + [True] * k


CHECKPOINT_WRAPPERS = {"selective": "_selective_checkpoint", "full": "_full_checkpoint"}


def count_checkpoint_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """
    Swap both module-level checkpoint wrappers for counting ones; the returned dict holds the calls per mode.
    """

    calls = dict.fromkeys(CHECKPOINT_WRAPPERS, 0)

    def counting(mode: str, orig: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Tensor:
            calls[mode] += 1
            return orig(*args, **kwargs)  # type: ignore[no-any-return]  # functools.partial of an untyped checkpoint

        return wrapper

    for mode, name in CHECKPOINT_WRAPPERS.items():
        monkeypatch.setattr(recurrence, name, counting(mode, getattr(recurrence, name)))
    return calls


@pytest.mark.parametrize("mode", ["selective", "full"])
def test_gradient_checkpointing_wraps_each_backprop_iteration(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch, mode: recurrence.CheckpointMode
) -> None:
    calls = count_checkpoint_calls(monkeypatch)
    adapter, layers = block_parts(tiny_model, 0)
    x = torch.randn(1, 4, 64, requires_grad=True)
    freqs = tiny_model.freqs_cis[:, :4]
    plain = iterate_core_block(x, x, freqs, None, 1, 3, adapter=adapter, layers=layers, gradient_checkpointing="none")
    assert calls == {"selective": 0, "full": 0}
    ckpt = iterate_core_block(x, x, freqs, None, 1, 3, adapter=adapter, layers=layers, gradient_checkpointing=mode)
    assert calls == {mode: 3, **{other: 0 for other in CHECKPOINT_WRAPPERS if other != mode}}
    torch.testing.assert_close(plain, ckpt)
    # the recompute in the backward gives the same gradient
    (plain_grad,) = torch.autograd.grad(plain.sum(), x)
    (ckpt_grad,) = torch.autograd.grad(ckpt.sum(), x)
    torch.testing.assert_close(ckpt_grad, plain_grad)


def test_gradient_checkpointing_mode_is_checked(tiny_model: RecurrentGPT) -> None:
    adapter, layers = block_parts(tiny_model, 0)
    x = torch.randn(1, 4, 64)
    with pytest.raises(ValueError, match="gradient_checkpointing must be one of none, selective, full, not True"):
        iterate_core_block(
            x, x, tiny_model.freqs_cis[:, :4], None, 1, 1, adapter=adapter, layers=layers,
            gradient_checkpointing=cast(recurrence.CheckpointMode, True),  # deliberately invalid at runtime
        )


@pytest.mark.parametrize("values", [[], [1, 2, 3], [True], [1.5], [float("nan")], [float("inf")], [1j], [1, False]])
@pytest.mark.parametrize("form", ["list", "tuple", "tensor"])
def test_canon_steps_rejects_malformed_depths(values: list[Any], form: str) -> None:
    if form == "tensor":
        steps: object = torch.tensor(values, dtype=torch.bool if any(isinstance(v, bool) for v in values) else None)
    else:
        steps = tuple(values) if form == "tuple" else values
    with pytest.raises(ValueError, match="num_steps"):
        canon_steps(steps)


@pytest.mark.parametrize("steps", [True, False, 1.5, float("nan"), float("inf"), 1j, "2", None])
def test_canon_steps_rejects_malformed_scalars(steps: object) -> None:
    with pytest.raises(ValueError, match="num_steps"):
        canon_steps(steps)


def test_integral_real_depths_and_list_disambiguation() -> None:
    assert canon_steps(3.0) == (3, 0)
    assert canon_steps([3.0]) == (3, 0)
    assert canon_steps((0.0, 2.0)) == (0, 2)
    assert canon_steps(torch.tensor([[[3.0, 2.0]]])) == (3, 2)
    assert normalize_num_steps([3, 2], 2) == [(3, 0), (2, 0)]
    assert normalize_num_steps((3, 2), 2) == [(3, 2), (3, 2)]


@pytest.mark.parametrize("steps", [(1, 2, 3), torch.tensor([1.5]), [(1, 0), (True, 0)]])
def test_invalid_explicit_depth_precedes_recurrent_execution(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch, steps: Any,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid explicit depth reached recurrence")
    monkeypatch.setattr(tiny_model, "run_core_blocks", forbidden)
    with pytest.raises(ValueError, match="num_steps"):
        tiny_model(torch.zeros((1, 2), dtype=torch.long), num_steps=steps)
