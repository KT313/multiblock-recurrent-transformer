# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Inference semantics: real nonzero fixed latents, distinct recurrence K/V and an unchanged legacy path."""

from typing import Any

import pytest
import torch
from torch import Tensor

from model.generation import GenerationState
from model.model import RecurrentGPT
from model.test_config import tiny_config


def generation_model() -> RecurrentGPT:
    torch.manual_seed(7)
    return RecurrentGPT(tiny_config(
        n_embd=32, intermediate_size=64, n_layers_in_prelude=1, n_layers_in_coda=1,
        n_layers_in_recurrent_block=[2, 1], mean_recurrence=[2, 3], mean_backprop_depth=[1, 1],
        model_max_sequence_length=32,
    )).eval()


def batch() -> tuple[Tensor, Tensor]:
    return torch.tensor([[0, 0, 5, 7, 9, 11, 13], [2, 3, 5, 7, 9, 11, 13]]), torch.tensor([
        [0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
    ])


@pytest.mark.parametrize("chunks", [(4, 1, 1, 1), (2, 3, 2), (7,)])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_cached_prefix_matches_nonzero_fixed_latent_reference(
    chunks: tuple[int, ...], explicit_positions: bool,
) -> None:
    model = generation_model()
    ids, mask = batch()
    cached, reference = GenerationState(seed=321), GenerationState(seed=321)
    positions = (mask.cumsum(-1) - 1).clamp(min=0) + 3 if explicit_positions else None
    end = 0
    snapshots: dict[int, Tensor] = {}
    with torch.inference_mode():
        for length in chunks:
            start, end = end, end + length
            kwargs: dict[str, Any] = {} if positions is None else {"position_ids": positions[:, start:end]}
            actual = model(
                ids[:, start:end], attention_mask=mask[:, :end], generation_state=cached,
                use_cache=True, return_logits=True, **kwargs,
            )["logits"]
            kwargs = {} if positions is None else {"position_ids": positions[:, :end]}
            expected = model(
                ids[:, :end], attention_mask=mask[:, :end], generation_state=reference,
                return_logits=True, **kwargs,
            )["logits"]
            assert actual is not None and expected is not None
            torch.testing.assert_close(actual, expected[:, start:end], atol=3e-5, rtol=2e-5)
            assert len(cached.slots) == 1 + 2 * 2 + 3 * 1 + 1
            assert not reference.slots
            for core, latent in cached.latents.items():
                assert torch.count_nonzero(latent) == latent.numel()
                assert torch.equal(latent, reference.latents[core])
                if core in snapshots:
                    assert torch.equal(latent[:, :start], snapshots[core])
                snapshots[core] = latent.clone()
            assert not torch.equal(cached.latents[0], cached.latents[1])
            first = cached.slots[("core", 0, 0, 0)].key
            second = cached.slots[("core", 0, 1, 0)].key
            assert first is not None and second is not None
            assert first.data_ptr() != second.data_ptr() and not torch.equal(first, second)
            assert all(slot.key is not None and slot.key.shape[1] == end for slot in cached.slots.values())


def test_fixed_noise_is_independent_of_prefill_chunking() -> None:
    model = generation_model()
    ids, mask = batch()
    whole, pieces = GenerationState(seed=9), GenerationState(seed=9)
    with torch.inference_mode():
        expected = model(ids, attention_mask=mask, generation_state=whole, use_cache=True, return_logits=True)["logits"]
        outputs = [
            model(ids[:, i:i + 1], attention_mask=mask[:, :i + 1], generation_state=pieces, use_cache=True, return_logits=True)["logits"]
            for i in range(ids.shape[1])
        ]
    assert all(output is not None for output in outputs)
    actual = torch.cat([output for output in outputs if output is not None], dim=1)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-5)
    assert all(torch.equal(whole.latents[i], pieces.latents[i]) for i in whole.latents)


@pytest.mark.parametrize("mutation", ["weights", "step", "dtype", "schedule", "mask", "batch", "positions", "mode", "cache_mode"])
def test_invalid_sessions_are_rejected_and_released(mutation: str) -> None:
    model = generation_model()
    ids, mask = batch()
    state = GenerationState(seed=4)
    with torch.no_grad():
        model(ids[:, :4], attention_mask=mask[:, :4], generation_state=state, use_cache=True)
        kwargs: dict[str, Any] = {"attention_mask": mask[:, :5], "generation_state": state, "use_cache": True}
        new_ids = ids[:, 4:5]
        if mutation == "weights":
            model.lm_head.weight.add_(0.01)
        elif mutation == "step":
            model.step += 1
        elif mutation == "dtype":
            model.double()
        elif mutation == "schedule":
            kwargs["num_steps"] = [(1, 0), (3, 0)]
        elif mutation == "mask":
            kwargs["attention_mask"] = torch.ones_like(mask[:, :5])
        elif mutation == "batch":
            new_ids = new_ids[:1]
            kwargs["attention_mask"] = mask[:1, :5]
        elif mutation == "positions":
            kwargs["position_ids"] = torch.full_like(new_ids, 32)
        elif mutation == "mode":
            model.train()
        else:
            kwargs["use_cache"] = False
        with pytest.raises(ValueError):
            model(new_ids, **kwargs)
        assert not state.slots and not state.latents
        with pytest.raises(ValueError, match="invalid"):
            model(new_ids, **kwargs)


def test_fixed_reference_rejects_prefix_edits_and_position_edits() -> None:
    model = generation_model()
    ids, mask = batch()
    for change_position in (False, True):
        state = GenerationState(seed=4)
        with torch.no_grad():
            model(ids[:, :4], attention_mask=mask[:, :4], generation_state=state)
            changed = ids[:, :5].clone()
            kwargs: dict[str, Any] = {}
            if change_position:
                kwargs["position_ids"] = (mask[:, :5].cumsum(-1) - 1).clamp(min=0) + 1
            else:
                changed[0, 2] += 1
            with pytest.raises(ValueError, match="prefix|positions"):
                model(changed, attention_mask=mask[:, :5], generation_state=state, **kwargs)


def test_head_projection_preserves_rng_and_full_default_scoring() -> None:
    model = generation_model()
    ids, mask = batch()
    with torch.no_grad():
        torch.manual_seed(8)
        full = model(ids, attention_mask=mask, return_logits=True)["logits"]
        after_full = torch.get_rng_state()
        torch.manual_seed(8)
        tail = model(ids, attention_mask=mask, return_logits=True, logits_to_keep=1)["logits"]
        assert torch.equal(after_full, torch.get_rng_state())
    assert full is not None and tail is not None
    assert full.shape == (2, 7, 512) and tail.shape == (2, 1, 512)
    torch.testing.assert_close(tail, full[:, -1:], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="without labels"):
        model(ids, labels=ids, logits_to_keep=1)
    with pytest.raises(ValueError, match="nonnegative"):
        model(ids, logits_to_keep=-1)
    assert model(ids, labels=ids)["loss"] is not None
    assert not any("cache" in key or "latent" in key for key in model.state_dict())


def test_fixed_sessions_do_not_advance_global_rng_during_forward() -> None:
    model = generation_model()
    ids, mask = batch()
    state = GenerationState(seed=5)
    before = torch.get_rng_state()
    with torch.no_grad():
        model(ids[:, :4], attention_mask=mask[:, :4], generation_state=state, use_cache=True)
        model(ids[:, 4:], attention_mask=mask, generation_state=state, use_cache=True)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.gpu
def test_cuda_cached_fixed_latents_match_reference() -> None:
    model = generation_model().to("cuda")
    ids, mask = (value.to("cuda") for value in batch())
    cached, reference = GenerationState(seed=321), GenerationState(seed=321)
    with torch.inference_mode():
        for end in range(4, 8):
            start = 0 if end == 4 else end - 1
            actual = model(
                ids[:, start:end], attention_mask=mask[:, :end], generation_state=cached,
                use_cache=True, return_logits=True, logits_to_keep=1,
            )["logits"]
            expected = model(
                ids[:, :end], attention_mask=mask[:, :end], generation_state=reference,
                return_logits=True, logits_to_keep=1,
            )["logits"]
            assert actual is not None and expected is not None
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            assert all(torch.equal(cached.latents[i], reference.latents[i]) for i in cached.latents)


@pytest.mark.gpu
def test_cuda_generation_isolation_and_live_weight_integrity() -> None:
    from evaluation.wrapper import isolated_inference

    model = generation_model().to("cuda").train()
    ids, mask = (value.to("cuda") for value in batch())
    model.step = 89
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    weights = [(p, p.data_ptr(), p.clone(), p.grad.clone()) for p in model.parameters() if p.grad is not None]
    for fail in (False, True):
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        try:
            with isolated_inference(model):
                state = GenerationState()
                model(ids, attention_mask=mask, generation_state=state, use_cache=True, return_logits=True)
                if fail:
                    raise RuntimeError("injected GPU generation failure")
        except RuntimeError as error:
            assert fail and str(error) == "injected GPU generation failure"
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert model.training and model.step == 89
        for parameter, pointer, value, gradient in weights:
            assert parameter.data_ptr() == pointer and torch.equal(parameter, value)
            assert parameter.grad is not None and torch.equal(parameter.grad, gradient)


def test_cached_generation_matches_original_native_recurrence_with_supplied_latents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent oracle: original forward/run_core_blocks, with only its random draws replaced.

    This does not run the new uncached _forward_generation reference. Nonzero stored per-core noise enters the
    original native pipeline, so an error shared by the new cached/reference branches cannot hide behind parity.
    """
    import model.model as native_module

    model = generation_model()
    ids, mask = batch()
    state = GenerationState(seed=719)
    with torch.no_grad():
        for end in range(4, 8):
            start = 0 if end == 4 else end - 1
            cached = model(
                ids[:, start:end], attention_mask=mask[:, :end], generation_state=state,
                position_ids=(mask[:, :end].cumsum(-1) - 1).clamp(min=0)[:, start:end],
                use_cache=True, return_logits=True, num_steps=[(2, 0), (3, 0)],
            )["logits"]
            core = 0

            def stored_initial_state(x: Tensor) -> Tensor:
                nonlocal core
                latent = state.latents[core]
                assert latent.shape == x.shape and latent.dtype == x.dtype
                assert torch.count_nonzero(latent) == latent.numel()
                core += 1
                return latent.clone()

            with monkeypatch.context() as patch:
                patch.setattr(native_module, "initialize_state", stored_initial_state)
                original = model(
                    ids[:, :end], attention_mask=mask[:, :end],
                    position_ids=(mask[:, :end].cumsum(-1) - 1).clamp(min=0),
                    num_steps=[(2, 0), (3, 0)], return_logits=True,
                )["logits"]
            assert core == 2
            assert cached is not None and original is not None
            torch.testing.assert_close(cached, original[:, start:end], atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize("ownership", ["parameter", "buffer", "all"])
def test_generation_rejects_unversioned_inference_owned_model_tensors(ownership: str) -> None:
    if ownership == "all":
        with torch.inference_mode():
            model = generation_model()
    else:
        model = generation_model()
        with torch.inference_mode():
            if ownership == "parameter":
                model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
            else:
                model.register_buffer("freqs_cis", model.freqs_cis.clone(), persistent=False)
    ids, mask = batch()
    state = GenerationState(seed=1)
    with torch.no_grad():
        # Legacy forward can still use these tensors; the cache must fail clearly rather than drop version guards.
        legacy = model(ids, attention_mask=mask, return_logits=True, num_steps=[(2, 0), (3, 0)])["logits"]
        assert legacy is not None and torch.isfinite(legacy).all()
        with pytest.raises(ValueError, match="outside torch.inference_mode.*torch.no_grad"):
            model(ids, attention_mask=mask, generation_state=state, use_cache=True)
    assert not state.slots and not state.latents


def test_generation_accepts_no_grad_model_construction() -> None:
    with torch.no_grad():
        model = generation_model()
        ids, mask = batch()
        output = model(ids, attention_mask=mask, generation_state=GenerationState(seed=1), use_cache=True, return_logits=True)
    assert output["logits"] is not None
