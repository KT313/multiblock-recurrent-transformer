# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The HF generation loop really decodes only new queries and exposes an explicit legacy path."""

from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import LogitsProcessor, LogitsProcessorList

from model.generation import GenerationState
from model.hf.modeling import RecurrentGPTConfig, RecurrentGPTForCausalLM, mask_padded_vocabulary
from model.test_generation import batch, generation_model


def wrapper() -> RecurrentGPTForCausalLM:
    result = RecurrentGPTForCausalLM(RecurrentGPTConfig.from_recurrent_config(generation_model().config))
    result.train(False)
    result.generation_config.pad_token_id = 0
    result.generation_config.eos_token_id = 2
    return result


def test_hf_cache_matches_fixed_reference_and_preserves_full_scoring() -> None:
    model = wrapper()
    ids, mask = batch()
    with torch.no_grad():
        first = model(ids[:, :4], attention_mask=mask[:, :4], use_cache=True, logits_to_keep=1)
        state = first.past_key_values
        assert isinstance(state, GenerationState)
        reference = GenerationState(seed=state.seed)
        expected = model.model(
            ids[:, :4], attention_mask=mask[:, :4], generation_state=reference, logits_to_keep=1, return_logits=True,
        )["logits"]
        assert expected is not None
        torch.testing.assert_close(first.logits, expected)
        for end in range(5, 8):
            prepared = model.prepare_inputs_for_generation(
                ids[:, :end], attention_mask=mask[:, :end], past_key_values=state,
                use_cache=True, logits_to_keep=1,
            )
            assert prepared["input_ids"].shape == (2, 1)
            got = model(**prepared)
            assert got.past_key_values is state
            expected = model.model(
                ids[:, :end], attention_mask=mask[:, :end], generation_state=reference, logits_to_keep=1, return_logits=True,
            )["logits"]
            torch.testing.assert_close(got.logits, expected, atol=3e-5, rtol=2e-5)
        scored = model(ids, attention_mask=mask, labels=ids)
        assert scored.logits.shape == (2, 7, 512) and scored.loss is not None
        assert scored.past_key_values is None


@pytest.mark.parametrize("use_cache", [False, True])
def test_generate_query_width_and_fresh_session(use_cache: bool) -> None:
    model = wrapper()
    ids, mask = batch()
    widths: list[int] = []
    states: list[GenerationState] = []

    def record(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        widths.append(kwargs["input_ids"].shape[1])
        if kwargs.get("generation_state") is not None:
            states.append(kwargs["generation_state"])

    handle = model.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        kwargs = dict(attention_mask=mask[:, :4], max_new_tokens=3, min_new_tokens=3, do_sample=False, use_cache=use_cache)
        torch.manual_seed(42)
        first = model.generate(ids[:, :4], **kwargs)
        torch.manual_seed(42)
        second = model.generate(ids[:, :4], **kwargs)
    finally:
        handle.remove()
    assert torch.equal(first, second)
    assert widths == ([4, 1, 1] * 2 if use_cache else [4, 5, 6] * 2)
    if use_cache:
        assert states[0] is states[1] is states[2]
        assert states[3] is states[4] is states[5]
        assert states[0] is not states[3]
    else:
        assert not states


class ForceMixedEOS(LogitsProcessor):
    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]  # HF uses FloatTensor in its stub
        scores = torch.full_like(scores, -torch.inf)
        scores[0, 2] = 0.0
        scores[1, 2 if input_ids.shape[1] >= 6 else 7] = 0.0
        return scores


def test_cached_generation_finished_rows_are_padded_without_cache_reordering() -> None:
    model = wrapper()
    ids, mask = batch()
    output = model.generate(
        ids[:, :4], attention_mask=mask[:, :4], max_new_tokens=5, do_sample=False,
        logits_processor=LogitsProcessorList([ForceMixedEOS()]), use_cache=True,
    )
    assert output[:, 4:].tolist() == [[2, 0, 0], [7, 7, 2]]


@pytest.mark.parametrize("change", ["ids", "positions", "rows"])
def test_prepare_rejects_prefix_changes(change: str) -> None:
    model = wrapper()
    ids, mask = batch()
    with torch.no_grad():
        state = model(ids[:, :4], attention_mask=mask[:, :4], use_cache=True).past_key_values
    assert isinstance(state, GenerationState)
    changed = ids[:, :5].clone()
    positions = (mask[:, :5].cumsum(-1) - 1).clamp(min=0)
    if change == "ids":
        changed[0, 2] += 1
    elif change == "positions":
        positions[:, :4] += 1
    else:
        changed = changed.flip(0)
    with pytest.raises(ValueError, match="prefix|positions"):
        model.prepare_inputs_for_generation(
            changed, attention_mask=mask[:, :5], position_ids=positions, past_key_values=state, use_cache=True,
        )
    assert not state.slots


@pytest.mark.parametrize("kwargs", [
    {"num_beams": 2}, {"prompt_lookup_num_tokens": 2}, {"cache_implementation": "static"}, {"prefill_chunk_size": 2},
])
def test_unsupported_hf_cache_modes_fail_before_decode(kwargs: dict[str, Any]) -> None:
    model = wrapper()
    with pytest.raises(ValueError, match="cached generation"):
        model.generate(torch.tensor([[1, 3]]), max_new_tokens=2, use_cache=True, **kwargs)


def test_padded_vocabulary_and_legacy_rng_contract() -> None:
    model = wrapper()
    ids, mask = batch()
    with torch.no_grad():
        torch.manual_seed(14)
        native = model.model(ids, attention_mask=mask, return_logits=True, num_steps=[(2, 0), (3, 0)])["logits"]
        rng = torch.get_rng_state()
        torch.manual_seed(14)
        output = model(ids, attention_mask=mask, use_cache=False)
        assert torch.equal(rng, torch.get_rng_state())
        assert native is not None
        torch.testing.assert_close(output.logits, mask_padded_vocabulary(native, 512, 512), atol=0, rtol=0)


def test_hf_generation_config_inherits_explicit_legacy_policy() -> None:
    from transformers import GenerationConfig

    model = wrapper()
    model.generation_config.use_cache = False
    generated = model.generate(
        torch.tensor([[1, 3]]), generation_config=GenerationConfig(num_beams=2, max_new_tokens=2),  # type: ignore[no-untyped-call]  # HF config constructor
    )
    assert generated.shape[0] == 1 and generated.shape[1] <= 4


def test_hf_generation_can_return_its_ephemeral_cache() -> None:
    model = wrapper()
    result = model.generate(torch.tensor([[1, 3]]), max_new_tokens=2, min_new_tokens=2, return_dict_in_generate=True)
    assert isinstance(result.past_key_values, GenerationState)
    assert result.past_key_values.get_seq_length() == 3
    assert result.sequences.shape == (1, 4)


@pytest.mark.parametrize("shared_shape", ["row", "vector"])
def test_shared_positions_match_expanded_rows_and_original_native(
    monkeypatch: pytest.MonkeyPatch, shared_shape: str,
) -> None:
    import model.model as native_module

    model = wrapper()
    ids, mask = batch()
    positions = torch.arange(3, 10)
    if shared_shape == "row":
        positions = positions.unsqueeze(0)
    expanded = positions.expand(ids.shape[0], -1)
    shared_state, expanded_state = GenerationState(seed=81), GenerationState(seed=81)
    with torch.no_grad():
        for end in range(4, 8):
            start = 0 if end == 4 else end - 1
            shared_inputs = model.prepare_inputs_for_generation(
                ids[:, :end], attention_mask=mask[:, :end], position_ids=positions[..., :end],
                past_key_values=shared_state, use_cache=True,
            )
            expanded_inputs = model.prepare_inputs_for_generation(
                ids[:, :end], attention_mask=mask[:, :end], position_ids=expanded[:, :end],
                past_key_values=expanded_state, use_cache=True,
            )
            shared = model(**shared_inputs).logits
            explicit = model(**expanded_inputs).logits
            torch.testing.assert_close(shared, explicit, atol=3e-5, rtol=2e-5)
            core = 0

            def stored_initial_state(x: Tensor) -> Tensor:
                nonlocal core
                latent = shared_state.latents[core]
                assert latent.shape == x.shape
                core += 1
                return latent.clone()

            with monkeypatch.context() as patch:
                patch.setattr(native_module, "initialize_state", stored_initial_state)
                original = model.model(
                    ids[:, :end], attention_mask=mask[:, :end], position_ids=positions[..., :end],
                    num_steps=[(2, 0), (3, 0)], return_logits=True,
                )["logits"]
            assert original is not None and core == 2
            torch.testing.assert_close(shared, original[:, start:end], atol=3e-5, rtol=2e-5)
