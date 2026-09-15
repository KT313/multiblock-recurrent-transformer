# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Behavioral controls for representation metrics and isolated post-update probes; CPU only."""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Literal, cast

import pytest
import torch

from model import RecurrentConfig, RecurrentGPT
from model.blocks.recurrence import core_block_forward
from training.backend.single_device import SingleDeviceBackend
from training.data.packing import PackedBatch
from training.representation_metrics import (
    _perturb_state, representation_metrics, sampled_correlation_metrics, select_probe_document,
    state_sensitivity_metrics, track_recurrence_metrics,
)


def test_representation_controls_and_independent_pair_oracle() -> None:
    generator = torch.Generator().manual_seed(912)
    identical = torch.arange(16).float().expand(7, -1)
    result = representation_metrics(identical)
    assert result['correlation'] == pytest.approx(1) and result['cosine'] == pytest.approx(1)
    assert result['dispersion'] == 0 and result['correlation_pairs'] == 21
    values = torch.randn((40, 256), generator=generator)
    result = representation_metrics(values)
    centered = values-values.mean(-1, keepdim=True)
    normalized = centered/centered.norm(dim=-1, keepdim=True)
    # Small test-only pair matrix verifies the bounded implementation's vector-sum reduction.
    pair_matrix = normalized @ normalized.T
    expected = pair_matrix[~torch.eye(40, dtype=torch.bool)].mean()
    torch.testing.assert_close(result['correlation'], expected, atol=1e-7, rtol=1e-5)
    assert result['correlation'].abs() < .05 and result['dispersion'] > .9


def test_zero_padding_nonfinite_and_repeated_tokens() -> None:
    values = torch.tensor([[1.,2.,3.], [1.,2.,3.], [float('nan')]*3, [0.,0.,0.]])
    stats = representation_metrics(values, torch.tensor([True,True,False,False]), token_ids=torch.tensor([5,5,7,8]))
    assert stats['tokens'] == 2 and stats['same_token_pair_fraction'] == 1
    assert stats['correlation'] == pytest.approx(1) and stats['nonfinite_fraction'] == 0
    zero = representation_metrics(torch.zeros(5, 8))
    assert zero['zero_vector_fraction'] == 1 and zero['zero_variance_fraction'] == 1
    assert zero['correlation_pairs'] == 0 and torch.isnan(zero['correlation']) and torch.isnan(zero['dispersion'])
    constant = representation_metrics(torch.ones(5, 8))
    assert constant['cosine'] == pytest.approx(1) and torch.isnan(constant['correlation'])
    invalid = representation_metrics(values)
    assert invalid['nonfinite_fraction'] == .25 and torch.isnan(invalid['correlation'])
    empty = representation_metrics(values, torch.zeros(4, dtype=torch.bool))
    assert empty['tokens'] == 0 and torch.isnan(empty['rms'])
    with pytest.raises(ValueError, match='ONE|one'):
        representation_metrics(torch.zeros(2, 4, 8))  # refusing accidental sequence/document pooling


def test_state_sensitivity_distinguishes_ignored_and_carried_state() -> None:
    state = torch.arange(1, 13).float().view(1, 3, 4)
    alternative = state+.1
    carried = state_sensitivity_metrics(state, alternative, 2*state, 2*alternative)
    assert carried['state_sensitivity'] == pytest.approx(1)
    ignored = state_sensitivity_metrics(state, alternative, torch.ones_like(state), torch.ones_like(state))
    assert ignored['state_sensitivity'] == 0 and ignored['state_perturbation'] > 0
    unchanged = state_sensitivity_metrics(state, state, state, state)
    assert torch.isnan(unchanged['state_sensitivity'])
    # Changes in output alone are not a dependence test without an effective input perturbation.
    assert torch.isnan(state_sensitivity_metrics(state, state, state, alternative)['state_sensitivity'])


def test_sampled_correlation_matches_existing_definition_and_reports_degeneracy() -> None:
    generator = torch.Generator().manual_seed(912)
    hidden = torch.randn(129, 32, generator=generator)
    sampled = sampled_correlation_metrics(hidden)
    original = representation_metrics(hidden)
    torch.testing.assert_close(sampled['correlation'], original['correlation'], rtol=0, atol=0)
    assert sampled['sampled_tokens'] == 64 and sampled['correlation_pairs'] == 2016
    constant = sampled_correlation_metrics(torch.ones(20, 8))
    assert constant['sampled_zero_variance_fraction'] == 1 and constant['correlation_pairs'] == 0
    assert torch.isnan(constant['correlation'])
    invalid = sampled_correlation_metrics(torch.full((4, 8), float('nan')))
    assert invalid['sampled_nonfinite_fraction'] == 1 and torch.isnan(invalid['correlation'])
    empty = sampled_correlation_metrics(torch.empty(0, 8))
    assert empty['sampled_tokens'] == 0 and torch.isnan(empty['correlation'])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_calculations_do_not_write_inputs_or_retain_autograd(dtype: torch.dtype) -> None:
    # Noncontiguous FP32 inputs exercise detach/float paths that can alias caller storage.
    source = torch.arange(96, dtype=dtype).reshape(8, 12).requires_grad_()
    hidden = source[:, ::2]
    mask = torch.tensor([True, False, True, True, True, False, True, True])
    ids = torch.arange(8)
    alternative = hidden.detach().clone()+.1
    inputs = (source, hidden, mask, ids, alternative)
    before = [(value.clone(), value._version) for value in inputs]
    result = representation_metrics(hidden, mask, token_ids=ids)
    result.update(sampled_correlation_metrics(hidden))
    result.update(state_sensitivity_metrics(hidden, alternative, hidden, alternative))
    with torch.no_grad():
        perturbed = _perturb_state(hidden, torch.Generator().manual_seed(234))
    assert perturbed.data_ptr() != hidden.data_ptr()
    assert all(not value.requires_grad and value.grad_fn is None for value in result.values())
    for value, (snapshot, version) in zip(inputs, before):
        assert torch.equal(value, snapshot) and value._version == version


def make_pack() -> PackedBatch:
    # Two documents followed by a longer padding tail; aggregate metadata intentionally mismatches this rank.
    ids = torch.tensor([[1,3,4,1,8,9,10,2,2,2,2,2]])
    labels = torch.tensor([[3,4,2,-100,-100,10,2,-100,-100,-100,-100,-100]])
    docs = torch.tensor([[0,0,0,1,1,1,1,2,2,2,2,2]], dtype=torch.int32)
    positions = torch.tensor([[0,1,2,0,1,2,3,0,1,2,3,4]])
    return PackedBatch(ids, labels, ['world-a','world-b'], positions, docs, 900, [1000,2000])


def test_selection_uses_local_document_boundaries_and_retains_prompt_context() -> None:
    batch = make_pack()
    selected = select_probe_document(batch, 3)
    assert selected is not None and selected.tolist() == [[1,8,9]]
    assert select_probe_document(batch._replace(labels=torch.full_like(batch.labels, -100)), 3) is None
    assert select_probe_document(batch, 1) is None


@pytest.mark.parametrize('precision', ['32', 'bf16-mixed'])
@pytest.mark.parametrize('scaling', ['none', 'inverse_sqrt_depth'])
def test_probe_is_repeatable_and_preserves_training_state(
    tiny_model: RecurrentGPT, precision: str, scaling: Literal['none', 'inverse_sqrt_depth'],
) -> None:
    tiny_model = RecurrentGPT(replace(tiny_model.config, residual_scaling=scaling))
    backend = SingleDeviceBackend('cpu', precision)
    tiny_model.train()
    tiny_model.transformer.coda[0].eval()  # preserve deliberately mixed child flags
    tiny_model.step, tiny_model.micro_batch_index = 7, 3
    before_modes = [module.training for module in tiny_model.modules()]
    for parameter in tiny_model.parameters():
        parameter.grad = torch.ones_like(parameter)*.01
    before = copy.deepcopy(tiny_model.state_dict())
    # state_dict omits the RoPE table (persistent=False); inspect every registered tensor and its version too.
    tensors = dict(tiny_model.named_parameters()) | dict(tiny_model.named_buffers())
    before_tensors = {name: (value, value.clone(), value._version) for name, value in tensors.items()}
    config = copy.deepcopy(tiny_model.config)
    gradients = []
    for parameter in tiny_model.parameters():
        assert parameter.grad is not None
        gradients.append(parameter.grad.clone())
    rng = torch.get_rng_state().clone()
    batch = make_pack()
    batch_before = [value.clone() for value in batch if isinstance(value, torch.Tensor)]
    first = track_recurrence_metrics(tiny_model, backend, batch, correlations='adapter,attention,mlp')
    second = track_recurrence_metrics(tiny_model, backend, make_pack(), correlations='adapter,attention,mlp')
    assert {'token_correlation','token_dispersion','state_sensitivity'} <= first.keys()
    assert first['recurrence_probe/step'] == 8 and first['recurrence_probe/tokens'] == 4
    assert first['recurrence_probe/residual_scale'] == pytest.approx(tiny_model.config.residual_scale)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.get_rng_state(), rng)
    assert before_modes == [module.training for module in tiny_model.modules()]
    assert (tiny_model.step, tiny_model.micro_batch_index) == (7,3)
    assert tiny_model.config == config
    after_tensors = dict(tiny_model.named_parameters()) | dict(tiny_model.named_buffers())
    assert after_tensors.keys() == before_tensors.keys()
    for name, value in after_tensors.items():
        original, snapshot, version = before_tensors[name]
        assert value is original and value._version == version and torch.equal(value, snapshot)
    for value, snapshot in zip((value for value in batch if isinstance(value, torch.Tensor)), batch_before):
        assert torch.equal(value, snapshot)
    assert all(not value.requires_grad and value.grad_fn is None for value in first.values())
    for name, value in tiny_model.state_dict().items():
        assert torch.equal(value, before[name])
    for parameter, gradient in zip(tiny_model.parameters(), gradients):
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, gradient)
    backend.is_main = False
    assert track_recurrence_metrics(tiny_model, backend, make_pack()) == {}


@pytest.mark.parametrize('failure_site', ['core', 'observer'])
def test_probe_restores_context_after_failure(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch, failure_site: str,
) -> None:
    backend = SingleDeviceBackend('cpu', '32')
    rng = torch.get_rng_state().clone()
    modes = [module.training for module in tiny_model.modules()]
    tensors = list(tiny_model.parameters()) + list(tiny_model.buffers())
    snapshots = [(value.clone(), value._version) for value in tensors]
    def fail(*args: object, **kwargs: object) -> torch.Tensor:
        torch.rand(1)
        raise RuntimeError('injected probe failure')
    if failure_site == 'core':
        monkeypatch.setattr('training.representation_metrics.core_block_forward', fail)
    else:
        # Fail inside the diagnostic attention observer, after core execution has started and hooks are installed.
        monkeypatch.setattr('training.representation_metrics.sampled_correlation_metrics', fail)
    with pytest.raises(RuntimeError, match='injected probe failure'):
        selector = 'adapter,attention,mlp' if failure_site == 'core' else 'attention,mlp'
        track_recurrence_metrics(tiny_model, backend, make_pack(), correlations=selector)
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == [module.training for module in tiny_model.modules()]
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in tiny_model.modules())
    for value, (snapshot, version) in zip(tensors, snapshots):
        assert torch.equal(value, snapshot) and value._version == version
    assert all(parameter.grad is None for parameter in tiny_model.parameters())


@pytest.mark.parametrize('selector', ['', None])
def test_disabled_details_do_not_calculate_correlations_or_install_hooks(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch, selector: str | None,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('disabled detailed correlations must not be calculated or observed')
    monkeypatch.setattr('training.representation_metrics.sampled_correlation_metrics', forbidden)
    monkeypatch.setattr('training.representation_metrics._watch_input', forbidden)
    monkeypatch.setattr('training.representation_metrics._watch_output', forbidden)
    result = track_recurrence_metrics(tiny_model, SingleDeviceBackend('cpu', '32'), make_pack(), correlations=selector)
    assert not any(key.startswith(('adapter/', 'attention/', 'mlp/')) for key in result)
    assert {'token_correlation', 'token_dispersion', 'state_sensitivity'} <= result.keys()


def test_scaled_probe_matches_normal_forward_with_identical_initial_states(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecurrentGPT(replace(tiny_model.config, residual_scaling='inverse_sqrt_depth')).eval()
    backend = SingleDeviceBackend('cpu', '32')
    batch = make_pack()
    probe = track_recurrence_metrics(model, backend, batch)
    generator = torch.Generator().manual_seed(233)
    def initialize(x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    monkeypatch.setattr('model.model.initialize_state', initialize)
    observed = []
    def record(module: torch.nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        observed.append(output.detach().clone())
    handle = model.transformer.ln_final.register_forward_hook(record)
    tokens = select_probe_document(batch, model.config.model_max_sequence_length)
    assert tokens is not None
    try:
        with torch.no_grad():
            model(tokens, num_steps=(2, 0))  # tiny's configured mean and probe depth, held fixed
    finally:
        handle.remove()
    expected = representation_metrics(observed[0][0])
    for metric in ('correlation', 'dispersion', 'rms'):
        torch.testing.assert_close(probe[f'representation/pre_head/{metric}'], expected[metric], rtol=0, atol=0)


@pytest.mark.parametrize('selector', ['adapter', 'attention', 'mlp', 'adapter,attention', 'adapter,mlp', 'attention,mlp',
                                     'adapter,attention,mlp'])
def test_selector_logs_only_requested_families(tiny_model: RecurrentGPT, selector: str) -> None:
    result = track_recurrence_metrics(tiny_model, SingleDeviceBackend('cpu', '32'), make_pack(), correlations=selector)
    actual = {key.split('/')[0] for key in result if key.startswith(('adapter/', 'attention/', 'mlp/'))}
    assert actual == set(selector.split(','))


def test_all_adapter_iterations_match_actual_tensors_and_exclude_perturbations(monkeypatch: pytest.MonkeyPatch) -> None:
    import training.representation_metrics as metrics_module
    torch.manual_seed(233)
    model = RecurrentGPT(RecurrentConfig(n_embd=16, intermediate_size=32, num_attention_heads=2,
        vocab_size=32, padding_multiple=1, model_max_sequence_length=16, n_layers_in_prelude=1,
        n_layers_in_coda=1, n_layers_in_recurrent_block=[2,1,2], mean_recurrence=[12,3,1],
        mean_backprop_depth=[1,1,1], use_custom_kernels=False))
    backend = SingleDeviceBackend('cpu', '32')
    original = core_block_forward
    adapter_ids = {id(adapter): index for index, adapter in enumerate(model.transformer.adapters)}
    calls: dict[int, list[dict[str, torch.Tensor]]] = {index: [] for index in adapter_ids.values()}
    merged: dict[int, list[torch.Tensor]] = {index: [] for index in adapter_ids.values()}
    submodules: dict[tuple[int, int, str], list[dict[str, torch.Tensor]]] = {}
    handles = []
    for index, raw_layers in enumerate(model.transformer.core_blocks):
        layers = cast(torch.nn.ModuleList, raw_layers)
        def observe(module: torch.nn.Module, inputs: tuple[Any, ...], core: int = index) -> None:
            merged[core].append(cast(torch.Tensor, inputs[0]).detach().clone())
        handles.append(layers[0].register_forward_pre_hook(observe))
        for layer_index, layer in enumerate(layers):
            for target in ('attention', 'mlp'):
                module = cast(torch.nn.Module, layer.attn if target == 'attention' else layer.mlp)
                key = (index, layer_index, target)
                submodules[key] = []
                def input_observer(module: torch.nn.Module, inputs: tuple[Any, ...],
                                   slot: tuple[int, int, str] = key) -> None:
                    submodules[slot].append({'input': cast(torch.Tensor, inputs[0]).detach().clone()})
                def output_observer(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any,
                                    slot: tuple[int, int, str] = key) -> None:
                    submodules[slot][-1]['output'] = cast(torch.Tensor, output).detach().clone()
                handles.append(module.register_forward_pre_hook(input_observer))
                handles.append(module.register_forward_hook(output_observer))
    def record(s: torch.Tensor, e: torch.Tensor, rotary: torch.Tensor, mask: Any, adapter: torch.nn.Module,
               layers: torch.nn.ModuleList, base: torch.Tensor | None = None, caches: Any = None) -> torch.Tensor:
        output = original(s, e, rotary, mask, adapter, layers, base, caches)
        calls[adapter_ids[id(adapter)]].append({'state_input': s.detach().clone(), 'block_input': e.detach().clone(),
                                              'core_output': output.detach().clone()})
        return output
    monkeypatch.setattr(metrics_module, 'core_block_forward', record)
    try:
        result = track_recurrence_metrics(model, backend, make_pack(), correlations='adapter,attention,mlp')
        for index, depth in enumerate((8,3,1)):
            assert len(calls[index]) == len(merged[index]) == depth+1  # existing final perturbation pass
            for iteration in range(1, depth+1):
                tensors = calls[index][iteration-1] | {'merged_output': merged[index][iteration-1]}
                for stage, tensor in tensors.items():
                    prefix = f'adapter/core_{index}/iter_{iteration}/{stage}'
                    expected = representation_metrics(tensor[0])
                    torch.testing.assert_close(result[f'{prefix}/correlation'], expected['correlation'], rtol=0, atol=0, equal_nan=True)
                    assert result[f'{prefix}/correlation_pairs'] == expected['correlation_pairs']
            assert not any(key.startswith(f'adapter/core_{index}/iter_{depth+1}/') for key in result)
            layer = cast(torch.nn.ModuleList, model.transformer.core_blocks[index])[0]
            assert len(layer._forward_pre_hooks) == 1  # original observer retained, diagnostic observer removed
        for (index, layer_index, target), observations in submodules.items():
            depth = (8,3,1)[index]
            assert len(observations) == depth+1
            for iteration in range(1, depth+1):
                for location in ('input', 'output'):
                    metric_key = f'{target}/core_{index}/iter_{iteration}/layer_{layer_index}/{location}/correlation'
                    expected_correlation = representation_metrics(observations[iteration-1][location][0])['correlation']
                    torch.testing.assert_close(result[metric_key], expected_correlation, rtol=0, atol=0, equal_nan=True)
            assert not any(key.startswith(f'{target}/core_{index}/iter_{depth+1}/') for key in result)
            child_layer = cast(torch.nn.ModuleList, model.transformer.core_blocks[index])[layer_index]
            child = cast(torch.nn.Module, child_layer.attn if target == 'attention' else child_layer.mlp)
            assert len(child._forward_pre_hooks) == len(child._forward_hooks) == 1
    finally:
        for handle in handles:
            handle.remove()
