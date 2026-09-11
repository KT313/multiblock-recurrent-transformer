"""Configuration, strict errors, explicit native execution, export isolation and the integrated CUDA trio."""
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from model import RecurrentGPT
from model.kernels import runtime
from model.test_config import tiny_config


def test_disabled_kernels_do_not_load_implementations(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected() -> bool:
        raise AssertionError('disabled kernels must not probe or import CUDA implementations')
    monkeypatch.setattr(runtime, 'kernels_available', unexpected)
    model = RecurrentGPT(tiny_config(use_custom_kernels=False))
    assert model._custom_head is None
    for layer in model.modules():
        if hasattr(layer, '_custom_mlp'):
            assert layer._custom_mlp is None
        if hasattr(layer, '_custom_rope'):
            assert layer._custom_rope is None


def test_unavailable_kernels_raise_with_disable_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, 'kernels_available', lambda: False)
    with pytest.raises(runtime.CustomKernelError, match='use_custom_kernels: false'):
        RecurrentGPT(tiny_config(use_custom_kernels=True))
    native = RecurrentGPT(tiny_config(use_custom_kernels=False))
    tokens = torch.tensor([[1, 2, 3, 4]])
    assert torch.isfinite(native(tokens, labels=tokens)['loss'])


def test_loading_error_keeps_original_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = ImportError('broken Triton installation')
    def broken() -> bool:
        raise failure
    monkeypatch.setattr(runtime, 'kernels_available', broken)
    with pytest.raises(runtime.CustomKernelError, match='broken Triton.*use_custom_kernels: false') as info:
        runtime.load_mlp()
    assert info.value.__cause__ is failure


def test_execution_error_has_guidance_and_never_retries() -> None:
    calls = []
    @runtime.kernel_errors('test')
    def broken() -> None:
        calls.append(1)
        raise RuntimeError('launch failure')
    with pytest.raises(runtime.CustomKernelError, match='launch failure.*use_custom_kernels: false'):
        broken()
    assert calls == [1]


def test_config_serializes_kernel_choice(tmp_path: Path) -> None:
    from model import RecurrentConfig
    from model.hf.modeling import RecurrentGPTConfig
    for enabled in (False, True):
        config = tiny_config(use_custom_kernels=enabled)
        path = tmp_path / 'model.json'
        config.to_json(path)
        assert RecurrentConfig.from_json(path).use_custom_kernels is enabled
        hf = RecurrentGPTConfig.from_recurrent_config(config)
        assert hf.to_recurrent_config().use_custom_kernels is enabled
    with pytest.raises(ValueError, match='boolean'):
        tiny_config(use_custom_kernels='false')


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('checkpoint', ['none', 'selective', 'full'])
def test_integrated_compiled_training_trio(checkpoint: str) -> None:
    from model.blocks.recurrence import check_checkpoint_mode
    config = tiny_config(num_attention_heads=2, bf16_residual_stream='core')
    tokens = torch.randint(0, 512, (1, 32), device='cuda')
    from model.layers.attention import document_attention_mask
    positions = torch.arange(32, device='cuda').view(1, -1) % 8
    mask = document_attention_mask(torch.arange(32, device='cuda').view(1, -1) // 8)
    models = []
    for enabled in (False, True):
        torch.manual_seed(5)
        model = RecurrentGPT(replace(config, use_custom_kernels=enabled),
                             gradient_checkpointing=check_checkpoint_mode(checkpoint)).cuda().train()
        models.append(model)
    candidate = models[1]
    assert candidate._custom_head is not None
    assert candidate._custom_head.__module__ == 'model.kernels.lm_head'
    assert any(getattr(layer, '_custom_mlp', None) is not None for layer in candidate.modules())
    assert any(getattr(layer, '_custom_rope', None) is not None for layer in candidate.modules())
    results: list[dict[str, Any]] = []
    for model in models:
        compiled = torch.compile(model, dynamic=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        torch.manual_seed(19)
        loss = None
        for step in range(2):
            model.step = step
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                model.micro_batch_index = micro
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = compiled(tokens, labels=tokens, position_ids=positions,
                                    attention_mask=mask, return_logits=False)['loss'] / 2
                loss.backward()
            optimizer.step()
        assert loss is not None
        results.append({'loss': loss.detach(), 'parameters': dict(model.named_parameters()),
                        'optimizer': list(optimizer.state.values())})
    torch.testing.assert_close(results[1]['loss'], results[0]['loss'], atol=1e-2, rtol=1e-2)
    for name, ref in results[0]['parameters'].items():
        actual = results[1]['parameters'][name]
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
    for actual, ref in zip(results[1]['optimizer'], results[0]['optimizer'], strict=True):
        assert actual.keys() == ref.keys()
        for key in ref:
            torch.testing.assert_close(actual[key], ref[key], atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
@pytest.mark.slow
def test_exported_cuda_model_can_coexist_with_original(tmp_path: Path) -> None:
    from model.hf.modeling import export_to_hf
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    torch.manual_seed(14)
    original = RecurrentGPT(tiny_config(use_custom_kernels=True)).cuda().train()
    export_to_hf(original, original.config, tmp_path / 'export')
    remote_class: Any = get_class_from_dynamic_module('hf_modeling.RecurrentGPTForCausalLM', str(tmp_path / 'export'))
    exported = remote_class.from_pretrained(tmp_path / 'export')
    torch.nn.Module.cuda(exported)
    exported.train()
    assert exported.model.config.use_custom_kernels is True
    assert original._custom_head is not None and exported.model._custom_head is not None
    assert original._custom_head.__module__ != exported.model._custom_head.__module__
    tokens = torch.tensor([[1, 2, 3, 4]], device='cuda')
    results = []
    for model in (original, exported.model):
        torch.manual_seed(26)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(tokens, labels=tokens, return_logits=False)['loss']
        loss.backward()
        results.append((loss.detach(), [p.grad for p in model.parameters()]))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for left, right in zip(results[0][1], results[1][1], strict=True):
        assert left is not None and right is not None
        torch.testing.assert_close(left, right, rtol=0, atol=0)
