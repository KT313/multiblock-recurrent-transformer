# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Real local scoring/generation through the shared session and HFLM's inner precision context."""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from evaluation.benchmarks import evaluate_on_benchmarks
from evaluation.prompts import Prompt
from evaluation.samples import generate_samples
from evaluation.session import inference_session
from evaluation.wrapper import RECURRENCE_ENV, hf_wrapper_around, isolated_inference
from model.execution import ExecutionPolicy
from model.kernels.runtime import CustomKernelError
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer


@pytest.mark.parametrize("use_cache", [False, True])
def test_bf16_samples_match_manual_context(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, use_cache: bool) -> None:
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    prompts = [Prompt("tok_3 tok_4", kind="continuation")]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = generate_samples(tiny_model, tokenizer, prompts, max_new_tokens=3, seed=19, use_cache=use_cache)
    actual = generate_samples(tiny_model, tokenizer, prompts, max_new_tokens=3, seed=19, use_cache=use_cache,
                              execution_policy=ExecutionPolicy("bf16-mixed"))
    assert actual == expected
    assert all(parameter.dtype == torch.float32 for parameter in tiny_model.parameters())


def test_session_setup_failure_restores_state(tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force a known incompatible execution request without loading/changing the native fixture's kernels.
    tiny_model.config.use_custom_kernels = True
    tiny_model.train()
    monkeypatch.setenv(RECURRENCE_ENV, "7,7")
    rng = torch.get_rng_state()
    with (
        pytest.raises(CustomKernelError, match="BF16 autocast.*use_custom_kernels: false"),
        inference_session(tiny_model, [1, 1], execution_policy=ExecutionPolicy("32")),
    ):
        pytest.fail("unsupported session entered")
    assert tiny_model.training and os.environ[RECURRENCE_ENV] == "7,7"
    assert torch.equal(rng, torch.get_rng_state()) and not torch.is_autocast_enabled("cpu")



def test_legacy_session_allows_caller_to_select_precision_inside(tiny_model: RecurrentGPT) -> None:
    tiny_model.config.use_custom_kernels = True
    # The legacy helper must not reject before a caller has entered their own execution context.
    with isolated_inference(tiny_model), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")


def test_nested_session_restores_outer_state(tiny_model: RecurrentGPT) -> None:
    with isolated_inference(tiny_model, [1, 1], seed=12, execution_policy=ExecutionPolicy("bf16-mixed")):
        rng = torch.get_rng_state()
        with (
            pytest.raises(RuntimeError, match="inner failure"),
            inference_session(tiny_model, [2, 2], seed=37, execution_policy=ExecutionPolicy("32")),
        ):
            assert torch.is_autocast_enabled("cpu") and os.environ[RECURRENCE_ENV] == "2,2"
            torch.rand(5)
            raise RuntimeError("inner failure")
        assert torch.equal(rng, torch.get_rng_state())
        assert not tiny_model.training and os.environ[RECURRENCE_ENV] == "1,1"
        assert torch.is_autocast_enabled("cpu")


def exercise_real_hflm(
    model: RecurrentGPT, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lm_eval")
    huggingface: Any = importlib.import_module("lm_eval.models.huggingface")
    device = next(model.parameters()).device
    observations: list[bool] = []
    scores: list[torch.Tensor] = []

    def simple_evaluate(**kwargs: Any) -> dict[str, Any]:
        harness = kwargs["model"]
        assert isinstance(harness, huggingface.HFLM)
        handle = harness.model.register_forward_pre_hook(
            lambda *args: observations.append(torch.is_autocast_enabled(device.type))
        )
        ids = torch.tensor([[1, 4, 5]], device=device)
        try:
            torch.manual_seed(23)
            scores.append(harness._model_call(ids).clone())
            result = harness._model_generate(ids, max_length=5, stop=[], do_sample=False)
            assert result.shape[0] == 1 and 3 < result.shape[1] <= 5
        finally:
            handle.remove()
        return {"results": {"offline": {"acc,none": 1.0}}}

    monkeypatch.setattr("evaluation.benchmarks._import_lm_eval", lambda: (SimpleNamespace(simple_evaluate=simple_evaluate), huggingface))
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    evaluate_on_benchmarks(model, tokenizer, ["offline"], batch_size=1, execution_policy=ExecutionPolicy("bf16-mixed"))
    assert observations and all(observations)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    with torch.autocast(device.type, dtype=torch.bfloat16), isolated_inference(model, seed=0):
        reference = hf_wrapper_around(model, tokenizer)
        torch.manual_seed(23)
        expected = reference(torch.tensor([[1, 4, 5]], device=device)).logits
    torch.testing.assert_close(scores[0], expected, rtol=0, atol=0)


def test_actual_cpu_hflm_inner_scoring_and_generation(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exercise_real_hflm(tiny_model, Tokenizer(tiny_tokenizer_dir), monkeypatch)


@pytest.mark.gpu
@pytest.mark.slow
def test_actual_cuda_samples_and_hflm_use_strict_kernels(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from model.kernels import mlp
    from model.layers import mlp as layers_mlp

    original = mlp.mlp_projection
    calls: list[bool] = []

    def observed(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16)
        return original(*args, **kwargs)

    monkeypatch.setattr(layers_mlp, "load_mlp", lambda: observed)
    model = RecurrentGPT(replace(tiny_model.config, use_custom_kernels=True)).cuda().train()
    model.load_state_dict(tiny_model.state_dict())
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    for use_cache in (False, True):
        calls.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = generate_samples(model, tokenizer, [Prompt("tok_3", kind="continuation")], max_new_tokens=2, use_cache=use_cache)
        calls.clear()
        actual = generate_samples(model, tokenizer, [Prompt("tok_3", kind="continuation")], max_new_tokens=2, use_cache=use_cache,
                                  execution_policy=ExecutionPolicy("bf16-mixed"))
        assert calls and all(calls) and actual == expected
    calls.clear()
    exercise_real_hflm(model, tokenizer, monkeypatch)
    assert calls and all(calls)
    assert model.training and all(parameter.dtype == torch.float32 for parameter in model.parameters())
    with pytest.raises(CustomKernelError, match="BF16 autocast"):
        generate_samples(model, tokenizer, [Prompt("tok_3", kind="continuation")], max_new_tokens=2,
                         execution_policy=ExecutionPolicy("32"))


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("fail", [False, True])
def test_session_restores_mixed_module_modes(tiny_model: RecurrentGPT, training: bool, fail: bool) -> None:
    tiny_model.train(training)
    tiny_model.transformer.wte.train(not training)
    flags = [module.training for module in tiny_model.modules()]
    try:
        with inference_session(tiny_model):
            assert not any(module.training for module in tiny_model.modules())
            if fail:
                raise RuntimeError("session failed")
    except RuntimeError as error:
        assert fail and str(error) == "session failed"
    assert [module.training for module in tiny_model.modules()] == flags
