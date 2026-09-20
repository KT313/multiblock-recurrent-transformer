# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Global RNG restoration and parity with the historical all-device seed paths."""

import importlib
import inspect
import os
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from evaluation.benchmarks import evaluate_on_benchmarks
from evaluation.prompts import Prompt
from evaluation.rng import preserve_rng, seed_model_rng
from evaluation.samples import generate_samples
from evaluation.session import inference_session
from evaluation.wrapper import RECURRENCE_ENV
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer


def draws() -> tuple[float, list[float], list[float]]:
    return random.random(), np.random.random(4).tolist(), torch.rand(4).tolist()


def next_draws(device: torch.device) -> tuple[float, list[float], list[float]]:
    with preserve_rng(device):
        return draws()


@pytest.mark.parametrize("fail", [False, True])
def test_nested_sessions_restore_all_global_states(tiny_model: RecurrentGPT, fail: bool) -> None:
    device = torch.device("cpu")
    expected = next_draws(device)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with inference_session(tiny_model, seed=11):
        random.seed(17)
        np.random.seed(18)
        draws()
        outer_expected = next_draws(device)
        try:
            with inference_session(tiny_model, seed=42):
                random.seed(1)
                np.random.seed(2)
                draws()
                if fail:
                    raise RuntimeError("nested RNG failure")
        except RuntimeError as error:
            assert fail and str(error) == "nested RNG failure"
        assert draws() == outer_expected
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert draws() == expected


@pytest.mark.parametrize("fail", [False, True])
def test_harness_seed_adapter_and_imports_restore_rng(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch, fail: bool,
) -> None:
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    expected = next_draws(torch.device("cpu"))
    observed: list[tuple[float, list[float], list[float]]] = []

    def construct(**kwargs: Any) -> object:
        draws()  # HFLM construction must precede the effective Torch reseed.
        return object()

    def evaluate(**kwargs: Any) -> dict[str, Any]:
        # Faithful harness defaults/conditional seeding; None must suppress all-device Torch seeding.
        random.seed(kwargs["random_seed"])
        np.random.seed(kwargs["numpy_random_seed"])
        if kwargs["torch_random_seed"] is not None:
            torch.manual_seed(kwargs["torch_random_seed"])
        observed.append(draws())
        if fail:
            raise RuntimeError("harness RNG failure")
        return {"results": {"offline": {"acc,none": 1.0}}}

    def load() -> tuple[Any, Any]:
        draws()  # first-use imports are inside restoration too
        return SimpleNamespace(simple_evaluate=evaluate), SimpleNamespace(HFLM=construct)

    monkeypatch.setattr("evaluation.benchmarks._import_lm_eval", load)
    with preserve_rng(torch.device("cpu")):
        random.seed(0)
        np.random.seed(1234)
        seed_model_rng(29, torch.device("cpu"))
        expected_inside = draws()
    if fail:
        with pytest.raises(RuntimeError, match="harness RNG failure"):
            evaluate_on_benchmarks(tiny_model, tokenizer, ["offline"], seed=29)
    else:
        evaluate_on_benchmarks(tiny_model, tokenizer, ["offline"], seed=29, recurrences=[None, [1, 1]])
    assert observed and all(value == expected_inside for value in observed)
    assert draws() == expected


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_native_samples_match_historical_seeding(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch,
    use_cache: bool, batch_size: int,
) -> None:
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    prompts = [Prompt("tok_3"), Prompt("tok_4 tok_5"), Prompt("tok_6")]
    actual = generate_samples(tiny_model, tokenizer, prompts, seed=31, max_new_tokens=5,
                              temperature=1.2, batch_size=batch_size, use_cache=use_cache)
    # Restore precisely the old session + per-batch calls; compare generated samples, not seed integers.
    def legacy(seed: int, device: torch.device) -> None:
        torch.manual_seed(seed)
    monkeypatch.setattr("evaluation.session.seed_model_rng", legacy)
    monkeypatch.setattr("evaluation.samples.seed_model_rng", legacy)
    expected = generate_samples(tiny_model, tokenizer, prompts, seed=31, max_new_tokens=5,
                                temperature=1.2, batch_size=batch_size, use_cache=use_cache)
    assert actual == expected


def test_native_cpu_never_calls_cuda_or_benchmark_imports(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    cuda_initialized = cast(Callable[[], bool], torch.cuda.is_initialized)
    before = cuda_initialized()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("CPU samples must not touch CUDA RNG/initialization or import lm-eval")

    for name in ("init", "_lazy_init", "manual_seed", "manual_seed_all", "get_rng_state", "set_rng_state"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr("evaluation.benchmarks._import_lm_eval", forbidden)
    generate_samples(tiny_model, tokenizer, [Prompt("tok_3"), Prompt("tok_4")], batch_size=1, max_new_tokens=2)
    assert cuda_initialized() == before


def test_restoration_failure_preserves_original_error_and_cleans_session(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RECURRENCE_ENV, "7,7")
    original = RuntimeError("model failed")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    def broken_restore(state: torch.Tensor) -> None:
        raise RuntimeError("injected restore failure")

    try:
        with (
            monkeypatch.context() as patch,
            pytest.raises(RuntimeError) as caught,
            inference_session(tiny_model, [1, 1]),
        ):
            patch.setattr(torch, "set_rng_state", broken_restore)
            random.seed(17)
            np.random.seed(18)
            raise original
        assert caught.value is original
        assert "injected restore failure" in original.__notes__[0]
        assert tiny_model.training and os.environ[RECURRENCE_ENV] == "7,7"
        assert random.getstate() == python_state
        np.testing.assert_equal(np.random.get_state(), numpy_state)
    finally:
        torch.set_rng_state(torch_state)


def test_real_harness_seed_adapter_matches_first_scoring_call(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package: Any = pytest.importorskip("lm_eval")
    evaluator: Any = importlib.import_module("lm_eval.evaluator")
    huggingface: Any = importlib.import_module("lm_eval.models.huggingface")
    parameters = inspect.signature(package.simple_evaluate).parameters
    assert parameters["random_seed"].default == 0
    assert parameters["numpy_random_seed"].default == parameters["fewshot_random_seed"].default == 1234
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    harnesses: list[Any] = []
    observations: list[tuple[tuple[float, list[float], list[float]], torch.Tensor]] = []

    def construct(**kwargs: Any) -> Any:
        harness = huggingface.HFLM(**kwargs)
        torch.rand(13)  # expose an incorrectly placed seed before HFLM construction
        harnesses.append(harness)
        return harness

    class FirstScore(RuntimeError):
        pass

    class OfflineTasks:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def load(self, tasks: object) -> None:
            # Stop at the first task boundary: the real simple_evaluate seeding/preamble already ran.
            values = draws()
            score = harnesses[-1]._model_call(torch.tensor([[1, 4, 5]])).clone()
            observations.append((values, score))
            raise FirstScore("scored offline")

    monkeypatch.setattr(evaluator, "TaskManager", OfflineTasks)
    monkeypatch.setattr("evaluation.benchmarks._import_lm_eval",
                        lambda: (package, SimpleNamespace(HFLM=construct)))
    with inference_session(tiny_model, seed=37) as session:
        harness = construct(pretrained=session.hf_wrapper(tokenizer), tokenizer=tokenizer.processor,
                            batch_size=1, add_bos_token=True,
                            max_length=tiny_model.config.model_max_sequence_length,
                            mixed_precision_dtype=session.mixed_precision_dtype)
        with pytest.raises(FirstScore):
            package.simple_evaluate(model=harness, tasks=["offline"], torch_random_seed=37)
    with pytest.raises(FirstScore):
        evaluate_on_benchmarks(tiny_model, tokenizer, ["offline"], seed=37, batch_size=1)
    assert len(observations) == 2 and observations[0][0] == observations[1][0]
    torch.testing.assert_close(observations[0][1], observations[1][1], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("fail", [False, True])
def test_two_cuda_devices_keep_rng_states_and_next_draws(tiny_model: RecurrentGPT, fail: bool) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices; single-device coverage cannot qualify this test")
    model = tiny_model.to("cuda:1")
    states = torch.cuda.get_rng_state_all()
    expected = [torch.rand(5, device=f"cuda:{index}") for index in range(2)]
    torch.cuda.set_rng_state_all(states)
    try:
        with inference_session(model, seed=43):
            torch.rand(7, device="cuda:1")
            seed_model_rng(44, torch.device("cuda:1"))  # sampling's per-batch path
            with inference_session(model, seed=45):
                torch.rand(3, device="cuda:1")
            if fail:
                raise RuntimeError("CUDA evaluation failure")
    except RuntimeError as error:
        assert fail and str(error) == "CUDA evaluation failure"
    assert all(torch.equal(a, b) for a, b in zip(states, torch.cuda.get_rng_state_all(), strict=True))
    for index, value in enumerate(expected):
        assert torch.equal(torch.rand(5, device=f"cuda:{index}"), value)


@pytest.mark.gpu
def test_cpu_session_preserves_pending_cuda_seed_in_fresh_process() -> None:
    # CUDA must start uninitialized: the pytest process may already have run GPU tests/fixtures.
    code = """
import torch
from evaluation.rng import seed_model_rng
from evaluation.session import inference_session
model = torch.nn.Linear(2, 2)
assert not torch.cuda.is_initialized()
torch.cuda.manual_seed_all(981)  # caller's pending seed, applied only when CUDA initializes
with inference_session(model, seed=42):
    seed_model_rng(43, torch.device('cpu'))
    torch.rand(5)
assert not torch.cuda.is_initialized()
torch.cuda.init()
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        assert torch.cuda.initial_seed() == 981
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


@pytest.mark.gpu
def test_scoped_cuda_seed_matches_historical_cpu_and_model_streams() -> None:
    device = torch.device("cuda:0")
    with preserve_rng(device):
        torch.manual_seed(912)
        expected_cpu, expected_cuda = torch.rand(9), torch.rand(9, device=device)
        seed_model_rng(912, device)
        assert torch.equal(torch.rand(9), expected_cpu)
        assert torch.equal(torch.rand(9, device=device), expected_cuda)


def test_restore_error_is_not_swallowed_by_a_callers_handled_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_restore(state: torch.Tensor) -> None:
        raise RuntimeError("restore failed on successful evaluation")

    monkeypatch.setattr(torch, "set_rng_state", broken_restore)
    try:
        raise ValueError("already handled by caller")
    except ValueError:
        with pytest.raises(RuntimeError, match="restore failed on successful evaluation"), preserve_rng(torch.device("cpu")):
            pass
