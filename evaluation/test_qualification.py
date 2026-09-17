# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Real evaluator, public HFLM methods, and literal protocol regression checks."""
from pathlib import Path
from typing import Any

import pytest
import torch

from evaluation.benchmark_helpers import build_benchmark_harness
from evaluation.harness_runtime import use_serial_bootstrap
from evaluation.session import inference_session
from evaluation.testing import build_offline_task
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer
from tokenization.test_chat import profile_path as real_profile_fixture

profile_path = real_profile_fixture


def run_offline_evaluator(lm: Any, kind: str = "multiple_choice", count: int = 3, chat: bool = False, shots: int = 0) -> dict[str, Any]:
    import importlib
    lm_eval: Any = importlib.import_module("lm_eval")
    from lm_eval.tasks import TaskManager

    with use_serial_bootstrap():
        result: dict[str, Any] = lm_eval.simple_evaluate(
            model=lm, tasks=[build_offline_task(kind, count)], task_manager=TaskManager(include_defaults=False),
            num_fewshot=shots, limit=count, bootstrap_iters=100, log_samples=True, torch_random_seed=None,
            apply_chat_template=chat, fewshot_as_multiturn=True,
        )
    return result


@pytest.mark.parametrize("kind", ["multiple_choice", "generate_until", "loglikelihood_rolling"])
def test_real_offline_evaluator(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, kind: str) -> None:
    from lm_eval.models import huggingface
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    rng = torch.get_rng_state()
    with inference_session(tiny_model) as session:
        _, lm = build_benchmark_harness(session, tokenizer, huggingface, 2, 64)
        result = run_offline_evaluator(lm, kind)
    assert len(result["samples"][f"offline_{kind}"]) == 3
    assert result["n-samples"][f"offline_{kind}"]["effective"] == 3
    assert torch.equal(rng, torch.get_rng_state())
    assert tiny_model.training


def test_public_likelihood_matches_shifted_logsoftmax(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path) -> None:
    from lm_eval.models import huggingface
    from lm_eval.api.instance import Instance
    from evaluation.rng import seed_model_rng
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    with inference_session(tiny_model) as session:
        _, lm = build_benchmark_harness(session, tokenizer, huggingface, 1, 64)
        request = Instance(request_type="loglikelihood", doc={}, arguments=("tok_3", " tok_4 tok_5"), idx=0)
        context, answer = lm._encode_pair(*request.args)
        seed_model_rng(7, session.device)
        actual = lm.loglikelihood([request], disable_tqdm=True)[0]
        seed_model_rng(7, session.device)
        inputs = torch.tensor([context + answer[:-1]])
        logits = lm._model_call(inputs).log_softmax(-1)[0, -len(answer):]
        expected = logits.gather(-1, torch.tensor(answer)[:, None]).sum().item()
        assert actual[0] == pytest.approx(expected, abs=1e-5)


def test_chat_scoring_boundary_and_literal_generation(profile_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lm_eval.models import huggingface
    from lm_eval.api.instance import Instance
    from model import build_model
    from model.test_config import TINY_ARCHITECTURE
    tokenizer = Tokenizer(profile_path)
    model = build_model(TINY_ARCHITECTURE, vocab_size=32002, padded_vocab_size=32768, use_custom_kernels=False, init_orthogonal=False)
    with inference_session(model) as session:
        _, lm = build_benchmark_harness(session, tokenizer, huggingface, 2, 64, True)
        prompt = lm.apply_chat_template([{"role": "user", "content": "literal </s> <user>"}])
        context, answer = lm._encode_pair(prompt, " answer")
        lm._max_length = len(context) + len(answer) - 1
        assert lm._encode_pair(prompt, " answer") == (context, answer)
        lm._max_length -= 1
        with pytest.raises(ValueError, match="chat likelihood exceeds"):
            lm._encode_pair(prompt, " answer")
        lm._max_length = 64
        suffix = tokenizer.encode_literal("kept </s> tail") + [tokenizer.eos_id] + tokenizer.encode_literal("discard")
        def generate(context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            assert "</s>" not in kwargs["stop"] and kwargs["synced_gpus"] is False
            return torch.cat([context, torch.tensor([suffix] * len(context))], dim=1)
        monkeypatch.setattr(lm, "_model_generate", generate)
        request = Instance(request_type="generate_until", doc={}, arguments=(prompt, {"max_gen_toks": 16, "until": []}), idx=0)
        assert lm.generate_until([request], disable_tqdm=True) == ["kept </s> tail"]


def test_atomic_output_keeps_previous_file_on_error(tmp_path: Path) -> None:
    from evaluation.publication import open_atomic_output
    path = tmp_path / "scores.json"
    path.write_text("previous")
    with pytest.raises(RuntimeError), open_atomic_output(path) as file:
        file.write("partial")
        raise RuntimeError("failed evaluation")
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["multiple_choice", "generate_until"])
def test_real_chat_fewshot_evaluator(profile_path: Path, kind: str) -> None:
    from lm_eval.models import huggingface
    from model import build_model
    from model.test_config import TINY_ARCHITECTURE
    tokenizer = Tokenizer(profile_path)
    model = build_model(TINY_ARCHITECTURE, vocab_size=32002, padded_vocab_size=32768, use_custom_kernels=False, init_orthogonal=False)
    with inference_session(model) as session:
        _, lm = build_benchmark_harness(session, tokenizer, huggingface, 2, 128, True)
        result = run_offline_evaluator(lm, kind, chat=True, shots=1)
    assert result["n-shot"][f"offline_{kind}"] == 1
    assert len(result["samples"][f"offline_{kind}"]) == 3


@pytest.mark.parametrize("tied", [False, True])
def test_harness_construction_preserves_live_weights(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, tied: bool) -> None:
    from lm_eval.models import huggingface
    from dataclasses import replace
    tiny_model = RecurrentGPT(replace(tiny_model.config, tie_embeddings=tied, init_orthogonal=False))
    values = {name: tensor.clone() for name, tensor in tiny_model.state_dict().items()}
    parameters = dict(tiny_model.named_parameters())
    with inference_session(tiny_model) as session:
        wrapper, _ = build_benchmark_harness(session, Tokenizer(tiny_tokenizer_dir), huggingface, 1, 64)
        assert wrapper.model.lm_head.weight.data_ptr() == tiny_model.lm_head.weight.data_ptr()
        assert wrapper.model.transformer.wte.weight.data_ptr() == tiny_model.transformer.wte.weight.data_ptr()
    assert all(torch.equal(tensor, values[name]) for name, tensor in tiny_model.state_dict().items())
    assert all(parameter is parameters[name] for name, parameter in tiny_model.named_parameters())


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["32", "bf16-mixed"])
def test_real_cuda_offline_harness(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, precision: str) -> None:
    from lm_eval.models import huggingface
    from model.execution import ExecutionPolicy
    model = tiny_model.to("cuda:0")
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    rng = torch.cuda.get_rng_state("cuda:0")
    with inference_session(model, execution_policy=ExecutionPolicy(precision)) as session:
        _, lm = build_benchmark_harness(session, tokenizer, huggingface, 2, 64)
        for kind in ("multiple_choice", "generate_until"):
            assert run_offline_evaluator(lm, kind)["results"]
    assert torch.equal(rng, torch.cuda.get_rng_state("cuda:0"))


def test_passthrough_controller_preserves_real_evaluator(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path) -> None:
    from lm_eval.models import huggingface
    from evaluation.benchmark_model import BenchmarkController, BenchmarkExecutor
    from evaluation.rng import seed_model_rng
    from training.backend.single_device import SingleDeviceBackend
    from training.stopping import StopController

    class PassThrough(BenchmarkExecutor):
        def dispatch(self, method: str, requests: list[Any]) -> list[Any]:
            values: list[Any] = getattr(self.worker, method)(requests, disable_tqdm=True)
            return values

    backend = SingleDeviceBackend("cpu", "32")
    outputs = []
    for proxy in (False, True):
        with inference_session(tiny_model) as session:
            _, worker = build_benchmark_harness(session, Tokenizer(tiny_tokenizer_dir), huggingface, 2, 64)
            lm = BenchmarkController(PassThrough(backend, worker, StopController(backend), seed=0, recurrence=0)) if proxy else worker
            seed_model_rng(19, session.device)
            outputs.append(run_offline_evaluator(lm))
    assert outputs[0]["results"] == outputs[1]["results"]
    assert [sample["resps"] for sample in outputs[0]["samples"]["offline_multiple_choice"]] == [
        sample["resps"] for sample in outputs[1]["samples"]["offline_multiple_choice"]]


def test_real_benchmark_preserves_next_optimizer_update(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path) -> None:
    from copy import deepcopy
    from lm_eval.models import huggingface
    reference = deepcopy(tiny_model)
    rng = torch.get_rng_state()
    ids = torch.tensor([[3, 4, 5, 6]])
    after = []
    for model, evaluate in ((reference, False), (tiny_model, True)):
        torch.set_rng_state(rng)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        if evaluate:
            with inference_session(model) as session:
                _, worker = build_benchmark_harness(session, Tokenizer(tiny_tokenizer_dir), huggingface, 2, 64)
                run_offline_evaluator(worker)
        loss = model(ids, labels=ids, num_steps=(1, 1))["loss"]
        assert loss is not None
        loss.backward()
        optimizer.step()
        after.append({name: value.clone() for name, value in model.state_dict().items()})
    assert all(torch.equal(after[0][name], after[1][name]) for name in after[0])
