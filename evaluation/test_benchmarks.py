# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests of the lm-eval integration with a stubbed `lm_eval`; scoring a task needs the `eval` extra and network, but
the encoding of a context is checked through the real `HFLM` where the extra is installed.
"""

import importlib
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from evaluation.benchmarks import (
    BOOTSTRAP_ITERS,
    DEFAULT_TASKS,
    EVAL_EXTRA_HINT,
    benchmarks_path,
    evaluate_on_benchmarks,
    flatten_results,
)
from evaluation.wrapper import hf_wrapper_around
from model.hf.modeling import RecurrentGPTForCausalLM
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer

RESULTS: dict[str, dict[str, Any]] = {
    "arc_easy": {"acc,none": 0.25, "acc_stderr,none": 0.02, "acc_norm,none": 0.3, "acc_norm_stderr,none": 0.02, "alias": "arc_easy"},
    "hellaswag": {"acc,none": 0.26, "acc_stderr,none": 0.01, "alias": "hellaswag", "samples": 40},
    "gsm8k": {"exact_match,strict-match": 0.1, "exact_match_stderr,strict-match": 0.01, "exact_match,flexible-extract": 0.2},
}


def stub_lm_eval(monkeypatch: pytest.MonkeyPatch, error: Exception | None = None) -> dict[str, Any]:
    """
    A fake `lm_eval` package in `sys.modules` recording the `HFLM` and `simple_evaluate` arguments.
    """

    import evaluation.benchmark_model  # noqa: F401
    import torch

    calls: dict[str, Any] = {}
    package = types.ModuleType("lm_eval")
    models = types.ModuleType("lm_eval.models")
    huggingface = types.ModuleType("lm_eval.models.huggingface")

    class HFLM:
        def __init__(self, **kwargs: Any) -> None:
            calls["hflm"] = kwargs
            self.rank, self.world_size = 0, 1
            self.device = torch.device("cpu")

    def simple_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls["evaluate"] = kwargs
        if error is not None:
            raise error
        return {"results": RESULTS, "versions": {"arc_easy": 1, "hellaswag": 1}, "n-shot": {"arc_easy": 0}}

    huggingface.HFLM = HFLM  # type: ignore[attr-defined]
    package.simple_evaluate = simple_evaluate  # type: ignore[attr-defined]
    package.models = models  # type: ignore[attr-defined]
    models.huggingface = huggingface  # type: ignore[attr-defined]
    for name, module in (("lm_eval", package), ("lm_eval.models", models), ("lm_eval.models.huggingface", huggingface)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_flatten_results_keeps_the_metrics_without_stderr() -> None:
    assert flatten_results(RESULTS, "mean") == {
        "benchmark/mean/arc_easy/acc": 0.25,
        "benchmark/mean/arc_easy/acc_norm": 0.3,
        "benchmark/mean/hellaswag/acc": 0.26,
        "benchmark/mean/hellaswag/samples": 40.0,
        "benchmark/mean/gsm8k/exact_match_strict-match": 0.1,  # two filters of one metric stay two entries
        "benchmark/mean/gsm8k/exact_match_flexible-extract": 0.2,
    }
    assert set(flatten_results(RESULTS, "4-8")) == {f"benchmark/4-8/{k}" for k in ("arc_easy/acc", "arc_easy/acc_norm", "hellaswag/acc", "hellaswag/samples", "gsm8k/exact_match_strict-match", "gsm8k/exact_match_flexible-extract")}


def test_evaluate_on_benchmarks_runs_the_harness_on_the_wrapper(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = stub_lm_eval(monkeypatch)
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    tiny_model.train()
    path = benchmarks_path(tmp_path, 7)
    metrics = evaluate_on_benchmarks(
        tiny_model, tokenizer, ["arc_easy", "hellaswag"], num_fewshot=2, limit=40, batch_size=4, out_path=path, step=7,
        recurrences=[None, [2, 2]], seed=5,
    )
    assert metrics == flatten_results(RESULTS, "mean") | flatten_results(RESULTS, "2-2") and tiny_model.training
    wrapper = calls["hflm"]["pretrained"]
    assert isinstance(wrapper, RecurrentGPTForCausalLM) and not wrapper.training
    assert wrapper.model.transformer.wte.weight.data_ptr() == tiny_model.transformer.wte.weight.data_ptr()
    assert calls["hflm"]["tokenizer"] is tokenizer.processor and calls["hflm"]["batch_size"] == 4
    # `add_bos_token=True` only asks the tokenizer for special tokens: what makes a context start like a training
    # row is the tokenizer prepending BOS itself, so check the object that was handed over, not the flag
    assert calls["hflm"]["tokenizer"].encode("x", add_special_tokens=True)[0] == tokenizer.bos_id
    assert calls["hflm"]["max_length"] == tiny_model.config.model_max_sequence_length
    assert calls["evaluate"]["tasks"] == ["arc_easy", "hellaswag"]
    assert (calls["evaluate"]["num_fewshot"], calls["evaluate"]["limit"]) == (2, 40)
    assert calls["evaluate"]["model"].__class__.__name__ == "HFLM"
    # The adapter seeds Torch after HFLM construction and disables the harness all-device reseed.
    assert calls["evaluate"]["torch_random_seed"] is None
    assert calls["evaluate"]["random_seed"] == 0
    assert calls["evaluate"]["numpy_random_seed"] == calls["evaluate"]["fewshot_random_seed"] == 1234
    assert calls["evaluate"]["log_samples"] is False and calls["evaluate"]["bootstrap_iters"] == BOOTSTRAP_ITERS
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "benchmarks" / "step-00000007.json"
    assert (record["step"], record["tasks"], record["limit"], record["num_fewshot"]) == (7, ["arc_easy", "hellaswag"], 40, 2)
    assert record["seed"] == 5
    assert record["rng_seeds"] == {
        "random_seed": 0, "numpy_random_seed": 1234, "torch_random_seed": 5, "fewshot_random_seed": 1234,
    }
    assert record["torch_seed_owner"] == "evaluation_cpu_model_device"
    assert record["recurrences"] == [None, [2, 2]] and record["metrics"] == metrics
    assert record["results"] == {"mean": RESULTS, "2-2": RESULTS} and record["versions"] == {"arc_easy": 1, "hellaswag": 1}


def test_num_fewshot_default_leaves_every_task_at_its_own(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The default -1 becomes lm-eval's `num_fewshot=None` (gsm8k stays 5-shot); 0 is passed as the 0 it is.
    """

    calls = stub_lm_eval(monkeypatch)
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"])
    assert calls["evaluate"]["num_fewshot"] is None
    evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"], num_fewshot=0)
    assert calls["evaluate"].get("num_fewshot") == 0
    with pytest.raises(ValueError, match="num_fewshot must be >= -1"):
        evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"], num_fewshot=-2)


def test_default_tasks_are_the_settings_default() -> None:
    """
    One tuple, not two that drift apart (`training.settings` names it; importing it here keeps that module free of
    torch, which an import the other way round would pull in).
    """

    from training.settings import DEFAULT_BENCHMARK_TASKS, Settings

    assert DEFAULT_TASKS is DEFAULT_BENCHMARK_TASKS
    factory = Settings.__dataclass_fields__["benchmark_tasks"].default_factory
    assert callable(factory)
    assert list(DEFAULT_TASKS) == factory()


def test_evaluate_on_benchmarks_errors(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    with pytest.raises(ValueError, match="no benchmark tasks"):
        evaluate_on_benchmarks(tiny_model, tokenizer, [])
    with pytest.raises(ValueError, match="no recurrence setting"):
        evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"], recurrences=[])
    with pytest.raises(ValueError, match="2 blocks"):
        evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"], recurrences=[[4, 4, 4]])
    monkeypatch.setitem(sys.modules, "lm_eval", None)  # an import of a None entry raises ImportError
    with pytest.raises(ImportError, match=re.escape(EVAL_EXTRA_HINT)):
        evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"])
    stub_lm_eval(monkeypatch, error=RuntimeError("no network"))
    tiny_model.train()
    with pytest.raises(RuntimeError, match="no network"):
        evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"])
    assert tiny_model.training  # the model is restored on the error path too


def test_real_harness_encodes_contexts_with_bos(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path) -> None:
    """
    The same check through the real `HFLM`, no stub and no network: the BOS has to survive lm-eval's own routing of
    `add_bos_token` as well as transformers' `add_bos_token` flag. A bump of either package that drops it fails
    here, instead of quietly costing every benchmark the few points a missing BOS costs.
    """

    pytest.importorskip("lm_eval", reason="needs lm_eval (uv sync)")
    # Not an importorskip: with lm_eval installed, an `lm_eval.models.huggingface` that will not import (a missing
    # accelerate) is the state this test is here to catch, because a benchmark run dies on the same import.
    # `Any`: lm_eval ships no stubs for the HFLM constructor, and the assertions below are about its behaviour
    huggingface: Any = importlib.import_module("lm_eval.models.huggingface")

    tokenizer = Tokenizer(tiny_tokenizer_dir)
    language_model = huggingface.HFLM(
        pretrained=hf_wrapper_around(tiny_model, tokenizer), tokenizer=tokenizer.processor, batch_size=2,
        add_bos_token=True, device="cpu",
    )
    assert language_model.tok_encode("tok_3 tok_4") == tokenizer.encode("tok_3 tok_4", bos=True)
    ids, mask = language_model.tok_batch_encode(["tok_3", "tok_3 tok_4 tok_5"])
    assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]  # left-padded to the longest context
    for row, row_mask in zip(ids.tolist(), mask.tolist(), strict=True):
        assert row[row_mask.index(1)] == tokenizer.bos_id


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_BENCHMARK_TESTS"), reason="set RUN_BENCHMARK_TESTS=1 (needs lm_eval and network)")
def test_real_harness_scores_arc_easy(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("lm_eval")
    metrics = evaluate_on_benchmarks(
        tiny_model, Tokenizer(tiny_tokenizer_dir), ["arc_easy"], limit=4, batch_size=2, out_path=benchmarks_path(tmp_path, 1)
    )
    assert "benchmark/mean/arc_easy/acc" in metrics and 0.0 <= metrics["benchmark/mean/arc_easy/acc"] <= 1.0


def test_harness_without_seed_arguments_fails_explicitly(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_lm_eval(monkeypatch)

    def incompatible(model: object, tasks: object) -> None:
        pytest.fail("unsupported harness must be rejected before evaluation")

    monkeypatch.setattr(sys.modules["lm_eval"], "simple_evaluate", incompatible)
    with pytest.raises(RuntimeError, match="lacks required RNG seed arguments.*torch_random_seed"):
        evaluate_on_benchmarks(tiny_model, Tokenizer(tiny_tokenizer_dir), ["offline"])


def test_metadata_marks_unavailable_harness_settings(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_lm_eval(monkeypatch)
    path = tmp_path / "result.json"
    evaluate_on_benchmarks(tiny_model, Tokenizer(tiny_tokenizer_dir), ["offline"], batch_size="auto:2", out_path=path)
    record = json.loads(path.read_text())["execution_metadata"]
    assert record["schema_version"] == 1
    settings = record["recurrences"]["mean"]
    assert settings["batching"]["requested"] == "auto:2"
    assert settings["batching"]["configured"] == {"value": None, "source": "unavailable"}
    assert settings["scoring_cache"]["response_cache_enabled"] == {"value": None, "source": "unavailable"}
    assert settings["precision"]["requested_policy"] is None
    assert settings["precision"]["hflm_mixed_precision_dtype"]["source"] == "unavailable"
    assert settings["generation"]["effective_per_request_use_cache"]["source"] == "unavailable"
    assert record["dependencies"]["triton"]["status"] == "not_applicable"


def test_metadata_reads_post_harness_automatic_selection_without_leaking_cache_paths(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from model.execution import ExecutionPolicy

    stub_lm_eval(monkeypatch)
    count = 0

    def simple_evaluate(**kwargs: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        lm = kwargs["model"]
        lm.batch_size = "auto"
        lm.batch_sizes = {0: count + 1, 1: count + 2}
        lm.max_batch_size = 64
        lm.mixed_precision_dtype = torch.bfloat16
        lm.softmax_dtype = None
        lm.max_length = 99
        lm.logits_cache = False
        return {"results": RESULTS, "config": {"use_cache": "/private/account/cache.db", "batch_sizes": [count + 1]}}

    monkeypatch.setattr(sys.modules["lm_eval"], "simple_evaluate", simple_evaluate)
    path = tmp_path / "result.json"
    evaluate_on_benchmarks(tiny_model, Tokenizer(tiny_tokenizer_dir), ["offline"], batch_size="auto", out_path=path,
                           recurrences=[None, [1, 1]], execution_policy=ExecutionPolicy("bf16-mixed"))
    text = path.read_text()
    assert "/private" not in text
    record = json.loads(text)
    assert record["execution_precision"] == "bf16-mixed"
    settings = record["execution_metadata"]["recurrences"]
    for label, first in (("mean", 2), ("1-1", 3)):
        metadata = settings[label]
        assert metadata["batching"]["automatic_schedule"]["value"] == {"0": first, "1": first + 1}
        assert metadata["batching"]["harness_reported_batch_sizes"] == [first]
        assert metadata["precision"]["hflm_mixed_precision_dtype"]["value"] == "torch.bfloat16"
        assert metadata["precision"]["hflm_softmax_dtype"] == {"value": None, "source": "hflm.softmax_dtype"}
        assert metadata["context"]["effective_cap"]["value"] == 99
        assert metadata["scoring_cache"]["response_cache_enabled"]["value"] is True
        assert metadata["scoring_cache"]["logits_cache"]["value"] is False


def test_real_local_hflm_metadata(
    tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lm_eval")
    huggingface: Any = importlib.import_module("lm_eval.models.huggingface")
    from evaluation.benchmarks import _import_lm_eval
    from model.execution import ExecutionPolicy
    package, _ = _import_lm_eval()
    original = package.simple_evaluate

    # Retain the real default invocation signature, but do no task/dataset loading.
    import functools

    @functools.wraps(original)
    def local_evaluate(**kwargs: Any) -> dict[str, Any]:
        assert isinstance(kwargs["model"], huggingface.HFLM)
        return {"results": RESULTS}

    monkeypatch.setattr(package, "simple_evaluate", local_evaluate)
    path = tmp_path / "result.json"
    evaluate_on_benchmarks(tiny_model, Tokenizer(tiny_tokenizer_dir), ["offline"], batch_size=2, out_path=path,
                           execution_policy=ExecutionPolicy("bf16-mixed"))
    record = json.loads(path.read_text())["execution_metadata"]
    metadata = record["recurrences"]["mean"]
    assert metadata["batching"]["configured"] == {"value": 2, "source": "hflm.batch_size"}
    assert metadata["precision"]["hflm_mixed_precision_dtype"]["value"] == "torch.bfloat16"
    assert metadata["precision"]["session_autocast_enabled"] is True
    assert metadata["context"]["effective_cap"]["value"] == tiny_model.config.model_max_sequence_length
    assert metadata["scoring_cache"]["logits_cache"]["value"] is True
    assert metadata["scoring_cache"]["response_cache_enabled"] == {
        "value": False, "source": "simple_evaluate.signature_default",
    }
    assert metadata["scoring_cache"]["cache_requests"]["value"] is False
    assert metadata["generation"]["wrapper_generation_config_use_cache"] is None  # current local HF config leaves it unset
    for package_name in ("torch", "transformers", "lm_eval"):
        assert record["dependencies"][package_name]["version"]


def test_dependency_metadata_does_not_import_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.metadata import dependency_versions

    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.delitem(sys.modules, "lm_eval", raising=False)
    dependencies = dependency_versions(custom_kernels=True)
    assert dependencies["triton"] == {"version": None, "status": "not_loaded"}
    assert dependencies["lm_eval"] == {"version": None, "status": "not_loaded"}
    assert "triton" not in sys.modules and "lm_eval" not in sys.modules
