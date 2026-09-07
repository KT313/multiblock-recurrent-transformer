# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests of the lm-eval integration with a stubbed `lm_eval` (the real harness needs the `eval` extra and network).
"""

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

    calls: dict[str, Any] = {}
    package = types.ModuleType("lm_eval")
    models = types.ModuleType("lm_eval.models")
    huggingface = types.ModuleType("lm_eval.models.huggingface")

    class HFLM:
        def __init__(self, **kwargs: Any) -> None:
            calls["hflm"] = kwargs

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
    assert calls["hflm"]["add_bos_token"] is True  # scored on inputs shaped like the training rows
    assert calls["hflm"]["max_length"] == tiny_model.config.model_max_sequence_length
    assert calls["evaluate"]["tasks"] == ["arc_easy", "hellaswag"]
    assert (calls["evaluate"]["num_fewshot"], calls["evaluate"]["limit"]) == (2, 40)
    assert calls["evaluate"]["model"].__class__.__name__ == "HFLM"
    # lm-eval reseeds torch on every call, so the seed of the isolated inference only reaches it as an argument
    assert calls["evaluate"]["torch_random_seed"] == 5
    assert calls["evaluate"]["log_samples"] is False and calls["evaluate"]["bootstrap_iters"] == BOOTSTRAP_ITERS
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "benchmarks" / "step-00000007.json"
    assert (record["step"], record["tasks"], record["limit"], record["num_fewshot"]) == (7, ["arc_easy", "hellaswag"], 40, 2)
    assert record["seed"] == 5
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
    assert calls["evaluate"]["num_fewshot"] == 0
    with pytest.raises(ValueError, match="num_fewshot must be >= -1"):
        evaluate_on_benchmarks(tiny_model, tokenizer, ["arc_easy"], num_fewshot=-2)


def test_default_tasks_are_the_settings_default() -> None:
    """
    One tuple, not two that drift apart (`training.settings` names it; importing it here keeps that module free of
    torch, which an import the other way round would pull in).
    """

    from training.settings import DEFAULT_BENCHMARK_TASKS, Settings

    assert DEFAULT_TASKS is DEFAULT_BENCHMARK_TASKS
    assert list(DEFAULT_TASKS) == Settings.__dataclass_fields__["benchmark_tasks"].default_factory()


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


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_BENCHMARK_TESTS"), reason="set RUN_BENCHMARK_TESTS=1 (needs the eval extra and network)")
def test_real_harness_scores_arc_easy(tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("lm_eval")
    metrics = evaluate_on_benchmarks(
        tiny_model, Tokenizer(tiny_tokenizer_dir), ["arc_easy"], limit=4, batch_size=2, out_path=benchmarks_path(tmp_path, 1)
    )
    assert "benchmark/mean/arc_easy/acc" in metrics and 0.0 <= metrics["benchmark/mean/arc_easy/acc"] <= 1.0
