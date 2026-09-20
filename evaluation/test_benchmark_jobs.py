# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from evaluation.benchmark_jobs import plan_benchmark_jobs, check_responses
from evaluation.benchmark_model import BenchmarkController, BenchmarkExecutor
from evaluation.distributed import EvaluationCancelled
from training.backend.single_device import SingleDeviceBackend
from training.stopping import StopController


class Oracle:
    rank = 0
    world_size = 1
    device = torch.device("cpu")

    def loglikelihood(self, requests: list[Any], **kwargs: Any) -> list[tuple[float, bool]]:
        import random
        import numpy as np
        return [(float(-int(item.args[0]) - random.random() - np.random.random() - torch.rand(()).item()), False) for item in requests]

    def generate_until(self, requests: list[Any], **kwargs: Any) -> list[str]:
        return [str(item.args[0]) for item in requests]


def requests(count: int, method: str = "loglikelihood") -> list[Any]:
    return [SimpleNamespace(args=(str(index), {} if method == "generate_until" else "target"), task_name="task", doc_id=index // 2) for index in range(count)]


def test_planner_preserves_occurrences_document_affinity_and_kwargs() -> None:
    source = requests(19)
    source.append(source[0])
    first = plan_benchmark_jobs(source, seed=2, recurrence=0, call=0, method="loglikelihood")
    assert sorted(position for job in first for position in job.positions) == list(range(20))
    assert first[0].positions[:3] == [0, 1, 19]
    assert first == plan_benchmark_jobs(source, seed=2, recurrence=0, call=0, method="loglikelihood")
    original = SimpleNamespace(args=("x", {"until": ["stop"]}), task_name=None, doc_id=None)
    job = plan_benchmark_jobs([original], seed=0, recurrence=0, call=0, method="generate_until")[0]
    job.requests[0].args[1]["until"].append("changed")
    assert original.args[1]["until"] == ["stop"]


def test_controller_empty_calls_order_rng_and_cancellation() -> None:
    from evaluation.rng import preserve_rng
    backend = SingleDeviceBackend("cpu", "32")
    worker = Oracle()
    with preserve_rng(worker.device):
        executor = BenchmarkExecutor(backend, worker, StopController(backend), seed=3, recurrence=0)
        controller = BenchmarkController(executor)
        rng = torch.get_rng_state()
        assert controller.generate_until([]) == []
        assert controller.generate_until(requests(19, "generate_until")) == [str(index) for index in range(19)]
        assert len(controller.loglikelihood(requests(19))) == 19
        assert torch.equal(rng, torch.get_rng_state())
        assert controller.rank == 0 and controller.world_size == 1
    stopped = BenchmarkExecutor(backend, worker, StopController(backend, lambda: True), seed=3, recurrence=0)
    with pytest.raises(EvaluationCancelled):
        stopped.dispatch("generate_until", requests(1, "generate_until"))


@pytest.mark.parametrize("method,values", [("loglikelihood", [(float("nan"), True)]),
    ("generate_until", [1]), ("loglikelihood_rolling", [float("inf")]), ("generate_until", [])])
def test_bad_worker_responses_fail(method: str, values: Any) -> None:
    with pytest.raises(ValueError):
        check_responses(method, values, 1)
