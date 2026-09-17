# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded torchrun qualification of the production distributed inference paths."""
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from evaluation.benchmark_model import BenchmarkExecutor
from evaluation.distributed import finish_publication
from evaluation.distributed_benchmarks import evaluate_distributed_benchmarks
from evaluation.distributed_samples import generate_distributed_samples
from evaluation.prompts import Prompt
from evaluation.samples import generate_samples
from evaluation.test_benchmark_jobs import Oracle, requests
from evaluation.testing import build_offline_group, build_offline_task
from model import build_model
from training.backend.ddp import DDPBackend
from training.backend.single_device import SingleDeviceBackend
from training.data.tokenizer import Tokenizer
from training.failure import exit_failed_worker, fatal_errors
from training.stopping import StopController


class LocalBackend(SingleDeviceBackend):
    """Serial comparison on the already assigned CPU, without constructing a second group."""
    def _check_launch_environment(self) -> None:
        pass


def main() -> None:
    torch.set_num_threads(1)
    root, token_path, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    use_cuda = len(sys.argv) > 4 and sys.argv[4] == "cuda"
    backend = DDPBackend(device=None if use_cuda else "cpu", precision="bf16-mixed" if use_cuda else "32")
    with fatal_errors(exit_failed_worker):
        model = build_model("config/model_architecture/tiny.yaml", use_custom_kernels=False)
        wrapped = backend.setup_model(model)
        model = backend.plain_model(wrapped)
        tokenizer = Tokenizer(token_path)
        local = LocalBackend(str(backend.device), backend.precision)
        reference: list[Any] = []
        prompts = [Prompt("tok_3 " * (1 + index % 3), kind="continuation") for index in range(5)]
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(backend.device) if use_cuda else None
        for cache in (True, False):
            for temperature in (0.0, 0.7):
                for recurrence in (None, [1, 1]):
                    reference.extend(generate_samples(
                        model, tokenizer, prompts, batch_size=2, max_new_tokens=2, recurrence=recurrence,
                        use_cache=cache, temperature=temperature, execution_policy=backend.execution_policy,
                    ))
        results: list[Any] = []
        for cache in (True, False):
            result = generate_distributed_samples(
                backend, model, tokenizer, root / f"samples-{cache}.jsonl", step=0, prompts=prompts,
                recurrences=[None, [1, 1]], batch_size=2, max_new_tokens=2, use_cache=cache, temperature=[0.0, 0.7],
                stop=StopController(backend), on_fatal_error=exit_failed_worker,
            )
            if backend.is_main:
                rows = [json.loads(line) for line in (root / f"samples-{cache}.jsonl").read_text().splitlines()]
                assert [row["decoding"]["temperature"] for row in rows] == [sample.temperature for sample in result.samples]
            results.extend(result.samples)
            assert result.completed
            assert all(count > 0 for count in result.jobs_per_rank)
        if backend.is_main:
            assert results == reference
        assert torch.equal(rng, torch.get_rng_state()) and model.training

        # all-skipped and cancellation on a non-main/idle worker use the same collective order
        skipped = generate_distributed_samples(
            backend, model, tokenizer, root / "empty.jsonl", step=0, prompts=[Prompt("tok_3 " * 1000)],
            recurrences=[None], max_new_tokens=2, stop=StopController(backend), on_fatal_error=exit_failed_worker,
        )
        assert skipped.completed and not skipped.samples
        counter = 0
        def request_stop() -> bool:
            nonlocal counter
            counter += 1
            return backend.rank == 1 and counter >= 2
        cancelled = generate_distributed_samples(
            backend, model, tokenizer, root / "cancelled.jsonl", step=0, prompts=prompts, batch_size=1,
            recurrences=[None, [1, 1]], max_new_tokens=2, stop=StopController(backend, request_stop),
            on_fatal_error=exit_failed_worker,
        )
        assert not cancelled.completed and not (root / "cancelled.jsonl").exists()

        # fixed jobs must be invariant to rank assignment, including Python/NumPy/Torch RNG
        worker = Oracle()
        if mode == "fail" and backend.rank == 1:
            def fail(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("injected inference worker failure")
            worker.loglikelihood = fail  # type: ignore[method-assign]
        executor = BenchmarkExecutor(backend, worker, StopController(backend), seed=19, recurrence=0,
                                     on_fatal_error=exit_failed_worker)
        if backend.is_main:
            actual = executor.dispatch("loglikelihood", requests(19))
            serial = BenchmarkExecutor(local, Oracle(), StopController(local), seed=19, recurrence=0)
            assert actual == serial.dispatch("loglikelihood", requests(19))
            assert executor.dispatch("generate_until", []) == []
            assert executor.dispatch("generate_until", requests(1, "generate_until")) == ["0"]
            executor.finish(False)
        else:
            assert not executor.serve()
        finish_publication(backend)

        # use the real official evaluator, replacing only dataset discovery with in-memory task fixtures
        from lm_eval.tasks import TaskManager
        load_tasks = TaskManager.load
        def load_offline(manager: Any, specs: Any) -> Any:
            if mode == "taskfail":
                raise RuntimeError("injected rank-zero task preparation failure")
            return load_tasks(manager, [build_offline_group() if name == "offline_group" else
                                        build_offline_task(name.removeprefix("offline_"), 9) for name in specs])
        with patch.object(TaskManager, "load", load_offline):
            stop_calls = 0
            def stop_benchmark() -> bool:
                nonlocal stop_calls
                stop_calls += 1
                return backend.rank == 1 and stop_calls >= 2
            cancelled_path = root / "cancelled-benchmark.json"
            if backend.is_main:
                cancelled_path.write_text("previous complete result")
            finish_publication(backend)
            cancelled_b = evaluate_distributed_benchmarks(
                backend, model, tokenizer, ["offline_multiple_choice"], stop=StopController(backend, stop_benchmark),
                recurrences=[None], out_path=cancelled_path, step=0, limit=9, batch_size=2,
                on_fatal_error=exit_failed_worker,
            )
            assert not cancelled_b.completed and cancelled_path.read_text() == "previous complete result"
            for limit in (1, 9):
                tasks = ["offline_group", "offline_generate_until", "offline_loglikelihood_rolling"]
                result_b = evaluate_distributed_benchmarks(
                    backend, model, tokenizer, tasks, stop=StopController(backend), recurrences=[None, [1, 1]],
                    out_path=root / f"bench-{limit}.json", step=0, batch_size=2, limit=limit,
                    on_fatal_error=exit_failed_worker,
                )
                assert result_b.completed
                if backend.is_main:
                    serial_b = evaluate_distributed_benchmarks(
                        local, model, tokenizer, tasks, stop=StopController(local), recurrences=[None, [1, 1]],
                        out_path=root / f"serial-{limit}.json", step=0, batch_size=2, limit=limit,
                    )
                    assert result_b.metrics == serial_b.metrics
        assert torch.equal(rng, torch.get_rng_state()) and model.training
        finish_publication(backend)
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(backend.device))
        (root / f"rank-{backend.rank}.json").write_text(json.dumps({"completed": True, "world_size": backend.world_size}))
    backend.shutdown()


if __name__ == "__main__":
    main()
