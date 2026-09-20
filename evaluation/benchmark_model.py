# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Rank-zero public LM controller and a synchronous worker service loop."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
import time
from typing import Any

import numpy as np
import torch
from lm_eval.api.model import LM

from evaluation.benchmark_jobs import BenchmarkJob, METHODS, check_responses, plan_benchmark_jobs
from evaluation.distributed import EvaluationCancelled, exchange, poll_stop, share_from_main
from evaluation.rng import preserve_rng, seed_model_rng
from training.backend.base import Backend
from training.failure import FatalHandler, fatal_errors
from training.stopping import StopController


@dataclass(frozen=True)
class Command:
    sequence: int
    method: str
    jobs: list[BenchmarkJob]


@dataclass(frozen=True)
class Reply:
    sequence: int
    rank: int
    job: int | None
    values: list[Any]


@dataclass
class WorkerStats:
    jobs: int = 0
    requests: int = 0
    seconds: float = 0
    plan_digest: str = ""
    seeds: list[int] = field(default_factory=list)
    input_tokens_with_padding: int = 0
    generated_tokens_with_padding: int = 0
    process_peak_cuda_bytes: int | None = None


class BenchmarkExecutor:
    def __init__(
        self, backend: Backend, worker: Any, stop: StopController, *, seed: int, recurrence: int,
        on_fatal_error: FatalHandler | None = None,
    ) -> None:
        self.backend, self.worker, self.stop = backend, worker, stop
        self.seed, self.recurrence, self.on_fatal_error = seed, recurrence, on_fatal_error
        self.sequence = 0
        self.call = 0
        self.plan_digest = ""
        self.stats = WorkerStats()
        if worker.rank != 0 or worker.world_size != 1:
            raise ValueError("local benchmark workers must be single-process HFLM instances")

    def execute_round(self, command: Command) -> list[Reply]:
        if command.sequence != self.sequence or command.method not in METHODS:
            raise RuntimeError("benchmark command sequence/method mismatch")
        self.sequence += 1
        job = command.jobs[self.backend.rank] if self.backend.rank < len(command.jobs) else None
        values: list[Any] = []
        if job is not None:
            started = time.monotonic()
            with preserve_rng(self.worker.device, self.on_fatal_error), fatal_errors(self.on_fatal_error):
                random.seed(job.seed)
                np.random.seed(job.seed % 2**32)
                seed_model_rng(job.seed, self.worker.device)
                values = check_responses(command.method, getattr(self.worker, command.method)(
                    job.requests, disable_tqdm=True), len(job.requests))
            self.stats.input_tokens_with_padding = getattr(self.worker, "inference_input_tokens_with_padding", 0)
            self.stats.generated_tokens_with_padding = getattr(self.worker, "inference_generated_tokens_with_padding", 0)
            if self.worker.device.type == "cuda":
                self.stats.process_peak_cuda_bytes = torch.cuda.max_memory_allocated(self.worker.device)
            self.stats.jobs += 1
            self.stats.requests += len(job.requests)
            self.stats.seconds += time.monotonic() - started
            # Bounded diagnostics retain a rolling digest and a small seed preview.
            self.stats.plan_digest = hashlib.sha256((self.stats.plan_digest + job.digest).encode()).hexdigest()
            if len(self.stats.seeds) < 16:
                self.stats.seeds.append(job.seed)
        replies = exchange(self.backend, Reply(command.sequence, self.backend.rank, None if job is None else job.index, values))
        for rank, reply in enumerate(replies):
            expected = command.jobs[rank] if rank < len(command.jobs) else None
            if (reply.rank != rank or reply.sequence != command.sequence
                    or reply.job != (None if expected is None else expected.index)):
                raise RuntimeError("benchmark response sequence/owner/job mismatch")
            check_responses(command.method, reply.values, 0 if expected is None else len(expected.requests))
        poll_stop(self.stop, "after benchmark round")
        return replies

    def dispatch(self, method: str, requests: list[Any]) -> list[Any]:
        jobs = plan_benchmark_jobs(requests, seed=self.seed, recurrence=self.recurrence, call=self.call, method=method)
        self.plan_digest = hashlib.sha256((self.plan_digest + method + str(self.call)
                                          + repr([(job.digest, job.seed) for job in jobs])).encode()).hexdigest()
        self.call += 1
        missing = object()
        ordered: list[Any] = [missing] * len(requests)
        for start in range(0, len(jobs), self.backend.world_size):
            command = share_from_main(self.backend, Command(self.sequence, method, jobs[start:start + self.backend.world_size]))
            replies = self.execute_round(command)
            if self.stop.requested:
                raise EvaluationCancelled()
            for rank, job in enumerate(command.jobs):
                for position, value in zip(job.positions, replies[rank].values, strict=True):
                    if not 0 <= position < len(ordered) or ordered[position] is not missing:
                        raise RuntimeError("benchmark duplicate/out-of-range response position")
                    ordered[position] = value
        if any(value is missing for value in ordered):
            raise RuntimeError("benchmark missing response")
        return ordered

    def finish(self, cancelled: bool) -> None:
        share_from_main(self.backend, Command(self.sequence, "cancel" if cancelled else "finish", []))

    def serve(self) -> bool:
        while True:
            command: Command = share_from_main(self.backend, None)
            if command.sequence != self.sequence:
                raise RuntimeError("benchmark service sequence mismatch")
            if command.method in ("finish", "cancel"):
                if command.jobs:
                    raise RuntimeError("finish command contains work")
                return command.method == "cancel"
            self.execute_round(command)


class BenchmarkController(LM):  # type: ignore[misc]
    """The official evaluator sees one LM; only public inference calls are distributed."""
    def __init__(self, executor: BenchmarkExecutor) -> None:
        super().__init__()
        self.executor = executor
        self.worker = executor.worker

    def __getattr__(self, name: str) -> Any:
        return getattr(self.worker, name)

    @property
    def tokenizer_name(self) -> str:
        return str(self.worker.tokenizer_name)

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        return str(self.worker.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt))

    def chat_template(self, chat_template: bool | str = False) -> Any:
        return self.worker.chat_template(chat_template)

    def loglikelihood(self, requests: list[Any]) -> list[Any]:
        return self.executor.dispatch("loglikelihood", requests)

    def loglikelihood_rolling(self, requests: list[Any]) -> list[Any]:
        return self.executor.dispatch("loglikelihood_rolling", requests)

    def generate_until(self, requests: list[Any]) -> list[Any]:
        return self.executor.dispatch("generate_until", requests)
