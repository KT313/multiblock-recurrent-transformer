# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""World-size-independent benchmark jobs; positions identify occurrences, not text."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from evaluation.distributed import MAX_JOB_BYTES, check_payload

PROTOCOL = "fixed_document_jobs_v1"
TARGET_REQUESTS = 8
MAX_GROUP_REQUESTS = 128
METHODS = ("loglikelihood", "loglikelihood_rolling", "generate_until")


@dataclass(frozen=True)
class WorkerRequest:
    args: tuple[Any, ...]
    task_name: str | None


@dataclass(frozen=True)
class BenchmarkJob:
    index: int
    positions: list[int]
    requests: list[WorkerRequest]
    seed: int
    digest: str


def plan_benchmark_jobs(requests: list[Any], *, seed: int, recurrence: int, call: int, method: str) -> list[BenchmarkJob]:
    if method not in METHODS:
        raise ValueError(f"unsupported benchmark method {method}")
    groups: dict[tuple[Any, ...], list[tuple[int, WorkerRequest]]] = {}
    for position, request in enumerate(requests):
        task, doc = getattr(request, "task_name", None), getattr(request, "doc_id", None)
        key = (task, doc) if task is not None and doc is not None else ("position", position)
        args = tuple(request.args)
        arity = 1 if method == "loglikelihood_rolling" else 2
        if (len(args) != arity or not isinstance(args[0], str)
                or (method == "loglikelihood" and not isinstance(args[1], str))
                or (method == "generate_until" and not isinstance(args[1], dict))):
            raise ValueError(f"{method} requires text requests and plain generation options")
        try:
            json.dumps(args, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("benchmark requests must contain only finite JSON data, never tensors") from error
        item = WorkerRequest(deepcopy(args), task)
        check_payload(item, MAX_JOB_BYTES)
        groups.setdefault(key, []).append((position, item))

    chunks: list[list[tuple[int, WorkerRequest]]] = []
    current: list[tuple[int, WorkerRequest]] = []
    for group in groups.values():
        if len(group) > MAX_GROUP_REQUESTS:
            raise ValueError(f"one benchmark document exceeds {MAX_GROUP_REQUESTS} requests")
        if current and len(current) + len(group) > TARGET_REQUESTS:
            chunks.append(current)
            current = []
        current.extend(group)
    if current:
        chunks.append(current)
    jobs = []
    for index, chunk in enumerate(chunks):
        # repr is used only for diagnostics; seeds depend on explicit stable integers/method.
        job_seed = int.from_bytes(hashlib.sha256(f"{PROTOCOL}:{seed}:{recurrence}:{call}:{method}:{index}".encode()).digest()[:8], "big") % (2**63 - 1)
        digest = hashlib.sha256(repr(chunk).encode()).hexdigest()
        job = BenchmarkJob(index, [item[0] for item in chunk], [item[1] for item in chunk], job_seed, digest)
        check_payload(job, MAX_JOB_BYTES)
        jobs.append(job)
    return jobs


def check_responses(method: str, values: Any, count: int) -> list[Any]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{method} returned the wrong response count/type")
    for value in values:
        if method == "generate_until":
            valid = isinstance(value, str)
        elif method == "loglikelihood":
            valid = (isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], bool)
                     and isinstance(value[0], (int, float)) and math.isfinite(value[0]))
        else:
            valid = isinstance(value, (int, float)) and math.isfinite(value)
        if not valid:
            raise ValueError(f"{method} returned an invalid response: {value!r}")
    check_payload(values, MAX_JOB_BYTES)
    return values
