# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Generate fixed sample batches across resident replicas, then publish in input order."""
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluation.distributed import agree_on_phase, check_payload, exchange, finish_publication, poll_stop, share_from_main
from evaluation.prompts import Prompt
from evaluation.sample_helpers import GeneratedSample, save_generated_samples, select_fitting_prompts
from evaluation.samples import generate_prompt_batch
from evaluation.session import inference_session
from evaluation.wrapper import Recurrence, check_recurrence
from model.model import RecurrentGPT
from training.backend.base import Backend
from training.data.tokenizer import Tokenizer
from training.failure import FatalHandler, fatal_errors
from training.stopping import StopController
from training.sample_settings import normalize_sample_temperatures


@dataclass(frozen=True)
class SampleJob:
    index: int
    recurrence: Recurrence
    seed: int
    temperature: float
    batch: list[tuple[Prompt, list[int]]]


@dataclass
class SamplePhaseResult:
    completed: bool
    samples: list[GeneratedSample]
    jobs_per_rank: list[int]


def plan_sample_jobs(
    fitting: list[tuple[Prompt, list[int]]], recurrences: Sequence[Recurrence], batch_size: int, temperatures: Sequence[float],
) -> list[SampleJob]:
    jobs: list[SampleJob] = []
    for temperature in temperatures:
        for recurrence in recurrences:
            for start in range(0, len(fitting), batch_size):
                jobs.append(SampleJob(len(jobs), recurrence, start, temperature, fitting[start:start + batch_size]))
    return jobs


def generate_distributed_samples(
    backend: Backend, model: RecurrentGPT, tokenizer: Tokenizer, out_path: Path, *, step: int,
    prompts: Sequence[Prompt], recurrences: Sequence[Recurrence], stop: StopController,
    batch_size: int = 8, max_new_tokens: int = 64, temperature: float | list[float] = 0, use_cache: bool = True,
    on_fatal_error: FatalHandler | None = None,
) -> SamplePhaseResult:
    with fatal_errors(on_fatal_error):
        temperatures = normalize_sample_temperatures(temperature)
        if batch_size < 1 or max_new_tokens < 1 or not recurrences:
            raise ValueError("sample batch size/token limit must be positive and recurrences nonempty")
        for recurrence in recurrences:
            check_recurrence(recurrence, model)
        agree_on_phase(backend, model, tokenizer, {
            "phase": "samples", "step": step, "recurrences": recurrences, "batch_size": batch_size,
            "max_new_tokens": max_new_tokens, "temperature": temperatures, "use_cache": use_cache, "seed": 0,
        })
        counts = [0] * backend.world_size
        if poll_stop(stop, "before sample jobs"):
            return SamplePhaseResult(False, [], counts)

        # rank zero formats and filters once; batch offsets are the historical seed units
        jobs = None
        if backend.is_main:
            fitting = select_fitting_prompts(prompts, tokenizer, max_new_tokens, model.config.model_max_sequence_length)
            jobs = plan_sample_jobs(fitting, recurrences, batch_size, temperatures)
        plan = share_from_main(backend, jobs)
        samples: list[GeneratedSample] = []
        for start in range(0, len(plan), backend.world_size):
            index = start + backend.rank
            reply: tuple[int, list[GeneratedSample]] | None = None
            if index < len(plan):
                job = plan[index]
                with inference_session(model, job.recurrence, execution_policy=backend.execution_policy,
                                       on_fatal_error=on_fatal_error) as session:
                    values = generate_prompt_batch(
                        session, session.hf_wrapper(tokenizer), tokenizer, job.batch, recurrence=job.recurrence,
                        seed=job.seed, max_new_tokens=max_new_tokens, temperature=job.temperature, use_cache=use_cache,
                    )
                reply = (job.index, values)
            replies = exchange(backend, reply)
            for rank, received in enumerate(replies):
                expected = start + rank
                if expected >= len(plan):
                    if received is not None:
                        raise RuntimeError("idle sample worker returned output")
                    continue
                if received is None or received[0] != expected or len(received[1]) != len(plan[expected].batch):
                    raise RuntimeError("sample worker returned wrong job/count")
                counts[rank] += 1
                if backend.is_main:
                    samples.extend(received[1])
            if poll_stop(stop, "after sample round") and start + backend.world_size < len(plan):
                return SamplePhaseResult(False, [], counts)

        # publish only complete phases; peers cannot start training before publication
        check_payload(samples)
        if backend.is_main:
            save_generated_samples(samples, out_path, step=step, max_new_tokens=max_new_tokens,
                                   seed=0, batch_size=batch_size, use_cache=use_cache, execution_policy=backend.execution_policy)
        finish_publication(backend)
        poll_stop(stop, "after sample publication")
        return SamplePhaseResult(True, samples, counts)
