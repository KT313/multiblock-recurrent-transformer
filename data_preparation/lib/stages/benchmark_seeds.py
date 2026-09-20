# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Read-only complete-example adapters and a bounded, replayable packed-key spool.

Exactly two renderings: an instruction tuple (prompt, empty input, labelled answer),
and pretrain text ``prompt + '\\nAnswer: ' + labelled answer``. Prompts include all
choices; no question-only/answer-only keys. Global key normalization is unchanged.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import Any, BinaryIO, cast

from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.benchmarks import BENCHMARKS, BLOOM_BENCHMARKS, bloom_benchmark_names
from data_preparation.lib.stages.exact_dedup import BLOOM_MAX_LOAD, expected_items, memory_mb_for
from data_preparation.lib.stages.global_dedup import GlobalFrontier, global_key

log = get_logger(__name__)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"benchmark example requires nonempty string {field}")
    return value


def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"benchmark example requires at least two {field}")
    return [_text(item, field) for item in value]


def complete_example(name: str, example: dict[str, Any]) -> tuple[str, str]:
    """Explicit benchmark structure -> complete prompt and answer; metadata is ignored.

    ARC preserves original choice labels. Others label choices A..D (HellaSwag/MMLU)
    or 1..2 (WinoGrande). MMLU subject and HellaSwag activity are part of the prompt.
    Redundant HellaSwag ctx_a/ctx_b are represented by its provided full ctx.
    """
    if name == "arc_challenge":
        prompt = _text(example.get("question"), "question")
        choices = example.get("choices")
        if not isinstance(choices, dict):
            raise ValueError("ARC benchmark requires structured choices {label, text}")
        labels = _strings(choices.get("label"), "choice labels")
        texts = _strings(choices.get("text"), "choice texts")
        answer = _text(example.get("answerKey"), "answerKey")
    elif name in ("mmlu", "hellaswag"):
        if name == "mmlu":
            prompt = f"Subject: {_text(example.get('subject'), 'subject')}\n{_text(example.get('question'), 'question')}"
            texts = _strings(example.get("choices"), "choices")
            index = example.get("answer")
        else:
            prompt = f"Activity: {_text(example.get('activity_label'), 'activity_label')}\n{_text(example.get('ctx'), 'ctx')}"
            texts = _strings(example.get("endings"), "endings")
            label = example.get("label")
            if not isinstance(label, str) or label not in ("0", "1", "2", "3"):
                raise ValueError("HellaSwag benchmark requires a labelled validation example (label 0..3)")
            index = int(label)
        if len(texts) != 4 or type(index) is not int or not 0 <= index < 4:
            raise ValueError(f"{name}: benchmark requires four choices and answer index 0..3")
        labels = list("ABCD")
        answer = labels[index]
    elif name == "winogrande":
        prompt = _text(example.get("sentence"), "sentence")
        texts = [_text(example.get(field), field) for field in ("option1", "option2")]
        labels = ["1", "2"]
        answer = _text(example.get("answer"), "answer")
    else:
        raise ValueError(f"unknown Bloom benchmark {name!r}")
    if len(labels) != len(texts) or len(set(labels)) != len(labels) or answer not in labels:
        raise ValueError(f"{name}: inconsistent benchmark choices/answer")
    prompt += "\n" + "\n".join(f"{label}. {text}" for label, text in zip(labels, texts, strict=True))
    return prompt, f"{answer}. {texts[labels.index(answer)]}"


def example_keys(name: str, example: dict[str, Any]) -> tuple[int, int]:
    prompt, answer = complete_example(name, example)
    return (global_key("instruct", {"instruction": prompt, "input": "", "output": answer}),
            global_key("pretrain", {"text": f"{prompt}\nAnswer: {answer}"}))


def benchmark_records(name: str, *, hf_token: str | None = None) -> Iterator[dict[str, Any]]:
    """Stream the pinned labelled split; load/iteration errors propagate without fallback."""
    from datasets import load_dataset

    registry, split = BLOOM_BENCHMARKS[name]
    definition = BENCHMARKS[registry]
    records: Any = load_dataset(definition.hf_id, definition.config, split=split,
                               revision=definition.revision, streaming=True, token=hf_token)
    yield from records


@dataclass
class BenchmarkSeeds:
    """Temporary binary spool: constant Python memory even for a large benchmark stream."""
    file: BinaryIO
    count: int = 0
    digest: str = hashlib.sha256().hexdigest()

    def keys(self) -> Iterator[int]:
        self.file.seek(0)
        while data := self.file.read(8):
            if len(data) != 8:
                raise ValueError("corrupt benchmark seed spool")
            yield int.from_bytes(data, "big", signed=True)

    def frontier(self, order: tuple[str, ...], memory_mb: int) -> GlobalFrontier:
        return GlobalFrontier(order, memory_mb, preseed_count=self.count, preseed_digest=self.digest)


@contextmanager
def load_benchmark_seeds(
    names: list[str], *, memory_mb: int, hf_token: str | None = None, should_stop: StopCheck | None = None,
) -> Iterator[BenchmarkSeeds]:
    """Fully validate and capacity-check before publishing any new dataset output.

    Repeated rows/keys are inserted directly and counted conservatively, never admitted
    through Bloom membership. Every record's representations therefore reach the filter.
    """
    names = bloom_benchmark_names(names)
    with TemporaryFile(mode="w+b") as file:
        seeds = BenchmarkSeeds(cast(BinaryIO, file))
        digest = hashlib.sha256()
        limit = BLOOM_MAX_LOAD * expected_items(memory_mb)
        for name in names:
            check_stop(should_stop)
            before = seeds.count
            for example in benchmark_records(name, hf_token=hf_token):
                check_stop(should_stop)
                for key in example_keys(name, example):
                    seeds.count += 1
                    if seeds.count > limit:
                        raise ValueError(
                            f"benchmark seeds exceed global Bloom capacity; raise bloom_dedup_memory_mb "
                            f"to at least {memory_mb_for(seeds.count)} and replay"
                        )
                    data = key.to_bytes(8, "big", signed=True)
                    file.write(data)
                    digest.update(data)
            if seeds.count == before:
                raise ValueError(f"benchmark {name!r} loaded no examples; refusing incomplete exclusion")
            log.info("benchmark seed set %s: %d complete-example key insertions", name, seeds.count - before)
        seeds.digest = digest.hexdigest()
        if names:
            log.info("benchmark seeds %s: %d keys; global Bloom %d MiB, nominal capacity %d, max load %g",
                     names, seeds.count, memory_mb, expected_items(memory_mb), BLOOM_MAX_LOAD)
        yield seeds
