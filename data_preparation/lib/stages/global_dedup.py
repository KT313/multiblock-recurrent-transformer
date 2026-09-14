# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Ordered dataset admission, independent of storage and source preprocessing.

Only eligible source-local survivors enter this component. The caller owns the dataset
lease, persists output with its frontier atomically, and streams committed global_hash
columns to recover. A failed publication poisons this instance: recover from the durable
frontier, never reuse an in-memory reservation. Batch size bounds extra Python memory.
Benchmark material is loaded by the preparation lifecycle and passed as immutable preseed keys.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from data_preparation.lib.stages.benchmarks import bloom_benchmark_policy
from data_preparation.lib.stages.exact_dedup import (
    BLOOM_MAX_LOAD, TARGET_FALSE_POSITIVE_RATE, SeenDocuments, memory_mb_for,
)
from data_preparation.lib.stages.row_pipeline import normalize_text

if TYPE_CHECKING:
    from data_preparation.lib.dataset_config import DatasetConfig

GLOBAL_KEY_POLICY = "normalized-tagged-json-sha256-64-v1"
GLOBAL_ORDER_POLICY = "validation-only-then-declaration-v1"
GLOBAL_FILTER_POLICY = f"rbloom-{version('rbloom')}-splitmix64-v1"
_EMPTY_DIGEST = hashlib.sha256().hexdigest()
Row = dict[str, Any]


def ordered_sources(config: DatasetConfig) -> tuple[str, ...]:
    validation = tuple(name for name in config.sources if not config.used_in_train(name))
    return validation + tuple(name for name in config.sources if config.used_in_train(name))


def global_policy(config: DatasetConfig) -> dict[str, Any]:
    """Frozen semantics for the dataset identity; local candidate hashes stay reusable."""
    if not config.bloom_deduplicate_across_sources:
        return {"enabled": False}
    return {
        "enabled": True,
        "key_policy": GLOBAL_KEY_POLICY,
        "filter_policy": GLOBAL_FILTER_POLICY,
        "order_policy": GLOBAL_ORDER_POLICY,
        "source_order": list(ordered_sources(config)),
        "memory_mb": config.bloom_dedup_memory_mb,
        "false_positive_rate": TARGET_FALSE_POSITIVE_RATE,
        **({"benchmark_seeds": bloom_benchmark_policy(config.bloom_deduplicate_across_sources_add_benchmarks)}
           if config.bloom_deduplicate_across_sources_add_benchmarks else {}),
    }


def global_key(kind: str, row: Mapping[str, Any]) -> int:
    """Lowercase/collapse whitespace separately per field; formats remain distinct.

    Tagged JSON arrays preserve field boundaries; structured answers enter the key.
    Missing/null instruction inputs equal empty inputs. Text never equals an instruction
    tuple, even if rendering that tuple would produce the same text. Source-local hashes
    and normalization overrides have no effect on this policy.
    """
    if kind == "pretrain":
        values = [row["text"]]
    elif kind == "instruct":
        input_text = row.get("input")
        values = [row["instruction"], "" if input_text is None else input_text, row["output"]]
    else:
        raise ValueError(f"unknown global key kind {kind!r}")
    if any(not isinstance(value, str) for value in values):
        raise ValueError("global dedup keys require string text/fields")
    payload = json.dumps([kind, *(normalize_text(value) for value in values)], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big", signed=True)


@dataclass(frozen=True)
class GlobalFrontier:
    """Durable progress bound to the exact committed retained-key stream and policy."""
    source_order: tuple[str, ...]
    memory_mb: int
    key_policy: str = GLOBAL_KEY_POLICY
    filter_policy: str = GLOBAL_FILTER_POLICY
    order_policy: str = GLOBAL_ORDER_POLICY
    false_positive_rate: float = TARGET_FALSE_POSITIVE_RATE
    source_index: int = 0
    source_candidates: int = 0
    candidates: int = 0
    retained: int = 0
    bloom_positive: int = 0
    key_digest: str = _EMPTY_DIGEST
    preseed_count: int = 0
    preseed_digest: str = _EMPTY_DIGEST

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_order"] = list(self.source_order)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GlobalFrontier:
        """Reject corrupt or incompatible persisted state instead of silently resetting it."""
        if set(payload) != {item.name for item in fields(cls)}:
            raise ValueError("corrupt global dedup frontier fields; recovery requires the exact schema")
        order = payload["source_order"]
        if not isinstance(order, list) or any(not isinstance(name, str) for name in order):
            raise ValueError("corrupt global dedup frontier source order")
        for key in ("memory_mb", "source_index", "source_candidates", "candidates", "retained",
                    "bloom_positive", "preseed_count"):
            if type(payload[key]) is not int or payload[key] < 0:
                raise ValueError(f"corrupt global dedup frontier {key}")
        for key in ("key_digest", "preseed_digest"):
            value = payload[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"corrupt global dedup frontier {key}")
        return cls(**{**payload, "source_order": tuple(order)})


PublishBatch = Callable[[list[Row], GlobalFrontier], None]


class GlobalAdmission:
    """Serial bounded admission with storage-owned atomic output/frontier commits.

    finish_source is permitted only after the source reached its retained budget or
    exhaustion. This decision belongs to the orchestration layer. A later top-up of an
    earlier source requires replay in a new dataset snapshot, never reopening this frontier.
    """
    def __init__(
        self, source_order: tuple[str, ...], *, memory_mb: int,
        frontier: GlobalFrontier | None = None, committed_keys: Iterable[int] = (),
        preseed_keys: Iterable[int] = (),
    ) -> None:
        if type(memory_mb) is not int or memory_mb < 1:
            raise ValueError("bloom_dedup_memory_mb must be a positive integer")
        if len(set(source_order)) != len(source_order) or not source_order:
            raise ValueError("global dedup source order must be nonempty and unique")
        expected = GlobalFrontier(source_order=source_order, memory_mb=memory_mb)
        current = frontier or expected
        if (current.source_order, current.memory_mb, current.key_policy, current.filter_policy, current.order_policy, current.false_positive_rate) != (
            source_order, memory_mb, GLOBAL_KEY_POLICY, GLOBAL_FILTER_POLICY, GLOBAL_ORDER_POLICY, TARGET_FALSE_POSITIVE_RATE,
        ):
            raise ValueError("global dedup recovery policy changed; replay a new dataset snapshot")
        if (not 0 <= current.source_index <= len(source_order) or min(
            current.source_candidates, current.candidates, current.retained, current.bloom_positive, current.preseed_count,
        ) < 0 or current.candidates != current.retained + current.bloom_positive
                or current.source_candidates > current.candidates):
            raise ValueError("corrupt global dedup commit frontier")
        self.frontier = replace(current, preseed_count=0)
        self.seen = SeenDocuments(memory_mb=memory_mb)
        self._poisoned = False
        self._digest = hashlib.sha256()
        seed_digest = hashlib.sha256()
        seed_count = 0
        for key in preseed_keys:
            seed_count += 1
            self.check_capacity(seed_count + current.retained)
            seed_digest.update(key.to_bytes(8, "big", signed=True))
            self.seen.add_all([key])
        if frontier is not None and (seed_count, seed_digest.hexdigest()) != (current.preseed_count, current.preseed_digest):
            raise ValueError("global dedup recovery preseed policy changed")
        self.frontier = replace(current, preseed_count=seed_count, preseed_digest=seed_digest.hexdigest())
        self.check_capacity(current.retained)
        count = 0
        for key in committed_keys:
            count += 1
            if count > current.retained:
                raise ValueError("global dedup recovery committed keys exceed frontier")
            self._digest.update(key.to_bytes(8, "big", signed=True))
            self.seen.add_all([key])
        if count != current.retained or self._digest.hexdigest() != current.key_digest:
            raise ValueError("global dedup recovery committed keys do not match frontier")

    def check_capacity(self, expected_retained: int) -> None:
        """Check all retained/preseeded keys plus expected additions before admission."""
        if type(expected_retained) is not int or expected_retained < 0:
            raise ValueError("global Bloom expected retained count must be a nonnegative integer")
        total = expected_retained + self.frontier.preseed_count
        if total > BLOOM_MAX_LOAD * self.seen.nominal_capacity:
            raise ValueError(
                f"global Bloom filter would contain {total:,} keys, exceeding {BLOOM_MAX_LOAD:g}x nominal capacity; "
                f"raise bloom_dedup_memory_mb to {memory_mb_for(total)} and replay a new dataset snapshot"
            )

    def _check_source(self, name: str) -> None:
        if self._poisoned:
            raise RuntimeError("global dedup publication failed; recover from the committed frontier before retry")
        index = self.frontier.source_index
        if index >= len(self.frontier.source_order) or self.frontier.source_order[index] != name:
            raise ValueError(f"global dedup priority violation for {name!r}: frontier is source index {index}")

    def commit_batch(self, name: str, kind: str, rows: list[Row], publish: PublishBatch) -> None:
        self._check_source(name)
        self.check_capacity(self.frontier.retained + len(rows))
        survivors: list[Row] = []
        try:
            for row in rows:
                key = global_key(kind, row)
                if self.seen.add_if_new(key):
                    survivors.append({**row, "global_hash": key})
                    self._digest.update(key.to_bytes(8, "big", signed=True))
            next_frontier = replace(
                self.frontier,
                source_candidates=self.frontier.source_candidates + len(rows),
                candidates=self.frontier.candidates + len(rows),
                retained=self.frontier.retained + len(survivors),
                bloom_positive=self.frontier.bloom_positive + len(rows) - len(survivors),
                key_digest=self._digest.hexdigest(),
            )
            publish(survivors, next_frontier)
        except BaseException:
            self._poisoned = True
            raise
        self.frontier = next_frontier

    def finish_source(self, name: str, publish: PublishBatch) -> None:
        self._check_source(name)
        next_frontier = replace(self.frontier, source_index=self.frontier.source_index + 1, source_candidates=0)
        try:
            publish([], next_frontier)
        except BaseException:
            self._poisoned = True
            raise
        self.frontier = next_frontier

    def statistics(self) -> dict[str, int | float]:
        items = self.seen.items_in_filter
        return {
            "candidates": self.frontier.candidates,
            "retained": self.frontier.retained,
            "bloom_positive": self.frontier.bloom_positive,
            "memory_mb": self.frontier.memory_mb,
            "preseed_count": self.frontier.preseed_count,
            "nominal_capacity": self.seen.nominal_capacity,
            "max_load": BLOOM_MAX_LOAD,
            "items_in_filter": items,
            "load": items / self.seen.nominal_capacity,
            "measured_false_positive_rate": self.seen.expected_false_positive_rate(items),
        }
