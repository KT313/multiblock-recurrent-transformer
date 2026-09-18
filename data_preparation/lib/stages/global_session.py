# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Invocation-local admission reuse; the manifest remains the recovery authority.

Only the exclusive dataset owner may share this session across source passes. A
borrowed admission is removed from the cache until all asynchronous writes succeed.
Replay or any exception discards it; speculative Bloom insertions cannot be undone.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.download_profile import measure
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.global_dedup import GlobalAdmission, GlobalFrontier, global_policy, ordered_sources
from data_preparation.lib.stages.global_output import committed_keys, source_frontier
from data_preparation.lib.storage.manifest import Manifest

log = get_logger(__name__)


@dataclass(frozen=True)
class _Stamp:
    frontier: GlobalFrontier
    dependencies: tuple[tuple[str, str | None], ...]
    active: tuple[str, str | None, str, GlobalFrontier] | None

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> _Stamp:
        frontier = source_frontier(manifest)
        start = GlobalFrontier.from_dict(manifest.extra["global_start"])
        dependencies = tuple((name, generation) for name, generation in manifest.extra["global_dependencies"])
        active = None
        if frontier.source_index == start.source_index + 1:
            dependencies += ((manifest.source, manifest.generation_id),)
        elif frontier != start:
            active = (manifest.source, manifest.generation_id, manifest.extra["candidate_generation"], start)
        return cls(frontier, dependencies, active)


class GlobalAdmissionSession:
    """One lazy Bloom filter bound to a dataset scope, policy and immutable seed spool."""

    def __init__(
        self, config: DatasetConfig, layout: DatasetLayout, seed_frontier: GlobalFrontier,
        seed_keys: Callable[[], Iterable[int]],
    ) -> None:
        self._binding = (layout.root.resolve(), layout.processed_scope, json.dumps(global_policy(config), sort_keys=True))
        self._seeds = (seed_frontier.preseed_count, seed_frontier.preseed_digest)
        self._seed_keys = seed_keys
        self._entry: tuple[_Stamp, GlobalAdmission] | None = None
        self.restorations = 0
        self.restored_keys = 0
        self.reuses = 0

    def invalidate(self) -> None:
        self._entry = None

    @contextmanager
    def pass_scope(self, config: DatasetConfig, layout: DatasetLayout, start: GlobalFrontier) -> Iterator[None]:
        """Also invalidate on validation failures that occur before borrowing admission."""
        try:
            binding = (layout.root.resolve(), layout.processed_scope, json.dumps(global_policy(config), sort_keys=True))
            if binding != self._binding or (start.preseed_count, start.preseed_digest) != self._seeds:
                raise ValueError("global admission session dataset/policy/preseed identity changed")
            yield
        except BaseException:
            self.invalidate()
            raise

    def acquire(self, config: DatasetConfig, layout: DatasetLayout, manifest: Manifest) -> GlobalAdmission:
        stamp = _Stamp.from_manifest(manifest)
        entry, self._entry = self._entry, None
        if entry is not None and entry[0] == stamp:
            self.reuses += 1
            log.info("%s: reusing global Bloom state (%d retained keys)", manifest.source, stamp.frontier.retained)
            return entry[1]
        del entry  # release an incompatible large filter before allocating its replacement
        order = ordered_sources(config)
        start = GlobalFrontier.from_dict(manifest.extra["global_start"])
        started = time.monotonic()
        with measure("global_filter_restore"):
            admission = GlobalAdmission(
                order, memory_mb=config.bloom_dedup_memory_mb, frontier=stamp.frontier,
                committed_keys=chain(committed_keys(layout, order[:start.source_index]),
                                     committed_keys(layout, (manifest.source,))),
                preseed_keys=self._seed_keys(),
            )
        self.restorations += 1
        self.restored_keys += stamp.frontier.retained
        log.info("%s: restored global Bloom state (%d retained keys) in %.2fs",
                 manifest.source, stamp.frontier.retained, time.monotonic() - started)
        return admission

    def remember(self, manifest: Manifest, admission: GlobalAdmission) -> None:
        """Call only after the writer joins and generation/statistics publication succeeds."""
        stamp = _Stamp.from_manifest(manifest)
        if stamp.frontier != admission.frontier:
            self.invalidate()
            raise ValueError("global admission session does not match the committed frontier")
        self._entry = stamp, admission
