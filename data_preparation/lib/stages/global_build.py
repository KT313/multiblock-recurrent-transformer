# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset-scoped final output from reusable, source-local processed candidates.

The source manifest is the atomic commit record: output shards are published first,
then their list, the global frontier and candidate/dependency generations in one
manifest replacement. Unlisted shards never refill the Bloom filter. Every writer
runs under the caller's exclusive dataset lease; readers use completed snapshots.
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Any


from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.global_dedup import GlobalAdmission, GlobalFrontier, ordered_sources
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import guarded_path
from data_preparation.lib.storage.parquet import build_row_table, publish_shard, shard_name

from data_preparation.lib.stages.global_output import (
    GLOBAL_BATCH_ROWS, candidate_rows, committed_keys,
    outputs_complete as outputs_complete, source_frontier as source_frontier,  # noqa: PLC0414 - compatibility exports
    build_generation_identity, inspect_owned_output, load_completed_candidates, restore_output_generation,
)


log = get_logger(__name__)


def build_global_source(
    config: DatasetConfig, name: str, layout: DatasetLayout, start: GlobalFrontier,
    *, rows_target: int, exhausted: bool, should_stop: StopCheck | None = None,
    batch_rows: int = GLOBAL_BATCH_ROWS, preseed_keys: Iterable[int] = (),
) -> tuple[GlobalFrontier, bool]:
    """Admit one source; return its frontier and whether budget/exhaustion finalized it.

    A changed candidate generation replays this source. Dependency generations force
    downstream replay even if an earlier replacement happened to retain the same keys.
    A matching partial generation resumes behind exactly its committed candidate offset.
    """

    # validate the source priority and guard all output paths
    if batch_rows < 1:
        raise ValueError("global batch_rows must be positive")
    order = ordered_sources(config)
    if start.source_index >= len(order) or order[start.source_index] != name:
        raise ValueError(f"{name}: global source priority does not match starting frontier")
    prior_names = order[:start.source_index]
    directory = guarded_path(layout.root, layout.processed_dir(name))
    for filename in ("MANIFEST.json", "MANIFEST.json.tmp"):
        guarded_path(layout.root, directory / filename)
    candidates_dir = DatasetLayout(layout.root).processed_dir(name)
    candidates = load_completed_candidates(config, name, layout, candidates_dir)
    identity = build_generation_identity(config, name, layout, start, prior_names, candidates)

    # inspect ownership and restore the matching committed generation
    manifest = inspect_owned_output(name, directory, identity, candidates, start)
    manifest, completed_frontier = restore_output_generation(
        config, name, directory, start, candidates, identity, manifest, rows_target=rows_target, exhausted=exhausted,
    )
    if completed_frontier is not None:
        return completed_frontier, True

    # rebuild admission from committed keys before recovering or extending the frontier
    frontier = source_frontier(manifest)
    own_keys = committed_keys(layout, (name,))
    admission = GlobalAdmission(
        order, memory_mb=config.bloom_dedup_memory_mb, frontier=frontier,
        committed_keys=chain(committed_keys(layout, prior_names), own_keys), preseed_keys=preseed_keys,
    )
    admission.check_capacity(start.retained + candidates.rows())
    if frontier.source_index == start.source_index + 1:
        manifest.complete_generation(directory)  # the final frontier committed before a crash in completion
        return frontier, True

    commit = partial(commit_global_batch, layout, directory, manifest, start)

    # admit candidate batches in their existing order
    rows = candidate_rows(candidates_dir, candidates, frontier.source_candidates)
    while True:
        check_stop(should_stop)
        batch = list(islice(rows, batch_rows))
        if not batch:
            break
        admission.commit_batch(name, "messages" if config.sources[name].instruction_format == "messages" else config.sources[name].kind, batch, commit)

    # finalize the source budget and publish filter statistics
    complete = manifest.rows() >= rows_target or exhausted
    if complete:
        admission.finish_source(name, commit)
    manifest.stats["global_filter"] = admission.statistics()
    manifest.stats["budget_shortfall"] = max(0, rows_target - manifest.rows())
    if complete:
        manifest.complete_generation(directory)
    else:
        manifest.save(directory)
    log.info("%s: global Bloom admission %s; retained %d/%d; shortfall %d", name,
             manifest.stats["global_dedup"], manifest.rows(), rows_target, manifest.stats["budget_shortfall"])
    metrics = manifest.stats["global_filter"]
    log.info("%s: global Bloom filter: %d MiB, %.3fx nominal load, measured FPR %.6f%%", name,
             metrics["memory_mb"], metrics["load"], metrics["measured_false_positive_rate"] * 100)
    check_stop(should_stop)
    return admission.frontier, complete


def commit_global_batch(
    layout: DatasetLayout, directory: Path, manifest: Manifest, start: GlobalFrontier,
    rows: list[dict[str, Any]], next_frontier: GlobalFrontier,
) -> None:
    """Publish rows before atomically recording their frontier and counters."""
    if rows:
        path = guarded_path(layout.root, directory / shard_name(len(manifest.shards)))
        guarded_path(layout.root, path.with_name(path.name + ".tmp"))
        publish_shard(build_row_table(rows), path)
        manifest.add_shard(path.name, len(rows), sum(int(row["tokens"]) for row in rows))
    manifest.extra["global_frontier"] = next_frontier.to_dict()
    manifest.stats["global_dedup"] = {
        "candidates": next_frontier.candidates - start.candidates,
        "retained": next_frontier.retained - start.retained,
        "bloom_positive": next_frontier.bloom_positive - start.bloom_positive,
    }
    manifest.save(directory)
