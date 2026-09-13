# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset-scoped final output from reusable, source-local processed candidates.

The source manifest is the atomic commit record: output shards are published first,
then their list, the global frontier and candidate/dependency generations in one
manifest replacement. Unlisted shards never refill the Bloom filter. Every writer
runs under the caller's exclusive dataset lease; readers use completed snapshots.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from dataclasses import replace
from itertools import chain, islice
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.global_dedup import GlobalAdmission, GlobalFrontier, global_policy, ordered_sources
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import OwnershipError, guarded_path
from data_preparation.lib.storage.parquet import publish_shard, shard_name

log = get_logger(__name__)
GLOBAL_BATCH_ROWS = 4096


def committed_keys(layout: DatasetLayout, names: tuple[str, ...]) -> Iterator[int]:
    """Stream only manifest-listed keys, bounded independently of corpus/shard size."""
    for name in names:
        directory = layout.processed_dir(name)
        manifest = Manifest.load(directory)
        if manifest is None:
            raise ValueError(f"global recovery: missing committed source {name!r}")
        for shard in manifest.shards:
            for batch in pq.ParquetFile(directory / shard.name).iter_batches(
                batch_size=GLOBAL_BATCH_ROWS, columns=["global_hash"],
            ):
                yield from cast(list[int], batch.column(0).to_pylist())


def candidate_rows(directory: Path, manifest: Manifest, skip: int) -> Iterator[dict[str, Any]]:
    for shard in manifest.shards:
        if skip >= shard.rows:
            skip -= shard.rows
            continue
        for batch in pq.ParquetFile(directory / shard.name).iter_batches(batch_size=GLOBAL_BATCH_ROWS):
            rows: list[dict[str, Any]] = batch.to_pylist()
            if skip:
                skipped = min(skip, len(rows))
                rows = rows[skipped:]
                skip -= skipped
            yield from rows
    if skip:
        raise ValueError("global recovery frontier exceeds candidate rows; replay required")


def source_frontier(manifest: Manifest) -> GlobalFrontier:
    try:
        frontier = GlobalFrontier.from_dict(manifest.extra["global_frontier"])
        start = GlobalFrontier.from_dict(manifest.extra["global_start"])
        consumed = frontier.candidates - start.candidates
        expected_offset = consumed if frontier.source_index == start.source_index else 0
        policy_fields = ("source_order", "memory_mb", "key_policy", "filter_policy", "order_policy",
                         "false_positive_rate", "preseed_count", "preseed_digest")
        if (any(getattr(frontier, field) != getattr(start, field) for field in policy_fields)
                or frontier.source_index not in (start.source_index, start.source_index + 1)
                or start.source_candidates != 0 or frontier.source_candidates != expected_offset
                or frontier.retained - start.retained != manifest.rows()
                or frontier.candidates != frontier.retained + frontier.bloom_positive
                or frontier.candidates < start.candidates or frontier.bloom_positive < start.bloom_positive):
            raise ValueError("inconsistent frontier counters/policy")
        return frontier
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{manifest.source}: corrupt global recovery frontier; repair/replay required") from error


def outputs_complete(config: DatasetConfig, layout: DatasetLayout) -> bool:
    if layout.processed_scope is None:
        return True
    order = ordered_sources(config)
    previous = GlobalFrontier(order, config.bloom_dedup_memory_mb)
    dependencies: list[list[str | None]] = []
    for index, name in enumerate(order):
        manifest = Manifest.load(layout.processed_dir(name))
        if manifest is None or not manifest.generation_complete:
            return False
        if index == 0 and config.bloom_deduplicate_across_sources_add_benchmarks:
            # Read-only completion never fetches benchmarks. The configured pinned policy
            # and durable initial seed identity must cover the entire frontier chain.
            try:
                initial = GlobalFrontier.from_dict(manifest.extra["global_start"])
            except (KeyError, TypeError, ValueError):
                return False
            if initial.preseed_count <= 0:
                return False
            previous = replace(previous, preseed_count=initial.preseed_count, preseed_digest=initial.preseed_digest)
        if (manifest.extra.get("global_dedup") != global_policy(config)
                or manifest.extra.get("global_start") != previous.to_dict()
                or manifest.extra.get("global_dependencies") != dependencies):
            return False
        previous = source_frontier(manifest)
        if previous.source_index != index + 1:
            return False
        dependencies.append([name, manifest.generation_id])
    return True


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
    candidates = Manifest.load(candidates_dir)
    if (candidates is None or not candidates.generation_complete
            or candidates.source_hash != config.processed_hash(name)):
        raise ValueError(f"{name}: complete source-local candidates are required before global admission")
    if candidates.generation_id is None:
        guarded_path(layout.root, candidates_dir)
        candidates.complete_generation(candidates_dir)
    dependencies = []
    for prior in prior_names:
        previous = Manifest.load(layout.processed_dir(prior))
        if previous is None or not previous.generation_complete:
            raise ValueError(f"{name}: prerequisite source {prior!r} is incomplete")
        dependencies.append([prior, previous.generation_id])
    identity = {
        "global_dedup": global_policy(config),
        "global_start": start.to_dict(),
        "global_dependencies": dependencies,
        "candidate_generation": candidates.generation_id,
    }
    manifest = Manifest.load(directory)
    if manifest is None and directory.exists() and any(directory.iterdir()):
        raise OwnershipError(f"{directory}: no global output manifest proves ownership; preserving unexpected files")
    if manifest is not None:
        if manifest.source != name or manifest.stage != "processed":
            raise OwnershipError(f"{directory}: global output manifest ownership does not match {name!r}")
        if not all(key in manifest.extra for key in (*identity, "global_frontier")):
            raise ValueError(f"{name}: corrupt global recovery metadata; repair/replay required")
        if (manifest.extra["candidate_generation"] == candidates.generation_id
                and manifest.extra["global_dependencies"] == dependencies
                and manifest.extra["global_start"] != start.to_dict()):
            raise ValueError(f"{name}: inconsistent global starting frontier; recovery cannot continue")
    compatible = manifest is not None and all(manifest.extra.get(key) == value for key, value in identity.items())
    if manifest is not None and compatible:
        frontier = source_frontier(manifest)
        if manifest.generation_complete and (manifest.rows() >= rows_target or exhausted):
            return frontier, True
        if manifest.generation_complete:
            compatible = False  # reopening a formerly exhausted priority source requires downstream replay
    if not compatible:
        if directory.exists():
            log.info("%s: replaying dataset-scoped output after candidate/dependency changes", name)
            shutil.rmtree(directory)
        manifest = Manifest(
            source=name, source_hash=config.processed_hash(name), stage="processed",
            tokenizer=candidates.tokenizer, token_count=candidates.token_count,
            hash_payload=config.processed_hash_payload(name),
            input_shards=list(candidates.input_shards), columns=[*candidates.columns, "global_hash"],
            shuffled=candidates.shuffled, shuffle_seed=candidates.shuffle_seed,
            extra={**identity, "global_frontier": start.to_dict()},
            stats={"local_candidates": candidates.rows()},
        )
        manifest.begin_generation(directory)
    assert manifest is not None
    frontier = source_frontier(manifest)
    own_keys = committed_keys(layout, (name,))
    admission = GlobalAdmission(
        order, memory_mb=config.bloom_dedup_memory_mb, frontier=frontier,
        committed_keys=chain(committed_keys(layout, prior_names), own_keys), preseed_keys=preseed_keys,
    )
    admission.check_capacity(start.retained + candidates.rows())
    if frontier.source_index == start.source_index + 1:
        # The final frontier committed before a crash in complete_generation.
        manifest.complete_generation(directory)
        return frontier, True

    def commit(rows: list[dict[str, Any]], next_frontier: GlobalFrontier) -> None:
        if rows:
            path = guarded_path(layout.root, directory / shard_name(len(manifest.shards)))
            guarded_path(layout.root, path.with_name(path.name + ".tmp"))
            publish_shard(pa.Table.from_pylist(rows), path)
            manifest.add_shard(path.name, len(rows), sum(int(row["tokens"]) for row in rows))
        manifest.extra["global_frontier"] = next_frontier.to_dict()
        manifest.stats["global_dedup"] = {
            "candidates": next_frontier.candidates - start.candidates,
            "retained": next_frontier.retained - start.retained,
            "bloom_positive": next_frontier.bloom_positive - start.bloom_positive,
        }
        manifest.save(directory)

    rows = candidate_rows(candidates_dir, candidates, frontier.source_candidates)
    while True:
        check_stop(should_stop)
        batch = list(islice(rows, batch_rows))
        if not batch:
            break
        admission.commit_batch(name, config.sources[name].kind, batch, commit)
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
