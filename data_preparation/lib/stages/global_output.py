# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Inspect and restore the committed identity of dataset-wide admission output."""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.global_dedup import GlobalFrontier, global_policy, ordered_sources
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import OwnershipError, guarded_path

log = get_logger("data_preparation.lib.stages.global_build")
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


def load_completed_candidates(config: DatasetConfig, name: str, layout: DatasetLayout, candidates_dir: Path) -> Manifest:
    """Require current, complete source-local input before inspecting final output."""
    candidates = Manifest.load(candidates_dir)
    if (candidates is None or not candidates.generation_complete
            or candidates.source_hash != config.processed_hash(name)):
        raise ValueError(f"{name}: complete source-local candidates are required before global admission")
    if candidates.generation_id is None:
        guarded_path(layout.root, candidates_dir)
        candidates.complete_generation(candidates_dir)
    return candidates


def build_generation_identity(
    config: DatasetConfig, name: str, layout: DatasetLayout, start: GlobalFrontier,
    prior_names: tuple[str, ...], candidates: Manifest,
) -> dict[str, Any]:
    """Record priority dependencies and the exact candidate generation used for admission."""
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
    return identity


def inspect_owned_output(
    name: str, directory: Path, identity: dict[str, Any], candidates: Manifest, start: GlobalFrontier,
) -> Manifest | None:
    """Reject unowned output and inconsistent recovery metadata before any replay deletion."""
    manifest = Manifest.load(directory)
    if manifest is None and directory.exists() and any(directory.iterdir()):
        raise OwnershipError(f"{directory}: no global output manifest proves ownership; preserving unexpected files")
    if manifest is not None:
        if manifest.source != name or manifest.stage != "processed":
            raise OwnershipError(f"{directory}: global output manifest ownership does not match {name!r}")
        if not all(key in manifest.extra for key in (*identity, "global_frontier")):
            raise ValueError(f"{name}: corrupt global recovery metadata; repair/replay required")
        if (manifest.extra["candidate_generation"] == candidates.generation_id
                and manifest.extra["global_dependencies"] == identity["global_dependencies"]
                and manifest.extra["global_start"] != start.to_dict()):
            raise ValueError(f"{name}: inconsistent global starting frontier; recovery cannot continue")
    return manifest


def restore_output_generation(
    config: DatasetConfig, name: str, directory: Path, start: GlobalFrontier, candidates: Manifest,
    identity: dict[str, Any], manifest: Manifest | None, *, rows_target: int, exhausted: bool,
) -> tuple[Manifest, GlobalFrontier | None]:
    """Reuse matching output, or replay it after preserving the existing ownership checks."""
    compatible = manifest is not None and all(manifest.extra.get(key) == value for key, value in identity.items())
    if manifest is not None and compatible:
        frontier = source_frontier(manifest)
        if manifest.generation_complete and (manifest.rows() >= rows_target or exhausted):
            return manifest, frontier
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
    return manifest, None
