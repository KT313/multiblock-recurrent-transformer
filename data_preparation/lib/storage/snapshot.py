# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Small immutable build descriptors, validated using manifests only.

IDs attest managed publication, not file contents. Manual edits retaining manifests are outside this
contract. Legacy adoption establishes identity from now onward and proves nothing about old checkpoints.
Writers must hold the existing exclusive dataset lease; status and training only read descriptors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.storage.atomic import write_atomically
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import guarded_path

log = get_logger(__name__)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetSnapshot:
    schema_version: int
    build_id: str
    config_hash: str
    sources: list[dict[str, Any]]
    tokenizer: dict[str, Any]
    processing: dict[str, Any] = field(default_factory=dict)


def _constituents(config: DatasetConfig, layout: DatasetLayout, *, adopt: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    tokenizer: dict[str, Any] = {}
    entries = [(name, layout.processed_dir(name), config.processed_hash(name), "processed") for name in config.sources]
    entries.append((config.tokenizer.name, layout.tokenizer_dir(config.tokenizer.name), config.tokenizer_hash(), "tokenizer"))
    for name, directory, expected_hash, stage in entries:
        manifest = Manifest.load(directory)
        if manifest is None or manifest.stage != stage or manifest.source_hash != expected_hash:
            raise RuntimeError(f"{stage} {name!r}: missing or stale constituent manifest at {directory}")
        if not manifest.generation_complete:
            raise RuntimeError(f"{stage} {name!r}: generation {manifest.generation_id} is incomplete; rerun prepare")
        if manifest.generation_id is None:
            if not adopt:
                raise RuntimeError(f"{stage} {name!r}: legacy generation identity is missing; run prepare to adopt existing bytes")
            guarded_path(layout.root, directory)
            log.warning("%s %s: adopting legacy bytes; this establishes identity from now onward only", stage, name)
            manifest.complete_generation(directory)
        reference = {"name": name, "generation_id": manifest.generation_id, "source_hash": manifest.source_hash}
        if stage == "processed":
            sources.append({**reference, "rows": manifest.rows()})
        else:
            tokenizer = reference
    return sources, tokenizer


def _load_snapshot(config: DatasetConfig, layout: DatasetLayout) -> DatasetSnapshot | None:
    path = layout.snapshot_path(config.config_hash())
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        snapshot = DatasetSnapshot(**payload)
        if snapshot.schema_version != SCHEMA_VERSION or not isinstance(snapshot.build_id, str) or not snapshot.build_id:
            raise ValueError("unsupported schema or missing build ID")
        return snapshot
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid dataset snapshot {path}: {error}") from error


def read_snapshot(config: DatasetConfig, layout: DatasetLayout, *, processing: dict[str, Any] | None = None) -> DatasetSnapshot:
    """Reject unknown, stale or incomplete identity without opening any Parquet file."""
    snapshot = _load_snapshot(config, layout)
    if snapshot is None:
        raise RuntimeError("dataset snapshot identity is missing; run prepare to publish/adopt the dataset")
    sources, tokenizer = _constituents(config, layout)
    if (snapshot.config_hash != config.config_hash() or snapshot.sources != sources or snapshot.tokenizer != tokenizer
            or snapshot.processing != (processing or {})):
        raise RuntimeError(
            f"dataset snapshot {snapshot.build_id} is stale: ordered source/tokenizer generations or processing differ "
            f"(recorded sources={snapshot.sources}, tokenizer={snapshot.tokenizer}; current sources={sources}, "
            f"tokenizer={tokenizer}); run prepare to publish a coherent snapshot"
        )
    return snapshot


def snapshot_problem(config: DatasetConfig, layout: DatasetLayout, *, processing: dict[str, Any] | None = None) -> str | None:
    try:
        read_snapshot(config, layout, processing=processing)
    except RuntimeError as error:
        return str(error)
    return None


def publish_snapshot(config: DatasetConfig, layout: DatasetLayout, *, processing: dict[str, Any] | None = None) -> DatasetSnapshot:
    """Publish after the caller verified data readiness, under the preparation lease. No-op retains its ID."""
    sources, tokenizer = _constituents(config, layout, adopt=True)
    previous = _load_snapshot(config, layout)
    processing = dict(processing or {})
    if (previous is not None and previous.config_hash == config.config_hash() and previous.sources == sources
            and previous.tokenizer == tokenizer and previous.processing == processing):
        return previous
    snapshot = DatasetSnapshot(SCHEMA_VERSION, uuid4().hex, config.config_hash(), sources, tokenizer, processing)
    path = layout.snapshot_path(config.config_hash())
    guarded_path(layout.root, path)
    with write_atomically(path) as temporary:
        temporary.write_text(json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n")
    return snapshot
