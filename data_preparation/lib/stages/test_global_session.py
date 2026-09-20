# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""A live admission cache may reuse only the exact durable prefix it represents."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from data_preparation.conftest import CfgFactory
from data_preparation.lib.dataset_config import DatasetConfig, SourceConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.stages import global_build
from data_preparation.lib.stages.global_build import build_global_source
from data_preparation.lib.stages.global_dedup import GlobalFrontier, global_key, ordered_sources
from data_preparation.lib.stages.global_session import GlobalAdmissionSession
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.parquet import build_row_table, publish_shard


def candidates(cfg_factory: CfgFactory, root: Path) -> tuple[DatasetConfig, DatasetLayout, GlobalFrontier]:
    cfg = cfg_factory({name: SourceConfig(kind="pretrain", loader="synthetic") for name in ("a", "b", "c")},
                      bloom_deduplicate_across_sources=True)
    cfg = replace(cfg, bloom_dedup_memory_mb=1)
    for name, texts in {"a": ["shared", "a"], "b": ["SHARED", "b"], "c": ["shared", "c"]}.items():
        directory = DatasetLayout(root).processed_dir(name)
        rows = [{"text": text, "tokens": 1, "hash": index, "source": name} for index, text in enumerate(texts)]
        path = publish_shard(build_row_table(rows), directory / "shard_00000.parquet")
        manifest = Manifest(source=name, source_hash=cfg.processed_hash(name), stage="processed",
                            columns=list(rows[0]), token_count="estimate")
        manifest.add_shard(path.name, len(rows), len(rows))
        manifest.complete_generation(directory)
    layout = DatasetLayout(root).for_config(cfg)
    return cfg, layout, GlobalFrontier(ordered_sources(cfg), cfg.bloom_dedup_memory_mb)


def output_rows(layout: DatasetLayout, name: str) -> list[dict[str, Any]]:
    manifest = Manifest.load(layout.processed_dir(name))
    assert manifest is not None
    return [row for shard in manifest.shards for row in pq.read_table(layout.processed_dir(name) / shard.name).to_pylist()]


def test_reuses_across_sources_and_recovers_once_after_complete_prefix(cfg_factory: CfgFactory, tmp_path: Path) -> None:
    cfg, layout, start = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    frontier, _ = build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=session)
    # A new invocation skips a complete prefix without allocating a filter.
    resumed = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    assert build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=resumed)[0] == frontier
    assert resumed.restorations == 0
    for name in ("b", "c"):
        frontier, complete = build_global_source(cfg, name, layout, frontier, rows_target=1, exhausted=True, session=resumed)
        assert complete
    assert (resumed.restorations, resumed.restored_keys, resumed.reuses) == (1, 2, 1)
    assert [row["text"] for row in output_rows(layout, "c")] == ["c"]


def test_replay_discards_keys_from_the_discarded_attempt(cfg_factory: CfgFactory, tmp_path: Path) -> None:
    cfg, layout, start = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    start, _ = build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=session)
    partial, complete = build_global_source(cfg, "b", layout, start, rows_target=10, exhausted=False, session=session)
    assert not complete and partial.retained == 3
    directory = DatasetLayout(tmp_path).processed_dir("b")
    local = Manifest.load(directory)
    assert local is not None
    local.begin_generation(directory)
    local.complete_generation(directory)
    finished, complete = build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True, session=session)
    assert complete and finished.retained == 3
    assert [row["text"] for row in output_rows(layout, "b")] == ["b"]
    assert (session.restorations, session.restored_keys, session.reuses) == (2, 2, 1)


@pytest.mark.parametrize("failure", ["flush", "statistics", "validation"])
def test_failed_pass_invalidates_even_after_admission(
    cfg_factory: CfgFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    cfg, layout, start = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    start, _ = build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=session)
    with monkeypatch.context() as patch:
        def fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("injected failure")
        if failure == "flush":
            patch.setattr(global_build._Writer, "flush", fail)
        elif failure == "statistics":
            from data_preparation.lib.stages.global_dedup import GlobalAdmission
            patch.setattr(GlobalAdmission, "statistics", fail)
        else:
            patch.setattr(global_build, "load_completed_candidates", fail)
        with pytest.raises(OSError, match="injected"):
            build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True, session=session)
    build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True, session=session)
    assert session.restorations == 2
    assert [row["text"] for row in output_rows(layout, "b")] == ["b"]


def test_binding_mismatch_and_corrupt_metadata_do_not_use_cache(cfg_factory: CfgFactory, tmp_path: Path) -> None:
    cfg, layout, start = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    start, _ = build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=session)
    with pytest.raises(ValueError, match="identity changed"):
        build_global_source(cfg, "b", layout, replace(start, preseed_count=1),
                            rows_target=1, exhausted=True, session=session)
    assert session._entry is None
    build_global_source(cfg, "b", layout, start, rows_target=10, exhausted=False, session=session)
    manifest = Manifest.load(layout.processed_dir("b"))
    assert manifest is not None
    manifest.extra["global_frontier"]["source_candidates"] += 1
    manifest.save(layout.processed_dir("b"))
    with pytest.raises(ValueError, match="corrupt global"):
        build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True, session=session)
    assert session._entry is None


def test_same_source_partial_and_already_partial_next_source(cfg_factory: CfgFactory, tmp_path: Path) -> None:
    cfg, layout, start = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    start, _ = build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=session)
    # A different call has already committed a partial b. The cached a frontier cannot cover it.
    partial, _ = build_global_source(cfg, "b", layout, start, rows_target=10, exhausted=False)
    resumed, _ = build_global_source(cfg, "b", layout, start, rows_target=10, exhausted=False, session=session)
    assert resumed == partial and session.restorations == 2 and session.restored_keys == 3
    # Same-generation continuation is reusable, even when there are no more candidate batches.
    resumed, complete = build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True, session=session)
    assert complete and resumed.retained == 3 and session.reuses == 1
    assert [row["text"] for row in output_rows(layout, "b")] == ["b"]


@pytest.mark.parametrize("changed_dependency", [False, True])
def test_preseeds_and_zero_retained_source_dependencies(
    cfg_factory: CfgFactory, tmp_path: Path, changed_dependency: bool,
) -> None:
    cfg, layout, frontier = candidates(cfg_factory, tmp_path)
    directory = DatasetLayout(tmp_path).processed_dir("b")
    local = Manifest.load(directory)
    assert local is not None
    rows = [{"text": text, "tokens": 1, "source": "b", "hash": index} for index, text in enumerate(("shared", "SHARED"))]
    publish_shard(build_row_table(rows), directory / local.shards[0].name)
    seed = global_key("pretrain", {"text": "shared"})
    frontier = replace(frontier, preseed_count=1, preseed_digest=hashlib.sha256(seed.to_bytes(8, "big", signed=True)).hexdigest())
    seeded = 0

    def seeds() -> tuple[int]:
        nonlocal seeded
        seeded += 1
        return (seed,)

    session = GlobalAdmissionSession(cfg, layout, frontier, seeds)
    for name in ("a", "b"):
        frontier, _ = build_global_source(cfg, name, layout, frontier, rows_target=1, exhausted=True, session=session)
    assert output_rows(layout, "b") == []
    if changed_dependency:
        previous = Manifest.load(layout.processed_dir("b"))
        assert previous is not None
        previous.begin_generation(layout.processed_dir("b"))
        previous.complete_generation(layout.processed_dir("b"))
    build_global_source(cfg, "c", layout, frontier, rows_target=1, exhausted=True, session=session)
    assert seeded == session.restorations == (2 if changed_dependency else 1)
    assert session.reuses == (1 if changed_dependency else 2)
    assert [row["text"] for row in output_rows(layout, "c")] == ["c"]


def test_reused_filter_still_checks_capacity(
    cfg_factory: CfgFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_preparation.lib.stages.global_dedup import GlobalAdmission

    cfg, layout, start = candidates(cfg_factory, tmp_path)
    session = GlobalAdmissionSession(cfg, layout, start, lambda: ())
    start, _ = build_global_source(cfg, "a", layout, start, rows_target=1, exhausted=True, session=session)
    def full(self: GlobalAdmission, expected_retained: int) -> None:
        raise ValueError("capacity exceeded")
    monkeypatch.setattr(GlobalAdmission, "check_capacity", full)
    with pytest.raises(ValueError, match="capacity exceeded"):
        build_global_source(cfg, "b", layout, start, rows_target=1, exhausted=True, session=session)
    assert session.reuses == 1 and session._entry is None
