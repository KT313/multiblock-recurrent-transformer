# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Identity checks open small manifests only; output bytes are intentionally not hashed."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_preparation.dataset_config import DatasetConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.lock import dataset_lock
from data_preparation.lib.storage.manifest import Manifest, ShardInfo
from data_preparation.lib.storage.snapshot import publish_snapshot, read_snapshot, snapshot_problem


@pytest.fixture
def prepared(cfg_factory: Callable[..., DatasetConfig], layout: DatasetLayout) -> DatasetConfig:
    config = cfg_factory({name: SourceConfig(kind="pretrain", loader="synthetic", seed=i) for i, name in enumerate(("a", "b"))})
    with dataset_lock(layout.root):
        for name in config.sources:
            manifest = Manifest(name, config.processed_hash(name), "processed", shards=[ShardInfo("data-00000.parquet", 10)])
            manifest.complete_generation(layout.processed_dir(name))
        Manifest(config.tokenizer.name, config.tokenizer_hash(), "tokenizer").complete_generation(layout.tokenizer_dir(config.tokenizer.name))
        publish_snapshot(config, layout)
    return config


def test_noop_stats_and_unrelated_source_retain_metadata_only_identity(
    prepared: DatasetConfig, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = read_snapshot(prepared, layout)
    # There are deliberately no Parquet files. Fail also on any attempt to enumerate them.
    def no_scan(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("identity validation must not scan data")
    monkeypatch.setattr(Path, "glob", no_scan)
    with dataset_lock(layout.root):
        manifest = Manifest.load(layout.processed_dir("a"))
        assert manifest is not None
        manifest.stats["diagnostic"] = 4
        manifest.save(layout.processed_dir("a"))
        Manifest("other", "other-hash", "processed").complete_generation(layout.processed_dir("other"))
        assert publish_snapshot(prepared, layout) == original
    assert read_snapshot(prepared, layout) == original


@pytest.mark.parametrize("kind", ["same_count_rebuild", "extension", "tokenizer"])
def test_managed_generation_invalidates_snapshot(prepared: DatasetConfig, layout: DatasetLayout, kind: str) -> None:
    original = read_snapshot(prepared, layout)
    directory = layout.tokenizer_dir(prepared.tokenizer.name) if kind == "tokenizer" else layout.processed_dir("a")
    with dataset_lock(layout.root):
        manifest = Manifest.load(directory)
        assert manifest is not None
        manifest.begin_generation(directory)
        with pytest.raises(RuntimeError, match="incomplete"):
            read_snapshot(prepared, layout)
        if kind == "extension":
            manifest.add_shard("data-00001.parquet", 3)
        manifest.complete_generation(directory)
        with pytest.raises(RuntimeError, match=f"snapshot {original.build_id} is stale"):
            read_snapshot(prepared, layout)
        replacement = publish_snapshot(prepared, layout)
    assert replacement.build_id != original.build_id


def test_source_order_changes_snapshot_even_if_config_hash_sorts_keys(prepared: DatasetConfig, layout: DatasetLayout) -> None:
    original = read_snapshot(prepared, layout)
    prepared.sources = dict(reversed(list(prepared.sources.items())))
    with pytest.raises(RuntimeError, match="stale"):
        read_snapshot(prepared, layout)
    with dataset_lock(layout.root):
        replacement = publish_snapshot(prepared, layout)
    assert replacement.build_id != original.build_id
    assert [source["name"] for source in replacement.sources] == ["b", "a"]


def test_legacy_adoption_is_explicit_and_status_does_not_write(prepared: DatasetConfig, layout: DatasetLayout) -> None:
    directory = layout.processed_dir("a")
    manifest = Manifest.load(directory)
    assert manifest is not None
    manifest.generation_id = None
    manifest.save(directory)
    before = (directory / "MANIFEST.json").read_bytes()
    assert "legacy generation identity" in str(snapshot_problem(prepared, layout))
    assert (directory / "MANIFEST.json").read_bytes() == before
    with dataset_lock(layout.root):
        adopted = publish_snapshot(prepared, layout)
    assert read_snapshot(prepared, layout) == adopted


def test_interrupted_generation_is_stable_and_cannot_be_published(prepared: DatasetConfig, layout: DatasetLayout) -> None:
    with dataset_lock(layout.root):
        directory = layout.processed_dir("a")
        manifest = Manifest.load(directory)
        assert manifest is not None
        manifest.begin_generation(directory)
        generation = manifest.generation_id
        manifest = Manifest.load(directory)
        assert manifest is not None
        manifest.begin_generation(directory)
        assert manifest.generation_id == generation
        with pytest.raises(RuntimeError, match="incomplete"):
            publish_snapshot(prepared, layout)
        manifest.complete_generation(directory)
        snapshot = publish_snapshot(prepared, layout)
        assert publish_snapshot(prepared, layout) == snapshot
