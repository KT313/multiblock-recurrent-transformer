# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Offline dataset-scope integration: priority, top-ups, recovery and immutable identity."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from data_preparation.conftest import CfgFactory
from data_preparation.dataset_config import DatasetConfig, DedupConfig, ProcessingConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.runner import prepare, status
from data_preparation.lib.stages import global_build
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.global_dedup import GlobalFrontier, global_policy
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.parquet import publish_shard
from data_preparation.lib.storage.snapshot import read_snapshot

Writer = Callable[..., Path]
ConfigFile = Callable[[DatasetConfig], Path]


def _rows(layout: DatasetLayout, name: str) -> list[dict[str, Any]]:
    manifest = Manifest.load(layout.processed_dir(name))
    assert manifest is not None
    return [row for shard in manifest.shards for row in pq.read_table(layout.processed_dir(name) / shard.name).to_pylist()]


def _config(cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path) -> DatasetConfig:
    sources = {}
    for name, texts in {"a": ["shared", "a unique", "a second", "a third"],
                        "b": ["SHARED", "b unique", "b second", "b third"],
                        "val": ["shared", "validation unique"]}.items():
        directory = tmp_path / name
        write_local(directory, [{"text": text} for text in texts])
        sources[name] = SourceConfig(kind="pretrain", loader="local", path=str(directory),
                                     rows=1 if name == "val" else None)
    return cfg_factory(sources, tokens=4, token_count="estimate", training_target_sequence_length=1,
                       processing=ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="none")),
                       bloom_deduplicate_across_sources=True)


def test_priority_scope_noop_and_raw_preservation(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    cfg = _config(cfg_factory, write_local, tmp_path)
    path = config_file(cfg)
    root = tmp_path / "dataset"
    report = prepare(path, root, assume_yes=True)
    assert report.complete
    layout = DatasetLayout(root).for_config(cfg)
    assert [row["text"] for row in _rows(layout, "val")] == ["shared", "validation unique"]
    assert all(row["text"].lower() != "shared" for name in ("a", "b") for row in _rows(layout, name))
    snapshot = read_snapshot(cfg, layout, processing=global_policy(cfg))
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != ".build.lock"}
    assert prepare(path, root, assume_yes=True).complete
    assert prepare(path, root, assume_yes=True, sources=["b"]).complete
    assert status(path, root).complete
    assert read_snapshot(cfg, layout, processing=global_policy(cfg)).build_id == snapshot.build_id
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != ".build.lock"}
    # Removing the high-priority validation source creates independent final output;
    # local candidates restore the previously rejected shared sample without raw writes.
    removed = replace(cfg, sources={name: source for name, source in cfg.sources.items() if name != "val"},
                      stages=[replace(stage, val={"a": 1.0}) for stage in cfg.stages])
    raw_before = {p: p.read_bytes() for p in (root / "sources").rglob("*") if p.is_file()}
    assert prepare(config_file(removed), root, assume_yes=True, steps=["build"]).complete
    other = DatasetLayout(root).for_config(removed)
    assert other.processed_dir("a") != layout.processed_dir("a")
    assert any(row["text"] == "shared" for row in _rows(other, "a"))
    assert {p: p.read_bytes() for p in raw_before} == raw_before
    assert read_snapshot(cfg, layout, processing=global_policy(cfg)).build_id == snapshot.build_id


def test_partial_scope_rejected_before_mutation(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    cfg = _config(cfg_factory, write_local, tmp_path)
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="complete dataset scope"):
        prepare(config_file(cfg), root, assume_yes=True, sources=["b"])
    assert not (root / "sources").exists()
    assert not (root / "tokenizers").exists()


def test_real_admission_failure_and_retry(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(cfg_factory, write_local, tmp_path)
    path = config_file(cfg)
    root = tmp_path / "dataset"
    original = publish_shard
    failed = False

    def fail_after_shard(table: Any, destination: Path) -> Path:
        nonlocal failed
        result = original(table, destination)
        if not failed:
            failed = True
            raise OSError("global publication interrupted")
        return result

    monkeypatch.setattr(global_build, "publish_shard", fail_after_shard)
    with pytest.raises(OSError, match="publication interrupted"):
        prepare(path, root, assume_yes=True)
    assert not status(path, root).complete
    monkeypatch.setattr(global_build, "publish_shard", original)
    assert prepare(path, root, assume_yes=True).complete
    resumed = DatasetLayout(root).for_config(cfg)
    baseline_root = tmp_path / "baseline"
    assert prepare(path, baseline_root, assume_yes=True).complete
    baseline = DatasetLayout(baseline_root).for_config(cfg)
    assert {name: _rows(resumed, name) for name in cfg.sources} == {name: _rows(baseline, name) for name in cfg.sources}


def test_ordered_zero_yield_topup_reaches_later_unique_rows(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    for name, texts in {"a": ["shared", "a unique"], "b": ["shared"] * 5 + ["b unique", "b second"]}.items():
        write_local(tmp_path / name, [{"text": text} for text in texts])
    cfg = cfg_factory({name: SourceConfig(kind="pretrain", loader="local", path=str(tmp_path / name)) for name in ("a", "b")},
                      tokens=2, training_target_sequence_length=1, token_count="estimate",
                      processing=ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="none")),
                      bloom_deduplicate_across_sources=True)
    root = tmp_path / "dataset"
    assert prepare(config_file(cfg), root, assume_yes=True).complete
    layout = DatasetLayout(root).for_config(cfg)
    assert [row["text"] for row in _rows(layout, "b")] == ["b unique", "b second"]


def test_corrupt_frontier_fails_visibly(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    cfg = _config(cfg_factory, write_local, tmp_path)
    root = tmp_path / "dataset"
    path = config_file(cfg)
    assert prepare(path, root, assume_yes=True).complete
    manifest_path = DatasetLayout(root).for_config(cfg).processed_dir("a") / "MANIFEST.json"
    payload = json.loads(manifest_path.read_text())
    payload["extra"]["global_frontier"]["retained"] = "corrupt"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="corrupt global"):
        prepare(path, root, assume_yes=True)


def test_reverse_candidate_worker_completion_preserves_output(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from data_preparation.lib.build import runner

    cfg = _config(cfg_factory, write_local, tmp_path)
    path = config_file(cfg)
    baseline_root, reversed_root = tmp_path / "baseline", tmp_path / "reversed"
    assert prepare(path, baseline_root, assume_yes=True, num_workers=3).complete
    completed_b = threading.Event()
    original = build_source
    completion: list[str] = []

    def reordered(config: DatasetConfig, name: str, layout: DatasetLayout, **kwargs: Any) -> Manifest:
        if name == "a":
            assert completed_b.wait(5), "b candidate worker never completed"
        result = original(config, name, layout, **kwargs)
        completion.append(name)
        if name == "b":
            completed_b.set()
        return result

    monkeypatch.setattr(runner, "build_source", reordered)
    assert prepare(path, reversed_root, assume_yes=True, num_workers=3).complete
    assert completion.index("b") < completion.index("a")
    left, right = DatasetLayout(baseline_root).for_config(cfg), DatasetLayout(reversed_root).for_config(cfg)
    assert {name: _rows(left, name) for name in cfg.sources} == {name: _rows(right, name) for name in cfg.sources}


def test_finish_frontier_recovers_before_generation_completion(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(cfg_factory, write_local, tmp_path)
    path, root = config_file(cfg), tmp_path / "dataset"
    original = Manifest.complete_generation
    failed_id: str | None = None

    def fail_once(manifest: Manifest, directory: Path) -> None:
        nonlocal failed_id
        if ".dataset-scopes" in directory.parts and failed_id is None:
            failed_id = manifest.generation_id
            raise OSError("before complete marker")
        original(manifest, directory)

    monkeypatch.setattr(Manifest, "complete_generation", fail_once)
    with pytest.raises(OSError, match="complete marker"):
        prepare(path, root, assume_yes=True)
    monkeypatch.setattr(Manifest, "complete_generation", original)
    assert prepare(path, root, assume_yes=True).complete
    first = Manifest.load(DatasetLayout(root).for_config(cfg).processed_dir("val"))
    assert first is not None and first.generation_id == failed_id


def test_disabled_switch_uses_local_outputs_and_keeps_cross_source_duplicates(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    cfg = replace(_config(cfg_factory, write_local, tmp_path), bloom_deduplicate_across_sources=False)
    root = tmp_path / "dataset"
    assert prepare(config_file(cfg), root, assume_yes=True).complete
    layout = DatasetLayout(root).for_config(cfg)
    assert layout.processed_scope is None
    assert all(any(row["text"].lower() == "shared" for row in _rows(layout, name)) for name in cfg.sources)
    assert not (root / ".dataset-scopes").exists()


def test_global_scope_symlink_refused_without_external_mutation(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    from data_preparation.lib.storage.ownership import OwnershipError

    cfg = _config(cfg_factory, write_local, tmp_path)
    root, outside = tmp_path / "dataset", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "keep").write_text("untouched")
    (root / ".dataset-scopes").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OwnershipError, match="symlink"):
        prepare(config_file(cfg), root, assume_yes=True)
    assert list(outside.iterdir()) == [outside / "keep"]
    assert (outside / "keep").read_text() == "untouched"


def test_complete_instruction_keys_and_no_benchmark_reads(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_preparation.lib.stages import build

    def unexpected_benchmark(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("global dedup must not load benchmarks")

    monkeypatch.setattr(build, "load_benchmark_ngrams", unexpected_benchmark)
    records = {
        "a": [{"instruction": "Q", "input": "", "output": "one"},
              {"instruction": "Q", "input": "", "output": "two"},
              {"instruction": "a b", "input": "c", "output": "d"}],
        "b": [{"instruction": " q ", "input": None, "output": "ONE"},
              {"instruction": "a", "input": "b c", "output": "d"},
              {"instruction": "unique", "input": "", "output": "answer"}],
    }
    for name, rows in records.items():
        write_local(tmp_path / name, rows)
    cfg = cfg_factory({name: SourceConfig(kind="instruct", loader="local", path=str(tmp_path / name), shuffle=False,
                                         fields={"instruction": "instruction", "input": "input", "output": "output"})
                       for name in records}, tokens=6, token_count="estimate", training_target_sequence_length=1,
                      processing=ProcessingConfig(dedup=DedupConfig(mode="exact", normalize=False)),
                      bloom_deduplicate_across_sources=True)
    root = tmp_path / "dataset"
    assert prepare(config_file(cfg), root, assume_yes=True).complete
    layout = DatasetLayout(root).for_config(cfg)
    assert len(_rows(layout, "a")) == 3
    assert [(row["instruction"], row["output"]) for row in _rows(layout, "b")] == [("a", "d"), ("unique", "answer")]


def test_reopened_higher_priority_source_reclaims_late_duplicate(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    write_local(tmp_path / "val", [{"text": "first validation"}])
    write_local(tmp_path / "a", [{"text": text} for text in ("late duplicate", "unique one", "unique two")])
    cfg = cfg_factory({
        "a": SourceConfig(kind="pretrain", loader="local", path=str(tmp_path / "a")),
        "val": SourceConfig(kind="pretrain", loader="local", path=str(tmp_path / "val"), rows=3),
    }, tokens=2, token_count="estimate", training_target_sequence_length=1,
        processing=ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="none")),
        bloom_deduplicate_across_sources=True)
    root, path = tmp_path / "dataset", config_file(cfg)
    assert prepare(path, root, assume_yes=True).complete  # validation source is honestly exhausted short
    layout = DatasetLayout(root).for_config(cfg)
    old_id = read_snapshot(cfg, layout, processing=global_policy(cfg)).build_id
    assert "late duplicate" in [row["text"] for row in _rows(layout, "a")]
    write_local(tmp_path / "val", [{"text": "late duplicate"}, {"text": "another validation"}])
    assert prepare(path, root, assume_yes=True, reopen=["val"]).complete
    assert "late duplicate" in [row["text"] for row in _rows(layout, "val")]
    assert "late duplicate" not in [row["text"] for row in _rows(layout, "a")]
    assert read_snapshot(cfg, layout, processing=global_policy(cfg)).build_id != old_id


def test_reordering_replays_owned_output_without_touching_raw(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    for name, texts in {"a": ["train shared", "a one", "a two"],
                        "b": ["train shared", "b one", "b two"], "val": ["validation"]}.items():
        write_local(tmp_path / name, [{"text": text} for text in texts])
    cfg = cfg_factory({name: SourceConfig(kind="pretrain", loader="local", path=str(tmp_path / name),
                                         rows=1 if name == "val" else None) for name in ("a", "b", "val")},
                      tokens=6, token_count="estimate", training_target_sequence_length=1,
                      processing=ProcessingConfig(min_chars=1, dedup=DedupConfig(mode="none")),
                      bloom_deduplicate_across_sources=True)
    root = tmp_path / "dataset"
    assert prepare(config_file(cfg), root, assume_yes=True).complete
    layout = DatasetLayout(root).for_config(cfg)
    old_id = read_snapshot(cfg, layout, processing=global_policy(cfg)).build_id
    assert "train shared" in [row["text"] for row in _rows(layout, "a")]
    # The initial request exceeds these finite sources, so their raw manifests
    # already prove exhaustion; replay needs no fetch merely to discover EOF.
    raw = {path: path.read_bytes() for path in (root / "sources").rglob("*") if path.is_file()}
    reordered = replace(cfg, sources={name: cfg.sources[name] for name in ("b", "a", "val")})
    assert prepare(config_file(reordered), root, assume_yes=True, steps=["build"]).complete
    other = DatasetLayout(root).for_config(reordered)
    assert "train shared" in [row["text"] for row in _rows(other, "b")]
    assert "train shared" not in [row["text"] for row in _rows(other, "a")]
    assert {path: path.read_bytes() for path in raw} == raw
    assert read_snapshot(cfg, layout, processing=global_policy(cfg)).build_id == old_id


def test_unreadable_global_manifest_is_reported_and_repaired(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
) -> None:
    from data_preparation.lib.build.repair import ConfirmationRequired

    cfg = _config(cfg_factory, write_local, tmp_path)
    root, path = tmp_path / "dataset", config_file(cfg)
    assert prepare(path, root, assume_yes=True).complete
    layout = DatasetLayout(root).for_config(cfg)
    expected = _rows(layout, "a")
    manifest = layout.processed_dir("a") / "MANIFEST.json"
    manifest.write_text("{ invalid json")
    report = status(path, root)
    assert not report.complete and "a" in report.needs_repair
    with pytest.raises(ConfirmationRequired):
        prepare(path, root, assume_yes=False, confirm=lambda message: False)
    assert manifest.read_text() == "{ invalid json"
    assert prepare(path, root, assume_yes=True).complete
    assert _rows(layout, "a") == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_scope_is_explicit_when_dataset_root_has_reserved_name(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile, enabled: bool,
) -> None:
    cfg = replace(_config(cfg_factory, write_local, tmp_path), bloom_deduplicate_across_sources=enabled)
    root, path = tmp_path / ".dataset-scopes", config_file(cfg)
    assert prepare(path, root, assume_yes=True).complete
    assert status(path, root).complete
    assert prepare(path, root, assume_yes=False).complete


def _interrupt_later_global_source(
    cfg: DatasetConfig, path: Path, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, *, finished: bool,
) -> None:
    from data_preparation.lib.build import runner

    original_save = Manifest.save

    def one_row_batches(
        config: DatasetConfig, name: str, scoped: DatasetLayout, start: GlobalFrontier, **kwargs: Any,
    ) -> tuple[GlobalFrontier, bool]:
        return global_build.build_global_source(config, name, scoped, start, batch_rows=1, **kwargs)

    def interrupt_after_commit(manifest: Manifest, directory: Path) -> Path:
        result = original_save(manifest, directory)
        if directory == layout.processed_dir("b") and not manifest.generation_complete:
            frontier = manifest.extra["global_frontier"]
            start = manifest.extra["global_start"]
            stop = (frontier["source_index"] == start["source_index"] + 1 if finished
                    else frontier["source_candidates"] == 1)
            if stop:
                raise OSError("later source frontier committed; interrupted")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(runner, "build_global_source", one_row_batches)
        patch.setattr(Manifest, "save", interrupt_after_commit)
        with pytest.raises(OSError, match="later source frontier committed"):
            prepare(path, layout.root, assume_yes=True)
    manifest = Manifest.load(layout.processed_dir("b"))
    assert manifest is not None and not manifest.generation_complete
    assert manifest.extra["global_start"]["source_index"] > 0
    assert manifest.extra["global_start"]["source_candidates"] == 0
    assert manifest.extra["global_start"]["candidates"] > 1


@pytest.mark.parametrize("corruption", ["unfinished_offset", "finished_offset", "boundary_offset"])
def test_later_source_candidate_offset_corruption_fails_before_mutation(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    cfg = _config(cfg_factory, write_local, tmp_path)
    path = config_file(cfg)
    layout = DatasetLayout(tmp_path / "dataset").for_config(cfg)
    _interrupt_later_global_source(cfg, path, layout, monkeypatch, finished=corruption == "finished_offset")
    manifest = Manifest.load(layout.processed_dir("b"))
    assert manifest is not None
    frontier = manifest.extra["global_frontier"]
    retained_state = (frontier["retained"], frontier["key_digest"])
    if corruption == "boundary_offset":
        manifest.extra["global_start"]["source_candidates"] = 1
    else:
        frontier["source_candidates"] += 1
        # The old cumulative bound accepts this offset although it skips a pending
        # candidate (unfinished) or violates the completed source boundary.
        assert frontier["source_candidates"] <= frontier["candidates"]
    assert (frontier["retained"], frontier["key_digest"]) == retained_state
    manifest.save(layout.processed_dir("b"))
    before = {p: p.read_bytes() for p in layout.root.rglob("*") if p.is_file() and p.name != ".build.lock"}
    with pytest.raises(ValueError, match="corrupt global recovery frontier"):
        global_build.source_frontier(manifest)
    with pytest.raises(ValueError, match="corrupt global recovery frontier"):
        prepare(path, layout.root, assume_yes=True, steps=["build"])
    assert before == {p: p.read_bytes() for p in layout.root.rglob("*") if p.is_file() and p.name != ".build.lock"}


@pytest.mark.parametrize("finished", [False, True])
def test_later_source_valid_candidate_offset_restart_matches_uninterrupted(
    cfg_factory: CfgFactory, write_local: Writer, tmp_path: Path, config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch, finished: bool,
) -> None:
    cfg = _config(cfg_factory, write_local, tmp_path)
    path = config_file(cfg)
    layout = DatasetLayout(tmp_path / "dataset").for_config(cfg)
    _interrupt_later_global_source(cfg, path, layout, monkeypatch, finished=finished)
    assert prepare(path, layout.root, assume_yes=True).complete
    baseline = DatasetLayout(tmp_path / "baseline").for_config(cfg)
    assert prepare(path, baseline.root, assume_yes=True).complete
    assert {name: _rows(layout, name) for name in cfg.sources} == {name: _rows(baseline, name) for name in cfg.sources}
