# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.build.assessment: every verdict of `assess_processed_folder` on a scratch
layout, the crash-leftover (M1) case that is resumable instead of broken, and the manifest-only mode the planner
uses (`check_files=False`).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from data_preparation.dataset_config import DatasetConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.runner import prepare, status
from data_preparation.lib.build.planner import source_ledger
from data_preparation.lib.build.repair import repair_broken_and_stale_folders
from data_preparation.lib.build.assessment import (
    ProcessedAssessment,
    ProcessedProblem,
    ShardList,
    assess_processed_folder,
    next_shard_to_write,
)
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.download import download
from data_preparation.lib.storage.manifest import MANIFEST_NAME, Manifest

CfgFactory = Callable[..., DatasetConfig]
Prep = Callable[[DatasetConfig], DatasetConfig]


def _built(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, rows: int = 8) -> DatasetConfig:
    """
    A config with one synthetic source `a`, downloaded (`rows` rows in shards of 4) and built.
    """

    cfg = with_tokenizer(cfg_factory({"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}))
    download(cfg, "a", layout, rows_needed=rows, shard_size=4)
    build_source(cfg, "a", layout, shard_size=4)
    return cfg


def _raw_shards(layout: DatasetLayout, name: str = "a") -> ShardList:
    manifest = Manifest.load(layout.raw_dir(name))
    assert manifest is not None
    return [[shard.name, shard.rows] for shard in manifest.shards]


def _assess(cfg: DatasetConfig, layout: DatasetLayout, raw_shards: ShardList | None, *, check_files: bool = True) -> ProcessedAssessment:
    return assess_processed_folder(cfg, "a", layout.processed_dir("a"), raw_shards, check_files=check_files)


# --- the verdicts, one folder state each --------------------------------------------------------------------------------


def test_healthy_folder_is_ok(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.repair, assessment.reason) == ("none", "nothing", "ok")
    assert assessment.manifest is not None and assessment.manifest.rows() == 8


def test_absent_folder_is_missing(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}))
    assessment = _assess(cfg, layout, [])
    assert (assessment.problem, assessment.repair) == ("absent", "nothing")
    assert assessment.manifest is None


def test_unreadable_manifest_is_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / MANIFEST_NAME).write_text("{ not json")
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.repair) == ("unreadable_manifest", "rebuild")
    assert assessment.reason == "unreadable manifest" and assessment.manifest is None


def test_shards_without_a_manifest_are_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / MANIFEST_NAME).unlink()
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.repair, assessment.reason) == ("no_manifest", "rebuild", "no manifest")


def test_raw_folder_being_deleted_is_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    assessment = _assess(cfg, layout, None)
    assert (assessment.problem, assessment.repair) == ("raw_deleted", "rebuild")
    assert assessment.reason == "built from a raw folder that is being deleted"


def test_wrong_hash_is_stale(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    manifest = Manifest.load(layout.processed_dir("a"))
    assert manifest is not None
    manifest.source_hash = "old-processing"
    manifest.save(layout.processed_dir("a"))
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.repair) == ("stale", "rebuild")
    assert assessment.reason == "stale: processing settings, dataset_max_sequence_length or the source changed"


def test_missing_listed_shard_is_broken_only_when_files_are_checked(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / "data-00000.parquet").unlink()
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.repair) == ("broken_shard", "rebuild")
    assert assessment.reason == "broken: missing shard data-00000.parquet"
    # the planner reads manifests only: file problems are the repair step's business
    manifest_only = _assess(cfg, layout, _raw_shards(layout), check_files=False)
    assert manifest_only.problem == "none"


def test_crash_leftover_next_shard_is_resumable(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """
    The M1 state: one unlisted file with exactly the name the resumed build publishes next, while raw shards
    are still uncovered. The build overwrites it, so the cheapest repair is to do nothing.
    """

    cfg = _built(cfg_factory, with_tokenizer, layout)
    manifest = Manifest.load(layout.processed_dir("a"))
    assert manifest is not None and next_shard_to_write(manifest) == "data-00002.parquet"
    (layout.processed_dir("a") / "data-00002.parquet").write_bytes(b"whatever the crashed build wrote")
    pending_raw = [*_raw_shards(layout), ["data-00002.parquet", 4]]  # a third raw shard the build has not covered
    assessment = _assess(cfg, layout, pending_raw)
    assert (assessment.problem, assessment.repair) == ("crash_leftover", "nothing")
    assert "data-00002.parquet" in assessment.reason and "overwrites" in assessment.reason


def test_the_same_stray_on_a_complete_folder_is_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """
    A folder that already covers every raw shard gets no further publish: nothing would overwrite the stray, the
    training resolver would refuse the folder, so it stays a rebuild.
    """

    cfg = _built(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / "data-00002.parquet").write_bytes(b"stray")
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.repair) == ("stray_shards", "rebuild")
    assert assessment.reason == "unlisted shard(s): data-00002.parquet"


def test_other_unlisted_shards_are_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / "data-00099.parquet").write_bytes(b"stray")
    pending_raw = [*_raw_shards(layout), ["data-00002.parquet", 4]]
    assessment = _assess(cfg, layout, pending_raw)  # not the next shard, even though a resume is pending
    assert (assessment.problem, assessment.repair) == ("stray_shards", "rebuild")
    assert assessment.reason == "unlisted shard(s): data-00099.parquet"
    (layout.processed_dir("a") / "data-00002.parquet").write_bytes(b"stray")  # two strays: more than one crash explains
    two = _assess(cfg, layout, pending_raw)
    assert two.problem == "stray_shards" and two.reason == "unlisted shard(s): data-00002.parquet, data-00099.parquet"


def test_covered_shards_must_be_a_prefix_of_the_raw_shards(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    changed = [["data-00000.parquet", 3], ["data-00001.parquet", 4]]  # row count of raw shard 0 differs
    assessment = _assess(cfg, layout, changed)
    assert (assessment.problem, assessment.repair) == ("raw_changed", "rebuild")
    assert assessment.reason == "built from raw shards that no longer exist"
    assert _assess(cfg, layout, []).problem == "raw_changed"  # the raw folder disappeared entirely


# --- every consumer reads the same verdict ------------------------------------------------------------------------------


def test_repair_and_planner_agree_with_the_verdict_on_canonical_states(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """
    One tree, one folder state per source: the repair step deletes exactly the folders whose cheapest repair is a
    rebuild, and the planner's manifest-only state matches the verdict's manifest-level knowledge (file-level
    problems are invisible to it by design; the repair dry report flags them in status).
    """

    names = ("a", "b", "c", "d", "e", "f", "g")
    cfg = with_tokenizer(cfg_factory({name: SourceConfig(kind="pretrain", loader="synthetic", seed=seed) for seed, name in enumerate(names)}))
    for name in names:
        download(cfg, name, layout, rows_needed=8, shard_size=4)
        build_source(cfg, name, layout, shard_size=4)
    _make_stale(layout.processed_dir("b"))
    (layout.processed_dir("c") / MANIFEST_NAME).write_text("{ not json")
    (layout.processed_dir("d") / MANIFEST_NAME).unlink()
    (layout.processed_dir("e") / "data-00000.parquet").unlink()
    (layout.processed_dir("f") / "data-00099.parquet").write_bytes(b"stray")
    _make_crash_leftover(layout.processed_dir("g"))

    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, dry_run=True)
    would_rebuild = {action.source for action in report.actions}
    assert not report.performed and all(action.kind == "processed" and action.action == "delete" for action in report.actions)
    for name in names:
        assessment = assess_processed_folder(cfg, name, layout.processed_dir(name), _raw_shards(layout, name))
        assert (assessment.repair == "rebuild") == (name in would_rebuild), name
    assert would_rebuild == {"b", "c", "d", "e", "f"}
    assert {name: source_ledger(cfg, name, layout).processed_problem for name in names} == {
        "a": "none",
        "b": "stale",
        "c": "unreadable_manifest",
        "d": "no_manifest",
        "e": "none",  # manifest-only: the missing shard file is the repair step's finding
        "f": "none",  # manifest-only: so is the stray
        "g": "behind_raw",  # the crash leftover looks like any pending build, which is exactly what heals it
    }


def test_status_dry_run_and_prepare_agree_on_the_crash_leftover(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, config_file: Callable[[DatasetConfig], Path]
) -> None:
    """
    The M1 state through the entry points: status and prepare --dry_run report a pending build and no
    repair, prepare resumes the build over the leftover and ends complete, all three from the same verdict.
    """

    cfg = with_tokenizer(cfg_factory({"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=6))
    path = config_file(cfg)
    download(cfg, "a", layout, rows_needed=8, shard_size=4)  # 2 raw shards; rows_needed(cfg) is 8 too
    build_source(cfg, "a", layout, shard_size=4)
    _make_crash_leftover(layout.processed_dir("a"))

    for report in (status(path, layout.root), prepare(path, layout.root, assume_yes=False, dry_run=True)):
        assert not report.complete and report.needs_repair == []
        state = next(source for source in report.sources if source.name == "a")
        assert state.satisfaction() == (False, "processed behind raw")
    healed = prepare(path, layout.root, assume_yes=False)
    assert healed.complete
    manifest = Manifest.load(layout.processed_dir("a"))
    assert manifest is not None and len(manifest.input_shards) == 2
    listed = {shard.name for shard in manifest.shards}
    assert {p.name for p in layout.processed_dir("a").glob("*.parquet")} == listed


def _make_stale(folder: Path) -> None:
    manifest = Manifest.load(folder)
    assert manifest is not None
    manifest.source_hash = "old-processing"
    manifest.save(folder)


def _make_crash_leftover(folder: Path) -> None:
    """
    Reconstruct the crash between publishing a shard and saving the manifest: the last processed shard file
    stays on disk, the manifest no longer lists it nor covers the raw shard it came from.
    """

    manifest = Manifest.load(folder)
    assert manifest is not None and len(manifest.shards) >= 2
    manifest.shards = manifest.shards[:-1]
    manifest.input_shards = manifest.input_shards[:-1]
    manifest.save(folder)


def test_every_problem_has_a_verdict_and_a_repair() -> None:
    """
    The verdict and the repair derive from the problem, so a consumer can never see them disagree.
    """

    problems: tuple[ProcessedProblem, ...] = (
        "none", "absent", "crash_leftover", "unreadable_manifest", "no_manifest",
        "raw_deleted", "stale", "broken_shard", "stray_shards", "raw_changed",
    )  # fmt: skip
    for problem in problems:
        assessment = ProcessedAssessment(problem, "x", None)
        assert assessment.repair == ("nothing" if assessment.problem in ("none", "absent", "crash_leftover", "behind_raw") else "rebuild")
    assert ProcessedAssessment("crash_leftover", "x", None).repair == "nothing"
    assert ProcessedAssessment("stray_shards", "x", None).repair == "rebuild"
