# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.build.assessment: every verdict of `assess_processed_folder` on a scratch
layout, the crash-leftover (M1) case that is resumable instead of broken, and the manifest-only mode the planner
uses (`check_files=False`)."""

from __future__ import annotations

from collections.abc import Callable

from data_preparation.dataset_config import DatasetConfig, SourceConfig
from data_preparation.layout import DatasetLayout
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
    """A config with one synthetic source `a`, downloaded (`rows` rows in shards of 4) and built."""
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
    assert (assessment.problem, assessment.verdict, assessment.repair, assessment.reason) == ("none", "ok", "nothing", "ok")
    assert assessment.manifest is not None and assessment.manifest.rows() == 8


def test_absent_folder_is_missing(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}))
    assessment = _assess(cfg, layout, [])
    assert (assessment.problem, assessment.verdict, assessment.repair) == ("absent", "missing", "nothing")
    assert assessment.manifest is None


def test_unreadable_manifest_is_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / MANIFEST_NAME).write_text("{ not json")
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.verdict, assessment.repair) == ("unreadable_manifest", "broken", "rebuild")
    assert assessment.reason == "unreadable manifest" and assessment.manifest is None


def test_shards_without_a_manifest_are_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / MANIFEST_NAME).unlink()
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.repair, assessment.reason) == ("no_manifest", "rebuild", "no manifest")


def test_raw_folder_being_deleted_is_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    assessment = _assess(cfg, layout, None)
    assert (assessment.problem, assessment.verdict, assessment.repair) == ("raw_deleted", "broken", "rebuild")
    assert assessment.reason == "built from a raw folder that is being deleted"


def test_wrong_hash_is_stale(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _built(cfg_factory, with_tokenizer, layout)
    manifest = Manifest.load(layout.processed_dir("a"))
    assert manifest is not None
    manifest.source_hash = "old-processing"
    manifest.save(layout.processed_dir("a"))
    assessment = _assess(cfg, layout, _raw_shards(layout))
    assert (assessment.problem, assessment.verdict, assessment.repair) == ("stale", "stale", "rebuild")
    assert assessment.reason == "stale: processing settings, max_seq_length or the source changed"


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
    """The M1 state: one unlisted file with exactly the name the resumed build publishes next, while raw shards
    are still uncovered — the build overwrites it, so the cheapest repair is to do nothing."""
    cfg = _built(cfg_factory, with_tokenizer, layout)
    manifest = Manifest.load(layout.processed_dir("a"))
    assert manifest is not None and next_shard_to_write(manifest) == "data-00002.parquet"
    (layout.processed_dir("a") / "data-00002.parquet").write_bytes(b"whatever the crashed build wrote")
    pending_raw = [*_raw_shards(layout), ["data-00002.parquet", 4]]  # a third raw shard the build has not covered
    assessment = _assess(cfg, layout, pending_raw)
    assert (assessment.problem, assessment.verdict, assessment.repair) == ("crash_leftover", "resumable", "nothing")
    assert "data-00002.parquet" in assessment.reason and "overwrites" in assessment.reason


def test_the_same_stray_on_a_complete_folder_is_broken(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """A folder that already covers every raw shard gets no further publish: nothing would overwrite the stray, the
    training resolver would refuse the folder, so it stays a rebuild."""
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
    assert (assessment.problem, assessment.verdict, assessment.repair) == ("raw_changed", "broken", "rebuild")
    assert assessment.reason == "built from raw shards that no longer exist"
    assert _assess(cfg, layout, []).problem == "raw_changed"  # the raw folder disappeared entirely


def test_every_problem_has_a_verdict_and_a_repair() -> None:
    """The verdict and the repair derive from the problem, so a consumer can never see them disagree."""
    problems: tuple[ProcessedProblem, ...] = (
        "none", "absent", "crash_leftover", "unreadable_manifest", "no_manifest",
        "raw_deleted", "stale", "broken_shard", "stray_shards", "raw_changed",
    )  # fmt: skip
    for problem in problems:
        assessment = ProcessedAssessment(problem, "x", None)
        assert assessment.verdict in ("ok", "missing", "resumable", "stale", "broken")
        assert assessment.repair == ("nothing" if assessment.verdict in ("ok", "missing", "resumable") else "rebuild")
    assert ProcessedAssessment("crash_leftover", "x", None).repair == "nothing"
    assert ProcessedAssessment("stray_shards", "x", None).repair == "rebuild"
