# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.build.repair: every raw / processed branch on a scratch layout, the single
confirmation for all queued raw deletions, the non-interactive and declined aborts, and the read-only dry run."""

from __future__ import annotations

import io
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_preparation.dataset_config import DatasetConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.repair import (
    CONFIRMATION_HEADER,
    CONFIRMATION_QUESTION,
    ConfirmationRequired,
    RepairAction,
    RepairError,
    RepairReport,
    repair_broken_and_stale_folders,
)
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.download import download
from data_preparation.lib.storage.manifest import MANIFEST_NAME, Manifest

CfgFactory = Callable[..., DatasetConfig]
Prep = Callable[[DatasetConfig], DatasetConfig]
Snapshot = dict[str, int]


def _synthetic(seed: int = 0) -> SourceConfig:
    return SourceConfig(kind="pretrain", loader="synthetic", seed=seed)


def _prepared(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, names: tuple[str, ...] = ("a",), rows: int = 8) -> DatasetConfig:
    """A config over synthetic sources ``names``, each downloaded (``rows`` rows in shards of 4) and built."""
    cfg = with_tokenizer(cfg_factory({name: _synthetic(seed=index) for index, name in enumerate(names)}))
    for name in names:
        download(cfg, name, layout, rows_needed=rows, shard_size=4)
        build_source(cfg, name, layout, shard_size=4)
    return cfg


def _edit_manifest(folder: Path, **changes: Any) -> None:
    manifest = Manifest.load(folder)
    assert manifest is not None
    for key, value in changes.items():
        setattr(manifest, key, value)
    manifest.save(folder)


def _snapshot(root: Path) -> Snapshot:
    """``{relative path: mtime_ns}`` of every file under ``root`` (proves a run touched nothing)."""
    return {str(path.relative_to(root)): path.stat().st_mtime_ns for path in sorted(root.rglob("*")) if path.is_file()}


def _kinds(report: RepairReport) -> list[tuple[str, str, str]]:
    return [(action.source, action.kind, action.action) for action in report.actions]


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def _recording_confirm(prompts: list[str], answer: bool) -> Callable[[str], bool]:
    """A ``confirm`` callable that records every message it is asked and answers ``answer``."""

    def confirm(message: str) -> bool:
        prompts.append(message)
        return answer

    return confirm


# --- nothing to do ------------------------------------------------------------------------------------------------------


def test_healthy_folders_are_left_alone(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, ("a", "b"))
    (layout.root / "sources" / "other" / "raw").mkdir(parents=True)  # a source no config uses is never inspected
    (layout.root / "sources" / "other" / "raw" / "data-00000.parquet").write_bytes(b"not ours")
    before = _snapshot(layout.root)
    calls: list[str] = []
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=_recording_confirm(calls, True))
    assert report.actions == [] and report.describe() == "nothing to repair" and calls == []
    assert report.raw_deleted() == [] and report.processed_deleted() == []
    assert _snapshot(layout.root) == before


def test_missing_folders_are_not_a_problem(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"a": _synthetic()}))
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert report.actions == []


# --- raw folder branches ------------------------------------------------------------------------------------------------


def test_raw_shards_without_a_manifest_are_an_error(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    (layout.raw_dir("a") / MANIFEST_NAME).unlink()
    with pytest.raises(RepairError, match="holds shards but no manifest"):
        repair_broken_and_stale_folders(cfg, layout, assume_yes=True)
    assert (layout.raw_dir("a") / "data-00000.parquet").exists() and layout.processed_dir("a").exists()


def test_stale_raw_is_deleted_with_its_processed_folder_after_confirmation(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    _edit_manifest(layout.raw_dir("a"), source_hash="somebody-else")
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=lambda message: True)
    assert _kinds(report) == [("a", "raw", "delete"), ("a", "processed", "delete")]
    assert report.actions[0].reason == "stale: source identity or tokenizer changed"
    assert report.actions[1].reason == "built from a raw folder that is being deleted"
    assert not layout.raw_dir("a").exists() and not layout.processed_dir("a").exists()
    assert [action.source for action in report.raw_deleted()] == ["a"] and [action.folder for action in report.processed_deleted()] == [layout.processed_dir("a")]


def test_outdated_raw_is_queued_only_when_the_cap_was_raised(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)  # max_seq_length 64
    _edit_manifest(layout.raw_dir("a"), truncated_at_tokens=64)
    assert repair_broken_and_stale_folders(cfg, layout, assume_yes=True).actions == []  # equal cap: fine
    _edit_manifest(layout.raw_dir("a"), truncated_at_tokens=128)
    assert repair_broken_and_stale_folders(cfg, layout, assume_yes=True).actions == []  # lowered cap: fine
    _edit_manifest(layout.raw_dir("a"), truncated_at_tokens=32)
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=True)
    assert _kinds(report) == [("a", "raw", "delete"), ("a", "processed", "delete")]
    assert report.actions[0].reason == "outdated: max_seq_length 32 -> 64"
    assert not layout.raw_dir("a").exists()


def test_broken_raw_shard_mid_folder_truncates_after_the_one_confirmation(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, caplog: pytest.LogCaptureFixture
) -> None:
    """Truncating to the good prefix before a mid-folder broken shard drops the healthy shards after it too — that
    loss of downloaded rows joins the one confirmation instead of happening silently."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout, rows=12)  # 3 raw shards
    raw = layout.raw_dir("a")
    (raw / "data-00001.parquet").write_bytes(b"corrupt")
    calls: list[str] = []
    with caplog.at_level("WARNING", logger="data_preparation"):
        report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=_recording_confirm(calls, True))
    assert len(calls) == 1, "healthy shards after the broken one would be dropped: the one question is asked"
    assert calls[0].startswith(CONFIRMATION_HEADER) and calls[0].endswith(CONFIRMATION_QUESTION)
    assert "1 healthy shard(s) after the broken one are discarded and re-downloaded next run" in calls[0]
    assert _kinds(report) == [("a", "raw", "truncate"), ("a", "processed", "delete")]
    assert report.actions[0].needs_confirmation and not report.actions[1].needs_confirmation
    assert report.actions[0].reason.startswith("broken: unreadable shard data-00001.parquet") and "dropping 2 shard(s) data-00001.parquet..data-00002.parquet, keeping 1" in report.actions[0].reason
    assert report.actions[1].reason == "built from raw shards that no longer exist"
    assert "data-00001.parquet" in caplog.text
    manifest = Manifest.load(raw)
    assert manifest is not None and [shard.name for shard in manifest.shards] == ["data-00000.parquet"] and manifest.rows_fetched == 4
    assert sorted(path.name for path in raw.glob("*.parquet")) == ["data-00000.parquet"]
    assert not layout.processed_dir("a").exists() and report.raw_deleted() == []


def test_truncation_dropping_only_the_broken_tail_shard_asks_nothing(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, rows=12)  # 3 raw shards
    (layout.raw_dir("a") / "data-00002.parquet").write_bytes(b"corrupt")
    calls: list[str] = []
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=_recording_confirm(calls, False))
    assert calls == [], "only the broken shard itself is dropped: a repair, not a deletion"
    assert _kinds(report) == [("a", "raw", "truncate"), ("a", "processed", "delete")]
    assert not report.actions[0].needs_confirmation and "healthy shard(s)" not in report.actions[0].reason
    manifest = Manifest.load(layout.raw_dir("a"))
    assert manifest is not None and [shard.name for shard in manifest.shards] == ["data-00000.parquet", "data-00001.parquet"]


def test_refused_truncation_past_healthy_shards_changes_nothing(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, rows=12)
    (layout.raw_dir("a") / "data-00001.parquet").write_bytes(b"corrupt")
    before = _snapshot(layout.root)
    with pytest.raises(ConfirmationRequired) as info:
        repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=lambda message: False)
    assert _snapshot(layout.root) == before, "refused: not even the prefix truncation of that folder"
    assert _kinds(info.value.report) == [("a", "raw", "would_truncate"), ("a", "processed", "would_delete")]
    assert [action.source for action in info.value.report.raw_confirmations_planned()] == ["a"]
    assert info.value.report.raw_deletions_planned() == [], "a truncation is not a deletion"
    # non-interactive without --yes: the same abort with nothing changed
    monkeypatch.setattr(sys, "stdin", io.StringIO())  # not a tty
    with pytest.raises(ConfirmationRequired) as info2:
        repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert not info2.value.interactive and _snapshot(layout.root) == before
    # --yes proceeds without a question
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=True)
    assert _kinds(report) == [("a", "raw", "truncate"), ("a", "processed", "delete")]
    manifest = Manifest.load(layout.raw_dir("a"))
    assert manifest is not None and [shard.name for shard in manifest.shards] == ["data-00000.parquet"]


def test_dry_run_reports_would_truncate_with_the_healthy_loss_note(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, rows=12)
    (layout.raw_dir("a") / "data-00001.parquet").write_bytes(b"corrupt")
    before = _snapshot(layout.root)

    def confirm(message: str) -> bool:
        raise AssertionError("a dry run never asks")

    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, dry_run=True, confirm=confirm)
    assert _snapshot(layout.root) == before
    assert _kinds(report) == [("a", "raw", "would_truncate"), ("a", "processed", "would_delete")]
    assert "1 healthy shard(s) after the broken one are discarded and re-downloaded next run" in report.actions[0].reason
    assert report.actions[0].needs_confirmation


def test_one_prompt_covers_deletions_and_confirmable_truncations(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, ("a", "b"), rows=12)  # 3 raw shards each
    _edit_manifest(layout.raw_dir("a"), source_hash="changed")
    (layout.raw_dir("b") / "data-00001.parquet").write_bytes(b"corrupt")
    prompts: list[str] = []
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=_recording_confirm(prompts, True))
    assert len(prompts) == 1, "one question per run, deletions and truncations together"
    assert "a: stale: source identity or tokenizer changed" in prompts[0]
    assert "b: broken: unreadable shard data-00001.parquet" in prompts[0] and "re-downloaded next run" in prompts[0]
    assert not layout.raw_dir("a").exists()
    manifest = Manifest.load(layout.raw_dir("b"))
    assert manifest is not None and len(manifest.shards) == 1
    assert [action.source for action in report.raw_confirmations_planned()] == ["a", "b"]


def test_processed_covering_only_the_kept_prefix_survives_a_truncation(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = with_tokenizer(cfg_factory({"a": _synthetic()}))
    download(cfg, "a", layout, rows_needed=4, shard_size=4)
    build_source(cfg, "a", layout, shard_size=4)  # covers data-00000 only
    download(cfg, "a", layout, rows_needed=12, shard_size=4)  # two more raw shards, not built yet
    (layout.raw_dir("a") / "data-00002.parquet").unlink()
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "raw", "truncate")]
    assert layout.processed_dir("a").exists()


def test_broken_first_raw_shard_queues_the_folder_for_deletion(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    (layout.raw_dir("a") / "data-00000.parquet").unlink()
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=True)
    assert _kinds(report) == [("a", "raw", "delete"), ("a", "processed", "delete")]
    assert report.actions[0].reason == "broken: missing shard data-00000.parquet"
    assert not layout.raw_dir("a").exists() and not layout.processed_dir("a").exists()


def test_raw_without_a_resume_offset_cannot_be_truncated(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    manifest = Manifest.load(layout.raw_dir("a"))
    assert manifest is not None
    manifest.shards[0].offset = None  # a legacy manifest: no safe resume point
    manifest.save(layout.raw_dir("a"))
    (layout.raw_dir("a") / "data-00001.parquet").write_bytes(b"corrupt")
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=True)
    assert _kinds(report)[0] == ("a", "raw", "delete") and report.actions[0].reason.startswith("broken: unreadable shard data-00001.parquet")


# --- processed folder branches -----------------------------------------------------------------------------------------


def test_stale_processed_is_deleted_without_confirmation(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    _edit_manifest(layout.processed_dir("a"), source_hash="old-processing")
    raw_before = _snapshot(layout.raw_dir("a"))
    calls: list[str] = []
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=_recording_confirm(calls, False))
    assert calls == [] and _kinds(report) == [("a", "processed", "delete")]
    assert report.actions[0].reason.startswith("stale:") and not layout.processed_dir("a").exists()
    assert _snapshot(layout.raw_dir("a")) == raw_before


def test_broken_processed_shard_deletes_the_folder(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    processed = layout.processed_dir("a")
    first_shard = sorted(processed.glob("data-*.parquet"))[0]
    first_shard.unlink()
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")] and report.actions[0].reason == f"broken: missing shard {first_shard.name}"
    assert not processed.exists() and layout.raw_dir("a").exists()


def test_processed_shards_without_a_manifest_are_deleted(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    (layout.processed_dir("a") / MANIFEST_NAME).unlink()
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")] and report.actions[0].reason == "no manifest"
    assert not layout.processed_dir("a").exists()


def test_an_unlisted_processed_shard_deletes_the_folder(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """A shard on disk that the manifest does not list (a crash between publishing and saving) used to be invisible
    here while the training resolver refused the folder — the repair step must be the one that heals it."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    folder = layout.processed_dir("a")
    existing = sorted(folder.glob("data-*.parquet"))[0]
    stray = folder / "data-00099.parquet"
    stray.write_bytes(existing.read_bytes())
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")]
    assert report.actions[0].reason == "unlisted shard(s): data-00099.parquet"
    assert not folder.exists()


def test_crash_leftover_next_shard_is_left_for_the_resumed_build(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Callable[[Path], list[dict[str, Any]]]
) -> None:
    """The M1 regression: a build that crashes between publishing a shard and saving the manifest leaves one
    unlisted file — exactly the next shard the resumed build writes. The repair step leaves it alone (deleting the
    folder would redo the whole build for one file), the resumed build overwrites it and completes the folder."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout, rows=12)  # 3 raw shards, fully built
    processed = layout.processed_dir("a")
    complete_rows = read_rows(processed)
    manifest = Manifest.load(processed)
    assert manifest is not None and [shard.name for shard in manifest.shards] == ["data-00000.parquet", "data-00001.parquet", "data-00002.parquet"]
    # reconstruct the crash state by hand: the shard file of the third raw shard is on disk, the manifest save never ran
    manifest.shards = manifest.shards[:2]
    manifest.extra["input_shards"] = manifest.extra["input_shards"][:2]
    manifest.save(processed)
    stray = processed / "data-00002.parquet"
    assert stray.exists()

    before = _snapshot(layout.root)
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert report.actions == [], "the crash leftover is the resumed build's to overwrite, not corruption"
    assert _snapshot(layout.root) == before and stray.exists()

    resumed = build_source(cfg, "a", layout, shard_size=4)  # the resumed build overwrites the stray and completes
    assert [shard.name for shard in resumed.shards] == ["data-00000.parquet", "data-00001.parquet", "data-00002.parquet"]
    assert len(resumed.extra["input_shards"]) == 3 and read_rows(processed) == complete_rows, "no data lost"
    assert repair_broken_and_stale_folders(cfg, layout, assume_yes=False).actions == []


def test_the_same_stray_on_a_complete_folder_is_still_deleted(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """The next-shard name is only harmless while raw shards are uncovered; on a folder that covers every raw shard
    no build would overwrite it, so the folder is rebuilt as before."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout)  # 2 raw shards, fully built
    folder = layout.processed_dir("a")
    (folder / "data-00002.parquet").write_bytes(b"stray")
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")]
    assert report.actions[0].reason == "unlisted shard(s): data-00002.parquet"
    assert not folder.exists()


def test_an_unreadable_processed_manifest_deletes_the_folder(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """Derived data is deleted without asking; a corrupt processed MANIFEST.json used to abort the run instead."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    folder = layout.processed_dir("a")
    (folder / "MANIFEST.json").write_text("{ not json")
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")]
    assert report.actions[0].reason == "unreadable manifest"
    assert not folder.exists()



def test_processed_built_from_a_raw_folder_that_disappeared_is_deleted(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    shutil.rmtree(layout.raw_dir("a"))
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")] and report.actions[0].reason == "built from raw shards that no longer exist"


def test_processed_input_shards_must_be_a_prefix_of_the_raw_shards(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    processed = layout.processed_dir("a")
    manifest = Manifest.load(processed)
    assert manifest is not None
    manifest.extra["input_shards"] = [["data-00000.parquet", 3], ["data-00001.parquet", 4]]  # row count of shard 0 differs
    manifest.save(processed)
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")] and report.actions[0].reason == "built from raw shards that no longer exist"


def test_leftover_temporary_folder_is_removed(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    leftover = layout.processed_dir("a").with_name("a.tmp")
    leftover.mkdir()
    (leftover / "data-00000.parquet").write_bytes(b"junk")
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert report.actions == [RepairAction("a", leftover, "processed", "delete", "leftover of an interrupted all-at-once build")]
    assert not leftover.exists() and layout.processed_dir("a").exists()


# --- interrupted rename-aside swaps (all-at-once builds) -----------------------------------------------------------------


def test_complete_tmp_next_to_missing_processed_finishes_the_swap(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, read_rows: Callable[[Path], list[dict[str, Any]]]
) -> None:
    """A crash between the two renames of the all-at-once swap leaves the replaced folder aside as ``.old`` and the
    complete ``.tmp`` as the only copy of the data: the repair step renames the ``.tmp`` into place and removes the
    ``.old`` instead of discarding the completed build."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    processed = layout.processed_dir("a")
    rows_before = read_rows(processed)
    temporary = processed.with_name("a.tmp")
    processed.rename(temporary)  # the crash state by hand: old aside, new not yet in place
    old = processed.with_name("a.old")
    old.mkdir()
    (old / "data-00000.parquet").write_bytes(b"replaced")
    tmp_snapshot = _snapshot(temporary)

    dry = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, dry_run=True)
    assert _kinds(dry) == [("a", "processed", "would_swap"), ("a", "processed", "would_delete")]
    assert temporary.exists() and old.exists() and not processed.exists(), "a dry run touches nothing"

    calls: list[str] = []
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=_recording_confirm(calls, False))
    assert calls == [], "derived data: no confirmation"
    assert _kinds(report) == [("a", "processed", "swap"), ("a", "processed", "delete")]
    assert report.actions[0] == RepairAction("a", temporary, "processed", "swap", "complete build of an interrupted swap; renaming it into place")
    assert report.actions[1] == RepairAction("a", old, "processed", "delete", "leftover of a completed folder swap")
    assert processed.exists() and not temporary.exists() and not old.exists()
    assert _snapshot(processed) == tmp_snapshot and read_rows(processed) == rows_before, "renamed, not rewritten: no data lost"
    assert repair_broken_and_stale_folders(cfg, layout, assume_yes=False).actions == []


def test_incomplete_tmp_next_to_missing_processed_is_still_deleted(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """Only a ``.tmp`` whose own manifest says the build finished (current hash, every shard verifies) is swapped
    into place; one without a manifest is the leftover of an interrupted build as before."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    processed = layout.processed_dir("a")
    temporary = processed.with_name("a.tmp")
    processed.rename(temporary)
    (temporary / MANIFEST_NAME).unlink()  # nothing says the build finished
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")]
    assert report.actions[0].reason == "leftover of an interrupted all-at-once build"
    assert not temporary.exists() and not processed.exists()


def test_complete_tmp_next_to_an_existing_processed_folder_is_deleted(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """A crash after the ``.tmp`` was completed but before the swap began: the old folder is still in place, so the
    ``.tmp`` is removed as a leftover (the next build rebuilds it) instead of replacing a folder that exists."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    processed = layout.processed_dir("a")
    temporary = processed.with_name("a.tmp")
    shutil.copytree(processed, temporary)
    before = _snapshot(processed)
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _kinds(report) == [("a", "processed", "delete")]
    assert report.actions[0].reason == "leftover of an interrupted all-at-once build"
    assert not temporary.exists() and _snapshot(processed) == before


def test_leftover_old_folder_is_removed_without_confirmation(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """A crash after the new folder was renamed into place leaves the replaced ``.old`` behind: derived data that
    was already replaced, deleted without asking."""
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    processed = layout.processed_dir("a")
    old = processed.with_name("a.old")
    old.mkdir()
    (old / "data-00000.parquet").write_bytes(b"replaced")
    before = _snapshot(processed)
    dry = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, dry_run=True)
    assert dry.actions == [RepairAction("a", old, "processed", "would_delete", "leftover of a completed folder swap")]
    assert old.exists()
    calls: list[str] = []
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=_recording_confirm(calls, False))
    assert calls == [] and report.actions == [RepairAction("a", old, "processed", "delete", "leftover of a completed folder swap")]
    assert not old.exists() and _snapshot(processed) == before


# --- confirmation --------------------------------------------------------------------------------------------------------


def test_one_prompt_for_two_queued_raw_folders(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, ("a", "b", "c"))
    _edit_manifest(layout.raw_dir("a"), source_hash="changed")
    _edit_manifest(layout.raw_dir("c"), truncated_at_tokens=16)
    prompts: list[str] = []

    def confirm(message: str) -> bool:
        prompts.append(message)
        assert layout.raw_dir("a").exists() and layout.raw_dir("c").exists(), "asked before anything is deleted"
        return True

    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=confirm)
    assert prompts == [
        f"{CONFIRMATION_HEADER}\n  a: stale: source identity or tokenizer changed\n  c: outdated: max_seq_length 16 -> 64\n{CONFIRMATION_QUESTION}"
    ]
    assert prompts[0].endswith("Continue? [y/N] ")
    assert [action.source for action in report.raw_deleted()] == ["a", "c"] and [action.source for action in report.processed_deleted()] == ["a", "c"]
    assert not layout.raw_dir("a").exists() and not layout.raw_dir("c").exists()
    assert layout.raw_dir("b").exists() and layout.processed_dir("b").exists()


def test_assume_yes_deletes_without_asking(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    _edit_manifest(layout.raw_dir("a"), source_hash="changed")

    def confirm(message: str) -> bool:
        raise AssertionError("must not be asked")

    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=True, confirm=confirm)
    assert [action.source for action in report.raw_deleted()] == ["a"] and not layout.raw_dir("a").exists()


def test_non_interactive_run_without_assume_yes_aborts_with_the_list_and_deletes_nothing(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, ("a", "b"))
    _edit_manifest(layout.raw_dir("a"), source_hash="changed")
    _edit_manifest(layout.processed_dir("b"), source_hash="stale-too")  # a processed deletion waits for the same answer
    monkeypatch.setattr(sys, "stdin", io.StringIO())  # not a tty
    before = _snapshot(layout.root)
    with pytest.raises(ConfirmationRequired) as info:
        repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert _snapshot(layout.root) == before
    err = info.value
    assert isinstance(err, RepairError) and not err.interactive
    assert err.message == f"{CONFIRMATION_HEADER}\n  a: stale: source identity or tokenizer changed\n{CONFIRMATION_QUESTION}"
    assert str(err).startswith(err.message.rstrip()) and "--yes" in str(err)
    assert _kinds(err.report) == [("a", "raw", "would_delete"), ("a", "processed", "would_delete"), ("b", "processed", "would_delete")]
    assert err.report.raw_deleted() == [] and [action.source for action in err.report.raw_deletions_planned()] == ["a"]


def test_declined_answer_on_the_terminal_aborts_and_deletes_nothing(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    _edit_manifest(layout.raw_dir("a"), source_hash="changed")
    monkeypatch.setattr(sys, "stdin", _Terminal())
    prompts: list[str] = []

    def answer_no(prompt: str) -> str:
        prompts.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", answer_no)
    before = _snapshot(layout.root)
    with pytest.raises(ConfirmationRequired) as info:
        repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert prompts == [info.value.message] and info.value.interactive and "nothing was changed" in str(info.value)
    assert _snapshot(layout.root) == before
    # an explicit "yes" on the terminal proceeds
    monkeypatch.setattr("builtins.input", lambda prompt: " Yes ")
    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False)
    assert [action.source for action in report.raw_deleted()] == ["a"] and not layout.raw_dir("a").exists()


def test_confirm_callable_declining_aborts(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout)
    _edit_manifest(layout.raw_dir("a"), source_hash="changed")
    with pytest.raises(ConfirmationRequired):
        repair_broken_and_stale_folders(cfg, layout, assume_yes=False, confirm=lambda message: False)
    assert layout.raw_dir("a").exists() and layout.processed_dir("a").exists()


# --- dry run and the report ------------------------------------------------------------------------------------------------


def test_dry_run_reports_everything_and_touches_nothing(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _prepared(cfg_factory, with_tokenizer, layout, ("a", "b", "c"), rows=12)
    _edit_manifest(layout.raw_dir("a"), source_hash="changed")
    (layout.raw_dir("b") / "data-00002.parquet").write_bytes(b"corrupt")
    _edit_manifest(layout.processed_dir("c"), source_hash="stale")
    leftover = layout.processed_dir("c").with_name("c.tmp")
    leftover.mkdir()
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    before = _snapshot(layout.root)
    listing_before = sorted(str(path) for path in layout.root.rglob("*"))

    def confirm(message: str) -> bool:
        raise AssertionError("a dry run never asks")

    report = repair_broken_and_stale_folders(cfg, layout, assume_yes=False, dry_run=True, confirm=confirm)
    assert _snapshot(layout.root) == before and sorted(str(path) for path in layout.root.rglob("*")) == listing_before
    assert _kinds(report) == [
        ("a", "raw", "would_delete"),
        ("a", "processed", "would_delete"),
        ("b", "raw", "would_truncate"),
        ("b", "processed", "would_delete"),
        ("c", "processed", "would_delete"),
        ("c", "processed", "would_delete"),
    ]
    assert report.raw_deleted() == [] and report.processed_deleted() == [] and [action.source for action in report.raw_deletions_planned()] == ["a"]
    lines = report.describe().splitlines()
    assert lines[0] == f"would delete raw {layout.raw_dir('a')} (a): stale: source identity or tokenizer changed"
    assert lines[2].startswith(f"would truncate raw {layout.raw_dir('b')} (b): broken: unreadable shard data-00002.parquet")
    assert lines[5] == f"would delete processed {leftover} (c): leftover of an interrupted all-at-once build"
    # the real run afterwards performs exactly the planned actions
    performed = repair_broken_and_stale_folders(cfg, layout, assume_yes=True)
    assert _kinds(performed) == [(source, kind, action.replace("would_", "")) for source, kind, action in _kinds(report)]
    assert performed.describe().splitlines()[0] == f"delete raw {layout.raw_dir('a')} (a): stale: source identity or tokenizer changed"
    assert repair_broken_and_stale_folders(cfg, layout, assume_yes=False, dry_run=True).actions == []


def test_report_describe_and_helpers(tmp_path: Path) -> None:
    actions = [
        RepairAction("a", tmp_path / "raw", "raw", "delete", "stale: x"),
        RepairAction("a", tmp_path / "processed", "processed", "delete", "built from a raw folder that is being deleted"),
        RepairAction("b", tmp_path / "raw_b", "raw", "truncate", "broken: y"),
    ]
    report = RepairReport(actions)
    assert report.describe() == "\n".join(
        [
            f"delete raw {tmp_path / 'raw'} (a): stale: x",
            f"delete processed {tmp_path / 'processed'} (a): built from a raw folder that is being deleted",
            f"truncate raw {tmp_path / 'raw_b'} (b): broken: y",
        ]
    )
    assert report.raw_deleted() == actions[:1] and report.processed_deleted() == actions[1:2]
    planned = report.as_planned()
    assert [action.action for action in planned.actions] == ["would_delete", "would_delete", "would_truncate"]
    assert planned.raw_deleted() == [] and planned.raw_deletions_planned() == planned.actions[:1]
    assert RepairReport().describe() == "nothing to repair"
    assert json.dumps([action.reason for action in planned.actions])  # reasons are plain strings for the status output
