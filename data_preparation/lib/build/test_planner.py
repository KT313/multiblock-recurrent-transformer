# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the planner: `rows_needed` by hand (sequences, the split, validation-only sources), the download plan
(clamped, zero when exhausted, stale raw rejected, topped up when the build drops more than the margin), the
`SourceLedger` satisfaction cases and the status table."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any

import pytest

from data_preparation.dataset_config import (
    DatasetConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
)
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build import prepare, status
from data_preparation.lib.build.planner import (
    DatasetReport,
    Satisfaction,
    SourceLedger,
    SourceState,
    every_source_satisfies_its_budget,
    plan_downloads,
    raw_is_exhausted,
    read_ledgers,
    rows_needed,
    rows_sufficient,
    source_ledger,
    source_state,
    sources_with_pending_raw_shards,
    summarize_dataset_state,
    training_rows_after_split,
)
from data_preparation.lib.storage.manifest import Manifest

CfgFactory = Callable[..., DatasetConfig]
ConfigFile = Callable[[DatasetConfig], Path]
Writer = Callable[[Path, list[dict[str, Any]], str], Path]


def two_stage_cfg(tokens_a: int = 6400, tokens_b: int = 3200, rows_h: int = 8, block_size: int = 64, min_chars: int = 1) -> DatasetConfig:
    """`a` trained in both stages (weight 0.5 then 1.0) and validated on in the first (split), `b` only trained on
    in the first, `h` used only for validation (`rows`), instruct source `i` in a finetune stage (split)."""
    return DatasetConfig(
        name="two",
        tokenizer=TokenizerConfig(name="synthetic", kind="synthetic"),
        sources={
            "a": SourceConfig(kind="pretrain", loader="synthetic", seed=0),
            "b": SourceConfig(kind="pretrain", loader="synthetic", seed=1),
            "h": SourceConfig(kind="pretrain", loader="synthetic", seed=3, rows=rows_h),
            "i": SourceConfig(kind="instruct", loader="synthetic", seed=4),
        },
        stages=[
            StageConfig(name="s1", tokens=tokens_a, train={"a": 0.5, "b": 0.5}, val={"a": 1.0}),
            StageConfig(name="s2", tokens=tokens_b, train={"a": 1.0}, val={"h": 1.0}),
            StageConfig(name="ft", tokens=1280, train={"i": 1.0}, val={"i": 1.0}),
        ],
        block_size=block_size,
        max_seq_length=128,
        processing=ProcessingConfig(min_chars=min_chars),
    )


def _state(report: DatasetReport, name: str) -> SourceState:
    return next(source for source in report.sources if source.name == name)


# --- rows_needed ----------------------------------------------------------------------------------------------------


def test_rows_needed_counts_sequences_with_the_margin_and_the_split() -> None:
    cfg = two_stage_cfg()
    # a: max(ceil(6400 × 0.5 / 64), ceil(3200 / 64)) = 50 sequences; × 1.2 ÷ (1 − 0.05) = 63.16 -> 64
    assert cfg.sequence_budget("a") == 50 and cfg.validation_fraction_of("a") == 0.05 and rows_needed(cfg, "a") == 64
    # b: 50 sequences, not validated on: 50 × 1.2 = 60 exactly (no float slop from 1.2)
    assert cfg.validation_fraction_of("b") == 0.0 and rows_needed(cfg, "b") == 60
    # i: ceil(1280 / 64) = 20 sequences, split: 20 × 1.2 ÷ 0.95 = 25.26 -> 26
    assert rows_needed(cfg, "i") == 26


def test_rows_needed_takes_the_largest_stage_and_scales_with_block_size() -> None:
    overlap = two_stage_cfg(tokens_b=12800)  # stage 2 now dominates: ceil(12800 / 64) = 200 sequences
    assert overlap.sequence_budget("a") == 200 and rows_needed(overlap, "a") == 253  # 240 ÷ 0.95 = 252.6
    assert rows_needed(overlap, "b") == 60  # stage 1 unchanged
    wide = two_stage_cfg(block_size=128)  # half the sequences per stage
    assert wide.sequence_budget("a") == 25 and rows_needed(wide, "a") == 32  # 30 ÷ 0.95 = 31.58
    assert rows_needed(wide, "b") == 30


def test_rows_needed_of_a_validation_only_source_is_its_rows() -> None:
    assert rows_needed(two_stage_cfg(rows_h=8), "h") == 8
    assert rows_needed(two_stage_cfg(rows_h=20), "h") == 20


def test_rows_sufficient_and_training_rows_after_split() -> None:
    cfg = two_stage_cfg()
    assert rows_sufficient(cfg, "a") == 54  # 64 ÷ 1.2 = 53.3
    assert rows_sufficient(cfg, "b") == 50 and rows_sufficient(cfg, "h") == 7 and rows_sufficient(cfg, "i") == 22
    # the resolver holds ceil(0.05 × rows) out; 64 rows -> 4 validation, 60 training >= the 50-sequence budget
    assert training_rows_after_split(cfg, "a", 64) == 60 >= cfg.sequence_budget("a")
    assert training_rows_after_split(cfg, "b", 60) == 60  # not validated on: nothing held out
    assert training_rows_after_split(cfg, "a", 0) == 0


# --- plan_downloads -------------------------------------------------------------------------------------------------


def test_plan_downloads_on_an_empty_dir(layout: DatasetLayout) -> None:
    plan = plan_downloads(two_stage_cfg(), layout)
    assert [(s.name, s.rows_present, s.rows_needed, s.rows_to_fetch, s.reason) for s in plan.sources] == [
        ("a", 0, 64, 64, "raw missing"),
        ("b", 0, 60, 60, "raw missing"),
        ("h", 0, 8, 8, "raw missing"),
        ("i", 0, 26, 26, "raw missing"),
    ]
    assert plan.total_rows_to_fetch() == 158 and [s.name for s in plan.to_fetch()] == ["a", "b", "h", "i"]
    assert plan.summary() == "4 source(s) short, downloading 158 rows (a 64, b 60, h 8, i 26)"
    assert plan.describe().splitlines()[0].split() == ["source", "present", "needed", "fetch", "reason"]
    only = plan_downloads(two_stage_cfg(), layout, sources=["i", "a"])
    assert [s.name for s in only.sources] == ["a", "i"]  # config order
    with pytest.raises(ValueError, match="unknown sources"):
        plan_downloads(two_stage_cfg(), layout, sources=["nope"])


def test_rows_to_fetch_is_the_difference_clamped_at_zero(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    assert prepare(config_file(cfg), layout.root, assume_yes=False).complete
    plan = plan_downloads(cfg, layout)
    assert all(s.rows_to_fetch == 0 and s.reason == "enough rows" and s.rows_present >= s.rows_needed for s in plan.sources)
    assert plan.summary() == "nothing to download"

    bigger = two_stage_cfg(tokens_b=12800)  # same hashes, a larger budget for `a`: top up by the difference
    a = next(s for s in plan_downloads(bigger, layout).sources if s.name == "a")
    assert (a.rows_present, a.rows_needed, a.rows_to_fetch, a.reason) == (64, 253, 189, "rows 64 < 253")
    smaller = two_stage_cfg(tokens_a=640, tokens_b=320)
    assert all(s.rows_to_fetch == 0 for s in plan_downloads(smaller, layout).sources)


def test_rows_to_fetch_is_zero_when_the_loader_is_exhausted(layout: DatasetLayout, cfg_factory: CfgFactory, config_file: ConfigFile, write_local: Writer) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": f"tok_{i} tok_2 tok_3"} for i in range(4)], "parquet")
    cfg = cfg_factory({"v": SourceConfig(kind="pretrain", loader="local", path=str(src_dir), rows=10)})
    prepare(config_file(cfg), layout.root, assume_yes=False)
    v = next(s for s in plan_downloads(cfg, layout).sources if s.name == "v")
    assert (v.rows_present, v.rows_needed, v.rows_to_fetch, v.reason) == (4, 10, 0, "exhausted")


def test_raw_is_exhausted_honours_a_grown_check_limit() -> None:
    cfg = two_stage_cfg()
    manifest = Manifest(source="i", source_hash="x", stage="raw")
    assert not raw_is_exhausted(cfg, "i", manifest)
    manifest.extra["exhausted"] = True
    assert raw_is_exhausted(cfg, "i", manifest)  # exhausted by the loader
    manifest.extra["check_limit"] = 5
    cfg.sources["i"].check_limit = 5
    assert raw_is_exhausted(cfg, "i", manifest)  # exhausted by the limit that still applies
    cfg.sources["i"].check_limit = 10
    assert not raw_is_exhausted(cfg, "i", manifest)  # the limit grew: the download reads on
    cfg.sources["i"].check_limit = None
    assert not raw_is_exhausted(cfg, "i", manifest)


def test_plan_downloads_fetches_nothing_for_a_stale_or_outdated_raw_folder(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """The repair step (or a dry run's report) owns such a folder: the plan names the state and fetches nothing."""
    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)
    cfg.token_count = "estimate"  # part of the raw hash: every raw folder is stale
    stale_entry = next(s for s in plan_downloads(cfg, layout).sources if s.name == "a")
    assert stale_entry.rows_to_fetch == 0 and stale_entry.reason == "raw stale: the repair step deletes it after confirmation"
    outdated = two_stage_cfg()
    outdated.max_seq_length = 4096  # raised above the stored cap
    outdated_entry = next(s for s in plan_downloads(outdated, layout).sources if s.name == "a")
    assert outdated_entry.rows_to_fetch == 0 and outdated_entry.reason.startswith("raw outdated")
    a = source_state(cfg, "a", layout)  # the status table only reports it
    assert not a.satisfied and a.reason == "raw stale: the repair step deletes it after confirmation"



# --- the ledger -------------------------------------------------------------------------------------------------------


def _ledger(**overrides: Any) -> SourceLedger:
    """A ledger of a trained source with 100 raw rows fully built into 90 processed ones (budget 100 / 84)."""
    defaults: dict[str, Any] = dict(
        name="s", kind="pretrain", rows_needed=100, rows_sufficient=84, sequence_budget=70, raw_state="current",
        raw_rows=100, exhausted=False, skipped_malformed=0, dropped_too_long=0, processed_state="built",
        processed_rows=90, training_rows=90,
    )
    return SourceLedger(**{**defaults, **overrides})


def test_the_ledger_answers_both_questions_from_one_read(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """`plan_downloads`, the satisfaction check and the status table are three views of the same object."""
    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)
    ledger = source_ledger(cfg, "a", layout)
    assert (ledger.name, ledger.kind, ledger.raw_state, ledger.processed_state) == ("a", "pretrain", "current", "built")
    assert (ledger.rows_needed, ledger.rows_sufficient, ledger.raw_rows) == (64, 54, 64)
    assert ledger.satisfaction() is Satisfaction.OK and ledger.rows_to_fetch() == 0
    assert ledger.download() == next(s for s in plan_downloads(cfg, layout).sources if s.name == "a")
    assert ledger.state() == source_state(cfg, "a", layout)
    assert [led.name for led in read_ledgers(cfg, layout, sources=["i", "a"])] == ["a", "i"]  # config order


def test_the_satisfaction_cases() -> None:
    assert _ledger().satisfaction() is Satisfaction.OK
    assert _ledger(raw_state="stale").satisfaction() is Satisfaction.RAW_BROKEN
    assert _ledger(raw_state="outdated").satisfaction() is Satisfaction.RAW_BROKEN
    assert _ledger(raw_state="missing", raw_rows=0).satisfaction() is Satisfaction.RAW_MISSING
    for processed_state in ("missing", "stale", "behind_raw"):
        assert _ledger(processed_state=processed_state).satisfaction() is Satisfaction.NOT_BUILT
    assert _ledger(processed_rows=10).satisfaction() is Satisfaction.SHORT_BUT_FETCHABLE
    dry_small = _ledger(processed_rows=10, exhausted=True)
    assert dry_small.satisfaction() is Satisfaction.EXHAUSTED_SMALL and dry_small.satisfaction().satisfied
    dry_empty = _ledger(processed_rows=0, exhausted=True, skipped_malformed=100)
    assert dry_empty.satisfaction() is Satisfaction.EXHAUSTED_EMPTY and not dry_empty.satisfaction().satisfied
    assert "NOT ONE" in dry_empty.reason() and "100 malformed" in dry_empty.reason()
    assert "check the source's fields / converter / filter / language" in dry_empty.reason()
    assert dry_empty.epochs() is None and _ledger().epochs() == pytest.approx(70 / 90)


def test_rows_to_fetch_tops_up_from_the_observed_yield() -> None:
    """Raw is long enough but only 60 of 100 rows survived the build: the shortfall (84 − 60) is divided by the
    observed yield (0.6) and multiplied by the same 1.2 safety margin the first download uses."""
    short = _ledger(processed_rows=60, training_rows=60)
    assert short.rows_to_fetch() == 48 == ceil((84 - 60) * Fraction("1.2") / Fraction(60, 100))
    assert short.download().rows_target == 100 + 48, "the download takes a target, not an increment"
    assert "top-up: 60 of 84 rows survived 100 raw" in short.download().reason

    assert _ledger(raw_rows=50, processed_rows=45, training_rows=45).rows_to_fetch() == 50  # raw itself is short
    assert _ledger(processed_rows=20, exhausted=True).rows_to_fetch() == 0  # nothing left to fetch
    assert _ledger(processed_rows=20, processed_state="behind_raw").rows_to_fetch() == 0  # build first
    nothing_survives = _ledger(processed_rows=0)
    assert nothing_survives.rows_to_fetch() == 0  # no yield to extrapolate from: more raw would be dropped too
    assert nothing_survives.download().reason == "no row of 100 raw rows survives the build"


def test_the_top_up_is_capped_at_the_full_requirement(caplog: pytest.LogCaptureFixture) -> None:
    """A pathological yield (1 of 1,200 rows) extrapolates to a download nobody wants: the round asks for at most
    `rows_needed`, with a warning, and the next round measures the yield again on more data."""
    pathological = _ledger(raw_rows=1200, processed_rows=1, training_rows=1)
    assert ceil((84 - 1) * Fraction("1.2") * 1200) == 119_520  # what the yield extrapolates to
    with caplog.at_level("WARNING", logger="data_preparation"):
        assert pathological.rows_to_fetch() == 100 == pathological.rows_needed
    assert "capping this round at the full requirement of 100 rows" in caplog.text
    assert "top-up: 1 of 84 rows survived 1,200 raw" in pathological.download().reason
    caplog.clear()
    with caplog.at_level("WARNING", logger="data_preparation"):
        assert _ledger(processed_rows=60, training_rows=60).rows_to_fetch() == 48  # under the cap: unchanged, no warning
    assert caplog.text == ""


# --- satisfaction ---------------------------------------------------------------------------------------------------


def test_sources_are_satisfied_after_prepare(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and report.tokenizer_complete and report.missing() == []
    a = _state(report, "a")
    assert a.satisfied and a.reason == "ok" and a.state() == "complete" and not a.exhausted
    assert a.raw_rows == 64 and a.processed_rows >= 54 and a.rows_needed == 64
    assert a.epochs == pytest.approx(50 / training_rows_after_split(cfg, "a", a.processed_rows))
    h = _state(report, "h")
    assert h.satisfied and h.raw_rows == 8 and h.epochs is None  # not trained on: no budget to cycle
    assert every_source_satisfies_its_budget(cfg, layout) and every_source_satisfies_its_budget(cfg, layout, sources=["i"])
    assert sources_with_pending_raw_shards(cfg, layout) == []
    assert report.describe().endswith("dataset complete")


def test_not_satisfied_when_processed_is_missing_stale_or_behind_raw(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)

    (layout.processed_dir("b") / "MANIFEST.json").unlink()
    b = source_state(cfg, "b", layout)
    assert not b.satisfied and b.reason == "processed missing" and b.raw_rows == 60 and b.processed_rows == 0 and b.epochs is None
    assert not every_source_satisfies_its_budget(cfg, layout) and every_source_satisfies_its_budget(cfg, layout, sources=["a", "i"])
    assert sources_with_pending_raw_shards(cfg, layout) == ["b"]

    stale = two_stage_cfg(min_chars=2)  # changes every processed hash, no raw hash
    a = source_state(stale, "a", layout)
    assert not a.satisfied and a.reason == "processed stale" and a.raw_rows == 64

    bigger = two_stage_cfg(tokens_b=12800)
    prepare(config_file(bigger), layout.root, assume_yes=False, steps=["download"])  # raw topped up, processed not
    a = source_state(bigger, "a", layout)
    assert not a.satisfied and a.reason == "processed behind raw" and a.raw_rows == 253 and 0 < a.processed_rows < 253
    assert sources_with_pending_raw_shards(bigger, layout) == ["a", "b"]


def test_an_unreadable_processed_manifest_is_reported_not_raised(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """`Manifest.load` raises next to shards — right for raw, wrong for the derived processed folder the repair step
    deletes without asking: `status` (and `prepare --dry_run`, and training's auto-prepare) must report it."""
    cfg = two_stage_cfg()
    path = config_file(cfg)
    prepare(path, layout.root, assume_yes=False)
    (layout.processed_dir("b") / "MANIFEST.json").write_text("{ not json")

    b = source_state(cfg, "b", layout)
    assert not b.satisfied and b.processed_rows == 0 and b.raw_rows == 60
    assert b.reason == "processed manifest unreadable: the repair step deletes the folder and builds it again"
    assert not every_source_satisfies_its_budget(cfg, layout) and sources_with_pending_raw_shards(cfg, layout) == ["b"]
    assert [s.name for s in plan_downloads(cfg, layout).to_fetch()] == []  # raw is complete; the build is what is missing

    report = status(path, layout.root)  # used to crash with "unreadable manifest ... next to shards"
    assert not report.complete and "b" in report.missing() and "b" in report.needs_repair
    assert prepare(path, layout.root, assume_yes=False).complete  # and the repair step heals it


def test_short_processed_folder_is_not_satisfied(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)
    bigger = two_stage_cfg(tokens_b=12800)  # processed covers every raw shard but holds fewer rows than the new budget needs
    a = source_state(bigger, "a", layout)
    assert not a.satisfied and a.reason == f"processed rows {a.processed_rows} < {rows_sufficient(bigger, 'a')}" and a.epochs is None
    row = next(line for line in summarize_dataset_state(bigger, layout).table().splitlines() if line.startswith("a "))
    assert row.split()[5] == "-" and row.split()[6] == "incomplete"  # epochs only when complete


def test_exhausted_and_built_source_is_satisfied(layout: DatasetLayout, cfg_factory: CfgFactory, config_file: ConfigFile, write_local: Writer) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": f"tok_{i} tok_2 tok_3"} for i in range(4)], "parquet")
    cfg = cfg_factory({"v": SourceConfig(kind="pretrain", loader="local", path=str(src_dir), rows=10)})
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    v = _state(report, "v")
    assert report.complete and v.satisfied and v.exhausted and v.state() == "exhausted"
    assert v.reason == "exhausted at 4 of 9 rows" and v.processed_rows == 4  # 9 = ceil(10 ÷ 1.2)
    (layout.processed_dir("v") / "MANIFEST.json").unlink()
    assert not source_state(cfg, "v", layout).satisfied  # exhausted alone is not enough: the raw shards must be built


def test_report_table_and_missing(layout: DatasetLayout, config_file: ConfigFile) -> None:
    empty = DatasetReport()
    assert not empty.complete and empty.missing() == ["tokenizer"] and empty.describe().endswith("dataset INCOMPLETE")
    cfg = two_stage_cfg()
    report = summarize_dataset_state(cfg, layout)
    assert report.missing() == ["a", "b", "h", "i", "tokenizer"] and not report.complete
    lines = report.table().splitlines()
    assert lines[0].split() == ["source", "kind", "needed", "raw", "processed", "epochs", "state", "reason"]
    assert lines[1].split() == ["a", "pretrain", "64", "0", "0", "-", "incomplete", "raw", "missing"]
    assert lines[-1].split() == ["tokenizer", "tokenizer", "incomplete"]
    prepare(config_file(cfg), layout.root, assume_yes=False)
    complete = summarize_dataset_state(cfg, layout)
    assert complete.complete and complete.unsatisfied() == []
    assert complete.table().splitlines()[-1].split() == ["tokenizer", "tokenizer", "complete"]
    flagged = summarize_dataset_state(cfg, layout, needs_repair=["b", "b"])
    assert not flagged.complete and flagged.missing() == ["b"] and flagged.needs_repair == ["b"]
    assert "needs repair" in next(line for line in flagged.table().splitlines() if line.startswith("b "))
