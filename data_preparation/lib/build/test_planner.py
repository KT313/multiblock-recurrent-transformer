# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the planner: `rows_needed` by hand (tokens at a rate, the split, validation-only sources), the download
plan (clamped, zero when exhausted or served, stale or unreadable raw rejected, topped up when the build drops more
than the margin), the `SourceLedger` satisfaction cases and the status table.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.build.repair import ConfirmationRequired

from data_preparation.dataset_config import (
    DatasetConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
)
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.runner import prepare, status
from data_preparation.lib.build.planner import (
    DatasetReport,
    SourceLedger,
    every_source_satisfies_its_budget,
    plan_downloads,
    raw_is_exhausted,
    read_ledgers,
    source_ledger,
    sources_with_pending_raw_shards,
    summarize_dataset_state,
    training_rows_after_split,
)
from data_preparation.lib.storage.manifest import Manifest



CfgFactory = Callable[..., DatasetConfig]
ConfigFile = Callable[[DatasetConfig], Path]
Writer = Callable[[Path, list[dict[str, Any]], str], Path]


def two_stage_cfg(tokens_a: int = 6400, tokens_b: int = 3200, rows_h: int = 8, training_target_sequence_length: int = 64, min_chars: int = 1) -> DatasetConfig:
    """
    `a` trained in both stages (weight 0.5 then 1.0) and validated on in the first (split), `b` only trained on
    in the first, `h` used only for validation (`rows`), instruct source `i` in a finetune stage (split).
    """

    return DatasetConfig(
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
        training_target_sequence_length=training_target_sequence_length,
        dataset_max_sequence_length=128,
        processing=ProcessingConfig(min_chars=min_chars),
    )


def _state(report: DatasetReport, name: str) -> SourceLedger:
    return next(source for source in report.sources if source.name == name)


# --- rows_needed ----------------------------------------------------------------------------------------------------


def test_rows_needed_counts_tokens_at_the_rate_with_the_margin_and_the_split() -> None:
    cfg = two_stage_cfg()  # the default estimate of 500 tokens per row is clamped at the training length 64
    # a: 6400 × 0.5 + 3200 × 1.0 = 6400 tokens over the run, 100 rows at 64 tokens each; × 1.2 ÷ (1 − 0.05) = 126.3 -> 127
    assert cfg.token_budget("a") == 6400 and cfg.rows_budget("a") == 100 and cfg.validation_fraction_of("a") == 0.05 and cfg.rows_needed("a") == 127
    # b: 6400 × 0.5 = 3200 tokens = 50 rows, not validated on: 50 × 1.2 = 60 exactly (no float slop from 1.2)
    assert cfg.validation_fraction_of("b") == 0.0 and cfg.rows_needed("b") == 60
    # i: 1280 tokens = 20 rows, split: 20 × 1.2 ÷ 0.95 = 25.26 -> 26
    assert cfg.rows_needed("i") == 26
    # a measured mean below the training length takes more rows: 6400 ÷ 32 = 200; × 1.2 ÷ 0.95 = 252.6 -> 253
    assert cfg.tokens_per_row_rate("a", 32.0) == 32 and cfg.rows_needed("a", 32.0) == 253 and cfg.rows_sufficient("a", 32.0) == 211
    assert cfg.rows_needed("a", 500.0) == 127  # a measured mean above the training length is clamped like the estimate


def test_rows_needed_sums_the_stages_and_scales_with_the_training_length() -> None:
    overlap = two_stage_cfg(tokens_b=12800)  # stage 2 grows: 6400 × 0.5 + 12800 = 16000 tokens = 250 rows
    assert overlap.token_budget("a") == 16000 and overlap.rows_budget("a") == 250 and overlap.rows_needed("a") == 316  # 300 ÷ 0.95 = 315.8
    assert overlap.rows_needed("b") == 60  # stage 1 unchanged
    wide = two_stage_cfg(training_target_sequence_length=128)  # a row serves twice the tokens: half the rows
    assert wide.rows_budget("a") == 50 and wide.rows_needed("a") == 64  # 60 ÷ 0.95 = 63.16
    assert wide.rows_needed("b") == 30


def test_rows_needed_is_the_schema_formula(layout: DatasetLayout) -> None:
    """
    The planner delegates to `DatasetConfig.rows_needed`, the number the shuffled-build cap checks at config
    load, so the two views of the requirement cannot drift: the ledgers of an empty dataset (no measured rate
    yet) carry exactly the documented formula, budget × margin ÷ the training share, or rows × margin for a
    source used only for validation.
    """

    cfg = two_stage_cfg()
    formula = {
        name: ceil((source.rows or 0) * Fraction("1.2"))
        if not cfg.used_in_train(name)
        else ceil(cfg.rows_budget(name) * Fraction("1.2") / (1 - Fraction(str(cfg.validation_fraction_of(name)))))
        for name, source in cfg.sources.items()
    }
    assert formula == {"a": 127, "b": 60, "h": 10, "i": 26}, "the formula spelled out, not read from the config"
    assert {led.name: led.rows_needed for led in read_ledgers(cfg, layout)} == formula


def test_rows_needed_of_a_validation_only_source_is_its_rows_plus_the_margin() -> None:
    """
    `rows` of a validation-only source are delivered rows: the download adds the margin, the build has to keep
    `rows` of them.
    """

    assert two_stage_cfg(rows_h=8).rows_needed("h") == 10 and two_stage_cfg(rows_h=8).rows_sufficient("h") == 8  # 8 × 1.2 = 9.6
    assert two_stage_cfg(rows_h=20).rows_needed("h") == 24 and two_stage_cfg(rows_h=20).rows_sufficient("h") == 20


def test_rows_sufficient_and_training_rows_after_split() -> None:
    cfg = two_stage_cfg()
    assert cfg.rows_sufficient("a") == 106  # 127 ÷ 1.2 = 105.8
    assert cfg.rows_sufficient("b") == 50 and cfg.rows_sufficient("h") == 8 and cfg.rows_sufficient("i") == 22
    # the resolver holds ceil(0.05 × rows) out; 127 rows -> 7 validation, 120 training >= the 100-row budget
    assert training_rows_after_split(cfg, "a", 127) == 120 >= cfg.rows_budget("a")
    assert training_rows_after_split(cfg, "b", 60) == 60  # not validated on: nothing held out
    assert training_rows_after_split(cfg, "a", 0) == 0


# --- plan_downloads -------------------------------------------------------------------------------------------------


def test_plan_downloads_on_an_empty_dir(layout: DatasetLayout) -> None:
    plan = plan_downloads(two_stage_cfg(), layout)
    assert [(s.name, s.raw_rows, s.rows_needed, *s.rows_to_fetch) for s in plan.sources] == [
        ("a", 0, 127, 127, "raw missing"),
        ("b", 0, 60, 60, "raw missing"),
        ("h", 0, 10, 10, "raw missing"),
        ("i", 0, 26, 26, "raw missing"),
    ]
    assert plan.total_rows_to_fetch() == 223 and [s.name for s in plan.to_fetch()] == ["a", "b", "h", "i"]
    assert plan.summary() == "4 source(s) short, downloading 223 rows (a 127, b 60, h 10, i 26)"
    assert plan.describe().splitlines()[0].split() == ["source", "present", "needed", "fetch", "reason"]
    only = plan_downloads(two_stage_cfg(), layout, sources=["i", "a"])
    assert [s.name for s in only.sources] == ["a", "i"]  # config order
    with pytest.raises(ValueError, match="unknown sources"):
        plan_downloads(two_stage_cfg(), layout, sources=["nope"])


def test_rows_to_fetch_is_the_difference_clamped_at_zero(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    assert prepare(config_file(cfg), layout.root, assume_yes=False).complete
    plan = plan_downloads(cfg, layout)
    assert all(s.rows_to_fetch == (0, "budget served") for s in plan.sources) and plan.summary() == "nothing to download"
    a, b, h, i = plan.sources
    assert all(s.raw_rows >= s.rows_needed and s.tokens_per_row == 64 for s in (a, b, h))  # rows of the training length or longer: the clamp
    assert i.tokens_per_row < 64 and i.raw_rows == 26 < i.rows_needed  # short instruct rows measured below the estimate: served by the margin

    bigger = two_stage_cfg(tokens_b=12800)  # same hashes, a larger budget for `a`: top up by the difference
    a = next(s for s in plan_downloads(bigger, layout).sources if s.name == "a")
    assert (a.raw_rows, a.rows_needed, *a.rows_to_fetch) == (127, 316, 189, "rows 127 < 316")
    smaller = two_stage_cfg(tokens_a=640, tokens_b=320)
    assert all(s.rows_to_fetch[0] == 0 for s in plan_downloads(smaller, layout).sources)


def test_rows_to_fetch_is_zero_when_the_loader_is_exhausted(layout: DatasetLayout, cfg_factory: CfgFactory, config_file: ConfigFile, write_local: Writer) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": f"tok_{i} tok_2 tok_3"} for i in range(4)], "parquet")
    cfg = cfg_factory({"v": SourceConfig(kind="pretrain", loader="local", path=str(src_dir), rows=10)})
    prepare(config_file(cfg), layout.root, assume_yes=False)
    v = next(s for s in plan_downloads(cfg, layout).sources if s.name == "v")
    assert (v.raw_rows, v.rows_needed, *v.rows_to_fetch) == (4, 12, 0, "exhausted")


def test_the_measured_tokens_per_row_replace_the_estimate_once_raw_is_on_disk(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """
    Before the first shard the ledger plans at the config's estimate (clamped at the training length); afterwards at the
    raw manifest's mean tokens per row, so a source of short rows is asked for more rows than the estimate said.
    """

    cfg = two_stage_cfg(training_target_sequence_length=128)  # rows are cut at 128 tokens: the mean is measurably below the training length
    assert source_ledger(cfg, "a", layout).tokens_per_row == 128 == cfg.tokens_per_row_rate("a")
    prepare(config_file(cfg), layout.root, assume_yes=False)
    raw = Manifest.load(layout.raw_dir("a"))
    assert raw is not None and raw.tokens() is not None
    mean = raw.tokens() / raw.rows()  # type: ignore[operator]  # checked above
    a = source_ledger(cfg, "a", layout)
    assert a.tokens_per_row == pytest.approx(mean) and 64 < mean < 128
    assert (a.rows_needed, a.rows_sufficient, a.rows_budget) == (cfg.rows_needed("a", mean), cfg.rows_sufficient("a", mean), cfg.rows_budget("a", mean))
    assert a.rows_needed > cfg.rows_needed("a") == 64 and a.satisfaction() == (True, "ok")  # served: the margin covered the estimate


def test_raw_is_exhausted_honours_a_grown_check_limit() -> None:
    cfg = two_stage_cfg()
    manifest = Manifest(source="i", source_hash="x", stage="raw")
    assert not raw_is_exhausted(cfg, "i", manifest)
    manifest.exhausted = True
    assert raw_is_exhausted(cfg, "i", manifest)  # exhausted by the loader
    manifest.check_limit_reached = 5
    cfg.sources["i"].check_limit = 5
    assert raw_is_exhausted(cfg, "i", manifest)  # exhausted by the limit that still applies
    cfg.sources["i"].check_limit = 10
    assert not raw_is_exhausted(cfg, "i", manifest)  # the limit grew: the download reads on
    cfg.sources["i"].check_limit = None
    assert not raw_is_exhausted(cfg, "i", manifest)


def test_plan_downloads_fetches_nothing_for_a_stale_or_outdated_raw_folder(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """
    The repair step (or a dry run's report) owns such a folder: the plan names the state and fetches nothing.
    """

    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)
    cfg.sources["a"].seed = 9  # the synthetic seed is raw identity: the raw folder is stale
    stale_entry = next(s for s in plan_downloads(cfg, layout).sources if s.name == "a")
    assert stale_entry.rows_to_fetch == (0, "raw stale: source.seed: 0 -> 9; the repair step deletes it after confirmation")
    outdated = two_stage_cfg()
    outdated.dataset_max_sequence_length = 4096  # raised above the stored cap
    outdated_entry = next(s for s in plan_downloads(outdated, layout).sources if s.name == "a")
    assert outdated_entry.rows_to_fetch[0] == 0 and outdated_entry.rows_to_fetch[1].startswith("raw outdated")
    a = source_ledger(cfg, "a", layout)  # the status table only reports it
    assert a.satisfaction() == (False, "raw stale: source.seed: 0 -> 9; the repair step deletes it after confirmation")
    relabelled = two_stage_cfg()
    relabelled.token_count = "estimate"  # not raw identity: the same rows, counted differently, a choice for the repair step
    b = next(s for s in plan_downloads(relabelled, layout).sources if s.name == "b")
    assert b.raw_state == "tokenizer_changed" and b.raw_reason.startswith("token_count changed: tokenizer -> estimate; 60 rows")
    assert b.rows_to_fetch == (0, f"raw {b.raw_reason}; the repair step asks whether to keep it")
    assert b.satisfaction() == (False, f"raw {b.raw_reason}; the repair step asks whether to keep it") and b.raw_rows == 0


def test_an_unreadable_raw_manifest_is_a_reported_state_not_a_crash(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """
    `Manifest.load` raises for a manifest it cannot parse next to shards (the rows may have been expensive). The
    ledger turns that into an unsatisfied source nothing is planned for, so `status`, `prepare --dry_run` and a
    real `prepare` describe it instead of dying, and nobody deletes the folder: that is the user's call.
    """

    cfg = two_stage_cfg()
    path = config_file(cfg)
    prepare(path, layout.root, assume_yes=False)
    (layout.raw_dir("b") / "MANIFEST.json").write_text("{ not json")
    reason = "raw unreadable manifest next to shards; fix or delete the directory by hand"

    b = source_ledger(cfg, "b", layout)
    assert (b.raw_state, b.raw_rows, b.processed_rows, b.exhausted) == ("unreadable", 0, 0, False)
    assert b.satisfaction() == (False, reason) and b.rows_to_fetch == (0, reason) and b.epochs() is None
    assert b.state() == "incomplete" and not every_source_satisfies_its_budget(cfg, layout)
    assert [s.name for s in plan_downloads(cfg, layout).to_fetch()] == [] and sources_with_pending_raw_shards(cfg, layout) == []

    report = status(path, layout.root)
    assert not report.complete and report.missing() == ["b"] and report.needs_repair == []  # the repair step leaves it alone
    assert "b" in report.table() and reason in report.table()
    for dry_run in (True, False):
        assert not prepare(path, layout.root, assume_yes=True, dry_run=dry_run).complete
    assert (layout.raw_dir("b") / "data-00000.parquet").is_file() and layout.processed_dir("b").is_dir()  # nothing was deleted



# --- the ledger -------------------------------------------------------------------------------------------------------


def _ledger(**overrides: Any) -> SourceLedger:
    """
    A ledger of a trained source with 100 raw rows fully built into 90 processed ones (budget 100 / 84).
    """

    defaults: dict[str, Any] = dict(
        name="s", kind="pretrain", rows_needed=100, rows_sufficient=84, rows_budget=70, tokens_per_row=64.0, raw_state="current",
        raw_reason="current", raw_rows=100, exhausted=False, skipped_malformed=0, dropped_too_long=0, processed_problem="none", processed_reason="ok",
        processed_rows=90, training_rows=90,
    )
    return SourceLedger(**{**defaults, **overrides})


def test_the_ledger_answers_both_questions_from_one_read(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """
    `plan_downloads`, the satisfaction check and the status table are three views of the same object.
    """

    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)
    ledger = source_ledger(cfg, "a", layout)
    assert (ledger.name, ledger.kind, ledger.raw_state, ledger.processed_problem) == ("a", "pretrain", "current", "none")
    assert (ledger.rows_needed, ledger.rows_sufficient, ledger.raw_rows) == (127, 106, 127)
    assert ledger.satisfaction() == (True, "ok") and ledger.rows_to_fetch == (0, "budget served")
    assert ledger == next(s for s in plan_downloads(cfg, layout).sources if s.name == "a") == source_ledger(cfg, "a", layout)
    assert [led.name for led in read_ledgers(cfg, layout, sources=["i", "a"])] == ["a", "i"]  # config order


def test_missing_raw_rows_is_the_download_verdict() -> None:
    """
    The `download` command's completeness check: raw side only, so a source without a processed folder passes,
    while a raw folder not current, short of rows, or exhausted with nothing stored fails; the tokenizer counts.
    """

    unbuilt = dict(processed_problem="absent", processed_reason="missing", processed_rows=0, training_rows=0)
    report = DatasetReport(sources=[_ledger(**unbuilt)], tokenizer_complete=True)
    assert report.missing_raw_rows() == [] and not report.complete
    report.sources = [
        _ledger(name="short", raw_rows=40, **unbuilt),
        _ledger(name="stale", raw_state="stale", raw_reason="stale: x", raw_rows=0, **unbuilt),
        _ledger(name="dry_ok", exhausted=True, raw_rows=30, **unbuilt),
        _ledger(name="dry_empty", exhausted=True, raw_rows=0, skipped_malformed=30, **unbuilt),
        _ledger(name="built"),
    ]
    assert report.missing_raw_rows() == ["short", "stale", "dry_empty"]
    report.tokenizer_complete = False
    assert report.missing_raw_rows() == ["short", "stale", "dry_empty", "tokenizer"]


def test_the_satisfaction_cases() -> None:
    assert _ledger().satisfaction() == (True, "ok")
    stale = _ledger(raw_state="stale", raw_reason="stale: source identity or tokenizer changed")
    assert stale.satisfaction() == (False, "raw stale: source identity or tokenizer changed; the repair step deletes it after confirmation")
    outdated = _ledger(raw_state="outdated", raw_reason="outdated: dataset_max_sequence_length 64 -> 128")
    assert outdated.satisfaction() == (False, "raw outdated: dataset_max_sequence_length 64 -> 128; the repair step deletes it after confirmation")
    assert _ledger(raw_state="missing", raw_reason="missing", raw_rows=0).satisfaction() == (False, "raw missing")
    relabelled = _ledger(raw_state="tokenizer_changed", raw_reason="tokenizer changed: a (synthetic) -> b (synthetic); 5 rows ...")
    assert relabelled.satisfaction() == (False, f"raw {relabelled.raw_reason}; the repair step asks whether to keep it")
    for problem, reason in (("absent", "missing"), ("stale", "stale: processing settings, dataset_max_sequence_length or the source changed")):
        assert _ledger(processed_problem=problem, processed_reason=reason).satisfaction() == (False, f"processed {reason}")
        assert _ledger(processed_problem=problem, processed_reason=reason).build_pending
    assert _ledger(processed_problem="behind_raw", processed_reason="behind raw", processed_rows=10).satisfaction() == (False, "processed behind raw")
    assert _ledger(processed_problem="behind_raw", processed_reason="behind raw", processed_rows=10).build_pending
    assert _ledger(processed_problem="behind_raw", processed_reason="behind raw").satisfaction() == (True, "ok")  # served: the cap
    assert not _ledger(processed_problem="behind_raw", processed_reason="behind raw", unbuilt_shards=2).build_pending
    assert not _ledger(raw_state="missing", raw_reason="missing", raw_rows=0, processed_problem="absent", processed_reason="missing").build_pending
    assert _ledger(processed_rows=10).satisfaction() == (False, "processed rows 10 < 84")
    dry_small = _ledger(processed_rows=10, training_rows=9, exhausted=True)
    assert dry_small.satisfaction() == (True, "exhausted at 10 of 84 rows")
    dry_empty = _ledger(processed_rows=0, training_rows=0, exhausted=True, skipped_malformed=100)
    satisfied, reason = dry_empty.satisfaction()
    assert not satisfied and "NOT ONE" in reason and "100 malformed" in reason
    assert "check the source's fields / converter / filter / language" in reason
    assert dry_empty.epochs() is None and _ledger().epochs() == pytest.approx(70 / 90)


def test_an_exhausted_source_whose_rows_all_go_to_the_holdout_is_failed() -> None:
    """
    The training resolver holds `ceil(validation_fraction × rows)` out; an exhausted source must keep at least
    one training row after that split, or training would fail at startup with an empty range (decision D3).
    """

    all_validation = _ledger(processed_rows=2, training_rows=0, exhausted=True)
    assert all_validation.satisfaction() == (
        False,
        "exhausted, and 2 processed rows − 2 validation holdout leaves 0 training rows; "
        "lower the source's validation_fraction or give it more rows",
    )
    assert all_validation.epochs() is None
    # one surviving training row is enough: served, with the exhaustion warning
    one_left = _ledger(processed_rows=2, training_rows=1, exhausted=True)
    assert one_left.satisfaction() == (True, "exhausted at 2 of 84 rows")
    # the ledger's `training_rows` mirrors the training resolver's arithmetic
    cfg = two_stage_cfg()
    assert training_rows_after_split(cfg, "a", 1) == 0  # ceil(0.05 × 1) = 1: the single row is all validation
    assert training_rows_after_split(cfg, "a", 2) == 1


def test_rows_to_fetch_tops_up_from_the_observed_yield() -> None:
    """
    Raw is long enough but only 60 of 100 rows survived the build: the shortfall (84 − 60) is divided by the
    observed yield (0.6) and multiplied by the same 1.2 safety margin the first download uses.
    """

    short = _ledger(processed_rows=60, training_rows=60)
    assert short.rows_to_fetch[0] == 48 == ceil((84 - 60) * Fraction("1.2") / Fraction(60, 100))
    assert short.rows_target == 100 + 48, "the download takes a target, not an increment"
    assert "top-up: 60 of 84 rows survived 100 raw" in short.rows_to_fetch[1]

    assert _ledger(raw_rows=50, processed_rows=45, training_rows=45).rows_to_fetch[0] == 50  # raw itself is short
    assert _ledger(processed_rows=20, exhausted=True).rows_to_fetch[0] == 0  # nothing left to fetch
    assert _ledger(processed_rows=20, processed_problem="behind_raw", processed_reason="behind raw").rows_to_fetch[0] == 0  # build first
    nothing_survives = _ledger(processed_rows=0)
    assert nothing_survives.rows_to_fetch == (0, "no row of 100 raw rows survives the build")  # no yield to extrapolate from


def test_a_served_budget_is_never_re_downloaded() -> None:
    """
    A budget raised a little (or a measured rate a little below the estimate) leaves raw short of the new
    `rows_needed` while the processed rows already serve it: satisfied, and nothing to fetch. Before, the raw
    comparison came first and re-downloaded 150,000 rows nobody needed.
    """

    served = _ledger(rows_needed=1_200_000, rows_sufficient=1_000_000, raw_rows=1_050_000, processed_rows=1_040_000, training_rows=1_040_000)
    assert served.satisfaction() == (True, "ok") and served.rows_to_fetch == (0, "budget served")
    unbuilt = _ledger(rows_needed=1_200_000, rows_sufficient=1_000_000, raw_rows=1_050_000, processed_rows=0, processed_problem="absent", processed_reason="missing")
    assert unbuilt.rows_to_fetch == (150_000, "rows 1,050,000 < 1,200,000")  # nothing built yet: raw is what counts
    # the build is capped at the budget: a folder behind raw that serves it is served, one short of it builds first
    behind = _ledger(rows_needed=1_200_000, rows_sufficient=1_000_000, raw_rows=1_050_000, processed_rows=1_040_000, processed_problem="behind_raw", processed_reason="behind raw", unbuilt_shards=5)
    assert behind.rows_to_fetch == (0, "budget served") and behind.satisfaction() == (True, "ok, 5 raw shard(s) past the budget unbuilt") and not behind.build_pending
    short = _ledger(rows_needed=1_200_000, rows_sufficient=1_000_000, raw_rows=1_050_000, processed_rows=500_000, processed_problem="behind_raw", processed_reason="behind raw", unbuilt_shards=50)
    assert short.rows_to_fetch == (150_000, "rows 1,050,000 < 1,200,000") and short.build_pending  # raw short and the build has work


def test_the_top_up_is_capped_at_the_full_requirement(caplog: pytest.LogCaptureFixture) -> None:
    """
    A pathological yield (1 of 1,200 rows) extrapolates to a download nobody wants: the round asks for at most
    `rows_needed`, with a warning, and the next round measures the yield again on more data.
    """

    pathological = _ledger(raw_rows=1200, processed_rows=1, training_rows=1)
    assert ceil((84 - 1) * Fraction("1.2") * 1200) == 119_520  # what the yield extrapolates to
    with caplog.at_level("WARNING", logger="data_preparation"):
        assert pathological.rows_to_fetch[0] == 100 == pathological.rows_needed
    assert "capping this round at the full requirement of 100 rows" in caplog.text
    assert "top-up: 1 of 84 rows survived 1,200 raw" in pathological.rows_to_fetch[1]
    caplog.clear()
    with caplog.at_level("WARNING", logger="data_preparation"):
        assert _ledger(processed_rows=60, training_rows=60).rows_to_fetch[0] == 48  # under the cap: unchanged, no warning
    assert caplog.text == ""


# --- satisfaction ---------------------------------------------------------------------------------------------------


def test_sources_are_satisfied_after_prepare(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and report.tokenizer_complete and report.missing() == []
    a = _state(report, "a")
    assert a.satisfaction()[0] and a.satisfaction()[1] == "ok" and a.state() == "complete" and not a.exhausted
    assert a.raw_rows == 127 and a.processed_rows >= 106 and a.rows_needed == 127
    assert a.epochs() == pytest.approx(100 / training_rows_after_split(cfg, "a", a.processed_rows))
    h = _state(report, "h")
    assert h.satisfaction()[0] and h.raw_rows == 10 and h.epochs() is None  # not trained on: no budget to cycle
    assert every_source_satisfies_its_budget(cfg, layout) and every_source_satisfies_its_budget(cfg, layout, sources=["i"])
    assert sources_with_pending_raw_shards(cfg, layout) == []
    assert report.describe().endswith("dataset complete")


def test_not_satisfied_when_processed_is_missing_stale_or_behind_raw(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)

    (layout.processed_dir("b") / "MANIFEST.json").unlink()
    b = source_ledger(cfg, "b", layout)
    assert not b.satisfaction()[0] and b.satisfaction()[1] == "processed no manifest" and b.raw_rows == 60 and b.processed_rows == 0 and b.epochs() is None
    assert not every_source_satisfies_its_budget(cfg, layout) and every_source_satisfies_its_budget(cfg, layout, sources=["a", "i"])
    assert sources_with_pending_raw_shards(cfg, layout) == ["b"]

    stale = two_stage_cfg(min_chars=2)  # changes every processed hash, no raw hash
    a = source_ledger(stale, "a", layout)
    assert not a.satisfaction()[0] and a.satisfaction()[1] == "processed stale: processing.min_chars: 1 -> 2" and a.raw_rows == 127

    bigger = two_stage_cfg(tokens_b=12800)
    prepare(config_file(bigger), layout.root, assume_yes=False, steps=["download"])  # raw topped up, processed not
    a = source_ledger(bigger, "a", layout)
    assert not a.satisfaction()[0] and a.satisfaction()[1] == "processed behind raw" and a.raw_rows == 316 and 0 < a.processed_rows < 316
    assert sources_with_pending_raw_shards(bigger, layout) == ["a", "b"]


def test_an_unreadable_processed_manifest_is_reported_not_raised(layout: DatasetLayout, config_file: ConfigFile) -> None:
    """
    `Manifest.load` raises next to shards: right for raw, wrong for a processed folder the repair step deletes
    (after confirmation): `status` (and `prepare --dry_run`, and training's auto-prepare) must report it.
    """

    cfg = two_stage_cfg()
    path = config_file(cfg)
    prepare(path, layout.root, assume_yes=False)
    (layout.processed_dir("b") / "MANIFEST.json").write_text("{ not json")

    b = source_ledger(cfg, "b", layout)
    assert not b.satisfaction()[0] and b.processed_rows == 0 and b.raw_rows == 60
    assert b.satisfaction()[1] == "processed unreadable manifest"
    assert not every_source_satisfies_its_budget(cfg, layout) and sources_with_pending_raw_shards(cfg, layout) == ["b"]
    assert [s.name for s in plan_downloads(cfg, layout).to_fetch()] == []  # raw is complete; the build is what is missing

    report = status(path, layout.root)  # used to crash with "unreadable manifest ... next to shards"
    assert not report.complete and "b" in report.missing() and "b" in report.needs_repair
    with pytest.raises(ConfirmationRequired):  # the repair step heals it, but only after the user confirmed
        prepare(path, layout.root, assume_yes=False)
    assert prepare(path, layout.root, assume_yes=True).complete


def test_short_processed_folder_is_not_satisfied(layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = two_stage_cfg()
    prepare(config_file(cfg), layout.root, assume_yes=False)
    bigger = two_stage_cfg(tokens_b=12800)  # processed covers every raw shard but holds fewer rows than the new budget needs
    a = source_ledger(bigger, "a", layout)
    assert not a.satisfaction()[0] and a.satisfaction()[1] == f"processed rows {a.processed_rows} < {bigger.rows_sufficient('a')}" and a.epochs() is None
    row = next(line for line in summarize_dataset_state(bigger, layout).table().splitlines() if line.startswith("a "))
    assert row.split()[6] == "-" and row.split()[7] == "incomplete"  # epochs only when complete


def test_exhausted_and_built_source_is_satisfied(layout: DatasetLayout, cfg_factory: CfgFactory, config_file: ConfigFile, write_local: Writer) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": f"tok_{i} tok_2 tok_3"} for i in range(4)], "parquet")
    cfg = cfg_factory({"v": SourceConfig(kind="pretrain", loader="local", path=str(src_dir), rows=10)})
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    v = _state(report, "v")
    assert report.complete and v.satisfaction()[0] and v.exhausted and v.state() == "exhausted"
    assert v.satisfaction()[1] == "exhausted at 4 of 10 rows" and v.processed_rows == 4  # its `rows` are the delivered rows
    (layout.processed_dir("v") / "MANIFEST.json").unlink()
    assert not source_ledger(cfg, "v", layout).satisfaction()[0]  # exhausted alone is not enough: the raw shards must be built


def test_prepare_fails_an_exhausted_source_the_validation_holdout_would_empty(layout: DatasetLayout, cfg_factory: CfgFactory, config_file: ConfigFile, write_local: Writer) -> None:
    """
    End to end: a trained-and-validated source that runs dry with its every processed row going to the
    training-time holdout is reported FAILED by `prepare`; training would otherwise crash at startup. A val-only
    source holds nothing out, so any row still serves it (`test_exhausted_and_built_source_is_satisfied`).
    """

    src_dir = layout.root.parent / "one_row"
    write_local(src_dir, [{"text": "tok_1 tok_2 tok_3"}], "parquet")
    cfg = cfg_factory({"w": SourceConfig(kind="pretrain", loader="local", path=str(src_dir), validation_fraction=0.5)})
    assert cfg.used_in_train("w") and cfg.used_in_val("w") and cfg.validation_fraction_of("w") == 0.5
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    w = _state(report, "w")
    assert not report.complete and not w.satisfaction()[0] and w.exhausted and w.processed_rows == 1
    assert w.satisfaction()[1] == (
        "exhausted, and 1 processed rows − 1 validation holdout leaves 0 training rows; "
        "lower the source's validation_fraction or give it more rows"
    )


def test_report_table_and_missing(layout: DatasetLayout, config_file: ConfigFile) -> None:
    empty = DatasetReport()
    assert not empty.complete and empty.missing() == ["tokenizer"] and empty.describe().endswith("dataset INCOMPLETE")
    cfg = two_stage_cfg()
    report = summarize_dataset_state(cfg, layout)
    assert report.missing() == ["a", "b", "h", "i", "tokenizer"] and not report.complete
    lines = report.table().splitlines()
    assert lines[0].split() == ["source", "kind", "needed", "tokens/row", "raw", "processed", "epochs", "state", "reason"]
    assert lines[1].split() == ["a", "pretrain", "127", "64", "0", "0", "-", "incomplete", "raw", "missing"]
    assert lines[-1].split() == ["tokenizer", "tokenizer", "incomplete"]
    prepare(config_file(cfg), layout.root, assume_yes=False)
    complete = summarize_dataset_state(cfg, layout)
    assert complete.complete and complete.unsatisfied() == []
    assert complete.table().splitlines()[-1].split() == ["tokenizer", "tokenizer", "complete"]
    flagged = summarize_dataset_state(cfg, layout, needs_repair=["b", "b"])
    assert not flagged.complete and flagged.missing() == ["b"] and flagged.needs_repair == ["b"]
    assert "needs repair" in next(line for line in flagged.table().splitlines() if line.startswith("b "))
