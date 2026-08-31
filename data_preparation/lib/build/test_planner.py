# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the budget planner: arithmetic by hand, completeness rules against real manifests of tiny builds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from data_preparation.dataset_config import (
    DatasetConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
)
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build import build
from data_preparation.lib.build.planner import Plan, SourcePlan, budget_tokens_of, plan, rows_for_budget, stage_problems
from data_preparation.lib.storage.manifest import Manifest

CfgFactory = Callable[..., DatasetConfig]


def two_stage_cfg(tokens_a: int = 1000, tokens_b: int = 1000, rows_h: int = 8) -> DatasetConfig:
    """Source `a` in both stages (0.5 then 1.0), `b` only in the first, `h` used only for validation (`rows`),
    instruct source `i` in a finetune stage. `block_size` 1: the interim token budget equals `tokens × weight`."""
    return DatasetConfig(
        name="two",
        tokenizer=TokenizerConfig(name="synthetic", kind="synthetic"),
        sources={
            "a": SourceConfig(kind="pretrain", loader="synthetic", seed=0, describe_tokens_per_row=100),
            "b": SourceConfig(kind="pretrain", loader="synthetic", seed=1, describe_tokens_per_row=50),
            "h": SourceConfig(kind="pretrain", loader="synthetic", seed=3, rows=rows_h),
            "i": SourceConfig(kind="instruct", loader="synthetic", seed=4, describe_tokens_per_row=40),
        },
        stages=[
            StageConfig(name="s1", tokens=tokens_a, train={"a": 0.5, "b": 0.5}, val={"h": 1.0}),
            StageConfig(name="s2", tokens=tokens_b, train={"a": 1.0}, val={"h": 1.0}),
            StageConfig(name="ft", tokens=400, train={"i": 1.0}, val={"i": 1.0}),
        ],
        block_size=1,
        max_seq_length=128,  # above every describe_tokens_per_row, so the estimates are not clamped
        processing=ProcessingConfig(min_chars=1),
    )


def _source(result: Plan, name: str) -> SourcePlan:
    return next(s for s in result.sources if s.name == name)


def test_rows_for_budget() -> None:
    assert rows_for_budget(1000, 100) == 12  # 10 rows × 1.2
    assert rows_for_budget(1000, 100, margin=1.0) == 10
    assert rows_for_budget(0, 100) == 1
    assert rows_for_budget(1, 1e-12) >= 1


def test_budget_tokens_of_is_the_sequence_budget_times_block_size() -> None:
    cfg = two_stage_cfg()
    assert cfg.sequence_budget("a") == 1000 and budget_tokens_of(cfg, "a") == 1000  # max(1000 × 0.5, 1000 × 1.0), not the sum
    assert budget_tokens_of(cfg, "b") == 500 and budget_tokens_of(cfg, "h") == 0 and budget_tokens_of(cfg, "i") == 400
    wide = replace(cfg, block_size=64)
    assert budget_tokens_of(wide, "b") == 8 * 64  # ceil(500 / 64) sequences of 64 tokens


def test_plan_on_empty_dir_by_hand(layout: DatasetLayout) -> None:
    result = plan(two_stage_cfg(), layout)
    assert not result.complete and not result.tokenizer_complete
    assert [s.name for s in result.sources] == ["a", "b", "h", "i"]
    a, b, h, i = result.sources
    assert a.kind == "pretrain" and a.budget_tokens == 1000
    assert a.tokens_per_row == 100 and a.rows_needed == 12 and a.rows_present == 0 and a.rows_to_fetch == 12
    assert b.budget_tokens == 500 and b.rows_needed == 12 and b.rows_to_fetch == 12
    assert not a.complete and not a.manifest_current and "raw: manifest missing" in a.reason
    assert h.kind == "pretrain" and h.budget_tokens == 0 and h.rows_needed == 8 and h.rows_to_fetch == 8 and not h.complete
    assert i.kind == "instruct" and i.budget_tokens == 400 and i.tokens_per_row == 40 and i.rows_needed == 12 and not i.complete
    missing = result.missing()
    assert missing[0] == "tokenizer: missing or stale" and [line.split(":")[0] for line in missing[1:]] == ["source a", "source b", "source h", "source i"]
    assert "INCOMPLETE" in result.summary()


def test_plan_after_build_is_complete_and_uses_measured_tokens(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    result = build(cfg, layout)
    assert result.complete and result.tokenizer_complete and result.missing() == []
    a = _source(result, "a")
    raw, processed = Manifest.load(layout.raw_dir("a")), Manifest.load(layout.processed_dir("a"))
    assert raw is not None and processed is not None
    assert a.tokens_per_row == (processed.tokens() or 0) / raw.rows() and a.tokens_per_row != 100
    assert a.tokens_present >= 1000 and a.rows_present == raw.rows() and a.rows_to_fetch == 0 and a.manifest_current
    assert a.reason == "ok" and not a.exhausted
    h = _source(result, "h")
    assert h.complete and h.rows_present == 8 and h.tokens_per_row > 0 and h.epochs is None  # validation-only: no budget
    i = _source(result, "i")
    assert i.complete and i.tokens_present >= 400 and i.epochs is not None and i.epochs <= 1.0
    processed_i = Manifest.load(layout.processed_dir("i"))
    assert processed_i is not None and processed_i.extra["columns"] == ["instruction", "input", "output", "tokens", "hash"]
    assert a.epochs is not None and a.epochs <= 1.0
    assert result.summary().endswith("dataset complete")
    row_a = next(line for line in result.summary().splitlines() if line.startswith("a "))
    assert row_a.split()[6] == "-"  # fetch column: nothing to fetch for a complete source


def test_plan_uses_raw_token_counts_right_after_download(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    result = build(cfg, layout, steps={"tokenizer", "download"})
    a = _source(result, "a")
    raw = Manifest.load(layout.raw_dir("a"))
    assert raw is not None and raw.tokens() is not None
    assert a.tokens_per_row == (raw.tokens() or 0) / raw.rows() and a.tokens_per_row != 100  # measured, not the estimate
    assert not a.complete and a.reason == "processed: manifest missing"
    assert f"{a.tokens_per_row:.1f}" in result.summary()


def test_rows_to_fetch_is_clamped_at_zero(layout: DatasetLayout) -> None:
    build(two_stage_cfg(), layout)
    smaller = two_stage_cfg(tokens_a=100, tokens_b=100)  # same source hashes, a tenth of the budget
    result = plan(smaller, layout)
    a = _source(result, "a")
    assert a.rows_present > a.rows_needed and a.rows_to_fetch == 0 and a.complete
    assert a.tokens_present >= a.budget_tokens and a.reason == "ok"
    row_a = next(line for line in result.summary().splitlines() if line.startswith("a "))
    assert row_a.split()[6] == "-"


def test_stale_hash_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    cfg.processing = ProcessingConfig(min_chars=2)  # changes every processed hash; the raw dirs stay current
    result = plan(cfg, layout)
    for name in ("a", "i"):
        entry = _source(result, name)
        assert not entry.complete and not entry.manifest_current and entry.reason == "processed: manifest stale"
        assert entry.rows_present > 0 and entry.rows_to_fetch == 0, "the raw rows are kept and only rebuilt"
        assert stage_problems(cfg, name, layout) == {"processed": "processed: manifest stale"}


def test_missing_shard_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    for name in ("a", "i"):
        shard = next(layout.processed_dir(name).glob("data-*.parquet"))
        shard.unlink()
        entry = _source(plan(cfg, layout), name)
        assert not entry.complete and entry.reason == f"processed: missing shard {shard.name}"
        assert stage_problems(cfg, name, layout) == {"processed": f"processed: missing shard {shard.name}"}
    (raw_shard,) = layout.raw_dir("h").glob("data-*.parquet")
    raw_shard.unlink()
    h = _source(plan(cfg, layout), "h")
    assert not h.complete and h.reason == f"raw: missing shard {raw_shard.name}"


def test_processed_without_the_current_columns_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    for name in ("a", "i"):
        processed = Manifest.load(layout.processed_dir(name))
        assert processed is not None
        del processed.extra["columns"]
        processed.save(layout.processed_dir(name))
        entry = _source(plan(cfg, layout), name)
        assert not entry.complete and entry.manifest_current and entry.reason == "processed: predates the current columns"
        assert entry.rows_to_fetch == 0  # the rebuild needs no download
    assert build(cfg, layout).complete


def test_status_reports_processed_behind_raw_after_a_top_up(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    bigger = two_stage_cfg(tokens_a=5000, tokens_b=5000)
    build(bigger, layout, steps={"download"})  # raw topped up, processed not yet
    a = _source(plan(bigger, layout), "a")
    assert not a.complete and a.manifest_current and a.reason == "processed: behind raw"
    build(bigger, layout, steps={"process"})
    a = _source(plan(bigger, layout), "a")
    assert a.reason != "processed: behind raw"


def test_tokens_below_budget_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    bigger = two_stage_cfg(tokens_a=5000, tokens_b=5000)
    result = plan(bigger, layout)
    a = _source(result, "a")
    assert not a.complete and a.manifest_current and a.reason.startswith("tokens ") and a.rows_to_fetch > 0
    assert _source(result, "i").complete  # the finetune budget did not change


def test_validation_only_source_is_sized_by_its_rows(layout: DatasetLayout, cfg_factory: CfgFactory, write_local: Callable[[Path, list[dict[str, str]], str], Path]) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    more = two_stage_cfg(rows_h=20)  # `rows` is not part of any hash: the raw folder is topped up
    h = _source(plan(more, layout), "h")
    assert not h.complete and h.manifest_current and h.reason == "rows 8 < 20" and h.rows_to_fetch == 12
    result = build(more, layout)
    h = _source(result, "h")
    assert result.complete and h.complete and h.rows_present == 20 and h.reason == "ok"
    raw_h = Manifest.load(layout.raw_dir("h"))
    assert raw_h is not None and [s.rows for s in raw_h.shards] == [8, 12], "appended, nothing rewritten"

    # a local source shorter than its `rows` is exhausted and complete, with a warning-worthy reason
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": f"tok_{i} tok_2 tok_3"} for i in range(4)], "parquet")
    small = cfg_factory({"v": SourceConfig(kind="pretrain", loader="local", path=str(src_dir), rows=10)})
    result = build(small, layout)
    v = _source(result, "v")
    assert result.complete and v.complete and v.exhausted and v.reason == "exhausted at 4 of 10 rows"


def test_exhausted_source_is_complete(cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Callable[[Path, list[dict[str, str]], str], Path]) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": "tok_1 tok_2 tok_3"}] * 3 + [{"text": "tok_4 tok_5"}], "parquet")
    cfg = cfg_factory({"s": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=1000)
    result = build(cfg, layout)
    (s,) = result.sources
    assert result.complete and s.complete and s.exhausted and s.tokens_present == 5 and s.reason == "exhausted at 5 of 1000 tokens"
    assert s.epochs == 200.0  # 1000-token budget over 5 tokens on disk: the sampler cycles the source 200 times
    row = next(line for line in result.summary().splitlines() if line.startswith("s "))
    assert row.split()[8] == "200.00" and row.split()[9] == "exhausted"


def test_plan_dataclass_defaults() -> None:
    empty = Plan()
    assert empty.sources == [] and not empty.complete and empty.missing() == ["tokenizer: missing or stale"]


def test_estimate_is_clamped_to_max_seq_length(tmp_path: Path) -> None:
    """Pretrain token counts are capped at max_seq_length, so a prior above it would only cause a wasted round."""
    cfg = two_stage_cfg()
    cfg.sources["a"].describe_tokens_per_row = cfg.max_seq_length * 10
    result = plan(cfg, DatasetLayout(tmp_path / "empty"))
    entry = _source(result, "a")
    assert entry.tokens_per_row == cfg.max_seq_length
    assert entry.rows_needed == rows_for_budget(entry.budget_tokens, cfg.max_seq_length)
