# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the budget planner: arithmetic by hand, completeness rules against real manifests of tiny builds."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from data_preparation.lib.build import build
from data_preparation.lib.schema.dataset_config import (
    DatasetConfig,
    MixtureConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
)
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.build.planner import Plan, plan, rows_for_budget, stage_problems

CfgFactory = Callable[..., DatasetConfig]


def two_stage_cfg(tokens_a: int = 1000, tokens_b: int = 1000) -> DatasetConfig:
    """Source `a` in both stages (0.5 then 1.0), `b` only in the first, holdout `h` for validation, mixture `m`."""
    return DatasetConfig(
        name="two",
        tokenizer=TokenizerConfig(name="synthetic", kind="synthetic"),
        sources={
            "a": SourceConfig(kind="pretrain", loader="synthetic", seed=0, tokens_per_row_estimate=100),
            "b": SourceConfig(kind="pretrain", loader="synthetic", seed=1, tokens_per_row_estimate=50),
            "unused": SourceConfig(kind="pretrain", loader="synthetic", seed=2),
            "h": SourceConfig(kind="holdout", loader="synthetic", seed=3, rows=8),
            "i": SourceConfig(kind="instruct", loader="synthetic", seed=4, tokens_per_row_estimate=40),
        },
        mixtures={"m": MixtureConfig(sources={"i": 1.0}, max_tokens=64)},
        stages=[
            StageConfig(name="s1", tokens=tokens_a, train={"a": 0.5, "b": 0.5}, val={"h": 1.0}),
            StageConfig(name="s2", tokens=tokens_b, train={"a": 1.0}, val={"h": 1.0}),
            StageConfig(name="ft", tokens=400, train={"m": 1.0}, val={"m/validation": 1.0}),
        ],
        max_seq_length=64,
        processing=ProcessingConfig(min_chars=1),
    )


def test_rows_for_budget() -> None:
    assert rows_for_budget(1000, 100) == 12  # 10 rows × 1.2
    assert rows_for_budget(1000, 100, margin=1.0) == 10
    assert rows_for_budget(0, 100) == 1
    assert rows_for_budget(1, 1e-12) >= 1


def test_plan_on_empty_dir_by_hand(layout: DatasetLayout) -> None:
    result = plan(two_stage_cfg(), layout)
    assert not result.complete and not result.tokenizer_complete
    by_name = {s.name: s for s in result.sources}
    assert set(by_name) == {"a", "b"}  # `unused` is skipped
    a, b = by_name["a"], by_name["b"]
    assert a.budget_tokens == 1000  # max(1000 × 0.5, 1000 × 1.0), not the sum 1500
    assert a.tokens_per_row == 100 and a.rows_needed == 12 and a.rows_present == 0 and a.rows_to_fetch == 12
    assert b.budget_tokens == 500 and b.rows_needed == 12 and b.rows_to_fetch == 12
    assert not a.complete and not a.manifest_current and "raw: manifest missing" in a.reason
    (h,) = result.holdouts
    assert h.kind == "holdout" and h.rows_needed == 8 and h.rows_to_fetch == 8 and not h.complete
    (m,) = result.mixtures
    assert m.name == "m" and m.budget_tokens == 400 and not m.present and not m.current and not m.complete
    missing = result.missing()
    assert missing[0] == "tokenizer: missing or stale" and any(line.startswith("source a:") for line in missing)
    assert any(line.startswith("holdout h:") for line in missing) and any(line.startswith("mixture m:") for line in missing)
    assert "INCOMPLETE" in result.summary() and "unused" not in result.summary()


def test_plan_after_build_is_complete_and_uses_measured_tokens(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    result = build(cfg, layout)
    assert result.complete and result.tokenizer_complete and result.missing() == []
    a = next(s for s in result.sources if s.name == "a")
    raw, processed = Manifest.load(layout.source_dir("a", "raw")), Manifest.load(layout.source_dir("a", "processed"))
    assert raw is not None and processed is not None
    assert a.tokens_per_row == (processed.tokens() or 0) / raw.rows() and a.tokens_per_row != 100
    assert a.tokens_present >= 1000 and a.rows_present == raw.rows() and a.rows_to_fetch == 0 and a.manifest_current
    assert a.reason == "ok" and not a.exhausted
    (h,) = result.holdouts
    assert h.complete and h.rows_present == 8 and h.tokens_per_row > 0
    (m,) = result.mixtures
    assert m.complete and m.present and m.current and m.short_sources == []
    assert result.summary().endswith("dataset complete")
    row_a = next(line for line in result.summary().splitlines() if line.startswith("a "))
    assert row_a.split()[6] == "-"  # fetch column: nothing to fetch for a complete source


def test_rows_to_fetch_is_clamped_at_zero(layout: DatasetLayout) -> None:
    build(two_stage_cfg(), layout)
    smaller = two_stage_cfg(tokens_a=100, tokens_b=100)  # same source hashes, a tenth of the budget
    result = plan(smaller, layout)
    a = next(s for s in result.sources if s.name == "a")
    assert a.rows_present > a.rows_needed and a.rows_to_fetch == 0 and a.complete


def test_stale_hash_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    cfg.processing = ProcessingConfig(min_chars=2)  # changes every pretrain source hash
    result = plan(cfg, layout)
    a = next(s for s in result.sources if s.name == "a")
    assert not a.complete and not a.manifest_current and a.reason == "raw: manifest stale"
    assert a.rows_present == 0 and a.rows_to_fetch == a.rows_needed and a.tokens_per_row == 100  # back to the estimate
    assert stage_problems(cfg, "a", layout) == {s: f"{s}: manifest stale" for s in ("raw", "filtered", "processed")}


def test_missing_shard_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    shard = next(layout.source_dir("a", "processed").glob("data-*.parquet"))
    shard.unlink()
    result = plan(cfg, layout)
    a = next(s for s in result.sources if s.name == "a")
    assert not a.complete and a.reason == f"processed: missing shard {shard.name}"
    assert stage_problems(cfg, "a", layout) == {"processed": f"processed: missing shard {shard.name}"}
    (hold_shard,) = layout.holdout_dir("h").glob("data-*.parquet")
    hold_shard.unlink()
    (h,) = plan(cfg, layout).holdouts
    assert not h.complete and "missing shard" in h.reason
    mix_shard = next(layout.mixture_dir("two", "m", "train").glob("data-*.parquet"))
    mix_shard.unlink()
    (m,) = plan(cfg, layout).mixtures
    assert not m.complete and m.present and not m.current and "missing shard" in m.reason


def test_tokens_below_budget_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    bigger = two_stage_cfg(tokens_a=5000, tokens_b=5000)
    result = plan(bigger, layout)
    a = next(s for s in result.sources if s.name == "a")
    assert not a.complete and a.manifest_current and a.reason.startswith("tokens ") and a.rows_to_fetch > 0
    (m,) = result.mixtures
    assert m.complete  # the mixture budget (stage ft) did not change


def test_mixture_budget_change_is_not_complete(layout: DatasetLayout) -> None:
    cfg = two_stage_cfg()
    build(cfg, layout)
    cfg.stages[2] = StageConfig(name="ft", tokens=800, train={"m": 1.0}, val={"m/validation": 1.0})
    (m,) = plan(cfg, layout).mixtures
    assert not m.complete and m.budget_tokens == 800 and not m.current  # budget is part of the mixture hash


def test_exhausted_source_is_complete(cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Callable[[Path, list[dict[str, str]], str], Path]) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": "tok_1 tok_2 tok_3"}] * 3 + [{"text": "tok_4 tok_5"}], "parquet")
    cfg = cfg_factory({"s": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=1000)
    result = build(cfg, layout)
    (s,) = result.sources
    assert result.complete and s.complete and s.exhausted and s.tokens_present == 5 and s.reason == "exhausted at 5 of 1000 tokens"
    assert "exhausted" in result.summary()


def test_plan_dataclass_defaults() -> None:
    empty = Plan()
    assert empty.sources == [] and not empty.complete and empty.missing() == ["tokenizer: missing or stale"]



def test_estimate_is_clamped_to_max_seq_length(tmp_path: Path) -> None:
    """Token counts are capped at max_seq_length, so a prior above it would only cause a wasted refinement round."""
    cfg = two_stage_cfg()
    cfg.sources["a"].tokens_per_row_estimate = cfg.max_seq_length * 10
    result = plan(cfg, DatasetLayout(tmp_path / "empty"))
    entry = next(s for s in result.sources if s.name == "a")
    assert entry.tokens_per_row == cfg.max_seq_length
    assert entry.rows_needed == rows_for_budget(entry.budget_tokens, cfg.max_seq_length)
