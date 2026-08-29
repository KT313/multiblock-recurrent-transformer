# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.prepare: the `tiny` command and the inline planner (`build_dataset`)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_preparation import prepare
from data_preparation.lib.dataset_config import DatasetConfig, SourceConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.manifest import Manifest, verify_shards

REPO_ROOT = Path(__file__).resolve().parents[1]
CfgFactory = Callable[..., DatasetConfig]
Writer = Callable[[Path, list[dict[str, Any]], str], Path]


def test_rows_for_budget() -> None:
    assert prepare.rows_for_budget(1000, 100) == 12  # 10 rows × 1.2
    assert prepare.rows_for_budget(1000, 100, margin=1.0) == 10
    assert prepare.rows_for_budget(0, 100) == 1
    assert prepare.rows_for_budget(1, 1e-12) >= 1


def test_parser_defaults_and_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = prepare.build_parser().parse_args(["tiny"])
    assert args.dataset_config == Path("config/datasets/tiny.yaml") and args.dataset_dir == Path("dataset")
    assert args.hf_token is None and args.run is prepare.run_tiny
    seen: list[tuple[DatasetConfig, DatasetLayout, str | None]] = []
    monkeypatch.setattr(prepare, "build_dataset", lambda cfg, layout, hf_token=None: seen.append((cfg, layout, hf_token)))
    prepare.main(["tiny", "--dataset_config", str(REPO_ROOT / "config/datasets/tiny.yaml"), "--dataset_dir", str(tmp_path), "--hf_token", "t"])
    assert len(seen) == 1 and seen[0][0].name == "tiny" and seen[0][1] == DatasetLayout(tmp_path) and seen[0][2] == "t"


def test_missing_or_unknown_command_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        prepare.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        prepare.main(["no-such-command"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_tiny_build_is_complete(tiny_dataset_config: DatasetConfig, tiny_layout: DatasetLayout) -> None:
    """The session fixture built config/datasets/tiny.yaml; every stage directory has a current, verified manifest."""
    cfg, layout = tiny_dataset_config, tiny_layout
    tok = Manifest.load(layout.tokenizer_dir("synthetic"))
    assert tok is not None and tok.is_current(cfg.tokenizer_hash())
    for stage in ("raw", "filtered", "processed"):
        m = Manifest.load(layout.source_dir("synthetic_pretrain", stage))
        assert m is not None and m.is_current(cfg.source_hash("synthetic_pretrain")) and m.stage == stage
        assert verify_shards(layout.source_dir("synthetic_pretrain", stage), m) == []
    processed = Manifest.load(layout.source_dir("synthetic_pretrain", "processed"))
    assert processed is not None and (processed.tokens() or 0) >= cfg.source_budget_tokens("synthetic_pretrain")
    hold = Manifest.load(layout.holdout_dir("synthetic_val"))
    assert hold is not None and hold.rows() == 32 and verify_shards(layout.holdout_dir("synthetic_val"), hold) == []
    for split in ("train", "validation"):
        m = Manifest.load(layout.mixture_dir("tiny", "tiny_mixture", split))
        assert m is not None and m.is_current(cfg.mixture_hash("tiny_mixture")) and m.rows() > 0
        assert m.extra["short_sources"] == {}
        assert verify_shards(layout.mixture_dir("tiny", "tiny_mixture", split), m) == []


def test_build_dataset_is_idempotent(tiny_dataset_config: DatasetConfig, tiny_layout: DatasetLayout) -> None:
    root = tiny_layout.root
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*.parquet")}
    prepare.build_dataset(tiny_dataset_config, tiny_layout)
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*.parquet")} == before


def test_build_dataset_refines_a_bad_estimate(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    """tokens_per_row_estimate is 100x too high; the measured tokens/row of round 1 fixes the second download."""
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0, tokens_per_row_estimate=20_000),
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2, tokens_per_row_estimate=5_000),
    }
    from data_preparation.lib.dataset_config import MixtureConfig

    cfg = cfg_factory(sources, mixtures={"m": MixtureConfig(sources={"i": 1.0}, max_tokens=64)}, tokens=3000, max_seq_length=64)
    prepare.build_dataset(cfg, layout)
    processed = Manifest.load(layout.source_dir("p", "processed"))
    assert processed is not None and (processed.tokens() or 0) >= 3000
    train = Manifest.load(layout.mixture_dir("t", "m", "train"))
    assert train is not None and train.extra["short_sources"] == {}


def test_build_dataset_exhausted_source_warns_instead_of_looping(
    cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": "tok_1 tok_2 tok_3"}] * 3 + [{"text": "tok_4 tok_5"}], "parquet")
    cfg = cfg_factory({"s": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=1000)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        prepare.build_dataset(cfg, layout)
    assert "exhausted at 5 tokens, budget is 1000" in caplog.text
    processed = Manifest.load(layout.source_dir("s", "processed"))
    assert processed is not None and processed.rows() == 2


def test_build_dataset_skips_unused_sources(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    from data_preparation.lib.dataset_config import StageConfig

    sources = {"used": SourceConfig(kind="pretrain", loader="synthetic"), "unused": SourceConfig(kind="pretrain", loader="synthetic", seed=5)}
    cfg = cfg_factory(sources, tokens=500)
    cfg.stages = [StageConfig(name="s", tokens=500, train={"used": 1.0}, val={"used": 1.0})]
    prepare.build_dataset(cfg, layout)
    assert (layout.source_dir("used", "processed") / "MANIFEST.json").is_file()
    assert not layout.source_dir("unused", "raw").exists()
