# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `build` / `status`: refinement loop, repair of broken stages, idempotence, step/source filtering."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.build import runner as build_mod
from data_preparation.lib.build import build, status
from data_preparation.lib.schema.dataset_config import DatasetConfig, InstructMixtureConfig, SourceConfig, StageConfig
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest, verify_shards
from data_preparation.lib.stages.pretrain import process as real_process

CfgFactory = Callable[..., DatasetConfig]
Writer = Callable[[Path, list[dict[str, Any]], str], Path]


def all_mtimes(root: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_build_refines_a_bad_estimate(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    """Estimates are ~10x too high (but below the max_seq_length cap, which would otherwise clamp them); the measured
    tokens/row of round 1 sizes the second download correctly."""
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0, tokens_per_row_estimate=3_000),
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2, tokens_per_row_estimate=2_000),
    }
    cfg = cfg_factory(sources, instruct_mixtures={"m": InstructMixtureConfig(sources={"i": 1.0}, max_tokens=4096)}, tokens=3000, max_seq_length=4096)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        result = build(cfg, layout, max_rounds=3)
    assert result.complete
    processed = Manifest.load(layout.source_dir("p", "processed"))
    assert processed is not None and (processed.tokens() or 0) >= 3000
    train = Manifest.load(layout.instruct_mixture_dir("t", "m", "train"))
    assert train is not None and train.extra["short_sources"] == {}
    assert "p: round 1:" in caplog.text and "m: round 1: short sources ['i']" in caplog.text


def test_build_raises_after_max_rounds(cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", tokens_per_row_estimate=1000)}, tokens=1000)
    calls: list[int] = []

    def never_enough(*args: Any, **kwargs: Any) -> Manifest:
        manifest = real_process(*args, **kwargs)
        calls.append(manifest.rows())
        stub = Manifest(source="p", source_hash=manifest.source_hash, stage="processed")
        stub.add_shard("data-00000.parquet", rows=1, tokens=1)  # never enough tokens
        return stub

    monkeypatch.setattr(build_mod, "process", never_enough)
    with pytest.raises(RuntimeError, match="token budget 1000 not reached after 2 rounds"):
        build(cfg, layout, max_rounds=2)
    assert len(calls) == 2


def test_build_exhausted_source_warns_and_is_complete(cfg_factory: CfgFactory, layout: DatasetLayout, write_local: Writer, caplog: pytest.LogCaptureFixture) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": "tok_1 tok_2 tok_3"}] * 3 + [{"text": "tok_4 tok_5"}], "parquet")
    cfg = cfg_factory({"s": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=1000)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        result = build(cfg, layout)
    assert "s: exhausted at 5 tokens, budget is 1000" in caplog.text
    assert result.complete and result.sources[0].exhausted
    processed = Manifest.load(layout.source_dir("s", "processed"))
    assert processed is not None and processed.rows() == 2
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert status(cfg, layout).complete
    assert "s: exhausted at 5 of 1000 tokens" in caplog.text


def test_build_twice_writes_nothing(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0),
        "h": SourceConfig(kind="validation", loader="synthetic", seed=1, rows=4),
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2),
    }
    cfg = cfg_factory(sources, tokens=500)
    assert build(cfg, layout).complete
    before = all_mtimes(layout.root)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        assert build(cfg, layout).complete
    assert all_mtimes(layout.root) == before
    assert "p: complete, skipping" in caplog.text and "h: complete, skipping" in caplog.text and "auto: complete, skipping" in caplog.text


def test_build_repairs_missing_shards(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0),
        "h": SourceConfig(kind="validation", loader="synthetic", seed=1, rows=4),
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2),
    }
    cfg = cfg_factory(sources, tokens=500)
    build(cfg, layout)
    victims = [
        next(layout.source_dir("p", "processed").glob("data-*.parquet")),
        next(layout.validation_dir("h").glob("data-*.parquet")),
        next(layout.instruct_mixture_dir("t", "auto", "train").glob("data-*.parquet")),
    ]
    for victim in victims:
        victim.unlink()
    assert not status(cfg, layout).complete
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        result = build(cfg, layout)
    assert result.complete and all(v.is_file() for v in victims)
    assert "missing shard" in caplog.text and "removing" in caplog.text
    for directory in (layout.source_dir("p", "processed"), layout.validation_dir("h"), layout.instruct_mixture_dir("t", "auto", "train")):
        manifest = Manifest.load(directory)
        assert manifest is not None and verify_shards(directory, manifest) == []


def test_build_rebuilds_stale_hash(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    from data_preparation.lib.schema.dataset_config import ProcessingConfig

    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500)
    build(cfg, layout)
    cfg.processing = ProcessingConfig(min_chars=2)
    result = build(cfg, layout)
    assert result.complete
    raw = Manifest.load(layout.source_dir("p", "raw"))
    assert raw is not None and raw.is_current(cfg.source_hash("p"))


def test_build_steps_filter(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    # estimate 500 tokens/row vs ~220 real (uncapped: max_seq_length above the document length), so the first
    # estimate-sized download falls short of the budget
    cfg = cfg_factory(
        {"p": SourceConfig(kind="pretrain", loader="synthetic"), "h": SourceConfig(kind="validation", loader="synthetic", seed=1, rows=4)},
        tokens=500,
        max_seq_length=4096,
    )
    result = build(cfg, layout, steps={"tokenizer", "download"})
    assert not result.complete and result.tokenizer_complete
    assert (layout.source_dir("p", "raw") / "MANIFEST.json").is_file()
    assert not layout.source_dir("p", "filtered").exists() and not layout.validation_dir("h").exists()
    result = build(cfg, layout, steps={"filter", "process", "validation"})
    (p,) = result.sources
    assert (layout.source_dir("p", "processed") / "MANIFEST.json").is_file() and result.validations[0].complete
    assert not p.complete and p.reason.startswith("tokens ")  # the estimate-sized download cannot be topped up without `download`
    assert build(cfg, layout).complete
    with pytest.raises(ValueError, match="unknown steps"):
        build(cfg, layout, steps={"nope"})


def test_build_sources_filter(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    sources = {"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0), "b": SourceConfig(kind="pretrain", loader="synthetic", seed=1)}
    cfg = cfg_factory(sources, tokens=500)
    result = build(cfg, layout, sources=["a"])
    assert not result.complete and (layout.source_dir("a", "processed") / "MANIFEST.json").is_file()
    assert not layout.source_dir("b", "raw").exists()
    assert build(cfg, layout, sources=["b"]).complete
    with pytest.raises(ValueError, match="unknown sources"):
        build(cfg, layout, sources=["c"])


def test_build_skips_unused_sources(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    sources = {"used": SourceConfig(kind="pretrain", loader="synthetic"), "unused": SourceConfig(kind="pretrain", loader="synthetic", seed=5)}
    cfg = cfg_factory(sources, tokens=500)
    cfg.stages = [StageConfig(name="s", tokens=500, train={"used": 1.0}, val={"used": 1.0})]
    assert build(cfg, layout).complete
    assert (layout.source_dir("used", "processed") / "MANIFEST.json").is_file()
    assert not layout.source_dir("unused", "raw").exists()


def test_build_dry_run_writes_nothing(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")}, tokens=500)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        result = build(cfg, layout, dry_run=True)
    assert not result.complete and not layout.root.exists()
    assert "dry run" in caplog.text and "INCOMPLETE" in caplog.text


def test_build_failure_propagates(cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")}, tokens=500)

    def boom(*args: Any, **kwargs: Any) -> Manifest:
        raise OSError("network down")

    monkeypatch.setattr(build_mod, "download", boom)
    with caplog.at_level(logging.ERROR, logger="data_preparation"), pytest.raises(OSError, match="network down"):
        build(cfg, layout)
    assert "source p failed" in caplog.text
