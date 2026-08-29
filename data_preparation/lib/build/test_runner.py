# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `build` / `status`: refinement loop, repair of broken stages, idempotence, step/source filtering."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from data_preparation.conftest import REPO, REV, FakeHub
from data_preparation.lib.build import runner as build_mod
from data_preparation.lib.build import build, status
from data_preparation.lib.build.planner import plan
from data_preparation.lib.schema.dataset_config import DatasetConfig, InstructMixtureConfig, SourceConfig, StageConfig
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest, verify_shards
from data_preparation.lib.stages.pretrain import process as real_process
from data_preparation.lib.stages.shared import download as real_download

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
    assert not layout.source_dir("p", "processed").exists() and not layout.validation_dir("h").exists()
    result = build(cfg, layout, steps={"process", "validation"})
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


# --- github_code groups ------------------------------------------------------------------------------------------------


def _github_cfg(cfg_factory: CfgFactory, languages: list[str], **kwargs: Any) -> DatasetConfig:
    sources = {
        f"code_{lang.lower()}": SourceConfig(
            kind="pretrain", loader="github_code", hf_id=REPO, revision=REV, language=lang, tokens_per_row_estimate=2,
        )
        for lang in languages
    }
    return cfg_factory(sources, tokens=20, **kwargs)


def _code_rows(prefix: str, n: int) -> list[dict[str, Any]]:
    languages = ("Python", "Java", "Go")
    return [{"id": f"{prefix}{i}", "text": f"{prefix} code {i}", "language": languages[i % 3]} for i in range(n)]


def test_build_downloads_github_code_languages_in_one_pass(hub: FakeHub, cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    hub.add("data/a.parquet", _code_rows("a", 30))
    hub.add("data/b.parquet", _code_rows("b", 30))
    cfg = _github_cfg(cfg_factory, ["Python", "Java", "Go"])
    single: list[str] = []

    def spy_download(cfg: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        single.append(name)
        return real_download(cfg, name, *args, **kwargs)

    monkeypatch.setattr(build_mod, "download", spy_download)

    items = build_mod._work_items(cfg, layout, plan(cfg, layout), set(build_mod.STEPS), None, 1, None, 5, build_mod._Slots.create(1, 1))
    assert [(i.what, i.name) for i in items] == [("github_code group", "code_python, code_java, code_go")]  # the tokenizer is prepared before the items
    result = build(cfg, layout)
    assert result.complete and single == []  # the group pass replaced the per-source downloads
    assert len(hub.streams) == len(set(hub.streams))  # every repo file opened at most once for all three languages
    for name in ("code_python", "code_java", "code_go"):
        processed = Manifest.load(layout.source_dir(name, "processed"))
        assert processed is not None and (processed.tokens() or 0) >= cfg.source_budget_tokens(name)  # a third of the stage

    # `--sources` with one language uses the ordinary per-source path
    hub.streams.clear()
    bigger = _github_cfg(cfg_factory, ["Python", "Java", "Go"], max_seq_length=128)
    bigger.stages[0].tokens = 40
    items = build_mod._work_items(bigger, layout, plan(bigger, layout), set(build_mod.STEPS), {"code_python"}, 1, None, 5, build_mod._Slots.create(1, 1))
    assert [(i.what, i.name) for i in items] == [("source", "code_python")]
    build(bigger, layout, sources=["code_python"])
    assert single == ["code_python"] * len(single) and single


def test_github_code_group_raises_after_max_rounds(hub: FakeHub, cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    hub.add("data/a.parquet", _code_rows("a", 60))
    cfg = _github_cfg(cfg_factory, ["Python", "Java"])

    def never_enough(*args: Any, **kwargs: Any) -> Manifest:
        manifest = real_process(*args, **kwargs)
        stub = Manifest(source=manifest.source, source_hash=manifest.source_hash, stage="processed")
        stub.add_shard("data-00000.parquet", rows=1, tokens=1)
        return stub

    monkeypatch.setattr(build_mod, "process", never_enough)
    with pytest.raises(RuntimeError, match="code_python, code_java: token budget not reached after 2 rounds"):
        build(cfg, layout, max_rounds=2)


# --- overlapping download and processing -----------------------------------------------------------------------------


@dataclass
class _Trace:
    """Start/end instants of stubbed stages plus the largest number of them running at once, per stage kind."""

    events: list[tuple[str, str, str, float]] = field(default_factory=list)  # (stage, source, "start"|"end", t)
    running: dict[str, int] = field(default_factory=dict)
    peak: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enter(self, stage: str, name: str) -> None:
        with self.lock:
            self.running[stage] = self.running.get(stage, 0) + 1
            self.peak[stage] = max(self.peak.get(stage, 0), self.running[stage])
            self.events.append((stage, name, "start", time.monotonic()))

    def leave(self, stage: str, name: str) -> None:
        with self.lock:
            self.running[stage] -= 1
            self.events.append((stage, name, "end", time.monotonic()))

    def span(self, stage: str, name: str) -> tuple[float, float]:
        start = next(t for s, n, kind, t in self.events if (s, n, kind) == (stage, name, "start"))
        end = next(t for s, n, kind, t in self.events if (s, n, kind) == (stage, name, "end"))
        return start, end


def _slow_stages(monkeypatch: pytest.MonkeyPatch, delay: float, fail: str | None = None) -> _Trace:
    """Stub `download` / `process` in the runner with versions that sleep `delay` around the real stage."""
    trace = _Trace()

    def slow_download(cfg: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        trace.enter("download", name)
        try:
            if name == fail:
                raise OSError(f"{name}: network down")
            time.sleep(delay)
            return real_download(cfg, name, *args, **kwargs)
        finally:
            trace.leave("download", name)

    def slow_process(cfg: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        trace.enter("process", name)
        try:
            time.sleep(delay)
            return real_process(cfg, name, *args, **kwargs)
        finally:
            trace.leave("process", name)

    monkeypatch.setattr(build_mod, "download", slow_download)
    monkeypatch.setattr(build_mod, "process", slow_process)
    return trace


def _three_sources(cfg_factory: CfgFactory) -> DatasetConfig:
    sources = {f"s{i}": SourceConfig(kind="pretrain", loader="synthetic", seed=i, tokens_per_row_estimate=200) for i in range(3)}
    return cfg_factory(sources, tokens=600, max_seq_length=4096)


def test_build_overlaps_downloads_with_processing(cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _slow_stages(monkeypatch, delay=0.2)
    cfg = _three_sources(cfg_factory)
    assert build(cfg, layout, num_workers=1, max_parallel_downloads=1).complete
    # the stages ran back to back in less wall-clock time than their summed durations: some of them overlapped
    stage_time = sum(trace.span(stage, name)[1] - trace.span(stage, name)[0] for stage in ("download", "process") for name in ("s0", "s1", "s2"))
    wall = max(t for *_, t in trace.events) - min(t for *_, t in trace.events)
    assert wall < stage_time * 0.9, f"downloads and processing did not overlap ({wall:.2f} s wall for {stage_time:.2f} s of stages)"
    assert trace.peak["download"] == 1 and trace.peak["process"] == 1, "the bounds were respected"
    # some download ran while another source was being processed
    overlaps = [
        (d, p)
        for d in ("s0", "s1", "s2")
        for p in ("s0", "s1", "s2")
        if d != p and trace.span("download", d)[0] < trace.span("process", p)[1] and trace.span("download", d)[1] > trace.span("process", p)[0]
    ]
    assert overlaps


def test_build_respects_the_download_and_process_bounds(cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _slow_stages(monkeypatch, delay=0.1)
    cfg = _three_sources(cfg_factory)
    assert build(cfg, layout, num_workers=2, max_parallel_downloads=2).complete
    assert trace.peak["download"] <= 2 and trace.peak["process"] <= 2
    assert trace.peak["download"] == 2, "two sources downloaded at the same time"
    with pytest.raises(ValueError, match="must be >= 1"):
        build(cfg, layout, max_parallel_downloads=0)


def test_parallel_build_equals_sequential_build(cfg_factory: CfgFactory, layout: DatasetLayout, tmp_path: Path) -> None:
    """The files a source ends up with depend only on its own loader order, not on the interleaving."""
    cfg = _three_sources(cfg_factory)
    sequential = DatasetLayout(tmp_path / "sequential")
    build(cfg, sequential, num_workers=1, max_parallel_downloads=1)
    build(cfg, layout, num_workers=3, max_parallel_downloads=3)

    def snapshot(root: Path) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for path in sorted(root.rglob("*")):
            if path.name == "MANIFEST.json":
                manifest = Manifest.load(path.parent)
                assert manifest is not None
                out[str(path.relative_to(root))] = [(s.name, s.rows, s.tokens) for s in manifest.shards]
            elif path.suffix == ".parquet":
                out[str(path.relative_to(root))] = path.read_bytes()
        return out

    assert snapshot(layout.root) == snapshot(sequential.root)


def test_failing_item_aborts_the_build_and_stops_the_others(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trace = _slow_stages(monkeypatch, delay=0.1, fail="s1")
    cfg = _three_sources(cfg_factory)
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(OSError, match="s1: network down"):
        build(cfg, layout, num_workers=1, max_parallel_downloads=1)
    assert "source s1 failed" in caplog.text and "network down" in caplog.text  # with the traceback
    # whichever of s0 / s1 got the single download slot first, nothing starts after the failure: s2 never
    # downloads (cancelled before it started, or stopped at its first slot) and no stage starts after the failure
    started = {(s, n) for s, n, kind, _ in trace.events if kind == "start"}
    assert ("download", "s1") in started and ("download", "s2") not in started
    assert started <= {("download", "s0"), ("process", "s0"), ("download", "s1")}
    failed_at = trace.span("download", "s1")[1]
    assert all(t <= failed_at + 0.01 for s, n, kind, t in trace.events if kind == "start")
    assert not layout.source_dir("s2", "raw").exists()


def test_the_original_error_wins_over_items_that_merely_stopped(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With several items in flight the items that stop because of the failure may finish before the failing
    one; the build still raises the failure itself, never `BuildAborted`."""
    _slow_stages(monkeypatch, delay=0.05, fail="s0")
    original_error = build_mod.log.error

    def slow_error(*args: Any, **kwargs: Any) -> None:
        time.sleep(0.3)  # widen the window between `abort.set()` and the failing future completing
        original_error(*args, **kwargs)

    monkeypatch.setattr(build_mod.log, "error", slow_error)
    monkeypatch.setattr(build_mod.log, "exception", slow_error)
    cfg = _three_sources(cfg_factory)
    with pytest.raises(OSError, match="s0: network down"):
        build(cfg, layout, num_workers=3, max_parallel_downloads=3)


def test_build_removes_orphaned_filtered_dirs(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    orphan = layout.root / "sources" / "p" / "filtered"
    orphan.mkdir(parents=True)
    (orphan / "data-00000.parquet").write_bytes(b"old")
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")})
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert build(cfg, layout).complete
    assert not orphan.exists() and "removing orphaned" in caplog.text
