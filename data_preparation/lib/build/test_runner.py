# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `build` / `status`: refinement loop, repair of broken folders, idempotence, step/source filtering."""

from __future__ import annotations

import importlib
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
from data_preparation.lib.build.planner import budget_tokens_of, plan
from data_preparation.dataset_config import DatasetConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.storage.manifest import Manifest, verify_shards
from data_preparation.lib.stages.build import build_source as real_build
from data_preparation.lib.stages.download import download as real_download

CfgFactory = Callable[..., DatasetConfig]
Writer = Callable[[Path, list[dict[str, Any]], str], Path]


def all_mtimes(root: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p.name != ".build.lock"}


def test_build_refines_a_bad_estimate(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    """Estimates are ~10x too high (but below the max_seq_length cap, which would otherwise clamp them): the first
    download is too small; the tokens/row measured from it sizes the second one correctly."""
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0, describe_tokens_per_row=3_000),
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2, describe_tokens_per_row=2_000),
    }
    cfg = cfg_factory(sources, tokens=3000, max_seq_length=4096)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        result = build(cfg, layout, max_rounds=3)
    assert result.complete
    for name in ("p", "i"):
        processed = Manifest.load(layout.processed_dir(name))
        assert processed is not None and (processed.tokens() or 0) >= 3000
        assert f"{name}: round 1:" in caplog.text
    raw_i = Manifest.load(layout.raw_dir("i"))
    assert raw_i is not None and raw_i.rows() > 2  # the estimate alone would have stopped at ceil(3000/2000*1.2) = 2


def test_build_raises_after_max_rounds(cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", describe_tokens_per_row=1000)}, tokens=1000)
    calls: list[int] = []

    def never_enough(*args: Any, **kwargs: Any) -> Manifest:
        manifest = real_build(*args, **kwargs)
        calls.append(manifest.rows())
        stub = Manifest(source="p", source_hash=manifest.source_hash, stage="processed")
        stub.add_shard("data-00000.parquet", rows=1, tokens=1)  # never enough tokens
        return stub

    monkeypatch.setattr(build_mod, "build_source", never_enough)
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
    processed = Manifest.load(layout.processed_dir("s"))
    assert processed is not None and processed.rows() == 2
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert status(cfg, layout).complete
    assert "s: exhausted at 5 of 1000 tokens" in caplog.text


def test_build_twice_writes_nothing(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0),
        "h": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=4),  # used only for validation
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2),
    }
    cfg = cfg_factory(sources, tokens=500)
    assert build(cfg, layout).complete
    before = all_mtimes(layout.root)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        assert build(cfg, layout).complete
    assert all_mtimes(layout.root) == before
    assert "p: complete, skipping" in caplog.text and "h: complete, skipping" in caplog.text and "i: complete, skipping" in caplog.text


def test_build_repairs_missing_shards(cfg_factory: CfgFactory, layout: DatasetLayout, caplog: pytest.LogCaptureFixture) -> None:
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0),
        "h": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=4),
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2),
    }
    cfg = cfg_factory(sources, tokens=500)
    build(cfg, layout)
    victims = [next(layout.processed_dir(name).glob("data-*.parquet")) for name in ("p", "h", "i")]
    for victim in victims:
        victim.unlink()
    assert not status(cfg, layout).complete
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        result = build(cfg, layout)
    assert result.complete and all(v.is_file() for v in victims)
    assert "missing shard" in caplog.text and "removing" in caplog.text
    for directory in (layout.processed_dir(name) for name in ("p", "h", "i")):
        manifest = Manifest.load(directory)
        assert manifest is not None and verify_shards(directory, manifest) == []


def test_processing_change_rebuilds_processed_but_leaves_raw_untouched(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    """The raw shards are the bandwidth-expensive part: a processing-only edit must not re-download them."""
    from data_preparation.dataset_config import ProcessingConfig

    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500)
    build(cfg, layout)
    raw_dir, processed_dir = layout.raw_dir("p"), layout.processed_dir("p")
    raw_before = {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in raw_dir.iterdir()}
    processed_before = Manifest.load(processed_dir)
    assert processed_before is not None

    cfg.processing = ProcessingConfig(min_chars=2)
    result = build(cfg, layout)
    assert result.complete
    raw = Manifest.load(raw_dir)
    assert raw is not None and raw.is_current(cfg.raw_hash("p"))
    assert {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in raw_dir.iterdir()} == raw_before
    processed = Manifest.load(processed_dir)
    assert processed is not None and processed.is_current(cfg.processed_hash("p"))
    assert not processed.is_current(processed_before.source_hash)


def test_token_mode_change_makes_raw_stale_and_re_downloads(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`token_count` (and the tokenizer) are part of the raw hash: the stored counts depend on them, so a change
    re-downloads the source from offset 0 instead of recounting in place (task 7 asks for confirmation first)."""
    download_module = importlib.import_module("data_preparation.lib.stages.download")  # the package attribute `download` is the function

    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500, max_seq_length=4096)
    build(cfg, layout)
    raw_dir = layout.raw_dir("p")
    before = Manifest.load(raw_dir)
    assert before is not None and before.token_count == "tokenizer"

    offsets: list[int] = []
    original = download_module._fetch_rows

    def spy(source: Any, name: str, offset: int, *args: Any, **kwargs: Any) -> Any:
        offsets.append(offset)
        return original(source, name, offset, *args, **kwargs)

    monkeypatch.setattr(download_module, "_fetch_rows", spy)
    cfg.token_count = "estimate"
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        result = build(cfg, layout)
    assert result.complete and "raw: manifest stale; removing" in caplog.text
    raw = Manifest.load(raw_dir)
    assert raw is not None and raw.token_count == "estimate" and raw.tokenizer is None and raw.is_current(cfg.raw_hash("p"))
    assert offsets[0] == 0 and raw.tokens() != before.tokens()  # chars/4 differs from the tokenizer count


def test_build_steps_filter(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    # estimate 500 tokens/row vs ~220 real (uncapped: max_seq_length above the document length), so the first
    # estimate-sized download falls short of the budget
    cfg = cfg_factory(
        {"p": SourceConfig(kind="pretrain", loader="synthetic"), "h": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=4)},
        tokens=500,
        max_seq_length=4096,
    )
    result = build(cfg, layout, steps={"tokenizer", "download"})
    assert not result.complete and result.tokenizer_complete
    assert (layout.raw_dir("p") / "MANIFEST.json").is_file() and (layout.raw_dir("h") / "MANIFEST.json").is_file()
    assert not layout.processed_dir("p").exists() and not layout.processed_dir("h").exists()
    result = build(cfg, layout, steps={"process"})
    p, h = result.sources
    assert (layout.processed_dir("p") / "MANIFEST.json").is_file() and h.complete
    assert not p.complete and p.reason.startswith("tokens ")  # the estimate-sized download cannot be topped up without `download`
    assert build(cfg, layout).complete
    with pytest.raises(ValueError, match="unknown steps"):
        build(cfg, layout, steps={"nope"})


def test_build_sources_filter(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    sources = {"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0), "b": SourceConfig(kind="pretrain", loader="synthetic", seed=1)}
    cfg = cfg_factory(sources, tokens=500)
    result = build(cfg, layout, sources=["a"])
    assert not result.complete and (layout.processed_dir("a") / "MANIFEST.json").is_file()
    assert not layout.raw_dir("b").exists()
    assert build(cfg, layout, sources=["b"]).complete
    with pytest.raises(ValueError, match="unknown sources"):
        build(cfg, layout, sources=["c"])


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
            kind="pretrain", loader="github_code", hf_id=REPO, revision=REV, language=lang, describe_tokens_per_row=2,
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
        processed = Manifest.load(layout.processed_dir(name))
        assert processed is not None and (processed.tokens() or 0) >= budget_tokens_of(cfg, name)  # a third of the stage

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
        manifest = real_build(*args, **kwargs)
        stub = Manifest(source=manifest.source, source_hash=manifest.source_hash, stage="processed")
        stub.add_shard("data-00000.parquet", rows=1, tokens=1)
        return stub

    monkeypatch.setattr(build_mod, "build_source", never_enough)
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
    """Stub `download` / `build_source` in the runner with versions that sleep `delay` around the real step."""
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

    def slow_build(cfg: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        trace.enter("process", name)
        try:
            time.sleep(delay)
            return real_build(cfg, name, *args, **kwargs)
        finally:
            trace.leave("process", name)

    monkeypatch.setattr(build_mod, "download", slow_download)
    monkeypatch.setattr(build_mod, "build_source", slow_build)
    return trace


def _three_sources(cfg_factory: CfgFactory) -> DatasetConfig:
    sources = {f"s{i}": SourceConfig(kind="pretrain", loader="synthetic", seed=i, describe_tokens_per_row=200) for i in range(3)}
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
    assert not layout.raw_dir("s2").exists()


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


def test_interrupt_stops_running_items_within_a_shard_and_keeps_their_shards(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ctrl-C while `_run_all` waits: the request reaches the running download through `should_stop`, it stops at
    its next shard (published), the build re-raises the interrupt and logs the reason."""
    import concurrent.futures

    from data_preparation.lib.abort import check_stop

    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500)
    shards_done: list[int] = []

    def slow_download(cfg_: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        for i in range(50):  # a long download: one "shard" per tick, stop checked after each
            time.sleep(0.02)
            shards_done.append(i)
            check_stop(should_stop)
        return real_download(cfg_, name, *args, **kwargs)

    def interrupted_wait(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.1)
        raise KeyboardInterrupt

    monkeypatch.setattr(build_mod, "download", slow_download)
    monkeypatch.setattr(build_mod, "as_completed", interrupted_wait)
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(KeyboardInterrupt):
        build(cfg, layout)
    assert 1 <= len(shards_done) < 50, "stopped within a shard of the interrupt, not at the end"
    assert "source p stopped: build interrupted" in caplog.text
    assert not layout.processed_dir("p").exists()
    monkeypatch.setattr(build_mod, "as_completed", concurrent.futures.as_completed)
    assert build(cfg, layout).complete  # resumes and finishes; nothing was lost


def test_failure_stops_a_running_download_within_a_shard(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from data_preparation.lib.abort import check_stop

    cfg = _three_sources(cfg_factory)
    ticks: dict[str, int] = {}

    def download_stub(cfg_: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name == "s1":
            time.sleep(0.05)
            raise OSError("s1: network down")
        for i in range(100):
            time.sleep(0.01)
            ticks[name] = i + 1
            check_stop(should_stop)
        return real_download(cfg_, name, *args, **kwargs)

    monkeypatch.setattr(build_mod, "download", download_stub)
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(OSError, match="s1: network down"):
        build(cfg, layout, num_workers=1, max_parallel_downloads=3)
    assert 0 < ticks["s0"] < 100 and 0 < ticks["s2"] < 100, "the running downloads stopped within a shard"
    assert "source s0 stopped: source s1 failed" in caplog.text


def test_broken_raw_shard_is_truncated_not_redownloaded(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    download_module = importlib.import_module("data_preparation.lib.stages.download")  # the package attribute `download` is the function

    from functools import partial

    monkeypatch.setattr(build_mod, "download", partial(real_download, shard_size=10))
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0, describe_tokens_per_row=100)}, tokens=2000)
    build(cfg, layout)
    raw = layout.raw_dir("p")
    manifest = Manifest.load(raw)
    assert manifest is not None and len(manifest.shards) >= 2, "the test needs several raw shards"
    last = manifest.shards[-1]
    (raw / last.name).write_bytes(b"corrupt")
    offsets: list[int] = []
    original = download_module._fetch_rows

    def spy(source: Any, name: str, offset: int, *args: Any, **kwargs: Any) -> Any:
        offsets.append(offset)
        return original(source, name, offset, *args, **kwargs)

    monkeypatch.setattr(download_module, "_fetch_rows", spy)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        result = build(cfg, layout)
    assert result.complete and "kept the" in caplog.text and "removing" not in caplog.text
    kept_offset = manifest.shards[-2].offset
    assert offsets == [kept_offset], "resumed behind the last good shard instead of from 0"
    repaired = Manifest.load(raw)
    assert repaired is not None and repaired.rows() == manifest.rows() and [s.rows for s in repaired.shards] == [s.rows for s in manifest.shards]
