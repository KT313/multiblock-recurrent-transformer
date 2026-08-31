# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `prepare` / `status`: tiny end to end and idempotent, the download → build rounds, step and source
filters, failure and interrupt handling of the parallel helpers, the lock, dry runs, the raw-deletion confirmation,
parallel == sequential, `github_code` groups, repair of broken folders."""

from __future__ import annotations

import concurrent.futures
import importlib
import logging
import os
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from data_preparation.conftest import REPO, REV, FakeHub
from data_preparation.dataset_config import DatasetConfig, ProcessingConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted, check_stop
from data_preparation.lib.build import runner
from data_preparation.lib.build import prepare, status
from data_preparation.lib.build.lock import BuildLocked, build_lock
from data_preparation.lib.build.planner import plan_downloads, rows_needed, rows_sufficient
from data_preparation.lib.build.repair import ConfirmationRequired
from data_preparation.lib.stages.build import build_source as real_build
from data_preparation.lib.stages.download import download as real_download
from data_preparation.lib.storage.manifest import Manifest

CfgFactory = Callable[..., DatasetConfig]
ConfigFile = Callable[[DatasetConfig], Path]
Writer = Callable[[Path, list[dict[str, Any]], str], Path]

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY = REPO_ROOT / "config" / "datasets" / "tiny.yaml"


def all_mtimes(root: Path, *, include_lock: bool = False) -> dict[Path, int]:
    """`{file: mtime_ns}` of every file under `root` (the lock file is rewritten by every `prepare`)."""
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and (include_lock or p.name != ".build.lock")}


def _three_sources(cfg_factory: CfgFactory) -> DatasetConfig:
    sources = {f"s{i}": SourceConfig(kind="pretrain", loader="synthetic", seed=i) for i in range(3)}
    return cfg_factory(sources, tokens=600, name="three")


# --- end to end ------------------------------------------------------------------------------------------------------


def test_prepare_tiny_end_to_end_and_a_second_run_is_a_no_op(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "dataset"
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(TINY, root, assume_yes=False)
    assert report.complete and report.missing() == [] and "round 1: 2 source(s) short" in caplog.text
    assert "round 2" not in caplog.text and "dataset status:" in caplog.text
    assert status(TINY, root).complete
    layout = DatasetLayout(root)
    for name in ("synthetic_pretrain", "synthetic_instruct"):
        assert (layout.raw_dir(name) / "MANIFEST.json").is_file() and (layout.processed_dir(name) / "MANIFEST.json").is_file()

    before = all_mtimes(root)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        assert prepare(TINY, root, assume_yes=False).complete
    assert all_mtimes(root) == before
    assert "round 1: nothing to download" in caplog.text


def test_prepare_returns_the_report_of_every_source(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0),
        "h": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=4),  # used only for validation
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2),
    }
    cfg = cfg_factory(sources, tokens=500)
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and [s.name for s in report.sources] == ["p", "h", "i"]
    p, h, i = report.sources
    assert p.rows_needed == 600 and p.raw_rows == 600 and p.processed_rows >= 500 and p.epochs is not None
    assert h.rows_needed == 4 and h.raw_rows == 4 and h.epochs is None
    assert i.kind == "instruct" and i.satisfied
    processed_i = Manifest.load(layout.processed_dir("i"))
    assert processed_i is not None and processed_i.extra["columns"] == ["instruction", "input", "output", "tokens", "hash"]


# --- rounds ----------------------------------------------------------------------------------------------------------


def test_a_download_that_falls_short_gets_a_second_round(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A loader that returns fewer rows than asked (without being exhausted) leaves the source short after round 1;
    round 2 plans the difference and tops it up."""
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500)
    calls: list[int] = []

    def half_the_first_time(config: DatasetConfig, name: str, *args: Any, rows_needed: int, **kwargs: Any) -> Manifest:
        calls.append(rows_needed)
        asked = rows_needed // 2 if len(calls) == 1 else rows_needed
        return real_download(config, name, *args, rows_needed=asked, **kwargs)

    monkeypatch.setattr(runner, "download", half_the_first_time)
    needed = rows_needed(cfg, "p")  # 500 sequences × 1.2 ÷ 0.95 (the factory validates on the trained source too) = 632
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and calls == [needed, needed] == [632, 632]
    assert "round 1: 1 source(s) short, downloading 632 rows (p 632)" in caplog.text
    assert "round 2: 1 source(s) short, downloading 316 rows (p 316)" in caplog.text and "round 3" not in caplog.text
    raw = Manifest.load(layout.raw_dir("p"))
    assert raw is not None and raw.rows() == 632 and [s.rows for s in raw.shards] == [316, 316]  # appended, not rewritten


def test_a_source_still_short_after_max_rounds_is_reported(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=100)
    calls = 0

    def one_row_per_call(config: DatasetConfig, name: str, *args: Any, rows_needed: int, **kwargs: Any) -> Manifest:
        nonlocal calls
        calls += 1
        return real_download(config, name, *args, rows_needed=calls, **kwargs)

    monkeypatch.setattr(runner, "download", one_row_per_call)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert not report.complete and calls == runner.MAX_ROUNDS == 5
    assert f"round {runner.MAX_ROUNDS}:" in caplog.text and f"round {runner.MAX_ROUNDS + 1}" not in caplog.text
    (p,) = report.sources
    sufficient = rows_sufficient(cfg, "p")
    assert not p.satisfied and p.reason == f"processed rows 5 < {sufficient}" and f"p: processed rows 5 < {sufficient}" in caplog.text


def test_rounds_stop_when_nothing_more_can_be_fetched(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    """A source whose dedup drops more than the safety margin stays short although every row asked for is on disk:
    no second round (nothing to fetch), the report says so instead of looping."""
    src_dir = layout.root.parent / "dupes"
    unique = [{"text": f"tok_{i} tok_{i + 1} tok_3"} for i in range(10)]
    write_local(src_dir, unique + [unique[0]] * 690, "parquet")
    cfg = cfg_factory({"d": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=500)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(config_file(cfg), layout.root, assume_yes=False)
    (d,) = report.sources
    assert not report.complete and not d.exhausted and d.raw_rows == rows_needed(cfg, "d") == 632 and d.processed_rows == 10
    assert d.reason == f"processed rows 10 < {rows_sufficient(cfg, 'd')}" and "round 2" not in caplog.text
    assert plan_downloads(cfg, layout).total_rows_to_fetch() == 0


def test_exhausted_source_is_complete_with_a_warning(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": "tok_1 tok_2 tok_3"}] * 3 + [{"text": "tok_4 tok_5"}], "parquet")
    cfg = cfg_factory({"s": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=1000)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(config_file(cfg), layout.root, assume_yes=False)
    (s,) = report.sources
    assert report.complete and s.satisfied and s.exhausted and s.processed_rows == 2  # the duplicates went
    assert f"s: source exhausted (exhausted at 2 of {rows_sufficient(cfg, 's'):,} rows)" in caplog.text
    assert "round 2" not in caplog.text


# --- steps and sources -------------------------------------------------------------------------------------------------


def test_steps_download_then_build(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic"), "h": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=4)}, tokens=500)
    path = config_file(cfg)
    report = prepare(path, layout.root, assume_yes=False, steps=["tokenizer", "download"])
    assert not report.complete and report.tokenizer_complete and report.missing() == ["p", "h"]
    assert (layout.raw_dir("p") / "MANIFEST.json").is_file() and (layout.raw_dir("h") / "MANIFEST.json").is_file()
    assert not layout.processed_dir("p").exists() and not layout.processed_dir("h").exists()
    assert [s.reason for s in report.sources] == ["processed missing", "processed missing"]
    report = prepare(path, layout.root, assume_yes=False, steps=["build"])
    assert report.complete
    with pytest.raises(ValueError, match="unknown steps"):
        prepare(path, layout.root, assume_yes=False, steps=["nope"])


def test_sources_filter(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    sources = {"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0), "b": SourceConfig(kind="pretrain", loader="synthetic", seed=1)}
    path = config_file(cfg_factory(sources, tokens=500))
    report = prepare(path, layout.root, assume_yes=False, sources=["a"])
    assert not report.complete and report.missing() == ["b"] and (layout.processed_dir("a") / "MANIFEST.json").is_file()
    assert not layout.raw_dir("b").exists()
    assert prepare(path, layout.root, assume_yes=False, sources=["b"]).complete
    with pytest.raises(ValueError, match="unknown sources"):
        prepare(path, layout.root, assume_yes=False, sources=["c"])
    with pytest.raises(ValueError, match="must be >= 1"):
        prepare(path, layout.root, assume_yes=False, max_parallel_downloads=0)


# --- failures and interrupts -------------------------------------------------------------------------------------------


def test_failing_download_stops_the_other_downloads_within_a_shard_and_is_reraised(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ticks: dict[str, int] = {}

    def download_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name == "s1":
            time.sleep(0.05)
            raise OSError("s1: network down")
        for i in range(100):
            time.sleep(0.01)
            ticks[name] = i + 1
            check_stop(should_stop)
        return real_download(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download", download_stub)
    path = config_file(_three_sources(cfg_factory))
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(OSError, match="s1: network down"):
        prepare(path, layout.root, assume_yes=False, num_workers=1, max_parallel_downloads=3)
    assert 0 < ticks["s0"] < 100 and 0 < ticks["s2"] < 100, "the running downloads stopped within a shard"
    assert "source s1 failed" in caplog.text and "network down" in caplog.text  # with the traceback
    assert "source s0 stopped: source s1 failed" in caplog.text
    assert not layout.processed_dir("s0").exists()  # nothing was built after the failure


def test_jobs_not_started_yet_are_cancelled_after_a_failure(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []

    def download_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        started.append(name)
        if name == "s0":
            raise OSError("s0: network down")
        return real_download(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download", download_stub)
    path = config_file(_three_sources(cfg_factory))
    with pytest.raises(OSError, match="s0: network down"):
        prepare(path, layout.root, assume_yes=False, max_parallel_downloads=1)
    assert started == ["s0"] and not layout.raw_dir("s1").exists() and not layout.raw_dir("s2").exists()


def test_the_original_error_wins_over_jobs_that_merely_stopped(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Jobs that stop because of the failure may finish before the failing one; the failure itself is raised,
    never `BuildAborted`."""

    def download_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name == "s0":
            time.sleep(0.05)
            raise OSError("s0: network down")
        for _ in range(100):
            time.sleep(0.01)
            check_stop(should_stop)
        return real_download(config, name, *args, **kwargs)

    original_exception = runner.log.exception

    def slow_exception(*args: Any, **kwargs: Any) -> None:
        time.sleep(0.3)  # widen the window between the flag being raised and the failing future completing
        original_exception(*args, **kwargs)

    monkeypatch.setattr(runner, "download", download_stub)
    monkeypatch.setattr(runner.log, "exception", slow_exception)
    with pytest.raises(OSError, match="s0: network down"):
        prepare(config_file(_three_sources(cfg_factory)), layout.root, assume_yes=False, max_parallel_downloads=3)


def test_interrupt_in_the_wait_stops_the_download_within_a_shard_and_keeps_its_shards(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ctrl-C while `run_jobs` waits: the flag reaches the running download through `should_stop`, it stops at its
    next shard (published), `prepare` raises `BuildAborted` and the next run resumes."""
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500)
    shards_done: list[int] = []

    def slow_download(config: DatasetConfig, name: str, *args: Any, rows_needed: int, should_stop: Any = None, **kwargs: Any) -> Manifest:
        partial_manifest = real_download(config, name, *args, rows_needed=100, **kwargs)  # one shard on disk first
        for i in range(50):  # then a long download: one "shard" per tick, stop checked after each
            time.sleep(0.02)
            shards_done.append(i)
            check_stop(should_stop)
        return partial_manifest

    def interrupted_wait(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.2)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "download", slow_download)
    monkeypatch.setattr(runner, "as_completed", interrupted_wait)
    path = config_file(cfg)
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(BuildAborted, match="interrupted"):
        prepare(path, layout.root, assume_yes=False)
    assert 1 <= len(shards_done) < 50, "stopped within a shard of the interrupt, not at the end"
    assert "source p stopped: interrupted" in caplog.text
    raw = Manifest.load(layout.raw_dir("p"))
    assert raw is not None and raw.rows() == 100 and not layout.processed_dir("p").exists()
    monkeypatch.setattr(runner, "as_completed", concurrent.futures.as_completed)
    monkeypatch.setattr(runner, "download", real_download)
    assert prepare(path, layout.root, assume_yes=False).complete  # resumes behind the published shard
    raw = Manifest.load(layout.raw_dir("p"))
    assert raw is not None and [s.rows for s in raw.shards] == [100, rows_needed(cfg, "p") - 100]


def test_an_outer_should_stop_is_honoured(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    path = config_file(_three_sources(cfg_factory))
    with pytest.raises(BuildAborted):
        prepare(path, layout.root, assume_yes=False, should_stop=lambda: True)


def test_run_jobs_with_no_jobs_is_a_no_op() -> None:
    runner.run_jobs([], max_workers=1, description="nothing")
    flag = runner.StopFlag()
    assert not flag.should_stop()
    flag.stop("first")
    flag.stop("second")
    assert flag.should_stop() and flag.reason == "first"


# --- lock, dry run, confirmation -----------------------------------------------------------------------------------------


def test_prepare_fails_fast_while_another_build_holds_the_lock(layout: DatasetLayout, tmp_path: Path) -> None:
    with build_lock(layout.root), pytest.raises(BuildLocked, match=f"pid {os.getpid()}") as excinfo:
        prepare(TINY, layout.root, assume_yes=False)
    assert "remove" not in str(excinfo.value)  # the OS releases the lock when the holder dies
    assert not (layout.root / "sources").exists()


def test_dry_run_writes_nothing(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, caplog: pytest.LogCaptureFixture) -> None:
    path = config_file(cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")}, tokens=500))
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=False, dry_run=True)
    assert not report.complete and not layout.root.exists()
    assert "dry run, downloads planned:" in caplog.text and "raw missing" in caplog.text and "INCOMPLETE" in caplog.text

    prepare(path, layout.root, assume_yes=False)
    before = all_mtimes(layout.root, include_lock=True)
    assert prepare(path, layout.root, assume_yes=False, dry_run=True).complete
    assert all_mtimes(layout.root, include_lock=True) == before  # not even the lock file was touched


def test_unconfirmed_raw_deletion_raises_and_deletes_nothing(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500)
    prepare(config_file(cfg), layout.root, assume_yes=False)
    raw_dir = layout.raw_dir("p")
    before = all_mtimes(raw_dir)

    cfg.token_count = "estimate"  # part of the raw hash: the raw folder is stale
    path = config_file(cfg)
    asked: list[str] = []

    def decline(message: str) -> bool:
        asked.append(message)
        return False

    with pytest.raises(ConfirmationRequired) as excinfo:
        prepare(path, layout.root, assume_yes=False, confirm=decline)
    assert len(asked) == 1 and "p: stale" in asked[0] and excinfo.value.report.raw_deletions_planned()
    assert all_mtimes(raw_dir) == before, "nothing was deleted"
    assert not status(path, layout.root).complete  # status shows the pending repair

    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=True)
    assert report.complete and "without asking" in caplog.text
    raw = Manifest.load(raw_dir)
    assert raw is not None and raw.token_count == "estimate" and raw.is_current(cfg.raw_hash("p"))


def test_status_is_read_only_and_reports_would_repair(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, caplog: pytest.LogCaptureFixture) -> None:
    path = config_file(cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500))
    prepare(path, layout.root, assume_yes=False)
    victim = next(layout.processed_dir("p").glob("data-*.parquet"))
    victim.unlink()
    before = all_mtimes(layout.root, include_lock=True)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = status(path, layout.root)
    assert not report.complete and report.needs_repair == ["p"] and report.missing() == ["p"]
    assert "would repair:" in caplog.text and "would delete processed" in caplog.text
    assert all_mtimes(layout.root, include_lock=True) == before
    assert prepare(path, layout.root, assume_yes=False).complete and victim.is_file()


# --- parallel == sequential ----------------------------------------------------------------------------------------------


def test_parallel_prepare_equals_sequential_prepare(cfg_factory: CfgFactory, config_file: ConfigFile, tmp_path: Path) -> None:
    """The rows a source ends up with depend only on its own loader order, not on the interleaving."""
    path = config_file(_three_sources(cfg_factory))
    sequential, parallel = tmp_path / "sequential", tmp_path / "parallel"
    assert prepare(path, sequential, assume_yes=False, num_workers=1, max_parallel_downloads=1).complete
    assert prepare(path, parallel, assume_yes=False, num_workers=3, max_parallel_downloads=3).complete

    def hashes(root: Path, name: str) -> list[int]:
        directory = DatasetLayout(root).processed_dir(name)
        manifest = Manifest.load(directory)
        assert manifest is not None
        values: list[Any] = [h for shard in manifest.shards for h in pq.read_table(directory / shard.name, columns=["hash"]).column("hash").to_pylist()]
        return [int(h) for h in values]

    for name in ("s0", "s1", "s2"):
        assert hashes(sequential, name) == hashes(parallel, name) and len(hashes(sequential, name)) > 0
        assert (DatasetLayout(sequential).raw_dir(name) / "data-00000.parquet").read_bytes() == (DatasetLayout(parallel).raw_dir(name) / "data-00000.parquet").read_bytes()


def test_prepare_works_without_the_dashboard(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PREP_PROGRESS", "0")
    path = config_file(_three_sources(cfg_factory))
    assert prepare(path, layout.root, assume_yes=False, num_workers=2, max_parallel_downloads=2).complete


# --- github_code groups ------------------------------------------------------------------------------------------------


def _github_cfg(cfg_factory: CfgFactory, languages: list[str], **kwargs: Any) -> DatasetConfig:
    sources = {
        f"code_{lang.lower()}": SourceConfig(kind="pretrain", loader="github_code", hf_id=REPO, revision=REV, language=lang)
        for lang in languages
    }
    return cfg_factory(sources, tokens=15, name="code", **kwargs)


def _code_rows(prefix: str, n: int) -> list[dict[str, Any]]:
    languages = ("Python", "Java", "Go")
    return [{"id": f"{prefix}{i}", "text": f"{prefix} code {i}", "language": languages[i % 3]} for i in range(n)]


def test_github_code_languages_of_one_repo_download_in_one_pass(
    hub: FakeHub, cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.add("data/a.parquet", _code_rows("a", 30))
    hub.add("data/b.parquet", _code_rows("b", 30))
    cfg = _github_cfg(cfg_factory, ["Python", "Java", "Go"])
    single: list[str] = []

    def spy_download(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        single.append(name)
        return real_download(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download", spy_download)
    jobs = runner.download_jobs(plan_downloads(cfg, layout), cfg, layout, None)
    assert [(job.what, job.name) for job in jobs] == [("github_code group", "code_python, code_java, code_go")]
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and single == []  # the group pass replaced the per-source downloads
    assert len(hub.streams) == len(set(hub.streams))  # every repo file opened at most once for all three languages
    for name in ("code_python", "code_java", "code_go"):
        raw = Manifest.load(layout.raw_dir(name))
        assert raw is not None and raw.rows() >= 6  # 5 sequences each × 1.2

    # `--sources` with one language uses the ordinary per-source path
    hub.streams.clear()
    bigger = _github_cfg(cfg_factory, ["Python", "Java", "Go"])
    bigger.stages[0].tokens = 30
    jobs = runner.download_jobs(plan_downloads(bigger, layout, sources=["code_python"]), bigger, layout, None)
    assert [(job.what, job.name) for job in jobs] == [("source", "code_python")]
    prepare(config_file(bigger), layout.root, assume_yes=False, sources=["code_python"])
    assert single == ["code_python"]


# --- repairs on the way -----------------------------------------------------------------------------------------------


def test_processing_change_rebuilds_processed_but_leaves_raw_untouched(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    """The raw shards are the bandwidth-expensive part: a processing-only edit must not re-download them."""
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500)
    prepare(config_file(cfg), layout.root, assume_yes=False)
    raw_dir, processed_dir = layout.raw_dir("p"), layout.processed_dir("p")
    raw_before = all_mtimes(raw_dir)
    processed_before = Manifest.load(processed_dir)
    assert processed_before is not None

    cfg.processing = ProcessingConfig(min_chars=2)
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and all_mtimes(raw_dir) == raw_before
    processed = Manifest.load(processed_dir)
    assert processed is not None and processed.is_current(cfg.processed_hash("p")) and not processed.is_current(processed_before.source_hash)


def test_missing_processed_shards_are_repaired(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, caplog: pytest.LogCaptureFixture) -> None:
    sources = {
        "p": SourceConfig(kind="pretrain", loader="synthetic", seed=0),
        "h": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=4),
        "i": SourceConfig(kind="instruct", loader="synthetic", seed=2),
    }
    path = config_file(cfg_factory(sources, tokens=500))
    prepare(path, layout.root, assume_yes=False)
    victims = [next(layout.processed_dir(name).glob("data-*.parquet")) for name in ("p", "h", "i")]
    for victim in victims:
        victim.unlink()
    assert not status(path, layout.root).complete
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=False)
    assert report.complete and all(victim.is_file() for victim in victims)
    assert "missing shard" in caplog.text and "deleting" in caplog.text


def test_broken_raw_shard_is_truncated_not_redownloaded(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    download_module = importlib.import_module("data_preparation.lib.stages.download")  # the package attribute `download` is the function
    monkeypatch.setattr(runner, "download", partial(real_download, shard_size=10))
    path = config_file(cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=50))
    prepare(path, layout.root, assume_yes=False)
    raw = layout.raw_dir("p")
    manifest = Manifest.load(raw)
    assert manifest is not None and len(manifest.shards) >= 3, "the test needs several raw shards"
    last = manifest.shards[-1]
    (raw / last.name).write_bytes(b"corrupt")
    offsets: list[int] = []
    original = download_module._fetch_rows

    def spy(source: Any, name: str, offset: int, *args: Any, **kwargs: Any) -> Any:
        offsets.append(offset)
        return original(source, name, offset, *args, **kwargs)

    monkeypatch.setattr(download_module, "_fetch_rows", spy)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=False)
    assert report.complete and "truncating" in caplog.text
    assert offsets == [manifest.shards[-2].offset], "resumed behind the last good shard instead of from 0"
    repaired = Manifest.load(raw)
    assert repaired is not None and repaired.rows() == manifest.rows() and [s.rows for s in repaired.shards] == [s.rows for s in manifest.shards]


def test_prepare_logs_the_stop_reason_of_a_slow_build(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The build helper has the same stop semantics as the download helper."""
    ticks: dict[str, int] = {}
    lock = threading.Lock()

    def build_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name == "s1":
            time.sleep(0.05)
            raise RuntimeError("s1: build exploded")
        for i in range(100):
            time.sleep(0.01)
            with lock:
                ticks[name] = i + 1
            check_stop(should_stop)
        return real_build(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "build_source", build_stub)
    path = config_file(_three_sources(cfg_factory))
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(RuntimeError, match="s1: build exploded"):
        prepare(path, layout.root, assume_yes=False, num_workers=3)
    assert 0 < ticks["s0"] < 100 and 0 < ticks["s2"] < 100
    assert "source s0 stopped: source s1 failed" in caplog.text
