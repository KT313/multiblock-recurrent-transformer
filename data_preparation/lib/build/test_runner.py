# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `prepare` / `status`: tiny end to end and idempotent, the download + build rounds, step and source
filters, builds overlapping the downloads (a source built as soon as its own download finished, never before),
failure and interrupt handling across both pools, the lock, dry runs, the raw-deletion confirmation, parallel ==
sequential, `github_code` groups, repair of broken folders.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from data_preparation import prepare as prepare_cli
from data_preparation.conftest import REPO, REV, FakeHub
from data_preparation.dataset_config import DatasetConfig, ProcessingConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted, check_stop
from data_preparation.lib.build import runner
from data_preparation.lib.build.runner import prepare, status
from data_preparation.lib.build.lock import RunLocked, build_lock
from data_preparation.lib.build.planner import DatasetReport, DownloadPlan, SourceLedger, plan_downloads
from data_preparation.lib.build.repair import ConfirmationRequired
from data_preparation.lib.stages.build import build_source as real_build
from data_preparation.lib.stages.download import download as real_download
from data_preparation.lib.stages.download import download_github_code_group as real_group
from data_preparation.lib.storage.manifest import Manifest

CfgFactory = Callable[..., DatasetConfig]
ConfigFile = Callable[[DatasetConfig], Path]
Writer = Callable[[Path, list[dict[str, Any]], str], Path]

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY = REPO_ROOT / "config" / "datasets" / "tiny.yaml"


def all_mtimes(root: Path, *, include_lock: bool = False) -> dict[Path, int]:
    """
    `{file: mtime_ns}` of every file under `root` (the lock file is rewritten by every `prepare`).
    """

    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and (include_lock or p.name != ".build.lock")}


def _state(report: DatasetReport, name: str) -> SourceLedger:
    return next(source for source in report.sources if source.name == name)


def _three_sources(cfg_factory: CfgFactory) -> DatasetConfig:
    sources = {f"s{i}": SourceConfig(kind="pretrain", loader="synthetic", seed=i) for i in range(3)}
    return cfg_factory(sources, tokens=600)


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
    cfg = cfg_factory(sources, tokens=500, training_target_sequence_length=1)
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and [s.name for s in report.sources] == ["p", "h", "i"]
    p, h, i = report.sources
    assert p.rows_needed == 600 and p.raw_rows == 600 and p.processed_rows >= 500 and p.epochs() is not None
    assert h.rows_needed == 5 and h.rows_sufficient == 4 and h.raw_rows == 5 and h.epochs() is None  # 4 delivered rows, × 1.2 downloaded
    assert i.kind == "instruct" and i.satisfaction()[0]
    processed_i = Manifest.load(layout.processed_dir("i"))
    assert processed_i is not None and processed_i.columns == ["instruction", "input", "output", "tokens", "hash"]


# --- rounds ----------------------------------------------------------------------------------------------------------


def test_a_loader_that_falls_short_is_exhausted_until_reopened(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A loader that yields fewer rows than asked is latched exhausted, whatever the reason: the source is
    satisfied with the rows it has (warning, no second round), and a later run does not read on by itself even
    when the source grew. `prepare(reopen=[name])` clears the latch; the download then resumes at its offset.
    """

    src_dir = layout.root.parent / "growing"
    write_local(src_dir, [{"text": f"tok_{i} tok_2 tok_3"} for i in range(4)], "parquet")
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=10, training_target_sequence_length=1)
    needed, sufficient = cfg.rows_needed("p"), cfg.rows_sufficient("p")  # 10 rows × 1.2 ÷ 0.95 = 13, 11
    path = config_file(cfg)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=False)
    (p,) = report.sources
    assert report.complete and p.exhausted and p.satisfaction() == (True, f"exhausted at 4 of {sufficient} rows") and (needed, sufficient) == (13, 11)
    assert "round 2" not in caplog.text
    assert f"p: source exhausted (exhausted at 4 of {sufficient} rows); the training sampler cycles the rows on disk; rerun with --reopen p if the source has more rows" in caplog.text
    assert caplog.text.index("dataset status:") < caplog.text.index("p: source exhausted")  # the warning follows the table

    write_local(src_dir, [{"text": f"tok_{i} tok_5 tok_6"} for i in range(20)], "parquet")  # the source grew
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        assert _state(prepare(path, layout.root, assume_yes=False), "p").exhausted  # the latch holds
    assert "round 1: nothing to download" in caplog.text
    with pytest.raises(ValueError, match="unknown sources"):
        prepare(path, layout.root, assume_yes=False, reopen=["nope"])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=False, reopen=["p"])
    (p,) = report.sources
    assert "p: reopened; the next download reads on from offset 4" in caplog.text
    assert f"round 1: 1 source(s) short, downloading {needed - 4} rows (p {needed - 4})" in caplog.text
    assert report.complete and not p.exhausted and p.satisfaction() == (True, "ok") and p.raw_rows == p.processed_rows == needed
    raw = Manifest.load(layout.raw_dir("p"))
    assert raw is not None and raw.rows_fetched == needed and [s.rows for s in raw.shards] == [4, needed - 4]  # appended, not rewritten


def test_a_measured_rate_below_the_estimate_gets_a_second_round(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Round 1 is sized at the config's tokens-per-row estimate (500 by default, clamped at the dataset length); the raw
    shards then measure the real mean. Synthetic rows cut at 256 tokens average well under 256 ÷ 1.2, so the
    margin does not cover the difference and round 2 tops the source up at the measured rate.
    """

    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=25_600, tokens_per_row=500, dataset_max_sequence_length=256)
    first = cfg.rows_needed("p")  # 25600 ÷ 256 = 100 rows × 1.2 ÷ 0.95 = 127
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(config_file(cfg), layout.root, assume_yes=False)
    (p,) = report.sources
    assert first == 127 and f"round 1: 1 source(s) short, downloading {first} rows (p {first})" in caplog.text
    assert "round 2: 1 source(s) short" in caplog.text and "round 3" not in caplog.text
    assert report.complete and p.satisfaction() == (True, "ok") and p.rows_to_fetch == (0, "budget served")
    assert p.tokens_per_row < 256 / 1.2 and p.raw_rows > first and p.rows_needed == cfg.rows_needed("p", p.tokens_per_row) > first
    assert p.processed_rows >= p.rows_sufficient and p.epochs() == pytest.approx(p.rows_budget / p.training_rows)


def test_a_source_still_short_after_max_rounds_is_reported(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=100, training_target_sequence_length=1)
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
    sufficient = cfg.rows_sufficient("p")
    assert not p.satisfaction()[0] and p.satisfaction()[1] == f"processed rows 5 < {sufficient}" and f"p: processed rows 5 < {sufficient}" in caplog.text


def test_a_dedup_shortfall_beyond_the_margin_is_topped_up_in_later_rounds(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The 20 % safety margin does not always cover what the build drops: here four of every five rows are exact
    duplicates. Round 1 downloads `rows_needed` raw rows and lands far short of `rows_sufficient` processed ones;
    the next rounds see that raw is long enough but *processed* is not and top the source up from the yield it
    showed. No round asks for more than the full requirement (`_top_up_rows` caps it, so one bad yield
    measurement cannot ask the loader for billions of rows). Before the ledger the plan only looked at raw rows: it
    planned nothing, the round loop gave up, and `prepare` failed with no way to make progress (open finding H5).
    """

    src_dir = layout.root.parent / "dupes"
    rows = [{"text": f"tok_{i} tok_2 tok_3"} if i % 5 == 0 else {"text": "tok_1 tok_2 tok_3"} for i in range(3500)]
    write_local(src_dir, rows, "parquet")
    cfg = cfg_factory({"d": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=500, training_target_sequence_length=1)
    needed, sufficient = cfg.rows_needed("d"), cfg.rows_sufficient("d")
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        report = prepare(config_file(cfg), layout.root, assume_yes=False)
    (d,) = report.sources
    assert report.complete and d.satisfaction()[0] and not d.exhausted and d.satisfaction()[1] == "ok"
    assert (needed, sufficient) == (632, 527) and d.processed_rows >= sufficient
    assert d.raw_rows > needed, "the top-up fetched beyond the budget, sized from the observed yield"
    assert "round 2: 1 source(s) short" in caplog.text  # round 1 did not serve the budget
    planned = [int(line.split("downloading ")[1].split(" rows")[0].replace(",", "")) for line in caplog.text.splitlines() if "source(s) short, downloading" in line]
    assert len(planned) >= 2 and max(planned) <= needed, f"every round is capped at the full requirement, got {planned}"
    assert f"capping this round at the full requirement of {needed:,} rows" in caplog.text
    assert plan_downloads(cfg, layout).total_rows_to_fetch() == 0


def test_a_source_whose_rows_never_survive_the_build_is_a_failed_build(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A `fields` mapping naming columns the rows do not have rejects every row as malformed: the source runs dry
    with an empty processed folder. That used to count as satisfied: `prepare` and `status` said "dataset complete",
    exit 0, and the training run failed much later (open finding H2). A failed source is a failed build, so it
    is now unsatisfied with a reason that names the likely mistake.
    """

    src_dir = layout.root.parent / "wrong_fields"
    write_local(src_dir, [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)], "jsonl")
    source = SourceConfig(kind="instruct", loader="local", path=str(src_dir), fields={"instruction": "prompt", "output": "completion"})
    cfg = cfg_factory({"i": source}, tokens=100)
    path = config_file(cfg)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=False)
    (i,) = report.sources
    assert not report.complete and not i.satisfaction()[0] and i.exhausted and (i.raw_rows, i.processed_rows) == (0, 0)
    assert "NOT ONE" in i.satisfaction()[1] and "20 malformed" in i.satisfaction()[1]
    assert "check the source's fields / converter / filter / language" in i.satisfaction()[1]
    assert report.missing() == ["i"] and f"i: {i.satisfaction()[1]}" in caplog.text
    assert status(path, layout.root).describe().endswith("dataset INCOMPLETE"), "`status` says the same"
    with pytest.raises(SystemExit) as exit_code:
        prepare_cli.main(["prepare", "--dataset_config", str(path), "--dataset_dir", str(layout.root)])
    assert exit_code.value.code == 1


def test_exhausted_source_is_complete_with_a_warning(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, write_local: Writer, caplog: pytest.LogCaptureFixture
) -> None:
    src_dir = layout.root.parent / "small"
    write_local(src_dir, [{"text": "tok_1 tok_2 tok_3"}] * 3 + [{"text": "tok_4 tok_5"}], "parquet")
    cfg = cfg_factory({"s": SourceConfig(kind="pretrain", loader="local", path=str(src_dir))}, tokens=1000, training_target_sequence_length=1)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(config_file(cfg), layout.root, assume_yes=False)
    (s,) = report.sources
    assert report.complete and s.satisfaction()[0] and s.exhausted and s.processed_rows == 2  # the duplicates went
    assert f"s: source exhausted (exhausted at 2 of {cfg.rows_sufficient('s'):,} rows)" in caplog.text
    assert "round 2" not in caplog.text


# --- steps and sources -------------------------------------------------------------------------------------------------


def test_steps_download_then_build(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic"), "h": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=4)}, tokens=500)
    path = config_file(cfg)
    report = prepare(path, layout.root, assume_yes=False, steps=["tokenizer", "download"])
    assert not report.complete and report.tokenizer_complete and report.missing() == ["p", "h"]
    assert (layout.raw_dir("p") / "MANIFEST.json").is_file() and (layout.raw_dir("h") / "MANIFEST.json").is_file()
    assert not layout.processed_dir("p").exists() and not layout.processed_dir("h").exists()
    assert [s.satisfaction()[1] for s in report.sources] == ["processed missing", "processed missing"]
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
    with pytest.raises(ValueError, match="must be >= 1"):
        prepare(path, layout.root, assume_yes=False, pass_workers=0)


def test_a_repeated_source_is_selected_once(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    """
    `--sources a a` would otherwise inspect `a` twice: the repair step would list it twice and delete the same
    folder twice (the second `rmtree` on a directory that is gone). The selection is in config order.
    """

    sources = {"a": SourceConfig(kind="pretrain", loader="synthetic", seed=0), "b": SourceConfig(kind="pretrain", loader="synthetic", seed=1)}
    cfg = cfg_factory(sources, tokens=500)
    assert runner.checked_sources(cfg, ["b", "a", "b"]) == ["a", "b"] and runner.checked_sources(cfg, None) is None
    path = config_file(cfg)
    assert prepare(path, layout.root, assume_yes=False, sources=["a", "a"]).missing() == ["b"]
    stale = cfg_factory(sources, tokens=500, token_count="estimate")  # a different raw hash: `a` is deleted and fetched again
    assert prepare(config_file(stale), layout.root, assume_yes=True, sources=["a", "a"]).missing() == ["b"]
    prepare_cli.main(["prepare", "--dataset_config", str(path), "--dataset_dir", str(layout.root), "--sources", "a", "a"])  # the CLI too


# --- failures and interrupts -------------------------------------------------------------------------------------------


def test_failing_download_stops_the_other_downloads_within_a_shard_and_is_reraised(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The other downloads stop at their next shard; a download that stopped is never followed by its build.
    """

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
    """
    Jobs that stop because of the failure may finish before the failing one; the failure itself is raised,
    never `BuildAborted`.
    """

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
    """
    Ctrl-C while `wait_for_jobs` waits: the flag reaches the running download through `should_stop`, it stops at
    its next shard (published), `prepare` raises `BuildAborted` and the next run resumes.
    """

    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500, training_target_sequence_length=1)
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
    monkeypatch.setattr(runner, "wait", interrupted_wait)
    path = config_file(cfg)
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(BuildAborted, match="interrupted"):
        prepare(path, layout.root, assume_yes=False)
    assert 1 <= len(shards_done) < 50, "stopped within a shard of the interrupt, not at the end"
    assert "source p stopped: interrupted" in caplog.text
    raw = Manifest.load(layout.raw_dir("p"))
    assert raw is not None and raw.rows() == 100 and not layout.processed_dir("p").exists()
    monkeypatch.setattr(runner, "wait", concurrent.futures.wait)
    monkeypatch.setattr(runner, "download", real_download)
    assert prepare(path, layout.root, assume_yes=False).complete  # resumes behind the published shard
    raw = Manifest.load(layout.raw_dir("p"))
    assert raw is not None and [s.rows for s in raw.shards] == [100, cfg.rows_needed("p") - 100]


def test_an_outer_should_stop_is_honoured(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    path = config_file(_three_sources(cfg_factory))
    with pytest.raises(BuildAborted):
        prepare(path, layout.root, assume_yes=False, should_stop=lambda: True)


def test_a_failing_follow_up_raises_the_stop_flag_and_cancels_the_queued_jobs() -> None:
    """
    H6: an exception in the main-thread `on_success` follow-up must stop the pools like a job failure would;
    otherwise the pool exits block on downloads polling a flag nobody raised.
    """

    flag = runner.StopFlag()
    ran: list[str] = []

    def follow_up(job: runner.Job) -> None:
        raise RuntimeError("follow-up boom")

    pool = runner.JobPool("downloads", max_workers=1, flag=flag, total=2, on_success=follow_up)
    with pool:
        pool.submit(runner.Job("source", "a", ("a",), lambda stop: ran.append("a")))
        pool.submit(runner.Job("source", "b", ("b",), lambda stop: ran.append("b")))
        with pytest.raises(RuntimeError, match="follow-up boom"):
            runner.wait_for_jobs([pool], flag)
        assert flag.should_stop(), "the flag must be raised so running jobs stop at their next shard"


def test_a_second_interrupt_ends_the_process_without_waiting_for_the_running_job(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """
    Ctrl-C while the pool exit waits for a job that never reaches its next shard (a huge row group, a stalled
    listing): the executor is abandoned and the process ends with 130 instead of joining the thread at exit.
    """

    flag = runner.StopFlag()
    release = threading.Event()
    exits: list[int] = []

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "wait", interrupt)  # the first Ctrl-C lands in `wait_for_jobs`
    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "__exit__", interrupt)  # the second in the pool exit
    monkeypatch.setattr(os, "_exit", exits.append)
    pool = runner.JobPool("downloads", max_workers=1, flag=flag, total=1)
    with caplog.at_level(logging.WARNING, logger="data_preparation"), pool:
        pool.submit(runner.Job("source", "a", ("a",), lambda stop: release.wait(10)))
        failures = runner.wait_for_jobs([pool], flag)
    assert [type(error) for error in failures] == [BuildAborted] and flag.should_stop(), "the first interrupt: stop at the next shard"
    assert exits == [130] and not release.is_set(), "the second: ended while the job was still running"
    assert "second interrupt: ending without waiting for the running transfer" in caplog.text
    release.set()
    pool._executor.shutdown(wait=True)  # the test's own thread, not the process's exit


def test_a_round_with_nothing_to_do_is_a_no_op(cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    cfg = _three_sources(cfg_factory)  # no raw folders: nothing pending, and an empty plan: nothing to download
    runner.download_and_build_missing(DownloadPlan(), cfg, layout, steps=set(runner.STEPS), sources=None, max_parallel_downloads=1, num_workers=1, pass_workers=1)
    assert not layout.root.exists()
    flag = runner.StopFlag()
    assert not flag.should_stop()
    flag.stop("first")
    flag.stop("second")
    assert flag.should_stop() and flag.reason == "first"


# --- builds overlap the downloads ---------------------------------------------------------------------------------------


class _Events:
    """
    A thread-safe ordered log of `(what, name)` events from the stubs of a test.
    """

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def add(self, what: str, name: str) -> None:
        with self._lock:
            self.items.append((what, name))

    def index(self, what: str, name: str) -> int:
        return self.items.index((what, name))


def _wait_for(event: threading.Event, should_stop: Any, what: str, timeout: float = 10.0) -> None:
    """
    Block a stub until `event` is set, honouring the stop request; a `TimeoutError` instead of a hanging test.
    """

    deadline = time.monotonic() + timeout
    while not event.wait(0.02):
        check_stop(should_stop)
        if time.monotonic() > deadline:
            raise TimeoutError(what)


def test_a_source_is_built_while_other_downloads_still_run(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    s1's download does not return before s0 has been built: the build pool works while the download pool runs.
    """

    events = _Events()
    s0_built = threading.Event()

    def download_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        manifest = real_download(config, name, *args, should_stop=should_stop, **kwargs)
        if name == "s1":
            _wait_for(s0_built, should_stop, "s0 was not built while s1 was still downloading")
        events.add("download", name)
        return manifest

    def build_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        manifest = real_build(config, name, *args, **kwargs)
        events.add("build", name)
        if name == "s0":
            assert (layout.processed_dir("s0") / "MANIFEST.json").is_file()
            s0_built.set()
        return manifest

    monkeypatch.setattr(runner, "download", download_stub)
    monkeypatch.setattr(runner, "build_source", build_stub)
    report = prepare(config_file(_three_sources(cfg_factory)), layout.root, assume_yes=False, num_workers=1, max_parallel_downloads=2)
    assert report.complete
    assert events.index("build", "s0") < events.index("download", "s1"), events.items
    assert sorted(events.items) == sorted([(what, f"s{i}") for what in ("download", "build") for i in range(3)])


def test_a_source_is_never_built_before_its_own_download_finished(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _Events()

    def download_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        time.sleep(0.03 * int(name[1:]))  # s2 finishes last
        manifest = real_download(config, name, *args, **kwargs)
        events.add("download_end", name)
        return manifest

    def build_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        events.add("build_start", name)
        return real_build(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download", download_stub)
    monkeypatch.setattr(runner, "build_source", build_stub)
    assert prepare(config_file(_three_sources(cfg_factory)), layout.root, assume_yes=False, num_workers=3, max_parallel_downloads=3).complete
    for name in ("s0", "s1", "s2"):
        assert events.index("download_end", name) < events.index("build_start", name), events.items
    assert events.index("build_start", "s0") < events.index("download_end", "s2"), "s0 waited for the slowest download"


def test_github_code_group_members_are_built_after_the_group_pass(
    hub: FakeHub, cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.add("data/a.parquet", _code_rows("a", 30))
    hub.add("data/b.parquet", _code_rows("b", 30))
    events = _Events()

    def group_stub(config: DatasetConfig, names: list[str], *args: Any, **kwargs: Any) -> dict[str, Manifest]:
        manifests = real_group(config, names, *args, **kwargs)
        events.add("group_end", ", ".join(names))
        return manifests

    def build_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        events.add("build_start", name)
        return real_build(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download_github_code_group", group_stub)
    monkeypatch.setattr(runner, "build_source", build_stub)
    assert prepare(config_file(_github_cfg(cfg_factory, ["Python", "Java", "Go"])), layout.root, assume_yes=False, num_workers=3).complete
    group_end = events.index("group_end", "code_python, code_java, code_go")
    assert all(events.index("build_start", name) > group_end for name in ("code_python", "code_java", "code_go")), events.items


def test_a_failing_build_stops_the_running_downloads(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The mirror image of the failing-download test: one stop flag for both pools.
    """

    ticks: dict[str, int] = {}
    lock = threading.Lock()

    def download_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name != "s0":
            for i in range(100):
                time.sleep(0.01)
                with lock:
                    ticks[name] = i + 1
                check_stop(should_stop)
        return real_download(config, name, *args, should_stop=should_stop, **kwargs)

    def build_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        time.sleep(0.05)
        raise RuntimeError(f"{name}: build exploded")

    monkeypatch.setattr(runner, "download", download_stub)
    monkeypatch.setattr(runner, "build_source", build_stub)
    path = config_file(_three_sources(cfg_factory))
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(RuntimeError, match="s0: build exploded"):
        prepare(path, layout.root, assume_yes=False, num_workers=1, max_parallel_downloads=3)
    assert 0 < ticks["s1"] < 100 and 0 < ticks["s2"] < 100, "the running downloads stopped within a shard"
    assert "source s0 failed" in caplog.text and "source s1 stopped: source s0 failed" in caplog.text


def test_no_build_is_submitted_after_a_failure(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A download that completes after another job failed (it did not poll the flag) is not followed by its build.
    """

    started: list[str] = []

    def download_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name == "s1":
            time.sleep(0.05)
            raise OSError("s1: network down")
        time.sleep(0.3)  # finishes after the failure without ever looking at the flag
        return real_download(config, name, *args, **kwargs)

    def build_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        started.append(name)
        return real_build(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download", download_stub)
    monkeypatch.setattr(runner, "build_source", build_stub)
    with pytest.raises(OSError, match="s1: network down"):
        prepare(config_file(_three_sources(cfg_factory)), layout.root, assume_yes=False, max_parallel_downloads=3)
    assert started == [] and not layout.processed_dir("s0").exists() and not layout.processed_dir("s2").exists()
    assert (layout.raw_dir("s0") / "MANIFEST.json").is_file(), "the download that completed is kept"


def test_interrupt_stops_downloads_and_builds_within_a_shard_and_the_rerun_resumes(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Ctrl-C while a build (s0) and a download (s1) run side by side: both stop at their next shard, everything
    published is kept, `prepare` raises `BuildAborted`; the next run resumes both.
    """

    cfg = cfg_factory({"s0": SourceConfig(kind="pretrain", loader="synthetic", seed=0), "s1": SourceConfig(kind="pretrain", loader="synthetic", seed=1)}, tokens=500, training_target_sequence_length=1)
    build_started = threading.Event()
    ticks: dict[str, int] = {}

    def tick(name: str, should_stop: Any) -> None:
        for i in range(50):
            time.sleep(0.02)
            ticks[name] = i + 1
            check_stop(should_stop)

    def download_stub(config: DatasetConfig, name: str, *args: Any, rows_needed: int, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name == "s0":
            return real_download(config, name, *args, rows_needed=rows_needed, should_stop=should_stop, **kwargs)
        manifest = real_download(config, name, *args, rows_needed=100, should_stop=should_stop, **kwargs)  # one shard on disk first
        tick(name, should_stop)  # then a long download: one "shard" per tick
        return manifest

    def build_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        manifest = real_build(config, name, *args, should_stop=should_stop, **kwargs)  # the whole source, published
        build_started.set()
        tick(name, should_stop)  # then a long build
        return manifest

    def interrupted_wait(futures: Any, *, return_when: str) -> Any:
        if not build_started.is_set():
            return concurrent.futures.wait(futures, timeout=0.02, return_when=return_when)  # the loop hands s0 to the build pool
        time.sleep(0.2)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "download", download_stub)
    monkeypatch.setattr(runner, "build_source", build_stub)
    monkeypatch.setattr(runner, "wait", interrupted_wait)
    path = config_file(cfg)
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(BuildAborted, match="interrupted"):
        prepare(path, layout.root, assume_yes=False, num_workers=1, max_parallel_downloads=2)
    assert 1 <= ticks["s0"] < 50 and 1 <= ticks["s1"] < 50, f"both stopped within a shard of the interrupt: {ticks}"
    assert "source s0 stopped: interrupted" in caplog.text and "source s1 stopped: interrupted" in caplog.text
    raw = Manifest.load(layout.raw_dir("s1"))
    assert raw is not None and raw.rows() == 100 and (layout.processed_dir("s0") / "MANIFEST.json").is_file()

    monkeypatch.setattr(runner, "wait", concurrent.futures.wait)
    monkeypatch.setattr(runner, "download", real_download)
    monkeypatch.setattr(runner, "build_source", real_build)
    assert prepare(path, layout.root, assume_yes=False).complete
    raw = Manifest.load(layout.raw_dir("s1"))
    assert raw is not None and [s.rows for s in raw.shards] == [100, cfg.rows_needed("s1") - 100]  # resumed, not restarted


def test_steps_download_only_never_builds_as_a_follow_up(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    monkeypatch.setattr(runner, "build_source", lambda config, name, *args, **kwargs: built.append(name))
    report = prepare(config_file(_three_sources(cfg_factory)), layout.root, assume_yes=False, steps=["tokenizer", "download"])
    assert built == [] and not report.complete and all(source.satisfaction()[1] == "processed missing" for source in report.sources)


def test_pending_sources_that_need_no_download_are_built_right_away(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _three_sources(cfg_factory)
    path = config_file(cfg)
    prepare(path, layout.root, assume_yes=False, steps=["tokenizer", "download"])
    downloaded: list[str] = []
    built: list[str] = []

    def download_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        downloaded.append(name)
        return real_download(config, name, *args, **kwargs)

    def build_stub(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        built.append(name)
        return real_build(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download", download_stub)
    monkeypatch.setattr(runner, "build_source", build_stub)
    plan = plan_downloads(cfg, layout)
    assert plan.total_rows_to_fetch() == 0
    runner.download_and_build_missing(plan, cfg, layout, steps={"download", "build"}, sources=None, max_parallel_downloads=2, num_workers=2, pass_workers=1)
    assert downloaded == [] and sorted(built) == ["s0", "s1", "s2"] and status(path, layout.root).complete


# --- lock, dry run, confirmation -----------------------------------------------------------------------------------------


def test_prepare_fails_fast_while_another_build_holds_the_lock(layout: DatasetLayout, tmp_path: Path) -> None:
    with build_lock(layout.root), pytest.raises(RunLocked, match=f"pid {os.getpid()}") as excinfo:
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
    planned = [action for action in excinfo.value.report.actions if action.kind == "raw" and action.action == "delete"]
    assert len(asked) == 1 and "p: stale" in asked[0] and planned and not excinfo.value.report.performed
    assert all_mtimes(raw_dir) == before, "nothing was deleted"
    assert not status(path, layout.root).complete  # status shows the pending repair

    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=True)
    assert report.complete and "without asking" in caplog.text
    raw = Manifest.load(raw_dir)
    assert raw is not None and raw.token_count == "estimate" and raw.is_current(cfg.raw_hash("p"))


def test_a_raw_folder_of_another_config_needs_allow_foreign_raw(
    cfg_factory: CfgFactory, layout: DatasetLayout, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Raw folders are shared by source name: config b gives `p` another loader identity, so a's raw folder is
    stale for it. b's prepare refuses to delete it (`--yes` or not) until allowed, then downloads its own and
    records itself in the manifest; status names the flag.
    """

    def config(path: Path, seed: int) -> Path:
        cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=seed)}, tokens=500)
        path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
        return path

    a, b = config(tmp_path / "a.yaml", 0), config(tmp_path / "b.yaml", 1)
    assert prepare(a, layout.root, assume_yes=True).complete
    raw = Manifest.load(layout.raw_dir("p"))
    assert raw is not None and raw.dataset_config == "a.yaml"
    with pytest.raises(ConfirmationRequired, match="downloaded under dataset config a.yaml, deleting it needs --allow_foreign_raw"):
        prepare(b, layout.root, assume_yes=True)
    assert Manifest.load(layout.raw_dir("p")) == raw, "nothing was changed"
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        assert not status(b, layout.root).complete
    assert "would delete raw" in caplog.text and "deleting it needs --allow_foreign_raw" in caplog.text
    assert prepare(b, layout.root, assume_yes=True, allow_foreign_raw=True).complete
    replaced = Manifest.load(layout.raw_dir("p"))
    assert replaced is not None and replaced.dataset_config == "b.yaml" and replaced.source_hash != raw.source_hash


def test_dry_run_and_status_agree_on_a_tree_that_needs_a_repair(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Neither entry point changes the tree here, so they must not judge it differently: both end with the same
    assessment, counting the repairs the run left undone (all of them in a dry run) as incomplete.
    """

    path = config_file(cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=500))
    assert prepare(path, layout.root, assume_yes=False).complete
    next(layout.processed_dir("p").glob("data-*.parquet")).unlink()  # broken: the repair step would delete the folder
    before = all_mtimes(layout.root, include_lock=True)

    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        dry = prepare(path, layout.root, assume_yes=False, dry_run=True)
    assert "would repair:" in caplog.text
    assert dry.describe() == status(path, layout.root).describe()
    assert not dry.complete and dry.needs_repair == ["p"] and dry.missing() == ["p"]
    assert all_mtimes(layout.root, include_lock=True) == before, "neither of them touched the tree"
    assert prepare(path, layout.root, assume_yes=False).complete  # a real run repairs, rebuilds and reports no repair


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
    """
    The rows a source ends up with depend only on its own loader order, not on the interleaving.
    """

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
    return cfg_factory(sources, tokens=15, **kwargs)


def _code_rows(prefix: str, n: int) -> list[dict[str, Any]]:
    languages = ("Python", "Java", "Go")
    return [{"id": f"{prefix}{i}", "text": f"{prefix} code {i}", "language": languages[i % 3]} for i in range(n)]


def test_github_code_languages_of_one_repo_download_in_one_pass(
    hub: FakeHub, cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.add("data/a.parquet", _code_rows("a", 30))
    hub.add("data/b.parquet", _code_rows("b", 30))
    cfg = _github_cfg(cfg_factory, ["Python", "Java", "Go"], training_target_sequence_length=1)
    single: list[str] = []

    def spy_download(config: DatasetConfig, name: str, *args: Any, **kwargs: Any) -> Manifest:
        single.append(name)
        return real_download(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "download", spy_download)
    jobs = runner.download_jobs(plan_downloads(cfg, layout), cfg, layout, None)
    assert [(job.what, job.name) for job in jobs] == [("github_code group", "code_python, code_java, code_go")]
    assert jobs[0].sources == ("code_python", "code_java", "code_go")
    report = prepare(config_file(cfg), layout.root, assume_yes=False)
    assert report.complete and single == []  # the group pass replaced the per-source downloads
    assert len(hub.streams) == len(set(hub.streams))  # every repo file opened at most once for all three languages
    for name in ("code_python", "code_java", "code_go"):
        raw = Manifest.load(layout.raw_dir(name))
        assert raw is not None and raw.rows() >= 6  # 5 sequences each × 1.2

    # `--sources` with one language uses the ordinary per-source path
    hub.streams.clear()
    bigger = _github_cfg(cfg_factory, ["Python", "Java", "Go"], training_target_sequence_length=1)
    bigger.stages[0].tokens = 30
    jobs = runner.download_jobs(plan_downloads(bigger, layout, sources=["code_python"]), bigger, layout, None)
    assert [(job.what, job.name) for job in jobs] == [("source", "code_python")]
    prepare(config_file(bigger), layout.root, assume_yes=False, sources=["code_python"])
    assert single == ["code_python"]


# --- repairs on the way -----------------------------------------------------------------------------------------------


def test_processing_change_rebuilds_processed_but_leaves_raw_untouched(cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile) -> None:
    """
    The raw shards are the bandwidth-expensive part: a processing-only edit must not re-download them.
    """

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
    from data_preparation.lib.sources import loaders as loaders_mod

    monkeypatch.setattr(runner, "download", partial(real_download, shard_size=10))
    path = config_file(cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic", seed=0)}, tokens=50, training_target_sequence_length=1))
    prepare(path, layout.root, assume_yes=False)
    raw = layout.raw_dir("p")
    manifest = Manifest.load(raw)
    assert manifest is not None and len(manifest.shards) >= 3, "the test needs several raw shards"
    last = manifest.shards[-1]
    (raw / last.name).write_bytes(b"corrupt")
    offsets: list[int] = []
    original = loaders_mod.LOADERS["synthetic"]

    def spy(source: Any, offset: int, count: int, shared_parameters: Any) -> Any:
        offsets.append(offset)
        return original(source, offset, count, shared_parameters)

    monkeypatch.setitem(loaders_mod.LOADERS, "synthetic", spy)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        report = prepare(path, layout.root, assume_yes=False)
    assert report.complete and "truncating" in caplog.text
    assert offsets == [manifest.shards[-2].offset], "resumed behind the last good shard instead of from 0"
    repaired = Manifest.load(raw)
    assert repaired is not None and repaired.rows() == manifest.rows() and [s.rows for s in repaired.shards] == [s.rows for s in manifest.shards]


def test_prepare_logs_the_stop_reason_of_a_slow_build(
    cfg_factory: CfgFactory, layout: DatasetLayout, config_file: ConfigFile, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A failing build stops the other running builds at their next shard (the build pool has the download pool's
    stop semantics). s1 explodes only once s0 and s2 are building: builds start as their downloads finish, so
    without the gate s2 might still be downloading and be stopped there instead.
    """

    ticks: dict[str, int] = {}
    lock = threading.Lock()
    others_building = threading.Event()

    def build_stub(config: DatasetConfig, name: str, *args: Any, should_stop: Any = None, **kwargs: Any) -> Manifest:
        if name == "s1":
            _wait_for(others_building, should_stop, "s0 and s2 never started building")
            raise RuntimeError("s1: build exploded")
        for i in range(100):
            time.sleep(0.01)
            with lock:
                ticks[name] = i + 1
                if len(ticks) == 2:
                    others_building.set()
            check_stop(should_stop)
        return real_build(config, name, *args, **kwargs)

    monkeypatch.setattr(runner, "build_source", build_stub)
    path = config_file(_three_sources(cfg_factory))
    with caplog.at_level(logging.INFO, logger="data_preparation"), pytest.raises(RuntimeError, match="s1: build exploded"):
        prepare(path, layout.root, assume_yes=False, num_workers=3, max_parallel_downloads=3)
    assert 0 < ticks["s0"] < 100 and 0 < ticks["s2"] < 100
    assert "source s0 stopped: source s1 failed" in caplog.text
