# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.prepare: command registration and flags, exit codes (`status`, interrupt, unconfirmed
raw deletion, failures), `describe` output, `prepare` options, tiny end to end.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any

import pytest

from data_preparation import prepare
from data_preparation.lib.build.lock import build_lock
from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.planner import DatasetReport, SourceLedger
from data_preparation.lib.build.runner import STEPS
from data_preparation.lib.build.repair import ConfirmationRequired, RepairAction, RepairReport

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY = REPO_ROOT / "config" / "datasets" / "tiny.yaml"


@pytest.fixture(autouse=True)
def detached_data_preparation_handlers() -> Iterator[logging.Logger]:
    """
    The `data_preparation` logger (yielded) without the handler `configure_logging` adds to it, removed again
    afterwards, so a handler bound to a captured stderr never outlives its test (the dashboard tests would then log
    into a closed stream, and a failing handler used to take the whole run down with it). Autouse: every `main()`
    call configures the hierarchy. The sibling of `training/test_train.py`'s fixture.
    """

    logger = logging.getLogger("data_preparation")
    handlers_before, level = list(logger.handlers), logger.level
    yield logger
    for handler in list(logger.handlers):
        if handler not in handlers_before:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(level)


def test_commands_are_registered() -> None:
    parser = prepare.build_parser()
    args = parser.parse_args(["prepare", "--dataset_config", "x.yaml"])
    assert args.run is prepare.run_prepare and args.dataset_dir == Path("dataset") and args.sources is None and args.steps is None
    assert args.num_workers == 2 and args.pass_workers == 4 and args.max_parallel_downloads == 2 and args.hf_token is None and args.cache_dir is None
    assert not args.dry_run and not args.yes and args.reopen is None and not args.allow_foreign_raw
    args = parser.parse_args(["prepare", "--dataset_config", "x.yaml", "--sources", "a", "b", "--steps", "download", "build", "--reopen", "a", "--dry_run", "--yes", "--allow_foreign_raw", "--num_workers", "3", "--pass_workers", "5", "--max_parallel_downloads", "4"])
    assert args.sources == ["a", "b"] and args.steps == ["download", "build"] and args.reopen == ["a"] and args.dry_run and args.yes and args.allow_foreign_raw
    assert args.num_workers == 3 and args.pass_workers == 5 and args.max_parallel_downloads == 4
    assert parser.parse_args(["prepare", "--dataset_config", "x.yaml", "-y"]).yes
    args = parser.parse_args(["status", "--dataset_config", "x.yaml", "--dataset_dir", "d"])
    assert args.run is prepare.run_status and args.dataset_dir == Path("d")
    args = parser.parse_args(["describe", "--dataset_config", "x.yaml"])
    assert args.run is prepare.run_describe and args.dataset_config == Path("x.yaml")
    args = parser.parse_args(["tiny"])
    assert args.run is prepare.run_prepare and args.dataset_config == Path("config/datasets/tiny.yaml")
    args = parser.parse_args(["download", "--dataset_config", "x.yaml", "--sources", "a", "--yes", "--steps", "download"])
    assert args.run is prepare.run_download and args.sources == ["a"] and args.yes and args.steps == ["download"] and args.dataset_dir == Path("dataset")
    with pytest.raises(SystemExit):
        parser.parse_args(["download", "--dataset_config", "x.yaml", "--steps", "build"])  # download never builds
    for command in ("prepare", "download", "status", "describe"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])  # --dataset_config is required
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare", "--dataset_config", "x.yaml", "--steps", "nope"])
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--dataset_config", "x.yaml"])  # the old command name is gone
    assert STEPS == ("tokenizer", "download", "build")


def test_missing_or_unknown_command_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        prepare.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        prepare.main(["no-such-command"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_status_exit_codes(tmp_path: Path, tiny_layout: DatasetLayout, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path / "empty")])
    assert exc.value.code == 1
    assert "INCOMPLETE" in capsys.readouterr().out
    prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(tiny_layout.root)])  # exit 0
    assert "dataset complete" in capsys.readouterr().out


def test_an_unreadable_raw_manifest_is_reported_without_a_traceback(
    tmp_path: Path, tiny_layout: DatasetLayout, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """
    `status` and `prepare --dry_run` on a raw folder whose manifest does not parse print the status table with
    the state and leave the folder alone; they used to die with the `Manifest.load` traceback.
    """

    root = tmp_path / "dataset"
    shutil.copytree(tiny_layout.root, root)
    manifest = DatasetLayout(root).raw_dir("synthetic_pretrain") / "MANIFEST.json"
    manifest.write_text("{ not json")
    with pytest.raises(SystemExit) as exc:
        prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    out = capsys.readouterr().out
    assert exc.value.code == 1 and "INCOMPLETE" in out and "Traceback" not in out
    assert "synthetic_pretrain  pretrain" in out and "raw unreadable manifest next to shards; fix or delete the directory by hand" in out
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--dry_run"])  # a dry run exits 0
    assert "dataset INCOMPLETE" in caplog.text and "would leave raw" in caplog.text and "Traceback" not in caplog.text
    assert manifest.read_text() == "{ not json" and (manifest.parent / "data-00000.parquet").is_file()


def test_describe_prints_markdown_with_the_config_notes(capsys: pytest.CaptureFixture[str]) -> None:
    prepare.main(["describe", "--dataset_config", str(TINY)])
    out = capsys.readouterr().out
    assert out.startswith("# Dataset `tiny`\n") and out.endswith("\n") and "## Notes" in out
    assert "Synthetic dataset for the smoke run" in out  # the YAML's leading comment block
    assert "| `synthetic_pretrain` | pretrain | `synthetic` |" in out


def test_prepare_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--dry_run"])
    assert not root.exists()


def test_prepare_sources_and_steps_filters(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    layout = DatasetLayout(root)
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--steps", "tokenizer", "download"])
    assert (layout.tokenizer_dir("synthetic") / "MANIFEST.json").is_file()
    assert (layout.raw_dir("synthetic_pretrain") / "MANIFEST.json").is_file()
    assert not layout.processed_dir("synthetic_pretrain").exists() and not layout.processed_dir("synthetic_instruct").exists()
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--sources", "synthetic_instruct"])
    assert layout.processed_dir("synthetic_instruct").is_dir() and not layout.processed_dir("synthetic_pretrain").exists()
    with pytest.raises(SystemExit) as exc:
        prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    assert exc.value.code == 1
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--steps", "build"])
    prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])  # exit 0


def test_prepare_writes_the_build_log(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--steps", "tokenizer"])
    log_text = (root / "build.log").read_text()
    assert "preparing dataset config" in log_text and "dataset status:" in log_text


def test_prepare_failure_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(prepare, "prepare", boom)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 1 and "stage exploded" in caplog.text and "prepare failed" in caplog.text


def test_prepare_incomplete_result_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "prepare", lambda *a, **k: DatasetReport())
    with pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 1


def test_an_explicit_full_steps_list_is_not_a_partial_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `--steps tokenizer download build` runs everything, so the completeness check applies (it used to be skipped
    for any --steps); a strict subset is partial and exits 0 whatever the report says.
    """

    monkeypatch.setattr(prepare, "prepare", lambda *a, **k: DatasetReport())
    with pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path), "--steps", "build", "tokenizer", "download"])
    assert exc.value.code == 1
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path), "--steps", "tokenizer", "download"])  # exit 0
    prepare.main(["download", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path), "--steps", "tokenizer"])  # exit 0


def test_download_runs_tokenizer_and_download_only_and_prepare_builds_afterwards(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "dataset"
    layout = DatasetLayout(root)
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        prepare.main(["download", "--dataset_config", str(TINY), "--dataset_dir", str(root)])  # exit 0
    assert f"download complete: {root}" in caplog.text
    assert (layout.tokenizer_dir("synthetic") / "MANIFEST.json").is_file()
    assert (layout.raw_dir("synthetic_pretrain") / "MANIFEST.json").is_file() and (layout.raw_dir("synthetic_instruct") / "MANIFEST.json").is_file()
    assert not layout.processed_dir("synthetic_pretrain").exists() and not layout.processed_dir("synthetic_instruct").exists()
    with pytest.raises(SystemExit) as exc:
        prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])  # the dataset is not built
    assert exc.value.code == 1
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])  # exit 0
    assert layout.processed_dir("synthetic_pretrain").is_dir() and layout.processed_dir("synthetic_instruct").is_dir()


def test_download_names_the_sources_still_short_and_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """
    The download verdict is about the raw side: a source whose processed folder is missing is fine, a raw folder
    still short is not.
    """

    def ledger(name: str, **overrides: Any) -> SourceLedger:
        fields: dict[str, Any] = dict(
            name=name, kind="pretrain", rows_needed=100, rows_sufficient=84, rows_budget=70, tokens_per_row=64.0, raw_state="current", raw_reason="current",
            raw_rows=100, exhausted=False, skipped_malformed=0, dropped_too_long=0, processed_problem="absent", processed_reason="missing", processed_rows=0, training_rows=0,
        )
        return SourceLedger(**{**fields, **overrides})

    report = DatasetReport(sources=[ledger("done"), ledger("short", raw_rows=40), ledger("absent", raw_state="missing", raw_reason="missing", raw_rows=0)], tokenizer_complete=True)
    assert not report.complete  # nothing is built: prepare would call this incomplete
    monkeypatch.setattr(prepare, "prepare", lambda *a, **k: report)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["download", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 1 and "download incomplete: short, absent" in caplog.text and "download failed" in caplog.text
    report.sources = [ledger("done")]
    prepare.main(["download", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])  # exit 0: raw rows are there, nothing built


@pytest.mark.parametrize("error", [KeyboardInterrupt(), BuildAborted("interrupted")])
def test_interrupt_exits_130(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: BaseException) -> None:
    def interrupted(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(prepare, "prepare", interrupted)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 130 and "prepare interrupted; everything published so far is kept" in caplog.text


def test_unconfirmed_raw_deletion_exits_two_with_the_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    action = RepairAction("fineweb", tmp_path / "raw", "raw", "delete", "outdated: dataset_max_sequence_length 2048 -> 4096")
    message = "The following raw folders will be deleted and downloaded again:\n  fineweb: outdated: dataset_max_sequence_length 2048 -> 4096\nContinue? [y/N] "

    def refused(*args: object, **kwargs: object) -> None:
        raise ConfirmationRequired(RepairReport([action]), message, interactive=False)

    monkeypatch.setattr(prepare, "prepare", refused)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "fineweb: outdated: dataset_max_sequence_length 2048 -> 4096" in err and "rerun with --yes" in err
    assert not (tmp_path / "sources").exists()


def test_yes_flag_reaches_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def record(*args: object, **kwargs: object) -> DatasetReport:
        seen.update(kwargs)
        return DatasetReport(tokenizer_complete=True)

    monkeypatch.setattr(prepare, "prepare", record)
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path), "--yes", "--allow_foreign_raw", "--hf_token", "t", "--num_workers", "3", "--pass_workers", "2"])
    assert (seen["assume_yes"], seen["allow_foreign_raw"], seen["hf_token"], seen["num_workers"], seen["pass_workers"], seen["steps"]) == (True, True, "t", 3, 2, STEPS)
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert (seen["assume_yes"], seen["dry_run"], seen["allow_foreign_raw"]) == (False, False, False)


@pytest.mark.slow
def test_tiny_end_to_end(tmp_path: Path, tiny_dataset_config: DatasetConfig) -> None:
    root = tmp_path / "dataset"
    prepare.main(["tiny", "--dataset_dir", str(root)])
    prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    layout = DatasetLayout(root)
    assert all((layout.processed_dir(name) / "MANIFEST.json").is_file() for name in tiny_dataset_config.sources)
    shutil.rmtree(layout.processed_dir("synthetic_instruct"))
    with pytest.raises(SystemExit):
        prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root)])  # rebuilds from raw
    prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])


def test_prepare_exits_3_while_another_run_holds_the_build_lock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "dataset"
    with build_lock(root), pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    assert exc.value.code == prepare.EXIT_ALREADY_RUNNING
    err = capsys.readouterr().err
    assert "data preparation expects one run at a time on this system; one is already running (started " in err
    assert f"kill -INT {os.getpid()}" in err


def test_prepare_turns_the_tokenizer_thread_pool_on_unless_the_environment_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The CLI never forks after loading the tokenizer, so it lifts the library's `TOKENIZERS_PARALLELISM=false`
    guard; an explicit value in the environment is kept.
    """

    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path / "a"), "--dry_run"])
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true" and os.environ["RAYON_NUM_THREADS"] == str(prepare.TOKENIZER_POOL_THREADS)
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("RAYON_NUM_THREADS", "3")
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path / "b"), "--dry_run"])
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false" and os.environ["RAYON_NUM_THREADS"] == "3"
