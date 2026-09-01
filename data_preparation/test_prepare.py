# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.prepare: command registration and flags, exit codes (`status`, interrupt, unconfirmed
raw deletion, failures), `describe` output, `prepare` options, tiny end to end."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from data_preparation import prepare
from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build import STEPS, DatasetReport
from data_preparation.lib.build.repair import ConfirmationRequired, RepairAction, RepairReport

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY = REPO_ROOT / "config" / "datasets" / "tiny.yaml"


@pytest.fixture(autouse=True)
def detached_data_preparation_handlers() -> Iterator[logging.Logger]:
    """The `data_preparation` logger (yielded) without the handler `configure_logging` adds to it — removed again
    afterwards, so a handler bound to a captured stderr never outlives its test (the dashboard tests would then log
    into a closed stream, and a failing handler used to take the whole run down with it). Autouse: every `main()`
    call configures the hierarchy. The sibling of `training/test_train.py`'s fixture."""
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
    assert not args.dry_run and not args.yes
    args = parser.parse_args(["prepare", "--dataset_config", "x.yaml", "--sources", "a", "b", "--steps", "download", "build", "--dry_run", "--yes", "--num_workers", "3", "--pass_workers", "5", "--max_parallel_downloads", "4"])
    assert args.sources == ["a", "b"] and args.steps == ["download", "build"] and args.dry_run and args.yes
    assert args.num_workers == 3 and args.pass_workers == 5 and args.max_parallel_downloads == 4
    assert parser.parse_args(["prepare", "--dataset_config", "x.yaml", "-y"]).yes
    args = parser.parse_args(["status", "--dataset_config", "x.yaml", "--dataset_dir", "d"])
    assert args.run is prepare.run_status and args.dataset_dir == Path("d")
    args = parser.parse_args(["describe", "--dataset_config", "x.yaml"])
    assert args.run is prepare.run_describe and args.dataset_config == Path("x.yaml")
    args = parser.parse_args(["tiny"])
    assert args.run is prepare.run_prepare and args.dataset_config == Path("config/datasets/tiny.yaml")
    for command in ("prepare", "status", "describe"):
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


@pytest.mark.parametrize("error", [KeyboardInterrupt(), BuildAborted("interrupted")])
def test_interrupt_exits_130(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: BaseException) -> None:
    def interrupted(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(prepare, "prepare", interrupted)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 130 and "prepare interrupted; everything published so far is kept" in caplog.text


def test_unconfirmed_raw_deletion_exits_two_with_the_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    action = RepairAction("fineweb", tmp_path / "raw", "raw", "would_delete", "outdated: max_seq_length 2048 -> 4096")
    message = "The following raw folders will be deleted and downloaded again:\n  fineweb: outdated: max_seq_length 2048 -> 4096\nContinue? [y/N] "

    def refused(*args: object, **kwargs: object) -> None:
        raise ConfirmationRequired(RepairReport([action]), message, interactive=False)

    monkeypatch.setattr(prepare, "prepare", refused)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "fineweb: outdated: max_seq_length 2048 -> 4096" in err and "rerun with --yes" in err
    assert not (tmp_path / "sources").exists()


def test_yes_flag_reaches_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def record(*args: object, **kwargs: object) -> DatasetReport:
        seen.update(kwargs)
        return DatasetReport(tokenizer_complete=True)

    monkeypatch.setattr(prepare, "prepare", record)
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path), "--yes", "--hf_token", "t", "--num_workers", "3", "--pass_workers", "2"])
    assert (seen["assume_yes"], seen["hf_token"], seen["num_workers"], seen["pass_workers"], seen["steps"]) == (True, "t", 3, 2, STEPS)
    prepare.main(["prepare", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert (seen["assume_yes"], seen["dry_run"]) == (False, False)


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
