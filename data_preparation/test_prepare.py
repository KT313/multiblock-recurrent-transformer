# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.prepare: command registration, `status` exit codes, `describe` output, `build` options,
tiny end to end."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from data_preparation import prepare
from data_preparation.lib.build import STEPS
from data_preparation.lib.schema.dataset_config import DatasetConfig
from data_preparation.lib.schema.layout import DatasetLayout

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY = REPO_ROOT / "config" / "datasets" / "tiny.yaml"


def test_commands_are_registered() -> None:
    parser = prepare.build_parser()
    args = parser.parse_args(["build", "--dataset_config", "x.yaml"])
    assert args.run is prepare.run_build and args.dataset_dir == Path("dataset") and args.sources is None and args.steps is None
    assert args.num_workers == 1 and args.hf_token is None and not args.dry_run and args.cache_dir is None
    assert args.max_parallel_downloads == 2
    assert parser.parse_args(["build", "--dataset_config", "x.yaml", "--max_parallel_downloads", "4"]).max_parallel_downloads == 4
    args = parser.parse_args(["build", "--dataset_config", "x.yaml", "--sources", "a", "b", "--steps", "download", "process", "--dry_run", "--num_workers", "3"])
    assert args.sources == ["a", "b"] and args.steps == ["download", "process"] and args.dry_run and args.num_workers == 3
    args = parser.parse_args(["status", "--dataset_config", "x.yaml", "--dataset_dir", "d"])
    assert args.run is prepare.run_status and args.dataset_dir == Path("d")
    args = parser.parse_args(["describe", "--dataset_config", "x.yaml"])
    assert args.run is prepare.run_describe and args.dataset_config == Path("x.yaml")
    args = parser.parse_args(["tiny"])
    assert args.run is prepare.run_build and args.dataset_config == Path("config/datasets/tiny.yaml")
    for command in ("build", "status", "describe"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])  # --dataset_config is required
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--dataset_config", "x.yaml", "--steps", "nope"])
    assert set(STEPS) == {"tokenizer", "download", "process", "validation", "instruct_mixtures"}


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


def test_build_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    prepare.main(["build", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--dry_run"])
    assert not root.exists()


def test_build_sources_and_steps_filters(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    layout = DatasetLayout(root)
    prepare.main(["build", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--steps", "tokenizer", "download"])
    assert (layout.tokenizer_dir("synthetic") / "MANIFEST.json").is_file()
    assert (layout.source_dir("synthetic_pretrain", "raw") / "MANIFEST.json").is_file()
    assert not layout.source_dir("synthetic_pretrain", "processed").exists() and not layout.validation_dir("synthetic_val").exists()
    prepare.main(["build", "--dataset_config", str(TINY), "--dataset_dir", str(root), "--sources", "synthetic_val"])
    assert layout.validation_dir("synthetic_val").is_dir() and not layout.source_dir("synthetic_pretrain", "processed").exists()
    with pytest.raises(SystemExit) as exc:
        prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    assert exc.value.code == 1


def test_build_failure_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(prepare, "build", boom)
    with pytest.raises(SystemExit) as exc:
        prepare.main(["build", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 1 and "stage exploded" in caplog.text and "build failed" in caplog.text


def test_build_incomplete_result_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_preparation.lib.build.planner import Plan

    monkeypatch.setattr(prepare, "build", lambda *a, **k: Plan())
    with pytest.raises(SystemExit) as exc:
        prepare.main(["build", "--dataset_config", str(TINY), "--dataset_dir", str(tmp_path)])
    assert exc.value.code == 1


@pytest.mark.slow
def test_tiny_end_to_end(tmp_path: Path, tiny_dataset_config: DatasetConfig) -> None:
    root = tmp_path / "dataset"
    prepare.main(["tiny", "--dataset_dir", str(root)])
    prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    layout = DatasetLayout(root)
    assert (layout.source_dir("synthetic_pretrain", "processed") / "MANIFEST.json").is_file()
    assert all((layout.instruct_mixture_dir(tiny_dataset_config.name, "tiny_instruct", s) / "MANIFEST.json").is_file() for s in ("train", "validation"))
    shutil.rmtree(layout.validation_dir("synthetic_val"))
    with pytest.raises(SystemExit):
        prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
    prepare.main(["build", "--dataset_config", str(TINY), "--dataset_dir", str(root)])  # repairs
    prepare.main(["status", "--dataset_config", str(TINY), "--dataset_dir", str(root)])
