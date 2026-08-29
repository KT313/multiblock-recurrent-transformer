# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.prepare: every command is registered, forwards its options and dispatches to run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from data_preparation import prepare
from data_preparation.lib import make_tiny_dataset, prepare_flan_mixture


def test_every_command_is_registered_and_parses_defaults() -> None:
    parser = prepare.build_parser()
    for name, module in prepare.COMMANDS.items():
        args = parser.parse_args([name])
        assert args.command == name
        assert args.run is module.run
    assert set(prepare.COMMANDS) == {
        "download", "filter", "process", "fineweb-validation", "flan-mixture", "tokenizer", "tiny",
    }


def test_subcommand_options_match_the_module_parser() -> None:
    via_entry = prepare.build_parser().parse_args(["flan-mixture", "--total_examples", "12", "--dry_run"])
    via_module = prepare_flan_mixture.build_parser().parse_args(["--total_examples", "12", "--dry_run"])
    for key, value in vars(via_module).items():
        assert getattr(via_entry, key) == value
    assert via_entry.dataset_dir == Path("dataset")


def test_missing_or_unknown_command_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        prepare.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        prepare.main(["no-such-command"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_main_dispatches_to_the_module_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setitem(prepare.COMMANDS, "tiny", make_tiny_dataset)
    monkeypatch.setattr(make_tiny_dataset, "run", lambda args: seen.append(args))
    prepare.main(["tiny", "--out", str(tmp_path / "t")])
    assert len(seen) == 1 and seen[0].out == tmp_path / "t"


def test_tiny_command_end_to_end(tmp_path: Path) -> None:
    prepare.main(["tiny", "--out", str(tmp_path / "tiny")])
    assert (tmp_path / "tiny" / "tokenizer" / "tokenizer.json").exists()
    assert list((tmp_path / "tiny" / "pretrain" / "train").glob("*.parquet"))
