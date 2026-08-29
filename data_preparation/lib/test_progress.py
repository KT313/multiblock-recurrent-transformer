# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.progress: the no-op path, the env switch and the tqdm path on a fake TTY."""

from __future__ import annotations

import io
import sys

import pytest

from data_preparation.lib.progress import ENV_VAR, NoProgress, progress, progress_enabled, write_line


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_disabled_without_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert progress_enabled(io.StringIO()) is False
    assert progress_enabled(FakeTty()) is True


@pytest.mark.parametrize("value", ["0", "false", "OFF", " no "])
def test_env_switch_disables(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV_VAR, value)
    assert progress_enabled(FakeTty()) is False


def test_no_progress_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "0")
    bar = progress(total=10, desc="x")
    assert isinstance(bar, NoProgress)
    with bar as b:
        b.update(3)
        b.set_postfix(tokens=1)
        b.set_description("y")
    assert bar.n == 3
    assert list(progress([1, 2, 3], desc="it")) == [1, 2, 3]
    assert list(progress(desc="empty")) == []


def test_tqdm_path_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    stream = FakeTty()
    monkeypatch.setattr(sys, "stderr", stream)
    bar = progress(total=2, desc="dl", unit="row", leave=False)
    assert not isinstance(bar, NoProgress)
    bar.update(2)
    bar.set_postfix(file="a.parquet")
    bar.close()
    assert "dl" in stream.getvalue()
    assert [x * 2 for x in progress([1, 2], desc="it")] == [2, 4]


def test_write_line() -> None:
    stream = io.StringIO()
    write_line("hello", stream)
    assert stream.getvalue() == "hello\n"
