# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.progress: the enabling rule and the no-op bar.
"""

from __future__ import annotations

import io

import pytest

from data_preparation.lib.progress import ENV_VAR, NoProgress, progress_enabled


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


def test_no_progress_counts_updates() -> None:
    bar = NoProgress(total=10)
    with bar as b:
        b.update(3)
        b.set_postfix(tokens=1)
    assert bar.n == 3 and bar.total == 10 and NoProgress().total is None


def test_no_progress_counts_from_initial() -> None:
    bar = NoProgress(total=10, initial=6)
    bar.update(2)
    assert bar.n == 8 and bar.total == 10
