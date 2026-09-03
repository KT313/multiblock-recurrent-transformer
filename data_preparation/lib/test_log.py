# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.log.
"""

import logging
from collections.abc import Iterator

import pytest

from data_preparation.lib.log import ROOT_LOGGER_NAME, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _clean_root_logger() -> Iterator[None]:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers.clear()
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_get_logger_is_under_hierarchy() -> None:
    assert get_logger("common").name == "data_preparation.common"
    assert get_logger("data_preparation.lib.storage.manifest").name == "data_preparation.lib.storage.manifest"
    assert get_logger(ROOT_LOGGER_NAME).name == ROOT_LOGGER_NAME
    parent = get_logger("x").parent
    assert parent is not None and parent.name == ROOT_LOGGER_NAME


def test_configure_logging_idempotent_and_formats(capsys: pytest.CaptureFixture[str]) -> None:
    root = configure_logging()
    configure_logging(logging.DEBUG)
    assert len(root.handlers) == 1 and root.level == logging.DEBUG
    get_logger("t").info("hello %d", 3)
    err = capsys.readouterr().err
    assert err.endswith("INFO data_preparation.t: hello 3\n")
