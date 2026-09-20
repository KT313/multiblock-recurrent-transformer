# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The package facade keeps callers independent of implementation module locations."""

import subprocess
import sys

import pytest

import data_preparation
from data_preparation.lib import dataset_config, layout


def test_public_exports_are_the_original_classes_and_functions() -> None:
    for name in data_preparation.__all__:
        implementation = layout if name == "DatasetLayout" else dataset_config
        assert getattr(data_preparation, name) is getattr(implementation, name)
    with pytest.raises(AttributeError, match="has no attribute 'unknown_export'"):
        _ = data_preparation.unknown_export


def test_importing_package_does_not_load_preparation_dependencies() -> None:
    subprocess.run(
        [sys.executable, "-c", "import sys; import data_preparation; "
         "assert 'data_preparation.lib.dataset_config' not in sys.modules; "
         "assert 'data_preparation.lib.layout' not in sys.modules; "
         "assert 'data_preparation.lib.build.runner' not in sys.modules; "
         "from data_preparation import DatasetLayout; "
         "assert DatasetLayout.__name__ == 'DatasetLayout'; "
         "assert 'data_preparation.lib.dataset_config' not in sys.modules"],
        check=True, timeout=10,
    )
