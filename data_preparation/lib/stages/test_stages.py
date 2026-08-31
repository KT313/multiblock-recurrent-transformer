# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The `stages` facade exposes every step function of the split modules."""

from data_preparation.lib import stages
from data_preparation.lib.stages import build as stages_build
from data_preparation.lib.stages import shared as stages_shared


def test_facade_reexports_every_step() -> None:
    assert stages.download is stages_shared.download
    assert stages.download_github_code_group is stages_shared.download_github_code_group
    assert stages.prepare_tokenizer is stages_shared.prepare_tokenizer
    assert stages.build_source is stages_build.build_source
    assert set(stages.__all__) >= {"download", "download_github_code_group", "prepare_tokenizer", "build_source"}
