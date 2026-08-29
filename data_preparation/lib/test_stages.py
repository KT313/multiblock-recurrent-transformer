# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The `stages` facade exposes every stage function of the split modules."""

from data_preparation.lib import stages, stages_instruct, stages_pretrain, stages_shared


def test_facade_reexports_every_stage() -> None:
    assert stages.download is stages_shared.download
    assert stages.holdout is stages_shared.holdout
    assert stages.prepare_tokenizer is stages_shared.prepare_tokenizer
    assert stages.length_filter is stages_pretrain.length_filter
    assert stages.process is stages_pretrain.process
    assert stages.build_mixture is stages_instruct.build_mixture
    assert set(stages.__all__) >= {"download", "holdout", "prepare_tokenizer", "length_filter", "process", "build_mixture"}
