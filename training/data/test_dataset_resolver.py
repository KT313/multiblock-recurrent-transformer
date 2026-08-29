# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the dataset resolver: golden comparison against the pre-dataset-config run config of the thesis run,
auto-prepare, the hard error without it, cross-checks between run and dataset config, resume hash check."""

import logging
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.schema.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.lib.schema.layout import DatasetLayout
from training.data.dataset_resolver import (
    CHECKPOINT_HASH_KEY,
    INSTRUCT_DATA_SIGNATURE,
    ResolvedDataset,
    ResolvedStage,
    build_command,
    check_checkpoint_dataset_hash,
    resolve_dataset,
    resolve_entries,
    validate_settings,
)
from training.settings import DataEntry, Settings
from training.stage_manager import TrainingStage

REPO_ROOT = Path(__file__).resolve().parents[2]
CROW_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "crow_300m_final.yaml"
TINY_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "tiny.yaml"

# The last hand-written run config (git history before the dataset-config restructure, `config/crow_300m_final.yaml`)
# as (data_dir, weight, data_signature) per stage, with its per-stage data directories mapped onto the new layout
# (pretrain source -> sources/<src>/processed, the fineweb validation set -> sources/fineweb_val/validation, the
# finetune mixture -> instruct_mixtures/crow-300m-final/flan_instruct/{train,validation}).
_P = "dataset/sources/{}/processed"
_INSTRUCT = {"keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output"}
GOLDEN_CROW_STAGES: list[dict[str, Any]] = [
    {
        "name": "pretrain_phase1",
        "tokens": 3_300_000_000,
        "base_lr": 3e-4,
        "transition_pct": 0.10,
        "train": [
            (_P.format("fineweb_edu"), 0.65, None),
            (_P.format("wikipedia"), 0.09, None),
            (_P.format("books_gutenberg"), 0.06, None),
            (_P.format("github_code_clean_python"), 0.036, None),
            (_P.format("github_code_clean_javascript"), 0.024, None),
            (_P.format("github_code_clean_typescript"), 0.012, None),
            (_P.format("github_code_clean_java"), 0.012, None),
            (_P.format("github_code_clean_cpp"), 0.0096, None),
            (_P.format("github_code_clean_go"), 0.0084, None),
            (_P.format("github_code_clean_rust"), 0.006, None),
            (_P.format("github_code_clean_shell"), 0.0048, None),
            (_P.format("github_code_clean_sql"), 0.0036, None),
            (_P.format("github_code_clean_html"), 0.0036, None),
            (_P.format("peso"), 0.03, None),
            (_P.format("arxiv"), 0.02, None),
            (_P.format("openwebmath"), 0.03, None),
        ],
        "val": [("dataset/sources/fineweb_val/validation", 1.0, None)],
    },
    {
        "name": "pretrain_phase2",
        "tokens": 1_500_000_000,
        "base_lr": 1e-4,
        "transition_pct": 0.10,
        "train": [
            (_P.format("fineweb_edu"), 0.35, None),
            (_P.format("github_code_clean_python"), 0.084, None),
            (_P.format("github_code_clean_javascript"), 0.056, None),
            (_P.format("github_code_clean_typescript"), 0.028, None),
            (_P.format("github_code_clean_java"), 0.028, None),
            (_P.format("github_code_clean_cpp"), 0.0224, None),
            (_P.format("github_code_clean_go"), 0.0196, None),
            (_P.format("github_code_clean_rust"), 0.014, None),
            (_P.format("github_code_clean_shell"), 0.0112, None),
            (_P.format("github_code_clean_sql"), 0.0084, None),
            (_P.format("github_code_clean_html"), 0.0084, None),
            (_P.format("openwebmath"), 0.088, None),
            (_P.format("tinygsm"), 0.066, None),
            (_P.format("algebraic_stack"), 0.044, None),
            (_P.format("gsm8k"), 0.022, None),
            (_P.format("peso"), 0.09, None),
            (_P.format("arxiv"), 0.06, None),
        ],
        "val": [("dataset/sources/fineweb_val/validation", 1.0, None)],
    },
    {
        "name": "finetune",
        "tokens": 150_000_000,
        "base_lr": 5e-5,
        "transition_pct": 0.0,
        "train": [("dataset/instruct_mixtures/crow-300m-final/flan_instruct/train", 1.0, _INSTRUCT)],
        "val": [("dataset/instruct_mixtures/crow-300m-final/flan_instruct/validation", 1.0, _INSTRUCT)],
    },
]
CROW_BASE_LRS = [3e-4, 1e-4, 5e-5]


def _triples(entries: list[DataEntry]) -> list[tuple[str, float, dict[str, Any] | None]]:
    return [(e.data_dir, e.weight, e.data_signature) for e in entries]


def _settings(dataset_config: Path, dataset_dir: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "dataset_config": str(dataset_config),
        "dataset_dir": str(dataset_dir),
        "stage_base_lrs": [3e-4, 1e-4, 5e-5],
        "block_size": 256,
        "prepare_num_workers": 1,
    }
    return Settings(**(base | overrides))


@pytest.fixture(scope="module")
def crow_cfg() -> DatasetConfig:
    return load_dataset_config(CROW_DATASET_YAML)


# --- resolve_entries: golden comparison with the thesis run config ---------------------------------------------------


def test_crow_entries_match_the_previous_run_config(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("dataset"))
    assert [s.name for s in crow_cfg.stages] == [g["name"] for g in GOLDEN_CROW_STAGES]
    for stage, golden in zip(crow_cfg.stages, GOLDEN_CROW_STAGES):
        train, val = resolve_entries(crow_cfg, layout, stage)
        assert _triples(train) == golden["train"], stage.name
        assert _triples(val) == golden["val"], stage.name
        assert (stage.tokens, stage.transition_pct) == (golden["tokens"], golden["transition_pct"])
        prefixes = [e.prefix for e in train] + [e.prefix for e in val]
        assert len(set(prefixes)) == len(prefixes), stage.name
        assert sum(e.weight for e in train) == pytest.approx(1.0)


def test_entry_prefixes_and_signatures(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("/data"))
    train, val = resolve_entries(crow_cfg, layout, crow_cfg.stages[2])
    assert train[0].prefix == "finetune-flan_instruct" and val[0].prefix == "finetune-flan_instruct-validation"
    assert train[0].data_dir == "/data/instruct_mixtures/crow-300m-final/flan_instruct/train"
    assert train[0].data_signature == INSTRUCT_DATA_SIGNATURE and train[0].data_signature is not INSTRUCT_DATA_SIGNATURE
    train1, val1 = resolve_entries(crow_cfg, layout, crow_cfg.stages[0])
    assert train1[0].prefix == "pretrain_phase1-fineweb_edu" and train1[0].data_signature is None
    assert val1[0].prefix == "pretrain_phase1-fineweb_val" and val1[0].data_dir == "/data/sources/fineweb_val/validation"


def test_instruct_mixture_train_split_is_explicit_or_default(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("d"))
    stage = crow_cfg.stages[2]
    stage_explicit = type(stage)(name="ft", tokens=1, train={"flan_instruct/train": 1.0}, val={"flan_instruct": 1.0})
    train, val = resolve_entries(crow_cfg, layout, stage_explicit)
    assert train[0].data_dir == val[0].data_dir == "d/instruct_mixtures/crow-300m-final/flan_instruct/train"
    assert train[0].prefix == "ft-flan_instruct-train" and val[0].prefix == "ft-flan_instruct"


# --- resolve_dataset -------------------------------------------------------------------------------------------------


def test_resolve_on_prepared_tiny_dataset(tiny_dataset_dir: Path, tiny_layout: DatasetLayout) -> None:
    resolved = resolve_dataset(_settings(TINY_DATASET_YAML, tiny_dataset_dir, auto_prepare=False))
    assert isinstance(resolved, ResolvedDataset) and resolved.config.name == "tiny"
    assert resolved.config_hash == load_dataset_config(TINY_DATASET_YAML).config_hash()
    assert resolved.tokenizer_dir == str(tiny_layout.tokenizer_dir("synthetic"))
    assert [s.name for s in resolved.stages] == ["pretrain_a", "pretrain_b", "finetune"]
    assert [s.base_lr for s in resolved.stages] == pytest.approx([3e-4, 1e-4, 5e-5])
    assert [s.tokens for s in resolved.stages] == [8192, 8192, 4096]
    assert all(isinstance(s, ResolvedStage) for s in resolved.stages)
    assert resolved.stages[0].train_data[0].data_dir == str(tiny_layout.source_dir("synthetic_pretrain", "processed"))
    assert resolved.stages[2].val_data[0].data_dir == str(tiny_layout.instruct_mixture_dir("tiny", "tiny_instruct", "validation"))
    for stage in resolved.stages:
        for entry in stage.train_data + stage.val_data:
            assert list(Path(entry.data_dir).glob("*.parquet")), entry


def test_stage_manager_stages(tiny_dataset_dir: Path) -> None:
    resolved = resolve_dataset(_settings(TINY_DATASET_YAML, tiny_dataset_dir, auto_prepare=False))
    stages = resolved.stage_manager_stages()
    assert len(stages) == 3 and all(isinstance(s, TrainingStage) for s in stages)
    ts = stages[2]
    assert (ts.name, ts.tokens, ts.base_lr, ts.transition_pct) == ("finetune", 4096, 5e-5, 0.0)
    assert ts.train_data == [vars(e) for e in resolved.stages[2].train_data]
    assert ts.train_data[0]["prefix"] == "finetune-tiny_instruct" and ts.train_data[0]["weight"] == 1.0
    assert ts.val_data[0]["data_signature"] == INSTRUCT_DATA_SIGNATURE


def test_auto_prepare_off_on_empty_dir_raises_with_build_command(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    settings = _settings(TINY_DATASET_YAML, empty, auto_prepare=False)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_dataset(settings)
    message = str(excinfo.value)
    assert build_command(str(TINY_DATASET_YAML), str(empty)) in message
    assert f"python data_preparation/prepare.py build --dataset_config {TINY_DATASET_YAML} --dataset_dir {empty}" in message
    assert "tokenizer: missing or stale" in message and "source synthetic_pretrain" in message
    assert not empty.exists() or not any(empty.iterdir())  # nothing was written


@pytest.mark.slow
def test_auto_prepare_builds_tiny_on_empty_dir(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    empty = tmp_path / "empty"
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        resolved = resolve_dataset(_settings(TINY_DATASET_YAML, empty))
    assert "preparing missing data" in caplog.text
    assert caplog.text.count("dataset status:") == 2  # once before the build (incomplete), once after it (complete)
    layout = DatasetLayout(empty)
    assert list(layout.source_dir("synthetic_pretrain", "processed").glob("*.parquet"))
    assert list(layout.validation_dir("synthetic_val").glob("*.parquet"))
    assert list(layout.instruct_mixture_dir("tiny", "tiny_instruct", "validation").glob("*.parquet"))
    assert Path(resolved.tokenizer_dir).is_dir()
    # a second resolve finds everything complete and does not build again
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        again = resolve_dataset(_settings(TINY_DATASET_YAML, empty, auto_prepare=False))
    assert "preparing missing data" not in caplog.text
    assert again.config_hash == resolved.config_hash


class _FakeBackend:
    def __init__(self, is_main: bool) -> None:
        self.is_main = is_main
        self.barriers = 0

    def barrier(self) -> None:
        self.barriers += 1


@pytest.mark.slow
def test_auto_prepare_builds_on_main_rank_only_and_barriers(tmp_path: Path) -> None:
    worker = _FakeBackend(is_main=False)
    with pytest.raises(RuntimeError, match="still incomplete after preparing"):
        resolve_dataset(_settings(TINY_DATASET_YAML, tmp_path / "worker"), worker)  # nobody built it
    assert worker.barriers == 1 and not (tmp_path / "worker" / "sources").exists()
    main = _FakeBackend(is_main=True)
    resolved = resolve_dataset(_settings(TINY_DATASET_YAML, tmp_path / "main"), main)
    assert main.barriers == 1 and Path(resolved.tokenizer_dir).is_dir()


def test_still_incomplete_after_build_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import training.data.dataset_resolver as resolver_module

    monkeypatch.setattr(resolver_module, "build", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="still incomplete after preparing") as excinfo:
        resolve_dataset(_settings(TINY_DATASET_YAML, tmp_path / "x"))
    assert "tokenizer: missing or stale" in str(excinfo.value)


# --- cross-checks ---------------------------------------------------------------------------------------------------


def test_base_lrs_length_mismatch_raises(tmp_path: Path, crow_cfg: DatasetConfig) -> None:
    settings = _settings(CROW_DATASET_YAML, tmp_path, stage_base_lrs=[1e-3, 1e-4], block_size=2048)
    with pytest.raises(ValueError, match="stage_base_lrs has 2 entries but dataset config .* has 3 stages"):
        validate_settings(settings, crow_cfg)
    with pytest.raises(ValueError, match="stage_base_lrs has 2 entries"):
        resolve_dataset(settings)  # checked before any data is touched


def test_block_size_above_max_seq_length_raises(tmp_path: Path, crow_cfg: DatasetConfig) -> None:
    settings = _settings(CROW_DATASET_YAML, tmp_path, block_size=4096)
    with pytest.raises(ValueError, match="block_size 4096 exceeds max_seq_length 2048"):
        validate_settings(settings, crow_cfg)
    validate_settings(_settings(CROW_DATASET_YAML, tmp_path, block_size=2048), crow_cfg)  # equal is fine


# --- resume hash check ----------------------------------------------------------------------------------------------


def test_checkpoint_hash_check(caplog: pytest.LogCaptureFixture) -> None:
    check_checkpoint_dataset_hash({CHECKPOINT_HASH_KEY: "abc"}, "abc", allow_change=False)
    with pytest.raises(RuntimeError, match="hash abc, the current dataset config hashes to xyz"):
        check_checkpoint_dataset_hash({CHECKPOINT_HASH_KEY: "abc"}, "xyz", allow_change=False)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_checkpoint_dataset_hash({CHECKPOINT_HASH_KEY: "abc"}, "xyz", allow_change=True)
    assert "allow_dataset_change" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_checkpoint_dataset_hash({"step": 3}, "xyz", allow_change=False)  # older checkpoint format
    assert "older format" in caplog.text
