# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the dataset resolver: golden comparison against the pre-dataset-config run config of the thesis run
(now with the validation-split row ranges), the split arithmetic and its disjointness on the tiny dataset, the
manifest/footer cross-check and the direct filesystem check, auto-prepare, the hard error without it, cross-checks
between run and dataset config, resume checks of the hash and of the stored split."""

import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import fields
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from data_preparation.dataset_config import (
    DatasetConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
    load_dataset_config,
)
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build import DatasetReport
from data_preparation.lib.storage.manifest import MANIFEST_NAME
from training.checkpoint import CheckpointMetadata
from training.data.dataset_resolver import (
    INSTRUCT_DATA_SIGNATURE,
    DataEntry,
    ResolvedDataset,
    ResolvedStage,
    build_command,
    check_dataset_unchanged,
    check_entries_on_disk,
    processed_rows,
    resolve_dataset,
    resolve_entries,
    resolve_splits,
    validate_settings,
    validation_rows_of,
)
from training.data.datasets import ParquetTextDataset
from training.settings import Settings
from training.stage_manager import TrainingStage

REPO_ROOT = Path(__file__).resolve().parents[2]
CROW_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "crow_300m_final.yaml"
TINY_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "tiny.yaml"

# The last hand-written run config (git history before the dataset-config restructure, `config/crow_300m_final.yaml`)
# as (data_dir, weight, data_signature, skip_rows, max_rows) per stage, with its per-stage data directories mapped onto
# the new layout: every source -> processed/<source>; fineweb's held-out set is the validation_fraction split of the
# same folder; the former flan_instruct mixture is the eight instruct sources with the mixture shares as weights, in
# train and val alike, each split by validation_fraction. The golden test needs no data on disk: every source is
# pretended to hold GOLDEN_ROWS processed rows, of which 5 % = GOLDEN_VAL_ROWS are validation rows when the source is
# used in train and val.
_P = "dataset/processed/{}"
_INSTRUCT = {"keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output"}
GOLDEN_ROWS = 1000
GOLDEN_VAL_ROWS = 50
Entry = tuple[str, float, dict[str, Any] | None, int, int | None]


def _train_only(name: str, weight: float) -> Entry:
    return (_P.format(name), weight, None, 0, None)


def _train_split(name: str, weight: float, signature: dict[str, Any] | None = None) -> Entry:
    return (_P.format(name), weight, signature, GOLDEN_VAL_ROWS, None)


def _val_split(name: str, weight: float, signature: dict[str, Any] | None = None) -> Entry:
    return (_P.format(name), weight, signature, 0, GOLDEN_VAL_ROWS)


_INSTRUCT_SHARES = [
    ("flan", 0.40),
    ("metamath", 0.15),
    ("orca_math", 0.10),
    ("evol_code", 0.125),
    ("code_alpaca", 0.025),
    ("slimorca", 0.10),
    ("sharegpt", 0.05),
    ("wizardlm", 0.05),
]
GOLDEN_CROW_STAGES: list[dict[str, Any]] = [
    {
        "name": "pretrain_phase1",
        "tokens": 3_300_000_000,
        "base_lr": 3e-4,
        "transition_pct": 0.10,
        "train": [
            _train_split("fineweb_edu", 0.65),
            _train_only("wikipedia", 0.09),
            _train_only("books_gutenberg", 0.06),
            _train_only("github_code_clean_python", 0.036),
            _train_only("github_code_clean_javascript", 0.024),
            _train_only("github_code_clean_typescript", 0.012),
            _train_only("github_code_clean_java", 0.012),
            _train_only("github_code_clean_cpp", 0.0096),
            _train_only("github_code_clean_go", 0.0084),
            _train_only("github_code_clean_rust", 0.006),
            _train_only("github_code_clean_shell", 0.0048),
            _train_only("github_code_clean_sql", 0.0036),
            _train_only("github_code_clean_html", 0.0036),
            _train_only("peso", 0.03),
            _train_only("arxiv", 0.02),
            _train_only("openwebmath", 0.03),
        ],
        "val": [_val_split("fineweb_edu", 1.0)],
    },
    {
        "name": "pretrain_phase2",
        "tokens": 1_500_000_000,
        "base_lr": 1e-4,
        "transition_pct": 0.10,
        "train": [
            _train_split("fineweb_edu", 0.35),
            _train_only("github_code_clean_python", 0.084),
            _train_only("github_code_clean_javascript", 0.056),
            _train_only("github_code_clean_typescript", 0.028),
            _train_only("github_code_clean_java", 0.028),
            _train_only("github_code_clean_cpp", 0.0224),
            _train_only("github_code_clean_go", 0.0196),
            _train_only("github_code_clean_rust", 0.014),
            _train_only("github_code_clean_shell", 0.0112),
            _train_only("github_code_clean_sql", 0.0084),
            _train_only("github_code_clean_html", 0.0084),
            _train_only("openwebmath", 0.088),
            _train_only("tinygsm", 0.066),
            _train_only("algebraic_stack", 0.044),
            _train_only("gsm8k", 0.022),
            _train_only("peso", 0.09),
            _train_only("arxiv", 0.06),
        ],
        "val": [_val_split("fineweb_edu", 1.0)],
    },
    {
        "name": "finetune",
        "tokens": 150_000_000,
        "base_lr": 5e-5,
        "transition_pct": 0.0,
        "train": [_train_split(name, weight, _INSTRUCT) for name, weight in _INSTRUCT_SHARES],
        "val": [_val_split(name, weight, _INSTRUCT) for name, weight in _INSTRUCT_SHARES],
    },
]
CROW_BASE_LRS = [3e-4, 1e-4, 5e-5]


def _ranges(entries: list[DataEntry]) -> list[Entry]:
    return [(e.data_dir, e.weight, e.data_signature, e.skip_rows, e.max_rows) for e in entries]


def _rows_in(directory: Path) -> int:
    """Rows of the `data-*.parquet` shards of a processed folder (parquet footers only)."""
    return sum(pq.read_metadata(path).num_rows for path in sorted(directory.glob("data-*.parquet")))


def _settings(dataset_config: Path, dataset_dir: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "dataset_config": str(dataset_config),
        "model_architecture_config": "config/model_architecture/tiny.yaml",
        "dataset_dir": str(dataset_dir),
        "stage_base_lrs": [3e-4, 1e-4, 5e-5],
        "block_size": 256,
        "prepare_num_workers": 1,
    }
    return Settings(**(base | overrides))


def _synthetic_config(**source_overrides: Any) -> DatasetConfig:
    """A config with a train-only source `a`, a source `both` used in train and val (with `source_overrides`) and a
    validation-only source `held` (`rows` given); framework for the split-arithmetic tests."""
    return DatasetConfig(
        name="t",
        tokenizer=TokenizerConfig(name="synthetic", kind="synthetic"),
        sources={
            "a": SourceConfig(kind="pretrain", loader="synthetic"),
            "both": SourceConfig(kind="pretrain", loader="synthetic", **source_overrides),
            "held": SourceConfig(kind="pretrain", loader="synthetic", rows=10),
        },
        stages=[StageConfig(name="s", tokens=100, train={"a": 0.5, "both": 0.5}, val={"both": 0.5, "held": 0.5})],
        block_size=8,
        max_seq_length=8,
    )


@pytest.fixture(scope="module")
def crow_cfg() -> DatasetConfig:
    return load_dataset_config(CROW_DATASET_YAML)


# --- resolve_entries: golden comparison with the thesis run config ---------------------------------------------------


def test_crow_entries_match_the_previous_run_config(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("dataset"))
    validation_rows = {name: validation_rows_of(crow_cfg, name, GOLDEN_ROWS) for name in crow_cfg.sources}
    assert validation_rows["fineweb_edu"] == GOLDEN_VAL_ROWS and validation_rows["flan"] == GOLDEN_VAL_ROWS
    assert validation_rows["wikipedia"] == 0 and validation_rows["gsm8k"] == 0
    assert [s.name for s in crow_cfg.stages] == [g["name"] for g in GOLDEN_CROW_STAGES]
    for stage, golden in zip(crow_cfg.stages, GOLDEN_CROW_STAGES):
        train, val = resolve_entries(crow_cfg, layout, stage, validation_rows)
        assert _ranges(train) == golden["train"], stage.name
        assert _ranges(val) == golden["val"], stage.name
        assert (stage.tokens, stage.transition_pct) == (golden["tokens"], golden["transition_pct"])
        for entries in (train, val):  # a source in both train and val has one prefix in each list, with disjoint ranges
            prefixes = [e.prefix for e in entries]
            assert len(set(prefixes)) == len(prefixes), stage.name
        assert sum(e.weight for e in train) == pytest.approx(1.0)


FRAMEWORK_NEUTRAL_MODULES = (
    "training.settings",
    "training.stage_manager",
    "training.lr_schedule",
    "training.data.dataset_resolver",
    "data_preparation.dataset_config",
    "data_preparation.layout",
)


def test_framework_neutral_modules_do_not_load_torch() -> None:
    """The JAX/TPU-port readiness claim, enforced: importing any of these modules must not pull in torch — also
    not through `training/data/__init__.py`, which therefore re-exports nothing."""
    lines = ["import importlib, sys"]
    for module in FRAMEWORK_NEUTRAL_MODULES:
        lines.append(f"importlib.import_module({module!r}); assert 'torch' not in sys.modules, {module!r}")
    subprocess.run([sys.executable, "-c", "\n".join(lines)], check=True, cwd=REPO_ROOT)


def test_same_source_same_split_in_every_stage(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("dataset"))
    validation_rows = {name: validation_rows_of(crow_cfg, name, GOLDEN_ROWS) for name in crow_cfg.sources}
    fineweb: dict[str, tuple[int, int | None]] = {}
    for stage in crow_cfg.stages[:2]:
        train, val = resolve_entries(crow_cfg, layout, stage, validation_rows)
        (train_entry,) = [e for e in train if e.data_dir.endswith("fineweb_edu")]
        (val_entry,) = [e for e in val if e.data_dir.endswith("fineweb_edu")]
        fineweb[stage.name] = (train_entry.skip_rows, train_entry.max_rows)
        assert (val_entry.skip_rows, val_entry.max_rows) == (0, GOLDEN_VAL_ROWS)
    assert fineweb["pretrain_phase1"] == fineweb["pretrain_phase2"] == (GOLDEN_VAL_ROWS, None)


def test_entry_prefixes_and_signatures(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("/data"))
    validation_rows = {name: validation_rows_of(crow_cfg, name, GOLDEN_ROWS) for name in crow_cfg.sources}
    train, val = resolve_entries(crow_cfg, layout, crow_cfg.stages[2], validation_rows)
    assert train[0].prefix == "finetune-flan" and val[0].prefix == "finetune-flan"
    assert train[0].data_dir == val[0].data_dir == "/data/processed/flan"
    assert train[0].data_signature == INSTRUCT_DATA_SIGNATURE and train[0].data_signature is not INSTRUCT_DATA_SIGNATURE
    assert (train[0].skip_rows, train[0].max_rows) == (GOLDEN_VAL_ROWS, None)
    assert (val[0].skip_rows, val[0].max_rows) == (0, GOLDEN_VAL_ROWS)
    train1, val1 = resolve_entries(crow_cfg, layout, crow_cfg.stages[0], validation_rows)
    assert train1[0].prefix == "pretrain_phase1-fineweb_edu" and train1[0].data_signature is None
    assert val1[0].prefix == "pretrain_phase1-fineweb_edu" and val1[0].data_dir == "/data/processed/fineweb_edu"
    assert (train1[1].prefix, train1[1].skip_rows, train1[1].max_rows) == ("pretrain_phase1-wikipedia", 0, None)


# --- the split arithmetic --------------------------------------------------------------------------------------------


def test_validation_rows_of_by_usage() -> None:
    cfg = _synthetic_config()
    assert [validation_rows_of(cfg, "a", n) for n in (0, 1, 7, 100)] == [0, 0, 0, 0]  # train only: all rows train
    assert [validation_rows_of(cfg, "held", n) for n in (0, 1, 7, 100)] == [0, 1, 7, 100]  # val only: all rows val
    assert [validation_rows_of(cfg, "both", n) for n in (0, 1, 19, 20, 21, 44, 77)] == [0, 1, 1, 1, 2, 3, 4]  # 5 %


def test_validation_rows_of_uses_the_per_source_override() -> None:
    cfg = _synthetic_config(validation_fraction=0.5)
    assert validation_rows_of(cfg, "both", 7) == 4 and validation_rows_of(cfg, "both", 8) == 4
    assert validation_rows_of(cfg, "a", 7) == 0  # the override of `both` does not leak


def test_validation_rows_of_multiplies_the_decimal_not_the_float() -> None:
    assert ceil(0.07 * 100) == 8  # the float pitfall the implementation avoids
    assert validation_rows_of(_synthetic_config(validation_fraction=0.07), "both", 100) == 7


def test_crow_split_fractions(crow_cfg: DatasetConfig) -> None:
    assert validation_rows_of(crow_cfg, "fineweb_edu", 100) == 5 and validation_rows_of(crow_cfg, "fineweb_edu", 1) == 1
    assert all(validation_rows_of(crow_cfg, name, 1000) == 50 for name, _ in _INSTRUCT_SHARES)
    assert all(validation_rows_of(crow_cfg, name, 1000) == 0 for name in ("wikipedia", "gsm8k", "tinygsm", "arxiv"))


def test_resolve_splits_on_the_tiny_dataset(tiny_dataset_config: DatasetConfig, tiny_layout: DatasetLayout) -> None:
    splits = resolve_splits(tiny_dataset_config, tiny_layout)
    assert set(splits) == {"synthetic_pretrain", "synthetic_instruct"}
    for name, validation in splits.items():
        total = _rows_in(tiny_layout.processed_dir(name))
        assert validation == ceil(Fraction("0.05") * total)
        assert 1 <= validation < total  # both parts of the split are non-empty


@pytest.mark.parametrize("name", ["synthetic_pretrain", "synthetic_instruct"])
def test_split_ranges_are_disjoint_and_complete(
    name: str, tiny_dataset_config: DatasetConfig, tiny_layout: DatasetLayout, tiny_dataset_dir: Path
) -> None:
    """Reading the validation and the training range through `ParquetTextDataset` yields every row of the folder
    exactly once: no row in both, counts add up."""
    resolved = resolve_dataset(_settings(TINY_DATASET_YAML, tiny_dataset_dir, auto_prepare=False))
    directory = tiny_layout.processed_dir(name)
    total = _rows_in(directory)
    k = resolved.validation_rows[name]
    signature = dict(INSTRUCT_DATA_SIGNATURE) if tiny_dataset_config.sources[name].kind == "instruct" else None
    keys = signature["keys"] if signature else ["text"]

    def rows(entry: DataEntry) -> list[tuple[str, ...]]:
        dataset = ParquetTextDataset(entry.data_dir, entry.prefix, entry.data_signature, skip_rows=entry.skip_rows, max_rows=entry.max_rows)
        return [tuple(str(row[key]) for key in keys) for row in dataset]

    val_entries = [e for s in resolved.stages for e in s.val_data if e.data_dir == str(directory)]
    train_entries = [e for s in resolved.stages for e in s.train_data if e.data_dir == str(directory)]
    assert val_entries and train_entries
    assert {(e.skip_rows, e.max_rows) for e in val_entries} == {(0, k)}  # the same split in every stage
    assert {(e.skip_rows, e.max_rows) for e in train_entries} == {(k, None)}
    assert all(e.data_signature == signature for e in val_entries + train_entries)
    val_rows, train_rows = rows(val_entries[0]), rows(train_entries[0])
    assert len(val_rows) == k and len(train_rows) == total - k
    assert not set(val_rows) & set(train_rows)
    assert len(set(val_rows) | set(train_rows)) == total  # every row exactly once (rows are unique after dedup)


# --- rows on disk: manifest cross-check and the direct filesystem check ------------------------------------------------


def test_processed_rows_cross_checks_the_manifest(tmp_path: Path, tiny_pretrain_dir: Path) -> None:
    folder = tmp_path / "processed" / "synthetic_pretrain"
    shutil.copytree(tiny_pretrain_dir, folder)
    total = processed_rows(folder, "x")
    assert total == _rows_in(tiny_pretrain_dir) > 0
    # an unlisted shard on disk (the planner verifies only the listed ones)
    shutil.copy(folder / "data-00000.parquet", folder / "data-00001.parquet")
    with pytest.raises(RuntimeError, match=rf"x: {MANIFEST_NAME} of {re.escape(str(folder))} lists {total} rows .* hold {2 * total}"):
        processed_rows(folder, "x")
    (folder / "data-00001.parquet").unlink()
    # a listed shard whose count is wrong
    manifest_path = folder / MANIFEST_NAME
    payload = json.loads(manifest_path.read_text())
    payload["shards"][0]["rows"] += 1
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match=f"lists {total + 1} rows .* hold {total}"):
        processed_rows(folder, "x")
    manifest_path.unlink()
    with pytest.raises(RuntimeError, match=f"has no {MANIFEST_NAME}"):
        processed_rows(folder, "x")


def test_processed_rows_missing_or_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=f"what: processed folder {re.escape(str(tmp_path / 'nope'))} does not exist"):
        processed_rows(tmp_path / "nope", "what")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / MANIFEST_NAME).write_text("{}")
    with pytest.raises(FileNotFoundError, match="holds no data-\\*.parquet shard"):
        processed_rows(tmp_path / "empty", "what")


def _stage(train: list[DataEntry], val: list[DataEntry]) -> ResolvedStage:
    return ResolvedStage(name="s", tokens=1, base_lr=1e-4, transition_pct=0.0, train_data=train, val_data=val)


def test_check_entries_on_disk_names_the_stage_key(tmp_path: Path, tiny_pretrain_dir: Path) -> None:
    good = str(tiny_pretrain_dir)
    total = _rows_in(tiny_pretrain_dir)
    check_entries_on_disk([_stage([DataEntry("s-a", good, skip_rows=total - 1)], [DataEntry("s-a", good, max_rows=1)])])
    check_entries_on_disk([_stage([DataEntry("s-a", good)], [DataEntry("s-a", good)])])  # full range twice is fine
    missing = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match=f"stage 's' train entry 's-a': processed folder {re.escape(missing)} does not exist"):
        check_entries_on_disk([_stage([DataEntry("s-a", missing)], [])])
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="stage 's' val entry 's-b': .* holds no data-\\*.parquet shard"):
        check_entries_on_disk([_stage([], [DataEntry("s-b", str(tmp_path / "empty"))])])
    with pytest.raises(RuntimeError, match=rf"stage 's' train entry 's-a': row range \[{total}, end\) of .* is empty \({total} rows on disk\); the training part"):
        check_entries_on_disk([_stage([DataEntry("s-a", good, skip_rows=total)], [])])
    with pytest.raises(RuntimeError, match=r"stage 's' val entry 's-a': row range \[0, 0\) of .* is empty .*; the validation part"):
        check_entries_on_disk([_stage([], [DataEntry("s-a", good, max_rows=0)])])


def test_resolve_dataset_checks_the_disk_independently_of_the_planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A planner that claims completeness does not save a run whose processed folder is missing."""
    import training.data.dataset_resolver as resolver_module

    monkeypatch.setattr(resolver_module, "status", lambda config_path, dataset_dir: DatasetReport(tokenizer_complete=True))
    root = tmp_path / "ds"
    expected_folder = re.escape(str(DatasetLayout(root).processed_dir("synthetic_pretrain")))
    with pytest.raises(FileNotFoundError, match=f"source 'synthetic_pretrain' \\(stage keys pretrain_a.train, pretrain_b.train, pretrain_a.val, pretrain_b.val\\): processed folder {expected_folder} does not exist"):
        resolve_dataset(_settings(TINY_DATASET_YAML, root, auto_prepare=False))


def test_an_unlisted_shard_is_reported_as_repairable_and_healed_by_auto_prepare(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """A shard the manifest does not list used to be a dead end: status said complete, training refused, repair saw
    nothing. Now the repair step owns it — without auto_prepare the run fails saying the dataset needs repair, with
    auto_prepare the derived folder is rebuilt and the run proceeds."""
    root = tmp_path / "ds"
    shutil.copytree(tiny_dataset_dir, root)
    folder = DatasetLayout(root).processed_dir("synthetic_instruct")
    shutil.copy(folder / "data-00000.parquet", folder / "data-00001.parquet")
    with pytest.raises(RuntimeError, match="not prepared .*auto_prepare is off.*synthetic_instruct"):
        resolve_dataset(_settings(TINY_DATASET_YAML, root, auto_prepare=False))
    resolved = resolve_dataset(_settings(TINY_DATASET_YAML, root, auto_prepare=True))
    assert isinstance(resolved, ResolvedDataset) and not (folder / "data-00001.parquet").exists()


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
    assert resolved.stages[0].train_data[0].data_dir == str(tiny_layout.processed_dir("synthetic_pretrain"))
    assert resolved.stages[2].val_data[0].data_dir == str(tiny_layout.processed_dir("synthetic_instruct"))
    assert resolved.validation_rows == resolve_splits(resolved.config, tiny_layout)
    for stage in resolved.stages:
        for entry in stage.train_data + stage.val_data:
            assert list(Path(entry.data_dir).glob("*.parquet")), entry


def test_data_entry_defaults() -> None:
    entry = DataEntry(prefix="p", data_dir="d")
    assert (entry.weight, entry.data_signature, entry.skip_rows, entry.max_rows) == (1.0, None, 0, None)


def test_training_stages(tiny_dataset_dir: Path) -> None:
    """`training_stages()` hands the stage manager the budget, LR and transition of every stage and nothing about
    the data (the entries stay in `ResolvedStage`)."""
    resolved = resolve_dataset(_settings(TINY_DATASET_YAML, tiny_dataset_dir, auto_prepare=False))
    stages = resolved.training_stages()
    assert len(stages) == 3 and all(isinstance(s, TrainingStage) for s in stages)
    assert stages == [
        TrainingStage(name=s.name, tokens=s.tokens, base_lr=s.base_lr, transition_pct=s.transition_pct) for s in resolved.stages
    ]
    assert (stages[2].name, stages[2].tokens, stages[2].base_lr, stages[2].transition_pct) == ("finetune", 4096, 5e-5, 0.0)
    assert {f.name for f in fields(TrainingStage)} == {"name", "tokens", "base_lr", "transition_pct"}


def test_auto_prepare_off_on_empty_dir_raises_with_build_command(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    settings = _settings(TINY_DATASET_YAML, empty, auto_prepare=False)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_dataset(settings)
    message = str(excinfo.value)
    assert build_command(str(TINY_DATASET_YAML), str(empty)) in message
    assert f"python data_preparation/prepare.py prepare --dataset_config {TINY_DATASET_YAML} --dataset_dir {empty}" in message
    assert "Missing: synthetic_pretrain, synthetic_instruct, tokenizer" in message and "raw missing" in message
    assert not empty.exists() or not any(empty.iterdir())  # nothing was written


@pytest.mark.slow
def test_auto_prepare_builds_tiny_on_empty_dir(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    empty = tmp_path / "empty"
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        resolved = resolve_dataset(_settings(TINY_DATASET_YAML, empty))
    assert "preparing missing data" in caplog.text
    assert caplog.text.count("dataset status:") == 2  # once before the build (incomplete), once after it (complete)
    assert re.search(r"source synthetic_pretrain: \d+ processed rows, rows \[0, \d+\) validation \(5.0%\)", caplog.text)
    layout = DatasetLayout(empty)
    assert list(layout.processed_dir("synthetic_pretrain").glob("*.parquet"))
    assert list(layout.processed_dir("synthetic_instruct").glob("*.parquet"))
    assert Path(resolved.tokenizer_dir).is_dir()
    assert resolved.validation_rows == resolve_splits(resolved.config, layout) and min(resolved.validation_rows.values()) >= 1
    # a second resolve finds everything complete and does not build again
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="data_preparation"):
        again = resolve_dataset(_settings(TINY_DATASET_YAML, empty, auto_prepare=False))
    assert "preparing missing data" not in caplog.text
    assert again.config_hash == resolved.config_hash and again.validation_rows == resolved.validation_rows


def test_auto_prepare_forwards_the_stop_request_to_the_build(tmp_path: Path) -> None:
    """`should_stop` (the CLI's Ctrl-C) reaches `prepare`: a request that already says stop ends the in-process build
    at its first shard with `BuildAborted` — the dataset stays incomplete, nothing is deleted."""
    empty = tmp_path / "empty"
    with pytest.raises(BuildAborted):
        resolve_dataset(_settings(TINY_DATASET_YAML, empty), should_stop=lambda: True)
    assert not DatasetLayout(empty).processed_dir("synthetic_pretrain").exists()


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


def test_still_incomplete_after_prepare_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import training.data.dataset_resolver as resolver_module

    monkeypatch.setattr(resolver_module, "prepare", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="still incomplete after preparing") as excinfo:
        resolve_dataset(_settings(TINY_DATASET_YAML, tmp_path / "x"))
    assert "tokenizer" in str(excinfo.value) and "synthetic_pretrain" in str(excinfo.value)


def test_auto_prepare_never_deletes_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The training run hands `prepare` `assume_yes=False` and a `confirm` that declines: a stale raw folder is a
    hard error naming the folder, never a silent re-download."""
    import training.data.dataset_resolver as resolver_module

    seen: dict[str, Any] = {}

    def record(*args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(resolver_module, "prepare", record)
    with pytest.raises(RuntimeError, match="stop here"):
        resolve_dataset(_settings(TINY_DATASET_YAML, tmp_path / "x"))
    assert seen["assume_yes"] is False and seen["confirm"]("delete?") is False


# --- cross-checks ---------------------------------------------------------------------------------------------------


def test_base_lrs_length_mismatch_raises(tmp_path: Path, crow_cfg: DatasetConfig) -> None:
    settings = _settings(CROW_DATASET_YAML, tmp_path, stage_base_lrs=[1e-3, 1e-4], block_size=2048)
    with pytest.raises(ValueError, match="stage_base_lrs has 2 entries but dataset config .* has 3 stages"):
        validate_settings(settings, crow_cfg)
    with pytest.raises(ValueError, match="stage_base_lrs has 2 entries"):
        resolve_dataset(settings)  # checked before any data is touched


@pytest.mark.parametrize("block_size", [4096, 256])
def test_block_size_mismatch_raises_naming_both_files(tmp_path: Path, crow_cfg: DatasetConfig, block_size: int) -> None:
    settings = _settings(CROW_DATASET_YAML, tmp_path, block_size=block_size)
    expected = f"block_size {block_size} of the run config does not match block_size 2048 of dataset config {str(CROW_DATASET_YAML)!r}"
    with pytest.raises(ValueError, match=re.escape(expected)):
        validate_settings(settings, crow_cfg)
    with pytest.raises(ValueError, match=re.escape(expected)):
        resolve_dataset(settings)  # checked before any data is touched (tmp_path is empty)
    validate_settings(_settings(CROW_DATASET_YAML, tmp_path, block_size=2048), crow_cfg)  # equal is fine


# --- resume checks --------------------------------------------------------------------------------------------------


def _resolved(config_hash: str, validation_rows: dict[str, int]) -> ResolvedDataset:
    return ResolvedDataset(
        config=load_dataset_config(TINY_DATASET_YAML),
        config_hash=config_hash,
        tokenizer_dir="unused",
        stages=[],
        validation_rows=validation_rows,
    )


def _metadata(config_hash: str, validation_rows: dict[str, int]) -> CheckpointMetadata:
    return CheckpointMetadata(
        step=3,
        stage=0,
        rng={},
        settings={},
        model_config={},
        dataset_config_hash=config_hash,
        validation_rows=validation_rows,
        data_stream={},
    )


def test_check_dataset_unchanged_passes_on_identical_hash_and_split() -> None:
    check_dataset_unchanged(_metadata("abc", {"a": 3, "b": 0}), _resolved("abc", {"a": 3, "b": 0}), allow_change=False)


def test_check_dataset_unchanged_hash_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    rows = {"a": 3}
    with pytest.raises(RuntimeError, match="hash abc, the current dataset config hashes to xyz") as excinfo:
        check_dataset_unchanged(_metadata("abc", rows), _resolved("xyz", rows), allow_change=False)
    assert "allow_dataset_change: true" in str(excinfo.value) and "validation split" not in str(excinfo.value)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_dataset_unchanged(_metadata("abc", rows), _resolved("xyz", rows), allow_change=True)
    assert "hash abc" in caplog.text and "allow_dataset_change" in caplog.text


def test_check_dataset_unchanged_validation_rows_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    expected = {"a": 3, "b": 0}
    with pytest.raises(RuntimeError, match="validation rows per source: 'a': checkpoint 4, now 3\\)") as excinfo:
        check_dataset_unchanged(_metadata("h", {"a": 4, "b": 0}), _resolved("h", expected), allow_change=False)
    assert "'b'" not in str(excinfo.value) and "allow_dataset_change: true" in str(excinfo.value)
    assert "config hash" not in str(excinfo.value)  # only the differing part is reported
    with pytest.raises(RuntimeError, match="'b': checkpoint 0, now absent; 'c': checkpoint absent, now 1"):
        check_dataset_unchanged(_metadata("h", {"a": 3, "b": 0}), _resolved("h", {"a": 3, "c": 1}), allow_change=False)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_dataset_unchanged(_metadata("h", {"a": 4, "b": 0}), _resolved("h", expected), allow_change=True)
    assert "'a': checkpoint 4, now 3" in caplog.text and "allow_dataset_change" in caplog.text


def test_check_dataset_unchanged_reports_both_differences_at_once() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        check_dataset_unchanged(_metadata("abc", {"a": 4}), _resolved("xyz", {"a": 3}), allow_change=False)
    message = str(excinfo.value)
    assert "hash abc, the current dataset config hashes to xyz; the validation split differs" in message
    assert "'a': checkpoint 4, now 3" in message
