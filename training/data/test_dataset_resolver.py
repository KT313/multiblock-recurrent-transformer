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
from data_preparation.lib.build.planner import DatasetReport
from data_preparation.lib.storage.manifest import MANIFEST_NAME
from training.checkpoint import CheckpointMetadata
from training.data.dataset_resolver import (
    INSTRUCT_DATA_SIGNATURE,
    DataEntry,
    ResolvedDataset,
    ResolvedStage,
    build_command,
    check_dataset_unchanged,
    check_entries,
    check_validation_batches,
    entry_rows_in_range,
    loader_shards,
    processed_row_counts,
    processed_rows,
    resolve_dataset,
    resolve_splits,
    resolve_train_sources,
    resolve_val_entries,
    validate_settings,
    validation_batches_available,
    validation_rows_of,
)
from training.data.datasets import ParquetTextDataset
from training.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
CROW_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "crow_300m_final.yaml"
TINY_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "tiny.yaml"

# The last hand-written run config (git history before the dataset-config restructure, `config/crow_300m_final.yaml`)
# with its per-stage data directories mapped onto the new layout: every source -> processed/<source>; fineweb's
# held-out set is the validation_fraction split of the same folder; the former flan_instruct mixture is the eight
# instruct sources with the mixture shares as weights, in train and val alike, each split by validation_fraction.
# Since the continuous-stream design, training reads every source through ONE run-wide reader
# (`ResolvedDataset.train_sources`), so the per-stage train side is the WEIGHTS dict and the row ranges live on the
# run-wide source entries. The golden test needs no data on disk: every source is pretended to hold GOLDEN_ROWS
# processed rows, of which 5 % = GOLDEN_VAL_ROWS are validation rows when the source is used in train and val.
_P = "dataset/processed/{}"
_INSTRUCT = {"keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output"}
GOLDEN_ROWS = 1000
GOLDEN_VAL_ROWS = 50
Entry = tuple[str, float, dict[str, Any] | None, int, int | None]


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
        "train": {
            "fineweb_edu": 0.65,
            "wikipedia": 0.09,
            "books_gutenberg": 0.06,
            "github_code_clean_python": 0.036,
            "github_code_clean_javascript": 0.024,
            "github_code_clean_typescript": 0.012,
            "github_code_clean_java": 0.012,
            "github_code_clean_cpp": 0.0096,
            "github_code_clean_go": 0.0084,
            "github_code_clean_rust": 0.006,
            "github_code_clean_shell": 0.0048,
            "github_code_clean_sql": 0.0036,
            "github_code_clean_html": 0.0036,
            "peso": 0.03,
            "arxiv": 0.02,
            "openwebmath": 0.03,
        },
        "val": [_val_split("fineweb_edu", 1.0)],
    },
    {
        "name": "pretrain_phase2",
        "tokens": 1_500_000_000,
        "base_lr": 1e-4,
        "transition_pct": 0.10,
        "train": {
            "fineweb_edu": 0.35,
            "github_code_clean_python": 0.084,
            "github_code_clean_javascript": 0.056,
            "github_code_clean_typescript": 0.028,
            "github_code_clean_java": 0.028,
            "github_code_clean_cpp": 0.0224,
            "github_code_clean_go": 0.0196,
            "github_code_clean_rust": 0.014,
            "github_code_clean_shell": 0.0112,
            "github_code_clean_sql": 0.0084,
            "github_code_clean_html": 0.0084,
            "openwebmath": 0.088,
            "tinygsm": 0.066,
            "algebraic_stack": 0.044,
            "gsm8k": 0.022,
            "peso": 0.09,
            "arxiv": 0.06,
        },
        "val": [_val_split("fineweb_edu", 1.0)],
    },
    {
        "name": "finetune",
        "tokens": 150_000_000,
        "base_lr": 5e-5,
        "transition_pct": 0.0,
        "train": dict(_INSTRUCT_SHARES),
        "val": [_val_split(name, weight, _INSTRUCT) for name, weight in _INSTRUCT_SHARES],
    },
]
# sources used in train AND val: their run-wide reader starts after the GOLDEN_VAL_ROWS held-out rows
GOLDEN_SPLIT_SOURCES = {"fineweb_edu", *(name for name, _ in _INSTRUCT_SHARES)}
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


# --- golden comparison with the thesis run config --------------------------------------------------------------------


def test_crow_entries_match_the_previous_run_config(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("dataset"))
    validation_rows = {name: validation_rows_of(crow_cfg, name, GOLDEN_ROWS) for name in crow_cfg.sources}
    assert validation_rows["fineweb_edu"] == GOLDEN_VAL_ROWS and validation_rows["flan"] == GOLDEN_VAL_ROWS
    assert validation_rows["wikipedia"] == 0 and validation_rows["gsm8k"] == 0
    assert [s.name for s in crow_cfg.stages] == [g["name"] for g in GOLDEN_CROW_STAGES]
    for stage, golden in zip(crow_cfg.stages, GOLDEN_CROW_STAGES):
        assert stage.train == golden["train"], stage.name
        val = resolve_val_entries(crow_cfg, layout, stage, validation_rows)
        assert _ranges(val) == golden["val"], stage.name
        assert (stage.tokens, stage.transition_pct) == (golden["tokens"], golden["transition_pct"])
        prefixes = [e.prefix for e in val]
        assert len(set(prefixes)) == len(prefixes), stage.name
        assert sum(stage.train.values()) == pytest.approx(1.0)


def test_crow_train_sources_are_one_run_wide_reader_per_source(crow_cfg: DatasetConfig) -> None:
    """The train side of the thesis mixture as the continuous-stream design reads it: one entry per source used in
    training, in config order, prefix = the plain source name, range = validation holdout -> end (so stages sharing
    a source never re-read rows)."""
    layout = DatasetLayout(Path("dataset"))
    validation_rows = {name: validation_rows_of(crow_cfg, name, GOLDEN_ROWS) for name in crow_cfg.sources}
    sources = resolve_train_sources(crow_cfg, layout, validation_rows)
    assert [e.prefix for e in sources] == [n for n in crow_cfg.sources if crow_cfg.used_in_train(n)]  # config order
    assert {e.prefix for e in sources} == {name for golden in GOLDEN_CROW_STAGES for name in golden["train"]}
    instruct_names = {name for name, _ in _INSTRUCT_SHARES}
    for entry in sources:
        assert entry.data_dir == _P.format(entry.prefix)
        held_out = GOLDEN_VAL_ROWS if entry.prefix in GOLDEN_SPLIT_SOURCES else 0
        assert (entry.skip_rows, entry.max_rows) == (held_out, None), entry.prefix
        assert entry.data_signature == (_INSTRUCT if entry.prefix in instruct_names else None), entry.prefix


# Not in this list: `model.config`, the known gap — it is framework-neutral by intent but imports
# `model/layers/init.py` for its `Init` object, and that imports torch (checked: `import model.config` loads torch).
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
    """fineweb_edu is validated on in both pretrain stages and trained on through one run-wide reader: every
    stage's validation entry reads the same held-out rows `[0, GOLDEN_VAL_ROWS)` and the single train source
    starts right after them."""
    layout = DatasetLayout(Path("dataset"))
    validation_rows = {name: validation_rows_of(crow_cfg, name, GOLDEN_ROWS) for name in crow_cfg.sources}
    for stage in crow_cfg.stages[:2]:
        val = resolve_val_entries(crow_cfg, layout, stage, validation_rows)
        (val_entry,) = [e for e in val if e.data_dir.endswith("fineweb_edu")]
        assert (val_entry.skip_rows, val_entry.max_rows) == (0, GOLDEN_VAL_ROWS)
    (train_entry,) = [
        e for e in resolve_train_sources(crow_cfg, layout, validation_rows) if e.data_dir.endswith("fineweb_edu")
    ]
    assert (train_entry.skip_rows, train_entry.max_rows) == (GOLDEN_VAL_ROWS, None)


def test_entry_prefixes_and_signatures(crow_cfg: DatasetConfig) -> None:
    layout = DatasetLayout(Path("/data"))
    validation_rows = {name: validation_rows_of(crow_cfg, name, GOLDEN_ROWS) for name in crow_cfg.sources}
    val = resolve_val_entries(crow_cfg, layout, crow_cfg.stages[2], validation_rows)
    assert val[0].prefix == "finetune-flan" and val[0].data_dir == "/data/processed/flan"
    assert val[0].data_signature == INSTRUCT_DATA_SIGNATURE and val[0].data_signature is not INSTRUCT_DATA_SIGNATURE
    assert (val[0].skip_rows, val[0].max_rows) == (0, GOLDEN_VAL_ROWS)
    val1 = resolve_val_entries(crow_cfg, layout, crow_cfg.stages[0], validation_rows)
    assert val1[0].prefix == "pretrain_phase1-fineweb_edu" and val1[0].data_dir == "/data/processed/fineweb_edu"
    assert val1[0].data_signature is None
    sources = resolve_train_sources(crow_cfg, layout, validation_rows)
    flan = next(e for e in sources if e.prefix == "flan")
    assert flan.data_dir == "/data/processed/flan" and (flan.skip_rows, flan.max_rows) == (GOLDEN_VAL_ROWS, None)
    assert flan.data_signature == INSTRUCT_DATA_SIGNATURE and flan.data_signature is not INSTRUCT_DATA_SIGNATURE
    wikipedia = next(e for e in sources if e.prefix == "wikipedia")
    assert (wikipedia.skip_rows, wikipedia.max_rows) == (0, None) and wikipedia.data_signature is None


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


def test_processed_row_counts_reads_every_source_once_by_directory(
    tiny_dataset_config: DatasetConfig, tiny_layout: DatasetLayout
) -> None:
    rows = processed_row_counts(tiny_dataset_config, tiny_layout)
    assert set(rows) == {str(tiny_layout.processed_dir(name)) for name in ("synthetic_pretrain", "synthetic_instruct")}
    assert all(count == processed_rows(Path(directory), "x") for directory, count in rows.items())


def test_resolve_splits_on_the_tiny_dataset(tiny_dataset_config: DatasetConfig, tiny_layout: DatasetLayout) -> None:
    splits = resolve_splits(tiny_dataset_config, tiny_layout, processed_row_counts(tiny_dataset_config, tiny_layout))
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
    train_entries = [e for e in resolved.train_sources if e.data_dir == str(directory)]
    assert val_entries and len(train_entries) == 1  # ONE run-wide train reader per source
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


def _stage(val: list[DataEntry]) -> ResolvedStage:
    return ResolvedStage(name="s", tokens=1, base_lr=1e-4, transition_pct=0.0, train_weights={}, val_data=val)


def test_check_entries_names_the_entry_with_an_empty_range(tiny_pretrain_dir: Path) -> None:
    """The row counts come from `processed_row_counts` (which already refused a missing or shard-less folder), so
    the check is pure arithmetic over the mapping: every entry's range must hold a row."""
    good = str(tiny_pretrain_dir)
    total = _rows_in(tiny_pretrain_dir)
    rows = {good: total}
    check_entries([DataEntry("a", good, skip_rows=total - 1)], [_stage([DataEntry("s-a", good, max_rows=1)])], rows)
    check_entries([DataEntry("a", good)], [_stage([DataEntry("s-a", good)])], rows)  # full range twice is fine
    # an empty training range is the error that makes the stream's restart-on-exhaustion safe: a source that runs
    # dry mid-run is restarted, which would spin forever on a range without a single row
    with pytest.raises(RuntimeError, match=rf"train source 'a': row range \[{total}, end\) of .* is empty \({total} rows on disk\); the training part"):
        check_entries([DataEntry("a", good, skip_rows=total)], [], rows)
    with pytest.raises(RuntimeError, match=r"stage 's' val entry 's-a': row range \[0, 0\) of .* is empty .*; the validation part"):
        check_entries([], [_stage([DataEntry("s-a", good, max_rows=0)])], rows)


# --- one row per dataloader worker shard ------------------------------------------------------------------------------


def test_loader_shards_counts_workers_and_ranks() -> None:
    """`num_workers=0` loads in the calling process: one shard per rank, not zero."""
    assert loader_shards(0, 1) == 1
    assert loader_shards(0, 4) == 4
    assert loader_shards(1, 1) == 1
    assert loader_shards(8, 1) == 8
    assert loader_shards(8, 2) == 16


def test_entry_rows_in_range_clips_to_the_rows_on_disk(tiny_pretrain_dir: Path) -> None:
    total = _rows_in(tiny_pretrain_dir)
    good = str(tiny_pretrain_dir)
    assert entry_rows_in_range(DataEntry("s-a", good), total) == total
    assert entry_rows_in_range(DataEntry("s-a", good, skip_rows=2), total) == total - 2
    assert entry_rows_in_range(DataEntry("s-a", good, max_rows=3), total) == 3
    assert entry_rows_in_range(DataEntry("s-a", good, max_rows=total + 100), total) == total
    assert entry_rows_in_range(DataEntry("s-a", good, skip_rows=total + 5), total) == 0


def test_check_entry_shards_fails_when_a_source_is_smaller_than_the_world(tiny_pretrain_dir: Path) -> None:
    """An entry with fewer rows than its loader has shards leaves a shard empty; the restart of an exhausted
    loader (or mixture member) then gets a second `StopIteration` and the run dies mid-training — so it is a setup
    error naming the entry, its rows and the shard count. Train loaders run one worker per source and validation
    loaders in-process, so with one device everything is a single shard and only a larger world can starve one."""
    good = str(tiny_pretrain_dir)
    total = _rows_in(tiny_pretrain_dir)
    rows = {good: total}
    train = [DataEntry("a", good), DataEntry("narrow", good, skip_rows=total - 2)]
    stage = _stage([DataEntry("s-a", good, max_rows=1)])
    check_entries(train, [stage], rows)  # world size 1: one shard per loader, whatever the row count
    check_entries(train, [stage], rows, world_size=1)
    # the training range is what counts, not the folder: the validation split narrows `narrow` to 2 rows
    check_entries([DataEntry("narrow", good, skip_rows=total - 2)], [], rows, world_size=2)
    with pytest.raises(
        ValueError,
        match=(
            r"train source 'narrow': .* gives it 2 row\(s\) after the validation split, but its loader deals the "
            r"rows round-robin over 3 shards \(1 dataloader worker\(s\) × world size 3\), so 1 shard\(s\) would be "
            r"empty and the run would fail during training\. Give the source more rows, or lower the world size"
        ),
    ):
        check_entries([DataEntry("narrow", good, skip_rows=total - 2)], [], rows, world_size=3)
    # validation loaders read in-process (`num_workers=0`): one shard per rank
    val_only = _stage([DataEntry("s-b", good, max_rows=2)])
    check_entries([], [val_only], rows, world_size=2)
    with pytest.raises(
        ValueError,
        match=r"stage 's' val entry 's-b': .*over 4 shards \(0 dataloader worker\(s\) × world size 4\).*lower the world size",
    ):
        check_entries([], [val_only], rows, world_size=4)


# --- the validation data an evaluation needs --------------------------------------------------------------------------


def test_validation_batches_available_counts_only_finite_loaders(tiny_pretrain_dir: Path) -> None:
    """A one-entry validation loader is one finite epoch over its row range (`ceil(rows / micro_batch_size)`
    batches, the rows dealt over `world_size` shards); a loader over several entries mixes them through
    `WeightedMixtureDataset`, which restarts exhausted members and therefore never runs out (`None`)."""
    good = str(tiny_pretrain_dir)
    total = _rows_in(tiny_pretrain_dir)
    rows = {good: total}
    assert validation_batches_available([DataEntry("s-a", good, max_rows=7)], rows, micro_batch_size=2, world_size=1) == 4
    assert validation_batches_available([DataEntry("s-a", good, max_rows=8)], rows, micro_batch_size=2, world_size=1) == 4
    assert validation_batches_available([DataEntry("s-a", good, max_rows=8)], rows, micro_batch_size=2, world_size=4) == 1
    assert validation_batches_available([DataEntry("s-a", good, max_rows=3)], rows, micro_batch_size=2, world_size=4) == 0
    assert validation_batches_available([DataEntry("s-a", good)], rows, micro_batch_size=1, world_size=1) == total
    assert validation_batches_available([DataEntry("s-a", good, skip_rows=total - 1)], rows, 4, 1) == 1  # a short last batch
    mixture = [DataEntry("s-a", good, max_rows=1), DataEntry("s-b", good, max_rows=1)]
    assert validation_batches_available(mixture, rows, micro_batch_size=4, world_size=1) is None  # restarts, never short


def test_check_validation_batches_fails_at_setup_on_a_split_without_one_batch(
    tiny_pretrain_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Too little validation data is caught at setup, not at the first evaluation step: no batch at all is a hard
    error naming the stage, the entries and both numbers; fewer batches than `eval_iters` is a warning (`evaluate`
    averages the batches it gets), and enough data passes silently."""
    good = str(tiny_pretrain_dir)
    rows = {good: _rows_in(tiny_pretrain_dir)}
    stage = _stage([DataEntry("s-a", good, max_rows=4)])
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_validation_batches([stage], rows, micro_batch_size=2, eval_iters=2)  # exactly eval_iters batches
    assert caplog.text == ""
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_validation_batches([stage], rows, micro_batch_size=8, eval_iters=1)  # one short batch is still a batch
    assert caplog.text == ""
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_validation_batches([stage], rows, micro_batch_size=2, eval_iters=5)
    assert "stage s: its validation data (s-a) yields 2 micro-batch(es) of 2 rows, fewer than eval_iters (5)" in caplog.text
    # nothing at all reaches a rank (here: 4 rows dealt over 8 ranks, the last two get none) is the hard error
    with pytest.raises(RuntimeError, match=r"stage 's': its validation data \(s-a\) yields 0 micro-batches of 2 rows per rank \(world size 8\) but eval_iters is 1, so evaluation"):
        check_validation_batches([stage], rows, micro_batch_size=2, eval_iters=1, world_size=8)
    # a validation loader that mixes several sources restarts them and is never short, whatever the row counts are
    mixed = _stage([DataEntry("s-a", good, max_rows=1), DataEntry("s-b", good, max_rows=1)])
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        caplog.clear()
        check_validation_batches([mixed], rows, micro_batch_size=8, eval_iters=50)
    assert caplog.text == ""


def test_resolve_dataset_warns_about_the_short_tiny_finetune_split(
    tiny_dataset_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The tiny dataset's finetune validation split is a couple of rows — enough for one micro-batch (the run is
    fine, `evaluate` averages what it gets) but fewer than `eval_iters` batches, so the resolver says so."""
    settings = _settings(TINY_DATASET_YAML, tiny_dataset_dir, auto_prepare=False, micro_batch_size=2, eval_iters=2)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        resolved = resolve_dataset(settings)
    assert resolved.validation_rows["synthetic_instruct"] < 2 * 2  # fewer rows than eval_iters micro-batches
    assert "stage finetune: its validation data (finetune-synthetic_instruct) yields 1 micro-batch(es)" in caplog.text
    assert "stage pretrain_a" not in caplog.text  # the pretrain split is long enough


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
    assert [s.train_weights for s in resolved.stages] == [
        {"synthetic_pretrain": 1.0},
        {"synthetic_pretrain": 1.0},
        {"synthetic_instruct": 1.0},
    ]
    assert [e.prefix for e in resolved.train_sources] == ["synthetic_pretrain", "synthetic_instruct"]  # config order
    assert resolved.train_sources[0].data_dir == str(tiny_layout.processed_dir("synthetic_pretrain"))
    assert resolved.stages[2].val_data[0].data_dir == str(tiny_layout.processed_dir("synthetic_instruct"))
    assert resolved.validation_rows == resolve_splits(resolved.config, tiny_layout, resolved.rows_on_disk)
    assert resolved.rows_on_disk == processed_row_counts(resolved.config, tiny_layout)
    for entry in resolved.train_sources + [e for stage in resolved.stages for e in stage.val_data]:
        assert list(Path(entry.data_dir).glob("*.parquet")), entry


def test_data_entry_defaults() -> None:
    entry = DataEntry(prefix="p", data_dir="d")
    assert (entry.weight, entry.data_signature, entry.skip_rows, entry.max_rows) == (1.0, None, 0, None)


def test_resolved_stages_carry_the_schedule_of_every_stage(tiny_dataset_dir: Path) -> None:
    """`ResolvedDataset.stages` is what the stage manager takes: budget, LR, transition and sampling weights per
    stage next to the validation entries."""
    resolved = resolve_dataset(_settings(TINY_DATASET_YAML, tiny_dataset_dir, auto_prepare=False))
    stages = resolved.stages
    assert len(stages) == 3 and all(isinstance(s, ResolvedStage) for s in stages)
    assert (stages[2].name, stages[2].tokens, stages[2].base_lr, stages[2].transition_pct) == ("finetune", 4096, 5e-5, 0.0)
    assert stages[2].train_weights == {"synthetic_instruct": 1.0}


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
    assert resolved.validation_rows == resolve_splits(resolved.config, layout, resolved.rows_on_disk) and min(resolved.validation_rows.values()) >= 1
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
        self.world_size = 1  # the resolver reads it for `check_validation_batches`
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
        train_sources=[],
        validation_rows=validation_rows,
        rows_on_disk={},
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
