# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset config schema: the single definition of a dataset (`config/datasets/<name>.yaml`).

Framework-neutral (no torch). Loaded by `data_preparation/prepare.py` (materialises it under `dataset/`) and by
`training/train.py` (verifies / auto-prepares it and derives the per-stage data mixtures). The run config only
references the file; every data-related setting lives here. This file is the config reference: every field
carries a trailing comment, `DatasetConfig` lists the top-level keys.

Layout produced on disk (see `data_preparation/README.md`):

    dataset/sources/<source>/raw/     rows as downloaded (text truncated to max_seq_length tokens); shared, append-only
    dataset/processed/<source>/       rows after cleaning (what training reads); derived from raw, shared
    dataset/tokenizers/<name>/

Mixing (stage weights) and the validation split (`validation_fraction`) happen in the training dataloader; the
pipeline only downloads and cleans one folder per source.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any, Literal, Optional

from jsonargparse import ArgumentError, ArgumentParser

SourceKind = Literal["pretrain", "instruct"]
LoaderName = Literal["hf_files", "hf_split", "hf_stream", "github_code", "local", "synthetic"]
DedupMode = Literal["none", "exact", "minhash"]
TokenCountMode = Literal["tokenizer", "estimate"]

HUB_LOADERS: tuple[str, ...] = ("hf_files", "hf_split", "hf_stream", "github_code")  # loaders that need `hf_id`

DEFAULT_BENCHMARKS = [
    "gsm8k_test",
    "math_test",
    "humaneval",
    "mbpp_test",
    "arc_challenge_test",
    "hellaswag_test",
    "mmlu_test",
    "winogrande_test",
]
SAFETY_MARGIN = Fraction("1.2")  # rows downloaded = sequence budget × this (covers what the length filter / dedup drop); a Fraction so 50 × 1.2 is exactly 60

SHUFFLED_BUILD_MAX_ROWS = 1_000_000  # a shuffled source is built all-at-once in memory; `rows_needed` above this fails at config load
# derivation, keep as a comment: processed rows are TEXT bounded by max_seq_length tokens at download
# (~8-10 KB/row worst case), so 1M rows is a worst case of ~10 GB held once; typical instruct rows are far smaller.

# Top-level / source keys of the pre-restructure schema, with the hint shown when a YAML still uses them.
REMOVED_KEYS: dict[str, str] = {
    "instruct_mixtures": "mixing is the training dataloader's job: list the instruct sources with weights in the stage",
    "validation_tokens": "the validation split is made at training time (`validation_fraction`)",
    "max_chars": "rows are truncated to `max_seq_length` tokens when downloaded",
    "max_tokens": "rows longer than `max_seq_length` tokens are dropped when downloaded",
    "tokens_per_row_estimate": "the planner counts sequences; `describe_tokens_per_row` feeds only the describe table",
}


# --- hash annotations --------------------------------------------------------------------------------------------------
#
# Every field of every config dataclass declares which of the three manifest hashes it belongs to, once, right where
# it is defined; `hash_payload` walks the annotations and `raw_hash` / `processed_hash` / `config_hash` assemble
# their payloads from it. Before this, each hash built the dict of all non-default fields and then popped a
# hand-maintained list of names — a table that drifted from the field list three times.
#
#   raw       identity of the downloaded rows: which rows a loader yields, in what order, and how their stored token
#             counts are made. Keys `sources/<s>/raw/`, the bandwidth-expensive tree: a change makes it *stale*, so
#             it is deleted (after confirmation) and downloaded again.
#   processed derives `processed/<s>/` from the raw shards: a change rebuilds that folder, nothing is downloaded.
#   config    everything else that defines the training data; only `config_hash` (recorded in checkpoints so a resume
#             against different data is detected) counts it.
#   none      not hashed at all: how rows are fetched or described, resource knobs — nothing that changes the data.
#
# `raw` and `processed` select exactly their own fields (`processed_hash` folds the raw hash in as one value, so raw
# fields must not appear a second time); `config` counts every field annotated raw, processed *or* config.
HashName = Literal["raw", "processed", "config"]
HASH_ANNOTATIONS: tuple[str, ...] = ("raw", "processed", "config", "none")

_RAW: dict[str, Any] = {"hash": "raw"}
_PROCESSED: dict[str, Any] = {"hash": "processed"}
_CONFIG: dict[str, Any] = {"hash": "config"}
_UNHASHED: dict[str, Any] = {"hash": "none"}
# `max_cached_file_mb` says whether a Hub file is cached whole or read remotely by piece: traffic, not rows.
_LOAD_KWARGS: dict[str, Any] = {**_RAW, "hash_drop": ("max_cached_file_mb",)}


def _seed_hash(source: SourceConfig) -> str:
    """`seed` is loader identity — it generates the rows themselves — only for `loader: synthetic`; for every other
    loader it drives the build-time input inversions and the shuffle order, so it belongs to the processed hash."""
    return "raw" if source.loader == "synthetic" else "processed"


def _normalize_hash(dedup: DedupConfig) -> str:
    """`normalize` changes the hashed text of every mode that hashes at all (`exact` and the exact pass of
    `minhash`); with `mode: none` nothing is hashed and it cannot change a result."""
    return "none" if dedup.mode == "none" else "processed"


def _minhash_only(dedup: DedupConfig) -> str:
    """The Jaccard threshold, the permutation count and the n-gram size only change a MinHash/LSH result."""
    return "processed" if dedup.mode == "minhash" else "none"


@dataclass
class TokenizerConfig:
    """Which tokenizer defines "a token" for this dataset; saved to `dataset/tokenizers/<name>/`."""

    name: str = field(metadata=_RAW)  # directory name under `dataset/tokenizers/`
    kind: Literal["hf", "synthetic"] = field(default="hf", metadata=_RAW)  # hf = download `hf_id` from the Hub; synthetic = the tiny test tokenizer
    hf_id: Optional[str] = field(default=None, metadata=_RAW)  # required for kind=hf
    revision: Optional[str] = field(default=None, metadata=_RAW)  # Hub commit sha; pin it

    def __post_init__(self) -> None:
        if self.kind == "hf" and not self.hf_id:
            raise ValueError(f"tokenizer {self.name!r}: kind=hf requires hf_id")


@dataclass
class DedupConfig:
    """Deduplication of a source's rows (both kinds; instruct rows are hashed as instruction + input + output).

    `none` and `exact` apply to both kinds, `minhash` only to pretrain sources — an instruct source configured with
    it is rejected by `DatasetConfig._check_dedup_modes` instead of quietly getting exact dedup."""

    mode: DedupMode = field(default="exact", metadata=_PROCESSED)  # minhash (pretrain sources only) = exact dedup first, then MinHash/LSH near-duplicate removal (not for scale)
    normalize: bool = field(default=True, metadata={"hash": _normalize_hash})  # exact mode: hash lowercased, whitespace-collapsed text
    # A larger filter only lowers an already negligible false-positive rate: a resource knob, never a reason to
    # rebuild a processed folder, so it is in no hash.
    bloom_memory_mb: int = field(default=1024, metadata=_UNHASHED)  # exact mode: memory budget of the Bloom filter holding the seen hashes (per source)
    threshold: float = field(default=0.95, metadata={"hash": _minhash_only})  # minhash mode: Jaccard threshold
    num_perm: int = field(default=256, metadata={"hash": _minhash_only})  # minhash mode: permutations
    ngram: int = field(default=5, metadata={"hash": _minhash_only})  # minhash mode: word n-gram size

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("dedup.threshold must be in (0, 1]")
        if self.num_perm <= 0 or self.ngram <= 0:
            raise ValueError("dedup.num_perm and dedup.ngram must be positive")
        if self.bloom_memory_mb <= 0:
            raise ValueError("dedup.bloom_memory_mb must be positive")


@dataclass
class DecontaminationConfig:
    """Drop documents overlapping benchmark test sets (off by default; the thesis run skipped it)."""

    enabled: bool = field(default=False, metadata=_PROCESSED)  # off: documents are kept regardless of benchmark overlap
    benchmarks: list[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARKS), metadata=_PROCESSED)  # benchmark test sets to check against (lib/stages/benchmarks.py)
    ngram: int = field(default=13, metadata=_PROCESSED)  # word n-gram size compared between a document and the benchmarks
    threshold: float = field(default=0.1, metadata=_PROCESSED)  # share of a document's n-grams found in one benchmark


@dataclass
class ProcessingConfig:
    """Per-source processing options; the dataset-level block is the default, a pretrain source may override it."""

    min_chars: int = field(default=50, metadata=_PROCESSED)  # drop shorter texts (pretrain only; the upper bound is `max_seq_length` at download)
    dedup: DedupConfig = field(default_factory=DedupConfig, metadata=_PROCESSED)  # exact / minhash / none, see DedupConfig
    quality_filter: bool = field(default=False, metadata=_PROCESSED)  # prose heuristics (sentences, caps ratio, repetition); thesis run: off
    decontamination: DecontaminationConfig = field(default_factory=DecontaminationConfig, metadata=_PROCESSED)  # benchmark overlap filter, see DecontaminationConfig

    def __post_init__(self) -> None:
        if self.min_chars < 0:
            raise ValueError("processing: min_chars must be >= 0")


ALL_KINDS: frozenset[str] = frozenset(("pretrain", "instruct"))
ALL_LOADERS: frozenset[str] = frozenset(("hf_files", "hf_split", "hf_stream", "github_code", "local", "synthetic"))


@dataclass(frozen=True)
class FieldScope:
    """Where one `SourceConfig` field applies: the kinds and the loaders that may set it, and whether it is
    required wherever it applies. The defaults are "every kind, every loader, optional"."""

    kinds: frozenset[str] = ALL_KINDS
    loaders: frozenset[str] = ALL_LOADERS
    required: bool = False


_NOWHERE = FieldScope(kinds=frozenset(), loaders=frozenset())  # a field the table below forgot

# The single table of which `SourceConfig` field belongs to which kind and loader. `_check_field_scopes` is the only
# place that reads it, in both directions: a field set outside its scope is an error, and a required field missing
# inside its scope is one. A new field of `SourceConfig` that is not listed here applies nowhere, so it fails
# immediately instead of being silently accepted everywhere (`test_field_scopes_cover_every_source_field`).
SOURCE_FIELD_SCOPES: dict[str, FieldScope] = {
    # every kind, every loader
    "kind": FieldScope(),
    "loader": FieldScope(),
    "converter": FieldScope(),  # instruct row mapping, and the text builder of a pretrain source (gsm8k)
    "check_limit": FieldScope(),
    "rows": FieldScope(),
    "seed": FieldScope(),
    "shuffle": FieldScope(),
    "validation_fraction": FieldScope(),
    "describe_tokens_per_row": FieldScope(),
    "split": FieldScope(),  # only hf_split / hf_stream read it, but it is part of every source's raw hash
    # one kind only
    "text_field": FieldScope(kinds=frozenset({"pretrain"})),
    "processing": FieldScope(kinds=frozenset({"pretrain"})),
    "fields": FieldScope(kinds=frozenset({"instruct"})),
    "filter": FieldScope(kinds=frozenset({"instruct"})),
    "input_inversions": FieldScope(kinds=frozenset({"instruct"})),
    # one loader (family) only
    "hf_id": FieldScope(loaders=frozenset(HUB_LOADERS), required=True),
    "revision": FieldScope(loaders=frozenset(HUB_LOADERS)),
    "load_kwargs": FieldScope(loaders=frozenset(HUB_LOADERS)),
    "language": FieldScope(loaders=frozenset({"github_code"}), required=True),
    "path": FieldScope(loaders=frozenset({"local"}), required=True),
}

_NO_DEFAULT = object()  # a field without a default is always "set"


def _field_default(f: Field[Any]) -> Any:
    """The value a field has when a config does not mention it."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return _NO_DEFAULT


def _scope_text(scope: FieldScope) -> str:
    """The scope as the error message names it."""
    if not scope.kinds or not scope.loaders:
        return "no kind or loader: it is missing from SOURCE_FIELD_SCOPES"
    kinds = "kind " + "/".join(sorted(scope.kinds)) if scope.kinds != ALL_KINDS else ""
    loaders = "loader " + "/".join(sorted(scope.loaders)) if scope.loaders != ALL_LOADERS else ""
    return " with ".join(part for part in (kinds, loaders) if part) or "every kind and loader"


@dataclass
class SourceConfig:
    """One data source. `kind` selects the converter and the training-side formatting, `loader` how rows are
    fetched (see `lib/sources/loaders.py`). Both kinds go through the same download and build steps.

    Which field belongs to which kind and loader is the `SOURCE_FIELD_SCOPES` table, not a chain of ifs."""

    kind: SourceKind = field(metadata=_RAW)  # pretrain (one text column) | instruct (instruction / input / output)
    loader: LoaderName = field(default="hf_split", metadata=_RAW)  # how rows are fetched: hf_files | hf_split | hf_stream | github_code | local | synthetic (lib/sources/loaders.py)
    hf_id: Optional[str] = field(default=None, metadata=_RAW)  # Hub dataset id (hf_files / hf_split / hf_stream / github_code)
    revision: Optional[str] = field(default=None, metadata=_RAW)  # Hub commit sha; pin it so row order is stable across increments
    load_kwargs: dict[str, Any] = field(default_factory=dict, metadata=_LOAD_KWARGS)  # hf_files/github_code: {data_files: <glob>, max_cached_file_mb: <MB>}; else `load_dataset` kwargs
    split: str = field(default="train", metadata=_RAW)  # Hub split to read (hf_split / hf_stream)
    text_field: str = field(default="text", metadata=_RAW)  # pretrain: column holding the document
    language: Optional[str] = field(default=None, metadata=_RAW)  # github_code: language label of codeparrot/github-code-clean
    path: Optional[str] = field(default=None, metadata=_RAW)  # local: directory of parquet/jsonl files
    converter: Optional[str] = field(default=None, metadata=_RAW)  # named row converter (lib/sources/converters.py), e.g. gsm8k_question_answer
    fields: Optional[dict[str, str]] = field(default=None, metadata=_RAW)  # instruct: {instruction: <col>, input: <col>, output: <col>}
    filter: Optional[str] = field(default=None, metadata=_RAW)  # instruct: named row filter applied at download, e.g. sharegpt_quality
    check_limit: Optional[int] = field(default=None, metadata=_CONFIG)  # stop after inspecting this many source rows even if short of target (> 0; both kinds)
    rows: Optional[int] = field(default=None, metadata=_CONFIG)  # rows to download for a source used only in validation (required there, forbidden for train sources)
    seed: int = field(default=42, metadata={"hash": _seed_hash})  # synthetic generator seed; instruct: input-inversion and shuffle seed
    # hashed through the source's *effective* processing block (`source_processing`), not as a field of its own
    processing: Optional[ProcessingConfig] = field(default=None, metadata=_PROCESSED)  # pretrain: override of the dataset-level processing block
    input_inversions: float = field(default=0.0, metadata=_PROCESSED)  # instruct: share of rows turned into "given the output, what was the instruction?"
    shuffle: Optional[bool] = field(default=None, metadata=_PROCESSED)  # write processed/ in a seeded shuffled order; None = True for instruct, False for pretrain
    validation_fraction: Optional[float] = field(default=None, metadata=_CONFIG)  # override of the dataset-level validation_fraction for this source
    describe_tokens_per_row: int = field(default=500, metadata=_UNHASHED)  # only used by `describe` for its token table; never a planner input

    def __post_init__(self) -> None:
        self._check_field_scopes()
        self._check_values()

    def _check_field_scopes(self) -> None:
        """The one loop over `SOURCE_FIELD_SCOPES`: a field set outside the kind/loader it belongs to is an error
        naming the field and where it does apply, and a required field missing inside its scope is one too."""
        for f in fields(self):
            scope = SOURCE_FIELD_SCOPES.get(f.name, _NOWHERE)
            value = getattr(self, f.name)
            if self.kind in scope.kinds and self.loader in scope.loaders:
                if scope.required and not value:
                    where = f"loader {self.loader}" if scope.loaders != ALL_LOADERS else f"kind {self.kind}"
                    raise ValueError(f"{where} requires {f.name}")
            elif value != _field_default(f):
                raise ValueError(f"{f.name} only applies to {_scope_text(scope)}")

    def _check_values(self) -> None:
        """The rules about a field's value, which the scope table cannot express."""
        if self.loader == "hf_files" and not isinstance(self.load_kwargs.get("data_files"), str):
            raise ValueError("loader hf_files requires load_kwargs.data_files (a glob relative to the repo root)")
        max_cached_file_mb = self.load_kwargs.get("max_cached_file_mb")
        if max_cached_file_mb is not None and not _is_non_negative_number(max_cached_file_mb):
            raise ValueError("load_kwargs.max_cached_file_mb must be a non-negative number (MB)")
        if self.kind == "instruct" and self.loader != "synthetic" and self.fields is None and self.converter is None:
            raise ValueError("kind instruct requires fields or converter")
        if self.fields is not None and not {"instruction", "output"} <= set(self.fields):
            raise ValueError("fields must map at least instruction and output")
        if self.check_limit is not None and self.check_limit <= 0:
            raise ValueError("check_limit must be positive (omit it to read the whole source)")
        if self.rows is not None and self.rows <= 0:
            raise ValueError("rows must be positive")
        if not 0.0 <= self.input_inversions < 1.0:
            raise ValueError("input_inversions must be in [0, 1)")
        if self.validation_fraction is not None and not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.describe_tokens_per_row <= 0:
            raise ValueError("describe_tokens_per_row must be positive")


@dataclass
class StageConfig:
    """One training stage: token budget and the train/val weights over sources."""

    name: str = field(metadata=_CONFIG)  # stage label (checkpoints, logs); unique per config
    tokens: int = field(metadata=_CONFIG)  # training tokens of this stage (steps = tokens // (world_batch_size × block_size))
    train: dict[str, float] = field(metadata=_CONFIG)  # source name -> sampling weight (> 0, sum 1)
    val: dict[str, float] = field(metadata=_CONFIG)  # source name -> validation weight (> 0, sum 1)
    transition_pct: float = field(default=0.0, metadata=_CONFIG)  # fraction of this stage (at its end) blending into the next stage's data/LR

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError(f"stage {self.name}: tokens must be positive")
        if not 0.0 <= self.transition_pct < 1.0:
            raise ValueError(f"stage {self.name}: transition_pct must be in [0, 1)")
        _check_weights(f"stage {self.name}.train", self.train)
        _check_weights(f"stage {self.name}.val", self.val)


@dataclass
class DatasetConfig:
    """The whole dataset definition (`config/datasets/<name>.yaml`); this class is the config reference.

    Top-level keys:

    - `name`: dataset name (logs, checkpoints).
    - `tokenizer`: which tokenizer defines "a token" (`TokenizerConfig`); saved to `dataset/tokenizers/<name>/`.
    - `sources`: named data sources (`SourceConfig`), shared by every config under `dataset/sources/<source>/raw/`
      and `dataset/processed/<source>/`.
    - `stages`: the training stages in order (`StageConfig`): token budget, train/val weights over sources, transition.
    - `block_size`: training sequence length; the planner counts sequences with it; the run config must match.
    - `max_seq_length`: pretrain rows are truncated to this many tokens when downloaded, instruct rows longer than
      this are dropped; raising it above what raw was stored with re-downloads raw (after confirmation), lowering
      it costs nothing; `block_size` must be <= it.
    - `validation_fraction`: share of a source's rows held out when the source is used for training AND validation.
    - `always_range_requests`: read Hub files remotely by piece instead of caching whole files (traffic only).
    - `token_count`: how the `tokens` column is counted: with the tokenizer, or `estimate` (chars / 4).
    - `processing`: dataset-level processing defaults (`ProcessingConfig`); a pretrain source may override.

    Stage keys are plain source names in `train` and `val`. A source used only in `val` states `rows` (how many to
    download); a source used in `train` is sized by `sequence_budget` and must not give `rows`; a source used
    nowhere is rejected. A source in both `train` and `val` is split by the training resolver: its first
    `ceil(validation_fraction_of(source) × rows)` processed rows are validation, the rest training.
    """

    name: str = field(metadata=_CONFIG)  # non-empty path component
    tokenizer: TokenizerConfig = field(metadata=_RAW)  # see TokenizerConfig
    sources: dict[str, SourceConfig] = field(metadata=_CONFIG)  # source name -> SourceConfig; the names are the stage keys
    stages: list[StageConfig] = field(metadata=_CONFIG)  # in training order; at least one, unique names
    block_size: int = field(metadata=_CONFIG)  # training sequence length (sequences per stage = tokens ÷ block_size); <= max_seq_length
    max_seq_length: int = field(default=2048, metadata=_PROCESSED)  # token cap per stored row (pretrain: truncated, instruct: dropped); block_size must be <= this
    validation_fraction: float = field(default=0.05, metadata=_CONFIG)  # in [0, 1): held-out share of a source used in both train and val
    # Traffic only, not part of any hash: with it off, files up to load_kwargs.max_cached_file_mb are downloaded
    # whole into the Hub cache instead of being read remotely by piece.
    always_range_requests: bool = field(default=True, metadata=_UNHASHED)  # read every Hub file remotely by piece (row groups / stream prefix)
    token_count: TokenCountMode = field(default="tokenizer", metadata=_RAW)  # "estimate" = chars / 4
    # hashed through every source's *effective* processing block, not as a field of its own
    processing: ProcessingConfig = field(default_factory=ProcessingConfig, metadata=_PROCESSED)  # defaults for every source; see ProcessingConfig

    # --- validation ------------------------------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("name must be a non-empty path component")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.block_size > self.max_seq_length:
            raise ValueError(f"block_size ({self.block_size}) must be <= max_seq_length ({self.max_seq_length})")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not self.stages:
            raise ValueError("stages must contain at least one stage")
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        for stage in self.stages:
            for key in (*stage.train, *stage.val):
                self._check_stage_key(stage.name, key)
        self._check_source_usage()
        self._check_dedup_modes()
        self._check_shuffled_build_sizes()

    def _check_stage_key(self, stage_name: str, key: str) -> None:
        """A stage key is the plain name of a declared source."""
        if "/" in key:
            raise ValueError(
                f"stage {stage_name}: {key!r}: stage keys are plain source names (`<source>/validation` and mixture "
                "keys no longer exist; a source in both train and val is split by `validation_fraction`)"
            )
        if key not in self.sources:
            raise ValueError(f"stage {stage_name}: unknown source {key!r}")

    def _check_source_usage(self) -> None:
        """Whole-config pass: every source is used, `rows` exactly on the sources used only in validation."""
        for name, source in self.sources.items():
            in_train, in_val = self.used_in_train(name), self.used_in_val(name)
            if not in_train and not in_val:
                raise ValueError(f"source {name!r} is used by no stage (remove it or list it in a stage)")
            if in_train and source.rows is not None:
                raise ValueError(f"source {name!r} is used for training: it is sized by the sequence budget, drop `rows`")
            if not in_train and source.rows is None:
                raise ValueError(f"source {name!r} is used only for validation: give `rows` (how many rows to download)")
            if in_train and in_val and self.validation_fraction_of(name) <= 0.0:
                raise ValueError(
                    f"source {name!r} is used in train and val but its validation_fraction is 0: nothing would be held out "
                    "(list it only in `train`, or give a positive fraction)"
                )

    def _check_dedup_modes(self) -> None:
        """`dedup.mode: minhash` never reaches an instruct source: the near-duplicate pass runs only in the pretrain
        branch of the build (`lib/stages/build.py`), so such a source would silently be deduplicated exactly and the
        config would promise something it does not do. `processing` is a pretrain-only per-source field, so it is the
        dataset-level block that reaches an instruct source — a config that wants minhash for its pretrain sources
        gives each of them its own `processing`."""
        for name, source in self.sources.items():
            if source.kind == "instruct" and self.source_processing(name).dedup.mode == "minhash":
                raise ValueError(
                    f"source {name!r}: dedup.mode=minhash is not implemented for instruct sources (only pretrain "
                    "sources run the near-duplicate pass) — use dedup.mode=exact, and set minhash in the "
                    "`processing` block of each pretrain source instead of the dataset-level one"
                )

    def _check_shuffled_build_sizes(self) -> None:
        """A shuffled source is built all-at-once: every processed row is held in memory, shuffled, then written
        (`lib/stages/build.py`). A config can legally ask that of a huge source and OOM hours into the build, so a
        shuffled source whose planned row requirement (:meth:`rows_needed`, the planner's number) exceeds
        `SHUFFLED_BUILD_MAX_ROWS` is refused here — both `prepare.py` and training's auto-prepare load the config
        before any work."""
        for name in self.sources:
            if not self.shuffle_of(name):
                continue
            needed = self.rows_needed(name)
            if needed > SHUFFLED_BUILD_MAX_ROWS:
                raise ValueError(
                    f"{name}: shuffle=true builds all-at-once in memory; {needed:,} rows exceed the limit of "
                    f"{SHUFFLED_BUILD_MAX_ROWS:,}. Split the source or turn shuffle off. "
                    "(A read-time shuffle that would lift this limit is not implemented.)"
                )

    # --- source usage ----------------------------------------------------------------------------------------------

    def used_in_train(self, source_name: str) -> bool:
        """True if any stage trains on the source."""
        return any(source_name in stage.train for stage in self.stages)

    def used_in_val(self, source_name: str) -> bool:
        """True if any stage validates on the source."""
        return any(source_name in stage.val for stage in self.stages)

    def shuffle_of(self, source_name: str) -> bool:
        """Whether ``processed/<source>`` is written in a seeded shuffled order: the source's ``shuffle`` if set,
        else True for instruct sources (sorted by task in their repos) and False for pretrain sources."""
        source = self.sources[source_name]
        return source.shuffle if source.shuffle is not None else source.kind == "instruct"

    def validation_fraction_of(self, source_name: str) -> float:
        """Share of the source's processed rows the training resolver holds out for validation: the per-source
        override or the dataset default when the source is used in both train and val, 0.0 otherwise (a source
        used only in val is all validation, one used only in train all training)."""
        if not (self.used_in_train(source_name) and self.used_in_val(source_name)):
            return 0.0
        override = self.sources[source_name].validation_fraction
        return override if override is not None else self.validation_fraction

    # --- derived views ---------------------------------------------------------------------------------------------

    def source_processing(self, source_name: str) -> ProcessingConfig:
        """Effective processing options of a source (its override or the dataset-level block)."""
        override = self.sources[source_name].processing
        return override if override is not None else self.processing

    # --- budgets ---------------------------------------------------------------------------------------------------

    def sequence_budget(self, source_name: str) -> int:
        """Sequences (rows padded / truncated to ``block_size``) the whole run draws from the source: the integral
        of its sampling-weight schedule over the stage token budgets, rounded up.

        The trainer reads every source as ONE continuous stream for the whole run — a stage does not restart the
        source, it only changes the sampling weight — so stages sharing a source add up instead of overlapping.
        Each stage contributes its plain part ``(tokens − transition tokens) × weight`` plus, for the transition
        window at its end (``transition tokens = tokens × transition_pct``; none after the last stage), the
        trapezoid ``transition tokens × (weight + next stage's weight) / 2`` of the linear weight interpolation.
        Exact arithmetic from the config's own numbers (the YAML decimals as `Fraction`); 0 for a source not used
        in training.
        """
        total = Fraction(0)
        for stage, next_stage in zip(self.stages, [*self.stages[1:], None]):
            weight = Fraction(str(stage.train.get(source_name, 0.0)))
            if next_stage is None:
                total += stage.tokens * weight
                continue
            transition = Fraction(str(stage.transition_pct)) * stage.tokens
            next_weight = Fraction(str(next_stage.train.get(source_name, 0.0)))
            total += (stage.tokens - transition) * weight + transition * (weight + next_weight) / 2
        return ceil(total / self.block_size)

    def rows_needed(self, source_name: str) -> int:
        """Raw rows to download for the source — THE definition of the planner's row requirement
        (`lib/build/planner.py:rows_needed` delegates here, and `_check_shuffled_build_sizes` reads the same number,
        so the two cannot drift). A source used for training (and maybe validation): ``ceil(sequence_budget ×
        SAFETY_MARGIN ÷ (1 − validation_fraction_of(name)))`` — the margin covers what the length filter and the
        dedup drop, the division keeps the *training* part at the sequence budget after the training resolver holds
        ``validation_fraction`` of the processed rows out. A source used only for validation: its ``rows``. Exact
        `Fraction` arithmetic: 50 × 1.2 is 60, not 60.000000000000007."""
        source = self.sources[source_name]
        if not self.used_in_train(source_name):
            return int(source.rows or 0)
        held_out = Fraction(str(self.validation_fraction_of(source_name)))
        return ceil(self.sequence_budget(source_name) * SAFETY_MARGIN / (1 - held_out))

    # --- hashes (manifest keys; changing what goes into them invalidates data on disk) ------------------------------

    def raw_hash(self, source_name: str) -> str:
        """Hash of a source's ``raw/`` folder: every field annotated ``raw`` — the loader identity (kind, loader,
        repo, revision, files, split, text field, language, path, converter/fields/filter; ``seed`` only for
        ``loader: synthetic``, where it generates the rows) plus ``token_count`` and the tokenizer, on which the
        stored ``tokens`` column and the token-boundary truncation depend.

        Everything else is annotated ``processed``, ``config`` or ``none`` and stays out: ``max_seq_length`` (the raw
        manifest records what the rows were truncated at; only a raise re-downloads), processing options, budgets /
        ``rows`` / ``check_limit`` (how many rows are needed or read, not what is read), ``validation_fraction``,
        ``input_inversions``, ``shuffle``, the non-synthetic ``seed`` (inversions and shuffle order are build-time),
        ``describe_tokens_per_row``, ``load_kwargs.max_cached_file_mb`` (how a file is fetched). Raw shards are the
        bandwidth-expensive part of a dataset; nothing but a real change of the source may invalidate them.
        """
        payload = {
            "source": hash_payload(self.sources[source_name], "raw"),
            "token_count": self.token_count,
            "tokenizer": hash_payload(self.tokenizer, "raw"),
        }
        return _stable_hash(payload)

    def processed_hash(self, source_name: str) -> str:
        """Hash of a source's ``processed/`` folder: the raw hash, plus the ``processed`` fields as the build
        resolves them — ``max_seq_length`` (stored counts are clamped to it), the *effective* processing block (only
        the dedup fields of the active mode: a minhash threshold does not change an exact-dedup result), the
        ``input_inversions``, the resolved ``shuffle`` and the ``seed`` behind both. These four are written out
        rather than taken from :func:`hash_payload`, because the build uses their resolved values (``shuffle_of``,
        ``source_processing``) whether or not they were spelled in the YAML. A change rebuilds ``processed/`` from
        the raw shards (no download)."""
        source = self.sources[source_name]
        payload: dict[str, Any] = {
            "raw": self.raw_hash(source_name),
            "max_seq_length": self.max_seq_length,
            "processing": hash_payload(self.source_processing(source_name), "processed"),
            "input_inversions": source.input_inversions,
            "shuffle": self.shuffle_of(source_name),
            "seed": source.seed,
        }
        return _stable_hash(payload)

    def tokenizer_hash(self) -> str:
        """Hash of the tokenizer definition (the manifest key of `dataset/tokenizers/<name>/`)."""
        return _stable_hash(hash_payload(self.tokenizer, "raw"))

    def config_hash(self) -> str:
        """Hash of everything that defines the training data (recorded in checkpoints so a resume with different
        data is detected): every field annotated ``raw``, ``processed`` or ``config``, which leaves out the knobs
        that only change how the data are fetched or described (``always_range_requests``,
        ``load_kwargs.max_cached_file_mb``, ``describe_tokens_per_row``). The two processing blocks are replaced by
        the *effective* block of each source — the view ``processed_hash`` uses, so a Bloom budget change does not
        change this hash either."""
        payload = hash_payload(self, "config")
        payload.pop("processing", None)  # folded into every source's effective processing below
        for name, source_fields in payload["sources"].items():
            source_fields.pop("processing", None)
            source_fields["effective_processing"] = hash_payload(self.source_processing(name), "processed")
        return _stable_hash(payload)

    def overlap_warnings(self) -> list[str]:
        """Sources used only for validation that read the same Hub repo as a training source with the same or a
        nested ``data_files`` glob prefix — such a held-out set is likely not disjoint from the training data
        (prefer listing the training source in ``val`` too: its ``validation_fraction`` split never overlaps)."""
        warnings: list[str] = []
        val_only = [name for name in self.sources if self.used_in_val(name) and not self.used_in_train(name)]
        trained = [name for name in self.sources if self.used_in_train(name)]
        for val_name in val_only:
            val = self.sources[val_name]
            if val.hf_id is None:
                continue
            for train_name in trained:
                train = self.sources[train_name]
                if train.hf_id != val.hf_id:
                    continue
                a, b = _glob_prefix(val.load_kwargs.get("data_files")), _glob_prefix(train.load_kwargs.get("data_files"))
                if a.startswith(b) or b.startswith(a):
                    warnings.append(
                        f"validation-only source {val_name!r} reads {val.hf_id} like training source {train_name!r} "
                        f"(data_files {val.load_kwargs.get('data_files')!r} vs {train.load_kwargs.get('data_files')!r}): "
                        "the held-out rows may overlap the training data; consider validating on the training source "
                        "itself (validation_fraction split)"
                    )
        return warnings


# --- helpers ----------------------------------------------------------------------------------------------------------


def _is_non_negative_number(value: Any) -> bool:
    """True for ints/floats >= 0; bools are not numbers here (`True` would silently mean 1 MB)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and value >= 0


def _check_weights(what: str, weights: dict[str, float]) -> None:
    """Non-empty, every weight > 0 (a zero weight would list a source a stage never draws from), sum 1."""
    if not weights:
        raise ValueError(f"{what}: must not be empty")
    if any(w <= 0 for w in weights.values()):
        raise ValueError(f"{what}: weights must be > 0 (drop the key instead of a zero weight)")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{what}: weights sum to {total:.6f}, expected 1")


def _glob_prefix(pattern: Any) -> str:
    """The literal directory prefix of a ``data_files`` glob (``data/CC-MAIN-2013-20/*.parquet`` -> ``data/CC-MAIN-2013-20/``)."""
    text = "" if pattern is None else str(pattern)
    for i, char in enumerate(text):
        if char in "*?[":
            return text[:i]
    return text


def _stable_hash(payload: Any) -> str:
    """First 16 hex chars of the sha256 of the payload as sorted-key JSON (independent of dict insertion order)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def hash_payload(obj: Any, hash_name: HashName) -> dict[str, Any]:
    """The fields of ``obj`` that ``hash_name`` counts, as a JSON-ready dict; the input of the three hashes.

    A field is counted when its ``metadata["hash"]`` annotation says so (see "hash annotations" at the top of this
    file): ``raw`` and ``processed`` select exactly their own fields, ``config`` selects all three kinds. A field
    still holding its **default** value never enters — so adding a field with a default to the schema, or removing
    one, never invalidates data on disk; only a value explicitly set to something else changes a hash. A nested
    dataclass whose counted fields are all defaults is dropped for the same reason, and ``metadata["hash_drop"]``
    names dict keys of a field's value that are no part of the hash (``load_kwargs.max_cached_file_mb``).

    An unannotated field raises: a new schema field has to say which hash it belongs to.
    """
    out: dict[str, Any] = {}
    for f in fields(obj):
        if not _counts_for(f, obj, hash_name):
            continue
        value = getattr(obj, f.name)
        if _holds_default(f, value):
            continue
        hashed = _hashable(value, hash_name)
        if _is_dataclass_instance(value) and not hashed:
            continue  # every counted field of the nested block holds its default
        out[f.name] = _drop_keys(hashed, f.metadata.get("hash_drop", ()))
    return out


def field_hash_annotation(f: Field[Any], obj: Any) -> str:
    """Which hash ``f`` of ``obj`` belongs to: its ``metadata["hash"]``, or what the callable there answers for
    ``obj`` (the two conditional fields: ``SourceConfig.seed`` and the dedup fields of an inactive mode)."""
    annotation = f.metadata.get("hash")
    if annotation is None:
        raise TypeError(
            f"{type(obj).__name__}.{f.name} carries no `hash` metadata; annotate it with one of {HASH_ANNOTATIONS} "
            "(see 'hash annotations' in dataset_config.py) — a schema field must say which hash it belongs to"
        )
    name = annotation(obj) if callable(annotation) else annotation
    if name not in HASH_ANNOTATIONS:
        raise ValueError(f"{type(obj).__name__}.{f.name}: unknown hash annotation {name!r}; expected one of {HASH_ANNOTATIONS}")
    return str(name)


def _counts_for(f: Field[Any], obj: Any, hash_name: HashName) -> bool:
    """``config`` counts every hashed field; ``raw`` / ``processed`` count only their own (``processed_hash`` folds
    the raw hash in as a single value, so raw fields must not appear in it a second time)."""
    annotation = field_hash_annotation(f, obj)
    if hash_name == "config":
        return annotation != "none"
    return annotation == hash_name


def _holds_default(f: Field[Any], value: Any) -> bool:
    if f.default is not MISSING:
        return bool(value == f.default)
    if f.default_factory is not MISSING:
        return bool(value == f.default_factory())
    return False  # required field: never a default


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _drop_keys(value: Any, keys: Any) -> Any:
    """``value`` without the dict keys ``keys`` (``metadata["hash_drop"]``); the emptied dict itself stays."""
    if not keys or not isinstance(value, dict):
        return value
    return {k: v for k, v in value.items() if k not in keys}


def _hashable(value: Any, hash_name: HashName) -> Any:
    """Plain dicts/lists/scalars for JSON: nested dataclasses via `hash_payload`, tuples become lists."""
    if _is_dataclass_instance(value):
        return hash_payload(value, hash_name)
    if isinstance(value, dict):
        return {k: _hashable(v, hash_name) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(v, hash_name) for v in value]
    return value


def load_dataset_config(path: str | Path, overrides: Optional[list[str]] = None) -> DatasetConfig:
    """Load a dataset config YAML; `overrides` are jsonargparse `--key value` strings (nested keys with dots).

    An unknown key (for instance one of `REMOVED_KEYS`) raises a `ValueError` naming the file and the key instead
    of jsonargparse's usage dump and `sys.exit(2)`.
    """
    parser = ArgumentParser(description="Dataset config", exit_on_error=False)
    parser.add_class_arguments(DatasetConfig, nested_key=None)
    try:
        namespace = parser.parse_path(str(path))
        if overrides:
            namespace = parser.parse_args(overrides, namespace=namespace)
    except ArgumentError as error:
        raise ValueError(_load_error_message(path, str(error))) from error
    return DatasetConfig(**parser.instantiate(namespace).as_dict())


def _load_error_message(path: str | Path, error: str) -> str:
    """`<path>: <jsonargparse message>`, plus the hint of every removed key the message mentions."""
    message = f"dataset config {Path(path).as_posix()}: {error.strip()}"
    hints = [f"`{key}` was removed: {hint}" for key, hint in REMOVED_KEYS.items() if re.search(rf"\b{key}\b", error)]
    return message if not hints else message + "\n" + "\n".join(hints)
