# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Dataset config schema: the single definition of a dataset (`config/datasets/<name>.yaml`).

Framework-neutral (no torch). Loaded by `data_preparation/prepare.py` (materialises it under `dataset/`) and by
`training/train.py` (verifies / auto-prepares it and derives the per-stage data mixtures). The run config only
references the file; every data-related setting lives here. This file is the config reference: every field
carries a trailing comment, `DatasetConfig` lists the top-level keys.

Layout produced on disk (see `data_preparation/README.md`):

    dataset/sources/<source>/raw/     rows as downloaded (text truncated to dataset_max_sequence_length tokens); shared, append-only
    dataset/processed/<source>/       rows after cleaning (what training reads); derived from raw, shared
    dataset/tokenizers/<name>/

Mixing (stage weights) and the validation split (`validation_fraction`) happen in the training dataloader; the
pipeline only downloads and cleans one folder per source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from fractions import Fraction
from math import ceil, isfinite
from pathlib import Path
from typing import Any, Literal, Optional

from jsonargparse import ArgumentError, ArgumentParser

from data_preparation.lib.stages.benchmarks import benchmark_revisions
from data_preparation.lib.stages.truncation import TOKEN_RULE

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
DEFAULT_TOKENS_PER_ROW_ESTIMATE = 500  # `describe_tokens_per_row` until a source's first raw shard measures the real rate
SAFETY_MARGIN = Fraction("1.2")  # rows downloaded = rows budget × this (covers filter / dedup losses and a tokens-per-row estimate that ran high); a Fraction so 50 × 1.2 is exactly 60

SHUFFLED_BUILD_MAX_ROWS = 1_000_000  # a shuffled source is built all-at-once in memory; `rows_needed` above this fails at config load
# derivation, keep as a comment: processed rows are TEXT bounded by dataset_max_sequence_length tokens at download
# (~8-10 KB/row worst case), so 1M rows is a worst case of ~10 GB held once; typical instruct rows are far smaller.
MINHASH_BUILD_MAX_ROWS = 250_000  # a minhash source is built all-at-once too, plus its LSH index; the same check, a lower limit
# derivation: the LSH index holds every kept row at roughly 3-5 KB (num_perm 256, `lib/stages/fuzzy_dedup.py`) next
# to the texts above, so 250k rows is ~1 GB of index plus a worst case of ~2.5 GB of text held once.

# --- hash annotations --------------------------------------------------------------------------------------------------
#
# Every field of every config dataclass declares which of the four manifest hashes it belongs to, once, right where
# it is defined; `hash_payload` walks the annotations and `raw_hash` / `processed_hash` / `tokenizer_hash` /
# `config_hash` assemble their payloads from it, so no hash keeps a list of field names of its own.
#
# The rule for a folder's hash: it contains exactly the settings that change what that folder stores, nothing
# that merely changes how the rows are fetched, counted, described or used later.
#
#   raw       identity of the downloaded rows: which rows a loader yields, in what order, with which columns. Keys
#             `sources/<s>/raw/`, the bandwidth-expensive tree: a change makes it *stale*, so it is deleted (after
#             confirmation) and downloaded again. The tokenizer and `token_count` are NOT in it: they only make the
#             stored token counts and the truncation of pretrain texts, which the raw manifest records
#             (`tokenizer_hash`, `token_count`) so that a later change is offered as a choice instead of a re-download.
#   processed derives `processed/<s>/` from the raw shards: a change rebuilds that folder (after confirmation),
#             nothing is downloaded. The tokenizer and `token_count` enter here.
#   tokenizer the tokenizer definition, keys `tokenizers/<name>/`.
#   config    everything else that defines the training data; only `config_hash` (recorded in checkpoints so a resume
#             against different data is detected) counts it.
#   none      not hashed at all: how rows are fetched or described, resource knobs; nothing that changes the data.
#
# Each name selects exactly the fields annotated with it; the hashes nest instead of re-walking fields:
# `processed_hash` folds the raw hash in as one value and `config_hash` is composed of every source's processed hash,
# the tokenizer hash and the `config` fields. Every selected field enters with its value, default or not. The
# manifests record the exact payload each hash was computed from (`Manifest.hash_payload`), so a mismatch can be
# explained field by field (:func:`describe_hash_change`).
HashName = Literal["raw", "processed", "config", "tokenizer"]
HASH_ANNOTATIONS: tuple[str, ...] = ("raw", "processed", "config", "tokenizer", "none")

_RAW: dict[str, Any] = {"hash": "raw"}
_PROCESSED: dict[str, Any] = {"hash": "processed"}
_CONFIG: dict[str, Any] = {"hash": "config"}
_TOKENIZER: dict[str, Any] = {"hash": "tokenizer"}
_UNHASHED: dict[str, Any] = {"hash": "none"}
# `max_cached_file_mb` says whether a Hub file is cached whole or read remotely by piece: traffic, not rows.
_LOAD_KWARGS: dict[str, Any] = {**_RAW, "hash_drop": ("max_cached_file_mb",)}


def _seed_hash(source: SourceConfig) -> str:
    """
    `seed` is loader identity (it generates the rows themselves) only for `loader: synthetic`; for every other
    loader it drives the build-time input inversions and the shuffle order, so it belongs to the processed hash.
    """

    return "raw" if source.loader == "synthetic" else "processed"


def _reads_split(source: SourceConfig) -> bool:
    """
    Whether `split` selects the source's rows: only `hf_split` and `hf_stream` read it; `hf_files`,
    `github_code`, `local` and `synthetic` never look at it (they carry the default "train" unread).
    """

    return source.loader in ("hf_split", "hf_stream")


def _split_hash(source: SourceConfig) -> str:
    """
    `split` selects the rows only for the loaders that read a Hub split (:func:`_reads_split`); elsewhere it must
    not re-label a raw folder.
    """

    return "raw" if _reads_split(source) else "none"


def _text_field_hash(source: SourceConfig) -> str:
    """
    `text_field` names the column a pretrain row is stored from; an instruct row is built from `fields` /
    `converter` and never reads it.
    """

    return "raw" if source.kind == "pretrain" else "none"


def _normalize_hash(dedup: DedupConfig) -> str:
    """
    `normalize` changes the hashed text of every mode that hashes at all (`exact` and the exact pass of
    `minhash`); with `mode: none` nothing is hashed and it cannot change a result.
    """

    return "none" if dedup.mode == "none" else "processed"


def _minhash_only(dedup: DedupConfig) -> str:
    """
    The Jaccard threshold, the permutation count and the n-gram size only change a MinHash/LSH result.
    """

    return "processed" if dedup.mode == "minhash" else "none"


@dataclass
class TokenizerConfig:
    """
    Which tokenizer defines "a token" for this dataset; saved to `dataset/tokenizers/<name>/`.
    """

    name: str = field(metadata=_TOKENIZER)  # directory name under `dataset/tokenizers/`
    kind: Literal["hf", "synthetic"] = field(default="hf", metadata=_TOKENIZER)  # hf = download `hf_id` from the Hub; synthetic = the tiny test tokenizer
    hf_id: Optional[str] = field(default=None, metadata=_TOKENIZER)  # required for kind=hf
    revision: Optional[str] = field(default=None, metadata=_TOKENIZER)  # Hub commit sha; pin it

    def __post_init__(self) -> None:
        if self.kind == "hf" and not self.hf_id:
            raise ValueError(f"tokenizer {self.name!r}: kind=hf requires hf_id")


@dataclass
class DedupConfig:
    """
    Deduplication of a source's rows (both kinds; instruct rows are hashed as instruction + input + output).

    `none` and `exact` apply to both kinds, `minhash` only to pretrain sources; an instruct source configured with
    it is rejected by `DatasetConfig._check_dedup_modes` instead of quietly getting exact dedup.
    """

    mode: DedupMode = field(default="exact", metadata=_PROCESSED)  # minhash: exact dedup first, then MinHash/LSH near-duplicate removal (pretrain only, not for scale)
    normalize: bool = field(default=True, metadata={"hash": _normalize_hash})  # exact mode: hash lowercased, whitespace-collapsed text
    # A larger filter only lowers an already negligible false-positive rate: a resource knob, never a reason to
    # rebuild a processed folder, so it is in no hash.
    bloom_memory_mb: int = field(default=1024, metadata=_UNHASHED)  # exact mode: memory budget (MB) of the Bloom filter of seen hashes, per source
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
    """
    Drop documents overlapping benchmark test sets (off by default; the thesis run skipped it).
    """

    enabled: bool = field(default=False, metadata=_PROCESSED)  # off: documents are kept regardless of benchmark overlap
    benchmarks: list[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARKS), metadata=_PROCESSED)  # test sets to check (lib/stages/benchmarks.py)
    ngram: int = field(default=13, metadata=_PROCESSED)  # word n-gram size compared between a document and the benchmarks
    threshold: float = field(default=0.1, metadata=_PROCESSED)  # share of a document's n-grams found in one benchmark

    def __post_init__(self) -> None:
        if self.ngram <= 0:
            raise ValueError("decontamination.ngram must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("decontamination.threshold must be in [0, 1]")


@dataclass
class ProcessingConfig:
    """
    Per-source processing options; the dataset-level block is the default, a pretrain source may override it.
    """

    min_chars: int = field(default=50, metadata=_PROCESSED)  # drop shorter texts (pretrain only; the upper bound is `dataset_max_sequence_length` at download)
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
    """
    Where one `SourceConfig` field applies: the kinds and the loaders that may set it, and whether it is
    required wherever it applies. The defaults are "every kind, every loader, optional".
    """

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
    "split": FieldScope(),  # only hf_split / hf_stream read it (and only there it is raw identity, `_split_hash`)
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


def _field_default(dataclass_field: Field[Any]) -> Any:
    """
    The value a field has when a config does not mention it.
    """

    if dataclass_field.default is not MISSING:
        return dataclass_field.default
    if dataclass_field.default_factory is not MISSING:
        return dataclass_field.default_factory()
    return _NO_DEFAULT


def _scope_text(scope: FieldScope) -> str:
    """
    The scope as the error message names it.
    """

    if not scope.kinds or not scope.loaders:
        return "no kind or loader: it is missing from SOURCE_FIELD_SCOPES"
    kinds = "kind " + "/".join(sorted(scope.kinds)) if scope.kinds != ALL_KINDS else ""
    loaders = "loader " + "/".join(sorted(scope.loaders)) if scope.loaders != ALL_LOADERS else ""
    return " with ".join(part for part in (kinds, loaders) if part) or "every kind and loader"


@dataclass
class SourceConfig:
    """
    One data source. `kind` selects the converter and the training-side formatting, `loader` how rows are
    fetched (see `lib/sources/loaders.py`). Both kinds go through the same download and build steps.

    Which field belongs to which kind and loader is the `SOURCE_FIELD_SCOPES` table, not a chain of ifs.
    """

    kind: SourceKind = field(metadata=_RAW)  # pretrain (one text column) | instruct (instruction / input / output)
    loader: LoaderName = field(default="hf_split", metadata=_RAW)  # how rows are fetched: hf_files | hf_split | hf_stream | github_code | local | synthetic (lib/sources/loaders.py)
    hf_id: Optional[str] = field(default=None, metadata=_RAW)  # Hub dataset id (hf_files / hf_split / hf_stream / github_code)
    revision: Optional[str] = field(default=None, metadata=_RAW)  # Hub commit sha; pin it so row order is stable across increments
    load_kwargs: dict[str, Any] = field(default_factory=dict, metadata=_LOAD_KWARGS)  # hf_files/github_code: {data_files: <glob>, max_cached_file_mb: <MB>}; else `load_dataset` kwargs
    split: str = field(default="train", metadata={"hash": _split_hash})  # Hub split to read (hf_split / hf_stream)
    text_field: str = field(default="text", metadata={"hash": _text_field_hash})  # pretrain: column holding the document
    language: Optional[str] = field(default=None, metadata=_RAW)  # github_code: language label of codeparrot/github-code-clean
    path: Optional[str] = field(default=None, metadata=_RAW)  # local: directory of parquet/jsonl files
    converter: Optional[str] = field(default=None, metadata=_RAW)  # named row converter (lib/sources/converters.py), e.g. gsm8k_question_answer
    fields: Optional[dict[str, str]] = field(default=None, metadata=_RAW)  # instruct: {instruction: <col>, input: <col>, output: <col>}
    filter: Optional[str] = field(default=None, metadata=_RAW)  # instruct: named row filter applied at download, e.g. sharegpt_quality
    check_limit: Optional[int] = field(default=None, metadata=_CONFIG)  # stop after inspecting this many source rows even if short of target (> 0)
    rows: Optional[int] = field(default=None, metadata=_CONFIG)  # processed rows a validation-only source delivers (required there, forbidden for train sources; the download adds the safety margin)
    seed: int = field(default=42, metadata={"hash": _seed_hash})  # synthetic generator seed; instruct: input-inversion and shuffle seed
    # hashed through the source's *effective* processing block (`source_processing`), not as a field of its own
    processing: Optional[ProcessingConfig] = field(default=None, metadata=_PROCESSED)  # pretrain: override of the dataset-level processing block
    input_inversions: float = field(default=0.0, metadata=_PROCESSED)  # instruct: share of rows turned into "given the output, what was the instruction?"
    shuffle: Optional[bool] = field(default=None, metadata=_PROCESSED)  # write processed/ in a seeded shuffled order; None = True for instruct, False for pretrain
    validation_fraction: Optional[float] = field(default=None, metadata=_CONFIG)  # override of the dataset-level validation_fraction for this source
    describe_tokens_per_row: int = field(default=DEFAULT_TOKENS_PER_ROW_ESTIMATE, metadata=_UNHASHED)  # assumed mean tokens per stored row until the first raw shard measures it: sizes the first download (clamped at the training length) and the row columns of `describe`

    def __post_init__(self) -> None:
        self._check_field_scopes()
        self._check_values()

    def _check_field_scopes(self) -> None:
        """
        The one loop over `SOURCE_FIELD_SCOPES`: a field set outside the kind/loader it belongs to is an error
        naming the field and where it does apply, and a required field missing inside its scope is one too.
        """

        for source_field in fields(self):
            scope = SOURCE_FIELD_SCOPES.get(source_field.name, _NOWHERE)
            value = getattr(self, source_field.name)
            if self.kind in scope.kinds and self.loader in scope.loaders:
                if scope.required and not value:
                    where = f"loader {self.loader}" if scope.loaders != ALL_LOADERS else f"kind {self.kind}"
                    raise ValueError(f"{where} requires {source_field.name}")
            elif value != _field_default(source_field):
                raise ValueError(f"{source_field.name} only applies to {_scope_text(scope)}")

    def _check_values(self) -> None:
        """
        The rules about a field's value, which the scope table cannot express.
        """

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
    """
    One training stage: token budget and the train/val weights over sources.
    """

    name: str = field(metadata=_CONFIG)  # stage label (checkpoints, logs); unique per config
    tokens: int = field(metadata=_CONFIG)  # training tokens of this stage (steps = tokens // tokens per optimizer step)
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
    """
    The whole dataset definition (`config/datasets/<name>.yaml`); this class is the config reference.

    Top-level keys:

    - `tokenizer`: which tokenizer defines "a token" (`TokenizerConfig`); saved to `dataset/tokenizers/<name>/`.
    - `sources`: named data sources (`SourceConfig`), shared by every config under `dataset/sources/<source>/raw/`
      and `dataset/processed/<source>/`.
    - `stages`: the training stages in order (`StageConfig`): token budget, train/val weights over sources, transition.
    - `training_target_sequence_length`: the length the run trains at, what the download planner counts a row
      with: a row serves min(its tokens, this) of the token budget. At most `dataset_max_sequence_length`.
    - `dataset_max_sequence_length`: pretrain rows are truncated to this many tokens when downloaded, instruct rows longer than
      this are dropped; raising it above what raw was stored with re-downloads raw (after confirmation), lowering
      it costs nothing. A storage cap only: it keeps a stray 100k-token document from being stored whole.
    - `validation_fraction`: share of a source's rows held out when the source is used for training AND validation.
    - `always_range_requests`: read Hub files remotely by piece instead of caching whole files (traffic only).
    - `token_count`: how the `tokens` column is counted: with the tokenizer, or `estimate` (chars / 4).
    - `processing`: dataset-level processing defaults (`ProcessingConfig`); a pretrain source may override.

    Stage keys are plain source names in `train` and `val`. A source used only in `val` states `rows` (how many to
    deliver); a source used in `train` is sized by `token_budget` and must not give `rows`; a source used
    nowhere is rejected. A source in both `train` and `val` is split by the training resolver: its first
    `ceil(validation_fraction_of(source) × rows)` processed rows are validation, the rest training.
    """

    tokenizer: TokenizerConfig = field(metadata=_RAW)  # see TokenizerConfig
    sources: dict[str, SourceConfig] = field(metadata=_CONFIG)  # source name -> SourceConfig; the names are the stage keys
    stages: list[StageConfig] = field(metadata=_CONFIG)  # in training order; at least one, unique names
    # Sizes the downloads, changes no data: a row counts min(its tokens, this) towards the token budget, the run's
    # training_max_sequence_length cuts it there. Not hashed: a different target re-plans, it never rebuilds.
    training_target_sequence_length: int = field(metadata=_UNHASHED)  # the length the run trains at; <= dataset_max_sequence_length
    dataset_max_sequence_length: int = field(default=2048, metadata=_PROCESSED)  # token cap per stored row (pretrain: truncated, instruct: dropped); a storage cap, not the training length
    validation_fraction: float = field(default=0.05, metadata=_CONFIG)  # in [0, 1): held-out share of a source used in both train and val
    # Traffic only, not part of any hash: with it off, files up to load_kwargs.max_cached_file_mb are downloaded
    # whole into the Hub cache instead of being read remotely by piece.
    always_range_requests: bool = field(default=True, metadata=_UNHASHED)  # read every Hub file remotely by piece (row groups / stream prefix)
    token_count: TokenCountMode = field(default="tokenizer", metadata=_PROCESSED)  # "estimate" = chars / 4; the raw manifest records it, see "hash annotations"
    # hashed through every source's *effective* processing block, not as a field of its own
    processing: ProcessingConfig = field(default_factory=ProcessingConfig, metadata=_PROCESSED)  # defaults for every source; see ProcessingConfig

    # --- validation ------------------------------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.dataset_max_sequence_length <= 0:
            raise ValueError("dataset_max_sequence_length must be positive")
        if not 0 < self.training_target_sequence_length <= self.dataset_max_sequence_length:
            raise ValueError(
                f"training_target_sequence_length ({self.training_target_sequence_length}) must be positive and at most "
                f"dataset_max_sequence_length ({self.dataset_max_sequence_length}): rows are cut there when stored"
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not self.stages:
            raise ValueError("stages must contain at least one stage")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        for stage in self.stages:
            for key in (*stage.train, *stage.val):
                self._check_stage_key(stage.name, key)
        self._check_source_usage()
        self._check_dedup_modes()
        self._check_shuffled_build_sizes()

    def _check_stage_key(self, stage_name: str, key: str) -> None:
        """
        A stage key is the plain name of a declared source.
        """

        if key not in self.sources:
            raise ValueError(f"stage {stage_name}: unknown source {key!r}")

    def _check_source_usage(self) -> None:
        """
        Whole-config pass: every source is used, `rows` exactly on the sources used only in validation.
        """

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
        """
        `dedup.mode: minhash` never reaches an instruct source: the near-duplicate pass runs only in the pretrain
        branch of the build (`lib/stages/build.py`), so such a source would silently be deduplicated exactly and the
        config would promise something it does not do. `processing` is a pretrain-only per-source field, so it is the
        dataset-level block that reaches an instruct source; a config that wants minhash for its pretrain sources
        gives each of them its own `processing`.
        """

        for name, source in self.sources.items():
            if source.kind == "instruct" and self.source_processing(name).dedup.mode == "minhash":
                raise ValueError(
                    f"source {name!r}: dedup.mode=minhash is not implemented for instruct sources (only pretrain "
                    "sources run the near-duplicate pass); use dedup.mode=exact, and set minhash in the "
                    "`processing` block of each pretrain source instead of the dataset-level one"
                )

    def _check_shuffled_build_sizes(self) -> None:
        """
        A shuffled source is built all-at-once: every processed row is held in memory, shuffled, then written
        (`lib/stages/build.py`). A config can legally ask that of a huge source and OOM hours into the build, so a
        shuffled source whose planned row requirement (:meth:`rows_needed` at the config's tokens-per-row estimate) exceeds
        `SHUFFLED_BUILD_MAX_ROWS` is refused here; both `prepare.py` and training's auto-prepare load the config
        before any work. A pretrain source under `dedup.mode: minhash` takes the same all-at-once path and holds
        an LSH index of every kept row on top, so it is refused above the lower `MINHASH_BUILD_MAX_ROWS`.
        """

        for name, source in self.sources.items():
            needed = self.rows_needed(name)
            if self.shuffle_of(name) and needed > SHUFFLED_BUILD_MAX_ROWS:
                raise ValueError(
                    f"{name}: shuffle=true builds all-at-once in memory; {needed:,} rows exceed the limit of "
                    f"{SHUFFLED_BUILD_MAX_ROWS:,}. Split the source or turn shuffle off. "
                    "(A read-time shuffle that would lift this limit is not implemented.)"
                )
            if source.kind == "pretrain" and self.source_processing(name).dedup.mode == "minhash" and needed > MINHASH_BUILD_MAX_ROWS:
                raise ValueError(
                    f"{name}: dedup.mode=minhash builds all-at-once in memory with an LSH index of every kept row; "
                    f"{needed:,} rows exceed the limit of {MINHASH_BUILD_MAX_ROWS:,}. Use dedup.mode=exact or a smaller source."
                )

    # --- source usage ----------------------------------------------------------------------------------------------

    def used_in_train(self, source_name: str) -> bool:
        """
        True if any stage trains on the source.
        """

        return any(source_name in stage.train for stage in self.stages)

    def used_in_val(self, source_name: str) -> bool:
        """
        True if any stage validates on the source.
        """

        return any(source_name in stage.val for stage in self.stages)

    def shuffle_of(self, source_name: str) -> bool:
        """
        Whether processed/<source> is written in a seeded shuffled order: the source's shuffle if set,
        else True for instruct sources (sorted by task in their repos) and False for pretrain sources.
        """

        source = self.sources[source_name]
        return source.shuffle if source.shuffle is not None else source.kind == "instruct"

    def validation_fraction_of(self, source_name: str) -> float:
        """
        Share of the source's processed rows the training resolver holds out for validation: the per-source
        override or the dataset default when the source is used in both train and val, 0.0 otherwise (a source
        used only in val is all validation, one used only in train all training).
        """

        if not (self.used_in_train(source_name) and self.used_in_val(source_name)):
            return 0.0
        override = self.sources[source_name].validation_fraction
        return override if override is not None else self.validation_fraction

    # --- derived views ---------------------------------------------------------------------------------------------

    def source_processing(self, source_name: str) -> ProcessingConfig:
        """
        Effective processing options of a source (its override or the dataset-level block).
        """

        override = self.sources[source_name].processing
        return override if override is not None else self.processing

    # --- budgets ---------------------------------------------------------------------------------------------------

    def token_budget(self, source_name: str) -> int:
        """
        Tokens the whole run draws from the source: the integral of its sampling-weight schedule over the stage
        token budgets, rounded up.

        The trainer reads every source as ONE continuous stream for the whole run (a stage does not restart the
        source, it only changes the sampling weight), so stages sharing a source add up instead of overlapping.
        Each stage contributes its plain part (tokens − transition tokens) × weight plus, for the transition
        window at its end (transition tokens = tokens × transition_pct; none after the last stage), the
        trapezoid transition tokens × (weight + next stage's weight) / 2 of the linear weight interpolation.
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
        return ceil(total)

    def tokens_per_row_rate(self, source_name: str, tokens_per_row: float | None = None) -> Fraction:
        """
        Tokens one stored row serves the token budget with: a row serves min(its tokens,
        training_target_sequence_length), the run cuts it there. tokens_per_row is that mean measured over the
        rows on disk (:func:`lib.build.planner.measured_tokens_per_row`); before the first shard the source's
        describe_tokens_per_row estimate stands in, clamped at the target (a mean of capped lengths never exceeds
        the cap). An estimate that ran high is what the top-up rounds correct once the rows are measured.
        """

        rate = self.sources[source_name].describe_tokens_per_row if tokens_per_row is None else tokens_per_row
        return min(Fraction(rate), Fraction(self.training_target_sequence_length))

    def rows_budget(self, source_name: str, tokens_per_row: float | None = None) -> int:
        """
        Rows the whole run draws from the source: token_budget ÷ :meth:`tokens_per_row_rate`, rounded up (0 for
        a source not used in training). The status table's epochs divide it by the training rows on disk.
        """

        return ceil(self._rows_budget(source_name, tokens_per_row))

    def _rows_budget(self, source_name: str, tokens_per_row: float | None) -> Fraction:
        return self.token_budget(source_name) / self.tokens_per_row_rate(source_name, tokens_per_row)

    def rows_needed(self, source_name: str, tokens_per_row: float | None = None) -> int:
        """
        Raw rows to download for the source, the planner's row requirement (`_check_shuffled_build_sizes` reads
        the same number at the estimate). A source used for training: ceil(token_budget ÷ rate × SAFETY_MARGIN ÷
        (1 − validation_fraction_of(name))) with rate = :meth:`tokens_per_row_rate`; the margin covers what the
        length filter and the dedup drop and an estimate that ran high, the division keeps the *training* part at
        the budget after the validation holdout. A source used only for validation: ceil(rows × SAFETY_MARGIN),
        so that `rows` processed rows survive the build. Exact `Fraction` arithmetic: 50 × 1.2 is 60, not
        60.000000000000007.
        """

        source = self.sources[source_name]
        if not self.used_in_train(source_name):
            return ceil((source.rows or 0) * SAFETY_MARGIN)
        held_out = Fraction(str(self.validation_fraction_of(source_name)))
        return ceil(self._rows_budget(source_name, tokens_per_row) * SAFETY_MARGIN / (1 - held_out))

    def rows_sufficient(self, source_name: str, tokens_per_row: float | None = None) -> int:
        """
        Processed rows at which a source serves its budget: rows_needed ÷ SAFETY_MARGIN (the rows budget over the
        training share of the rows), or the `rows` of a validation-only source.
        """

        if not self.used_in_train(source_name):
            return int(self.sources[source_name].rows or 0)
        return ceil(self.rows_needed(source_name, tokens_per_row) / SAFETY_MARGIN)

    # --- hashes (manifest keys; changing what goes into them invalidates data on disk) ------------------------------

    def raw_hash(self, source_name: str) -> str:
        """
        Hash of a source's raw/ folder: every field annotated raw, i.e. the loader identity (kind, loader,
        repo, revision, files, language, path, converter/fields/filter; split only for the loaders that read a Hub
        split, text_field only for pretrain rows, seed only for loader: synthetic, where it generates the rows).

        Everything else is annotated processed, config, tokenizer or none and stays out: the tokenizer and
        token_count (they make the stored token counts and the truncation, which the raw manifest records itself
        so that a change is a choice, not a re-download: `lib/stages/download.py:inspect_raw`),
        dataset_max_sequence_length (the raw manifest records what the rows were truncated at; only a raise
        re-downloads), processing options, budgets / rows / check_limit (how many rows are needed or read, not what
        is read), validation_fraction, input_inversions, shuffle, the non-synthetic seed (inversions and shuffle
        order are build-time), describe_tokens_per_row (how many rows the first download plans, not what is read),
        load_kwargs.max_cached_file_mb (how a file is fetched). Raw shards are the bandwidth-expensive part of a
        dataset; nothing but a real change of the source may invalidate them.
        """

        return self.raw_hash_of(self.sources[source_name])

    def raw_hash_of(self, source: SourceConfig) -> str:
        """
        :meth:`raw_hash` of a source that need not be in the config (a github_code language stored without a
        source of its own): the same payload, so a later config entry with the same raw fields adopts its folder.
        """

        return _stable_hash(self.raw_hash_payload_of(source))

    def raw_hash_payload(self, source_name: str) -> dict[str, Any]:
        """
        The dict :meth:`raw_hash` hashes; the raw manifest records it so a mismatch can be explained field by field.
        """

        return self.raw_hash_payload_of(self.sources[source_name])

    def raw_hash_payload_of(self, source: SourceConfig) -> dict[str, Any]:
        return {"source": hash_payload(source, "raw")}

    def processed_hash(self, source_name: str) -> str:
        """
        Hash of a source's processed/ folder: the raw hash, plus everything the build resolves the rows with:
        the tokenizer definition, token_count and the counting rule (truncation.TOKEN_RULE) behind the tokens
        column, dataset_max_sequence_length (stored counts are clamped to it), the *effective* processing block as
        the build applies it (pretrain: only the dedup fields of the active mode, a minhash threshold does not
        change an exact-dedup result; instruct: the dedup block alone, since min_chars, the quality filter and the
        decontamination run in the pretrain branch only), the input_inversions (instruct only), the resolved
        shuffle and the seed behind both. These are written out rather than taken from :func:`hash_payload`,
        because the build uses their resolved values (shuffle_of, source_processing) whether or not they were
        spelled in the YAML. With decontamination on, the pinned Hub revisions of the benchmarks it checks against
        (`lib/stages/benchmarks.py`) enter too: a re-pin changes what the build filtered out. A change rebuilds
        processed/ from the raw shards after confirmation (no download).
        """

        return _stable_hash(self.processed_hash_payload(source_name))

    def processed_hash_payload(self, source_name: str) -> dict[str, Any]:
        """
        The dict :meth:`processed_hash` hashes; the processed manifest records it.
        """

        source = self.sources[source_name]
        processing = self.source_processing(source_name)
        payload: dict[str, Any] = {
            "raw": self.raw_hash(source_name),
            "max_seq_length": self.dataset_max_sequence_length,  # the key keeps the field's old name
            "tokenizer": hash_payload(self.tokenizer, "tokenizer"),
            "token_count": self.token_count,
            "token_rule": TOKEN_RULE,
            "shuffle": self.shuffle_of(source_name),
            "seed": source.seed,
        }
        if source.kind == "instruct":
            payload["processing"] = {"dedup": hash_payload(processing.dedup, "processed")}
            payload["input_inversions"] = source.input_inversions
            return payload
        payload["processing"] = hash_payload(processing, "processed")
        if processing.decontamination.enabled:
            payload["benchmark_revisions"] = benchmark_revisions(list(processing.decontamination.benchmarks))
        return payload

    def tokenizer_hash(self) -> str:
        """
        Hash of the tokenizer definition (the manifest key of `dataset/tokenizers/<name>/`; a raw manifest
        records it next to the rows it counted).
        """

        return _stable_hash(hash_payload(self.tokenizer, "tokenizer"))

    def config_hash(self) -> str:
        """
        Hash of everything that defines the training data (recorded in checkpoints so a resume with different
        data is detected), composed of the hashes below it: every source's processed_hash (which folds in its
        raw hash, the tokenizer and the *effective* processing block, so a Bloom budget change does not count here
        either) next to the source's own config fields, the tokenizer hash, and the config fields of the dataset
        (stages, validation fraction). The knobs that only change how (or how far ahead) data are fetched or
        described (always_range_requests, load_kwargs.max_cached_file_mb, describe_tokens_per_row) stay out.
        """

        payload = hash_payload(self, "config")
        payload["sources"] = {
            name: {"processed": self.processed_hash(name), **payload["sources"][name]} for name in self.sources
        }
        payload["tokenizer"] = self.tokenizer_hash()
        return _stable_hash(payload)

    def overlap_warnings(self) -> list[str]:
        """
        Sources used only for validation that read the same Hub repo as a training source with the same or a
        nested data_files glob prefix: such a held-out set is likely not disjoint from the training data (prefer
        listing the training source in val too: its validation_fraction split never overlaps). Two sources that
        both read a Hub split (:func:`_reads_split`) and name different ones are disjoint by construction and
        never warn; a `split` no loader reads says nothing about the rows.
        """

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
                if _reads_split(train) and _reads_split(val) and train.split != val.split:
                    continue  # two splits of one repo (train vs test) are disjoint by construction
                val_prefix = _glob_prefix(val.load_kwargs.get("data_files"))
                train_prefix = _glob_prefix(train.load_kwargs.get("data_files"))
                if val_prefix.startswith(train_prefix) or train_prefix.startswith(val_prefix):
                    warnings.append(
                        f"validation-only source {val_name!r} reads {val.hf_id} like training source {train_name!r} "
                        f"(data_files {val.load_kwargs.get('data_files')!r} vs {train.load_kwargs.get('data_files')!r}): "
                        "the held-out rows may overlap the training data; consider validating on the training source "
                        "itself (validation_fraction split)"
                    )
        return warnings


# --- helpers ----------------------------------------------------------------------------------------------------------


def _is_non_negative_number(value: Any) -> bool:
    """
    True for ints/floats >= 0; bools are not numbers here (`True` would silently mean 1 MB).
    """

    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and value >= 0


def _check_weights(what: str, weights: dict[str, float]) -> None:
    """
    Non-empty, every weight a finite number > 0 (a zero weight would list a source a stage never draws from), sum 1.

    Finiteness first: every comparison against NaN is False, so a NaN weight would pass both checks below and then
    drop its source out of the training mixture without a word (`BatchStream._pick_source` compares deficits).
    """

    if not weights:
        raise ValueError(f"{what}: must not be empty")
    unusable = sorted(name for name, weight in weights.items() if not isfinite(weight))
    if unusable:
        raise ValueError(f"{what}: weights must be finite numbers, got {unusable} with a nan or inf weight")
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError(f"{what}: weights must be > 0 (drop the key instead of a zero weight)")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{what}: weights sum to {total:.6f}, expected 1")


def _glob_prefix(pattern: Any) -> str:
    """
    The literal directory prefix of a data_files glob (data/CC-MAIN-2013-20/*.parquet -> data/CC-MAIN-2013-20/).
    """

    text = "" if pattern is None else str(pattern)
    for position, char in enumerate(text):
        if char in "*?[":
            return text[:position]
    return text


def _stable_hash(payload: Any) -> str:
    """
    First 16 hex chars of the sha256 of the payload as sorted-key JSON (independent of dict insertion order).
    """

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def describe_hash_change(stored: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """
    Why a folder's hash differs from the config's, one short line per field: `a.b.c: old -> new` for every
    changed, added ((absent) -> new) or removed (old -> (absent)) key of the flattened payloads, keys sorted.
    stored is the payload the manifest recorded (`Manifest.hash_payload`, JSON on disk; current is compared after
    the same JSON round trip); None (a manifest from before payloads were recorded) gives the single line
    "(no field detail recorded)". Two equal payloads under different hashes (the hash rule itself changed, or a
    manifest was edited) give "(no recorded field differs: the hash rule changed)".
    """

    if stored is None:
        return ["(no field detail recorded)"]
    old, new = _flatten(stored), _flatten(json.loads(json.dumps(current, default=str)))
    lines = []
    for key in sorted(old.keys() | new.keys()):
        if key not in new:
            lines.append(f"{key}: {json.dumps(old[key])} -> (absent)")
        elif key not in old:
            lines.append(f"{key}: (absent) -> {json.dumps(new[key])}")
        elif old[key] != new[key]:
            lines.append(f"{key}: {json.dumps(old[key])} -> {json.dumps(new[key])}")
    return lines or ["(no recorded field differs: the hash rule changed)"]


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    {dotted key: leaf value} of a nested dict; lists are leaves (a data_files glob, a benchmark list).
    """

    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def hash_payload(obj: Any, hash_name: HashName) -> dict[str, Any]:
    """
    The fields of obj annotated hash_name, as a JSON-ready dict; the input of the four hashes.

    A field is counted when its metadata["hash"] annotation names hash_name (see "hash annotations" at the
    top of this file), and it enters with its value whether or not that value is the default: a changed default
    therefore invalidates data built under the old one, as it should, and adding a field to the schema changes the
    hashes once (one rebuild). metadata["hash_drop"] names dict keys of a field's value that are no part of the
    hash (load_kwargs.max_cached_file_mb).

    An unannotated field raises: a new schema field has to say which hash it belongs to.
    """

    payload: dict[str, Any] = {}
    for dataclass_field in fields(obj):
        if field_hash_annotation(dataclass_field, obj) != hash_name:
            continue
        value = _hashable(getattr(obj, dataclass_field.name), hash_name)
        payload[dataclass_field.name] = _drop_keys(value, dataclass_field.metadata.get("hash_drop", ()))
    return payload


def field_hash_annotation(f: Field[Any], obj: Any) -> str:
    """
    Which hash f of obj belongs to: its metadata["hash"], or what the callable there answers for
    obj (the conditional fields: SourceConfig.seed / split / text_field and the dedup fields of an inactive mode).
    """

    annotation = f.metadata.get("hash")
    if annotation is None:
        raise TypeError(
            f"{type(obj).__name__}.{f.name} carries no `hash` metadata; annotate it with one of {HASH_ANNOTATIONS} "
            "(see 'hash annotations' in dataset_config.py); a schema field must say which hash it belongs to"
        )
    name = annotation(obj) if callable(annotation) else annotation
    if name not in HASH_ANNOTATIONS:
        raise ValueError(f"{type(obj).__name__}.{f.name}: unknown hash annotation {name!r}; expected one of {HASH_ANNOTATIONS}")
    return str(name)


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _drop_keys(value: Any, keys: Any) -> Any:
    """
    value without the dict keys keys (metadata["hash_drop"]); the emptied dict itself stays.
    """

    if not keys or not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key not in keys}


def _hashable(value: Any, hash_name: HashName) -> Any:
    """
    Plain dicts/lists/scalars for JSON: nested dataclasses via `hash_payload`, tuples become lists.
    """

    if _is_dataclass_instance(value):
        return hash_payload(value, hash_name)
    if isinstance(value, dict):
        return {key: _hashable(item, hash_name) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(item, hash_name) for item in value]
    return value


def load_dataset_config(path: str | Path, overrides: Optional[list[str]] = None) -> DatasetConfig:
    """
    Load a dataset config YAML; `overrides` are jsonargparse `--key value` strings (nested keys with dots).

    An unknown key raises a `ValueError` naming the file and the key (`<path>: <jsonargparse message>`) instead of
    jsonargparse's usage dump and `sys.exit(2)`.
    """

    parser = ArgumentParser(description="Dataset config", exit_on_error=False)
    parser.add_class_arguments(DatasetConfig, nested_key=None)
    try:
        namespace = parser.parse_path(str(path))
        if overrides:
            namespace = parser.parse_args(overrides, namespace=namespace)
    except ArgumentError as error:
        raise ValueError(f"dataset config {Path(path).as_posix()}: {str(error).strip()}") from error
    return DatasetConfig(**parser.instantiate(namespace).as_dict())
