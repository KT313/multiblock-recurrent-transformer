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
from math import ceil
from pathlib import Path
from typing import Any, Literal, Optional

from jsonargparse import ArgumentError, ArgumentParser, Namespace

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
SAFETY_MARGIN = 1.2  # rows downloaded = sequence budget × this (covers what the length filter / dedup drop; planner)

# Top-level / source keys of the pre-restructure schema, with the hint shown when a YAML still uses them.
REMOVED_KEYS: dict[str, str] = {
    "instruct_mixtures": "mixing is the training dataloader's job: list the instruct sources with weights in the stage",
    "validation_tokens": "the validation split is made at training time (`validation_fraction`)",
    "max_chars": "rows are truncated to `max_seq_length` tokens when downloaded",
    "max_tokens": "rows longer than `max_seq_length` tokens are dropped when downloaded",
    "tokens_per_row_estimate": "the planner counts sequences; `describe_tokens_per_row` feeds only the describe table",
}


@dataclass
class TokenizerConfig:
    """Which tokenizer defines "a token" for this dataset; saved to `dataset/tokenizers/<name>/`."""

    name: str  # directory name under `dataset/tokenizers/`
    kind: Literal["hf", "synthetic"] = "hf"  # hf = download `hf_id` from the Hub; synthetic = the tiny test tokenizer
    hf_id: Optional[str] = None  # required for kind=hf
    revision: Optional[str] = None  # Hub commit sha; pin it

    def __post_init__(self) -> None:
        if self.kind == "hf" and not self.hf_id:
            raise ValueError(f"tokenizer {self.name!r}: kind=hf requires hf_id")


@dataclass
class DedupConfig:
    """Deduplication of a source's rows (both kinds; instruct rows are hashed as instruction + input + output)."""

    mode: DedupMode = "exact"  # minhash = exact dedup first, then MinHash/LSH near-duplicate removal (not for scale)
    normalize: bool = True  # exact mode: hash lowercased, whitespace-collapsed text
    bloom_memory_mb: int = 1024  # exact mode: memory budget of the Bloom filter holding the seen hashes (per source)
    threshold: float = 0.95  # minhash mode: Jaccard threshold
    num_perm: int = 256  # minhash mode: permutations
    ngram: int = 5  # minhash mode: word n-gram size

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

    enabled: bool = False  # off: documents are kept regardless of benchmark overlap
    benchmarks: list[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARKS))  # benchmark test sets to check against (lib/stages/benchmarks.py)
    ngram: int = 13  # word n-gram size compared between a document and the benchmarks
    threshold: float = 0.1  # share of a document's n-grams found in one benchmark


@dataclass
class ProcessingConfig:
    """Per-source processing options; the dataset-level block is the default, a pretrain source may override it."""

    min_chars: int = 50  # drop shorter texts (pretrain only; the upper bound is `max_seq_length` at download)
    dedup: DedupConfig = field(default_factory=DedupConfig)  # exact / minhash / none, see DedupConfig
    quality_filter: bool = False  # prose heuristics (sentences, caps ratio, repetition); thesis run: off
    decontamination: DecontaminationConfig = field(default_factory=DecontaminationConfig)  # benchmark overlap filter, see DecontaminationConfig

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

    kind: SourceKind  # pretrain (one text column) | instruct (instruction / input / output)
    loader: LoaderName = "hf_split"  # how rows are fetched: hf_files | hf_split | hf_stream | github_code | local | synthetic (lib/sources/loaders.py)
    hf_id: Optional[str] = None  # Hub dataset id (hf_files / hf_split / hf_stream / github_code)
    revision: Optional[str] = None  # Hub commit sha; pin it so row order is stable across increments
    load_kwargs: dict[str, Any] = field(default_factory=dict)  # hf_files/github_code: {data_files: <glob>, max_cached_file_mb: <MB>}; else `load_dataset` kwargs
    split: str = "train"  # Hub split to read (hf_split / hf_stream)
    text_field: str = "text"  # pretrain: column holding the document
    language: Optional[str] = None  # github_code: language label of codeparrot/github-code-clean
    path: Optional[str] = None  # local: directory of parquet/jsonl files
    converter: Optional[str] = None  # named row converter (lib/sources/converters.py), e.g. gsm8k_question_answer
    fields: Optional[dict[str, str]] = None  # instruct: {instruction: <col>, input: <col>, output: <col>}
    filter: Optional[str] = None  # instruct: named row filter applied at download, e.g. sharegpt_quality
    check_limit: Optional[int] = None  # stop after inspecting this many source rows even if short of target (> 0; both kinds)
    rows: Optional[int] = None  # rows to download for a source used only in validation (required there, forbidden for train sources)
    seed: int = 42  # synthetic generator seed; instruct: input-inversion and shuffle seed
    processing: Optional[ProcessingConfig] = None  # pretrain: override of the dataset-level processing block
    input_inversions: float = 0.0  # instruct: share of rows turned into "given the output, what was the instruction?"
    shuffle: Optional[bool] = None  # write processed/ in a seeded shuffled order; None = True for instruct, False for pretrain
    validation_fraction: Optional[float] = None  # override of the dataset-level validation_fraction for this source
    describe_tokens_per_row: int = 500  # only used by `describe` for its token table; never a planner input

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

    name: str  # stage label (checkpoints, logs); unique per config
    tokens: int  # training tokens of this stage (steps = tokens // (world_batch_size × block_size))
    train: dict[str, float]  # source name -> sampling weight (> 0, sum 1)
    val: dict[str, float]  # source name -> validation weight (> 0, sum 1)
    transition_pct: float = 0.0  # fraction of this stage (at its end) blending into the next stage's data/LR

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

    name: str  # non-empty path component
    tokenizer: TokenizerConfig  # see TokenizerConfig
    sources: dict[str, SourceConfig]  # source name -> SourceConfig; the names are the stage keys
    stages: list[StageConfig]  # in training order; at least one, unique names
    block_size: int  # training sequence length (sequences per stage = tokens ÷ block_size); <= max_seq_length
    max_seq_length: int = 2048  # token cap per stored row (pretrain: truncated, instruct: dropped); block_size must be <= this
    validation_fraction: float = 0.05  # in [0, 1): held-out share of a source used in both train and val
    always_range_requests: bool = True  # read every Hub file remotely by piece (row groups / stream prefix); False: files
    # up to load_kwargs.max_cached_file_mb are downloaded whole into the Hub cache instead. Traffic only, not part of
    # source hashes.
    token_count: TokenCountMode = "tokenizer"  # "estimate" = chars / 4
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)  # defaults for every source; see ProcessingConfig

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

    def sources_of_kind(self, kind: SourceKind) -> list[str]:
        return [name for name, source in self.sources.items() if source.kind == kind]

    # --- budgets ---------------------------------------------------------------------------------------------------

    def sequence_budget(self, source_name: str) -> int:
        """Sequences (rows padded / truncated to ``block_size``) a stage draws from the source, maximised over the
        stages (folders are shared between stages, so max, not sum): ``ceil(stage.tokens × train weight ÷
        block_size)``; 0 for a source not used in training."""
        budget = 0
        for stage in self.stages:
            weight = stage.train.get(source_name, 0.0)
            if weight:
                budget = max(budget, ceil(stage.tokens * weight / self.block_size))
        return budget

    # --- hashes (manifest keys; changing what goes into them invalidates data on disk) ------------------------------

    def raw_hash(self, source_name: str) -> str:
        """Hash of a source's ``raw/`` folder: the loader identity — everything that determines **which rows** it
        holds and in what order (kind, loader, repo, revision, files, split, text field, language, path,
        converter/fields/filter; ``seed`` only for ``loader: synthetic``, where it generates the rows) — plus
        ``token_count`` and the tokenizer (the stored ``tokens`` column and the token-boundary truncation depend on
        them).

        Deliberately NOT part of it: ``max_seq_length`` (the raw manifest records what the rows were truncated at;
        only a raise re-downloads), processing options, budgets / ``rows`` / ``check_limit`` (how many rows are
        needed or read, not what is read), ``validation_fraction``, ``input_inversions``, ``shuffle``, the
        instruct ``seed`` (inversions and shuffle order are build-time), ``describe_tokens_per_row``,
        ``load_kwargs.max_cached_file_mb`` (how a file is fetched). Raw shards are the bandwidth-expensive part of a
        dataset; nothing but a real change of the source may invalidate them.
        """
        source = self.sources[source_name]
        source_fields = hash_fields(source)
        for key in ("processing", "check_limit", "rows", "validation_fraction", "input_inversions", "shuffle", "describe_tokens_per_row"):
            source_fields.pop(key, None)
        if source.loader != "synthetic":
            source_fields.pop("seed", None)
        load_kwargs = source_fields.get("load_kwargs")
        if load_kwargs is not None:
            load_kwargs.pop("max_cached_file_mb", None)
        return _stable_hash({"source": source_fields, "token_count": self.token_count, "tokenizer": hash_fields(self.tokenizer)})

    def processed_hash(self, source_name: str) -> str:
        """Hash of a source's ``processed/`` folder: the raw hash, ``max_seq_length`` (stored counts are clamped to
        it), the effective processing block with only the fields of the active dedup mode (a minhash threshold
        does not change an exact-dedup result), ``input_inversions``, the resolved ``shuffle`` and the ``seed``
        behind both. A change rebuilds ``processed/`` from the raw shards (no download)."""
        payload: dict[str, Any] = {
            "raw": self.raw_hash(source_name),
            "max_seq_length": self.max_seq_length,
            "processing": _processing_hash_fields(self.source_processing(source_name)),
            "input_inversions": self.sources[source_name].input_inversions,
            "shuffle": self.shuffle_of(source_name),
            "seed": self.sources[source_name].seed,
        }
        return _stable_hash(payload)

    def tokenizer_hash(self) -> str:
        """Hash of the tokenizer definition (the manifest key of `dataset/tokenizers/<name>/`)."""
        return _stable_hash(hash_fields(self.tokenizer))

    def config_hash(self) -> str:
        """Hash of everything that defines the training data (recorded in checkpoints so a resume with different
        data is detected): the config minus the knobs that only change how it is fetched or described
        (``always_range_requests``, ``load_kwargs.max_cached_file_mb``, ``describe_tokens_per_row``), with every
        source's processing block reduced to the fields that change its rows (:func:`_processing_hash_fields`, the
        view ``processed_hash`` uses — a Bloom budget change does not change the data)."""
        payload = hash_fields(self)
        payload.pop("always_range_requests", None)
        payload.pop("processing", None)  # folded into every source's effective processing below
        for name, source_fields in payload.get("sources", {}).items():
            source_fields.pop("describe_tokens_per_row", None)
            source_fields.pop("processing", None)
            source_fields["effective_processing"] = _processing_hash_fields(self.source_processing(name))
            load_kwargs = source_fields.get("load_kwargs")
            if load_kwargs is not None:
                load_kwargs.pop("max_cached_file_mb", None)
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


# Dedup fields that change the result of each mode. `bloom_memory_mb` (exact) is deliberately absent: it is a
# resource knob — a larger filter only lowers an already negligible false-positive rate, it does not change which
# rows a rebuild would keep in any material way — so changing it must not invalidate processed folders.
_DEDUP_RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    "none": ("mode",),
    "exact": ("mode", "normalize"),
    "minhash": ("mode", "normalize", "threshold", "num_perm", "ngram"),  # the exact pass runs first, keyed on `normalize`
}


def _processing_hash_fields(processing: ProcessingConfig) -> dict[str, Any]:
    """`hash_fields(processing)` with the dedup block reduced to the fields of the active mode."""
    out = hash_fields(processing)
    dedup = {k: v for k, v in hash_fields(processing.dedup).items() if k in _DEDUP_RESULT_FIELDS[processing.dedup.mode]}
    if dedup:
        out["dedup"] = dedup
    else:
        out.pop("dedup", None)
    return out


def _stable_hash(payload: Any) -> str:
    """First 16 hex chars of the sha256 of the payload as sorted-key JSON (independent of dict insertion order)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def hash_fields(obj: Any) -> dict[str, Any]:
    """`asdict(obj)` without the fields that still hold their default value (recursively for nested dataclasses).

    Manifests key on hashes of this, so adding a field with a default to the schema, or removing one, never
    invalidates data on disk; only a value that was explicitly set to something else changes the hash.
    """
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if _holds_default(f, value):
            continue
        out[f.name] = _hashable(value)
    return out


def _holds_default(f: Field[Any], value: Any) -> bool:
    if f.default is not MISSING:
        return bool(value == f.default)
    if f.default_factory is not MISSING:
        return bool(value == f.default_factory())
    return False  # required field: never a default


def _hashable(value: Any) -> Any:
    """Plain dicts/lists/scalars for JSON: nested dataclasses via `hash_fields`, tuples become lists."""
    if is_dataclass(value) and not isinstance(value, type):
        return hash_fields(value)
    if isinstance(value, dict):
        return {k: _hashable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(v) for v in value]
    return value


def dataset_config_fields() -> list[str]:
    """Top-level field names (used by the settings/CLI layer to recognise dataset-config keys)."""
    return [f.name for f in fields(DatasetConfig)]


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
    instantiated = parser.instantiate_classes(namespace)
    values = instantiated.as_dict() if isinstance(instantiated, Namespace) else instantiated
    return DatasetConfig(**values)


def _load_error_message(path: str | Path, error: str) -> str:
    """`<path>: <jsonargparse message>`, plus the hint of every removed key the message mentions."""
    message = f"dataset config {Path(path).as_posix()}: {error.strip()}"
    hints = [f"`{key}` was removed: {hint}" for key, hint in REMOVED_KEYS.items() if re.search(rf"\b{key}\b", error)]
    return message if not hints else message + "\n" + "\n".join(hints)
