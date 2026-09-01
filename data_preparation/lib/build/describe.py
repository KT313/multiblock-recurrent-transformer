# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``prepare.py describe``: render a dataset config as a Markdown document (``docs/data_mixture.md`` is generated
with it, so the documentation of the thesis mixture cannot drift from ``config/datasets/crow_300m_final.yaml``).

Pure function of the config file: tokenizer, sequence length and processing defaults, one table per stage (weights,
derived token budgets and sequence counts), the validation split per source and the source registry. The leading
comment block of the YAML file (the lines starting with ``#`` before the first key) is rendered as the "Notes"
section, so config-specific remarks live next to the config. Per-source budgets are the planner's arithmetic
(``DatasetConfig.sequence_budget``: the integral of the weight schedule over the run); the per-stage tables show
``stage.tokens × weight``; the rows-per-stage column is an estimate from ``describe_tokens_per_row`` (which
nothing else uses).
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

from data_preparation.dataset_config import DatasetConfig, ProcessingConfig, SourceConfig, TokenizerConfig

GENERATED_WITH = "uv run python data_preparation/prepare.py describe --dataset_config {config}"


def describe(cfg: DatasetConfig, config_path: str | Path, notes: str = "") -> str:
    """The Markdown document for ``cfg`` loaded from ``config_path``; ``notes`` is inserted verbatim after the
    header (see :func:`leading_comment`)."""
    config = Path(config_path).as_posix()
    lines: list[str] = [
        f"# Dataset `{cfg.name}`",
        "",
        f"Generated from `{config}` with",
        "",
        "```bash",
        GENERATED_WITH.format(config=config) + " > docs/data_mixture.md",
        "```",
        "",
        "Do not edit by hand: change the dataset config and regenerate. Token budgets are the stage budgets of the",
        "config times the stage weights; sequences are those tokens divided by `block_size` (what the training loader",
        "draws and what the planner sizes downloads with). The weights mix rows, not tokens: a row shorter than",
        "`block_size` realises fewer tokens than its sequence, so the last column estimates the tokens actually trained on",
        "from `describe_tokens_per_row` (a per-source estimate nothing but this document uses).",
        "",
    ]
    if notes.strip():
        lines += ["## Notes", "", notes.strip(), ""]
    lines += _general(cfg)
    lines += _stages(cfg)
    lines += _validation_split(cfg)
    lines += _sources(cfg)
    return "\n".join(lines).rstrip("\n") + "\n"


def leading_comment(config_path: str | Path) -> str:
    """The comment block at the top of a YAML file (``#`` lines before the first non-comment line), as prose."""
    out: list[str] = []
    for raw in Path(config_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            out.append(line[1:].strip())
        elif line:
            break  # first real YAML line ends the comment block
        else:
            out.append("")  # blank lines inside the block become paragraph breaks
    return "\n".join(out).strip()


# --- sections ----------------------------------------------------------------------------------------------------------


def _general(cfg: DatasetConfig) -> list[str]:
    total_tokens = sum(stage.tokens for stage in cfg.stages)
    if cfg.token_count == "tokenizer":
        token_count = f"`{cfg.token_count}` (real tokenizer counts)"
    else:
        token_count = f"`{cfg.token_count}` (chars / 4)"
    return [
        "## Tokenizer, sequence length and token counting",
        "",
        f"- tokenizer: {_tokenizer_label(cfg.tokenizer)}",
        f"- `max_seq_length`: {cfg.max_seq_length} (pretrain rows are truncated to this many tokens when downloaded, longer instruct rows are dropped)",
        f"- `block_size`: {cfg.block_size} (training sequence length; the run config must use the same value)",
        f"- `token_count`: {token_count}",
        f"- `validation_fraction`: {cfg.validation_fraction:.0%} of a source used for training and validation is held out",
        f"- training tokens over all stages: {_tokens(total_tokens)} ({_sequences(total_tokens, cfg.block_size)} sequences)",
        "",
        "## Processing defaults",
        "",
        *_processing_lines(cfg.processing),
        "",
    ]


def _tokenizer_label(tok: TokenizerConfig) -> str:
    """e.g. "`llama-32k` (hf, `hf-internal-testing/llama-tokenizer` @ `<sha>`)"."""
    label = f"`{tok.name}` ({tok.kind}"
    if tok.hf_id:
        label += f", `{tok.hf_id}`"
        if tok.revision:
            label += f" @ `{tok.revision}`"
    return label + ")"


def _processing_lines(p: ProcessingConfig) -> list[str]:
    return [
        f"- length filter: {p.min_chars} <= chars (pretrain only; rows are cut at `max_seq_length` tokens when downloaded)",
        f"- dedup: {_dedup_label(p)}",
        f"- quality filter: {_yn(p.quality_filter)}",
        f"- decontamination: {_decontamination_label(p)}",
    ]


def _dedup_label(p: ProcessingConfig) -> str:
    label = f"`{p.dedup.mode}`"
    if p.dedup.mode == "exact":
        label += f" (normalize: {_yn(p.dedup.normalize)}, Bloom filter {p.dedup.bloom_memory_mb} MB per source)"
    elif p.dedup.mode == "minhash":
        label += f" (threshold {p.dedup.threshold}, {p.dedup.num_perm} permutations, {p.dedup.ngram}-grams)"
    return label


def _decontamination_label(p: ProcessingConfig) -> str:
    decon = p.decontamination
    label = _yn(decon.enabled)
    if decon.enabled:
        label += f" ({decon.ngram}-grams, threshold {decon.threshold}, benchmarks: {', '.join(decon.benchmarks)})"
    return label


def _stages(cfg: DatasetConfig) -> list[str]:
    lines = ["## Stages", ""]
    for index, stage in enumerate(cfg.stages):
        lines += [
            f"### Stage {index + 1}: `{stage.name}` ({_tokens(stage.tokens)} tokens, transition {stage.transition_pct:.0%})",
            "",
            "| Train source | Weight | Tokens | Sequences | Tokens/row (est.) | Realised tokens (est.) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, weight in stage.train.items():
            tokens = stage.tokens * weight
            tokens_per_row = cfg.sources[name].describe_tokens_per_row
            realised = ceil(tokens / cfg.block_size) * min(tokens_per_row, cfg.block_size)
            lines.append(
                f"| `{name}` | {weight:.2%} | {_tokens(int(tokens))} | {_sequences(tokens, cfg.block_size)} | "
                f"{tokens_per_row} | {_tokens(realised)} |"
            )
        validation = ", ".join(f"`{name}` at {weight:.0%}" for name, weight in stage.val.items())
        lines += ["", f"Validation: {validation}", ""]
    return lines


def _validation_split(cfg: DatasetConfig) -> list[str]:
    """One line per source: how the training resolver splits its processed rows."""
    lines = [
        "## Validation split",
        "",
        "Decided at training time, never on disk: a source used only for training is all training rows, one used only",
        "for validation is all validation rows (`rows` says how many are downloaded), one used for both gives its first",
        "`validation_fraction` of processed rows to validation (deduplicated as one set, so the two never share a document).",
        "",
        "| Source | Used in | Held out |",
        "|---|---|---|",
    ]
    for name in cfg.sources:
        lines.append(f"| `{name}` | {_usage(cfg, name)} | {_held_out(cfg, name)} |")
    return lines + [""]


def _usage(cfg: DatasetConfig, name: str) -> str:
    in_train, in_val = cfg.used_in_train(name), cfg.used_in_val(name)
    if in_train and in_val:
        return "train + val"
    return "train only" if in_train else "val only"


def _held_out(cfg: DatasetConfig, name: str) -> str:
    if cfg.used_in_train(name) and cfg.used_in_val(name):
        return f"{cfg.validation_fraction_of(name):.0%} held out (the first rows of `processed/{name}`)"
    if cfg.used_in_val(name):
        rows = cfg.sources[name].rows
        return f"all rows ({rows:,} downloaded)" if rows is not None else "all rows"
    return "none"


def _sources(cfg: DatasetConfig) -> list[str]:
    lines = ["## Sources", "", "| Source | Kind | Loader | Origin | Revision | Details |", "|---|---|---|---|---|---|"]
    for name, src in cfg.sources.items():
        revision = f"`{src.revision[:12]}`" if src.revision else "-"
        details = _details(cfg, name) or "-"
        lines.append(f"| `{name}` | {src.kind} | `{src.loader}` | {_origin(src)} | {revision} | {details} |")
    return lines + [""]


# --- formatting helpers ------------------------------------------------------------------------------------------------


def _origin(src: SourceConfig) -> str:
    if src.loader == "synthetic":
        return f"generated (seed {src.seed})"
    if src.loader == "local":
        return f"`{src.path}`"
    origin = f"`{src.hf_id}`"
    if src.load_kwargs:
        origin += " " + ", ".join(f"{k}={v}" for k, v in src.load_kwargs.items())
    if src.split != "train":
        origin += f" split={src.split}"
    return origin


def _details(cfg: DatasetConfig, name: str) -> str:
    src = cfg.sources[name]
    parts: list[str] = []
    if src.language:
        parts.append(f"language {src.language}")
    if src.text_field != "text" and src.kind != "instruct":
        parts.append(f"text_field `{src.text_field}`")
    if src.fields:
        parts.append("fields " + ", ".join(f"{k}←`{v}`" for k, v in src.fields.items()))
    if src.converter:
        parts.append(f"converter `{src.converter}`")
    if src.filter:
        parts.append(f"filter `{src.filter}`")
    if src.check_limit is not None:
        parts.append(f"check_limit {src.check_limit:,}")
    if cfg.used_in_train(name):
        budget = cfg.sequence_budget(name)
        parts.append(f"budget {budget:,} sequences ({_tokens(budget * cfg.block_size)} tokens)")
    else:
        parts.append(f"rows {src.rows:,} (validation only)")
    if src.kind == "instruct":
        parts.append(f"input inversions {src.input_inversions:.0%}")
    if cfg.shuffle_of(name):
        parts.append(f"shuffled (seed {src.seed})")
    if src.processing is not None:
        override_lines = _processing_lines(src.processing)
        parts.append("processing override: " + "; ".join(line.removeprefix("- ") for line in override_lines))
    return ", ".join(parts)


def _sequences(tokens: float, block_size: int) -> str:
    """Sequences of ``block_size`` tokens, rounded up like the planner does: 1,611,329."""
    return f"{ceil(tokens / block_size):,}"


def _tokens(n: int) -> str:
    """Human-scale token count: 3.30B, 150.0M, 12.5K, 42."""
    if n >= 10**9:
        return f"{n / 1e9:.2f}B"
    if n >= 10**6:
        return f"{n / 1e6:.1f}M"
    if n >= 10**3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _yn(flag: bool) -> str:
    return "on" if flag else "off"


__all__ = ["GENERATED_WITH", "describe", "leading_comment"]
