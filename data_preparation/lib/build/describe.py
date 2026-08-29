# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``prepare.py describe``: render a dataset config as a Markdown document (``docs/data_mixture.md`` is generated
with it, so the documentation of the thesis mixture cannot drift from ``config/datasets/crow_300m_final.yaml``).

Pure function of the config file: tokenizer and processing defaults, one table per stage (weights and derived
token budgets), one per mixture (shares and derived example counts), validation sources and the source registry. The leading
comment block of the YAML file (the lines starting with ``#`` before the first key) is rendered as the "Notes"
section, so config-specific remarks live next to the config. Numbers are the same arithmetic the planner uses
(``DatasetConfig.source_budget_tokens`` / ``instruct_mixture_budget_tokens``); estimates use ``tokens_per_row_estimate``.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

from data_preparation.lib.schema.dataset_config import DatasetConfig, ProcessingConfig, SourceConfig

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
        "config times the mixture weights; row and example counts are estimates from `tokens_per_row_estimate` (the",
        "planner refines them with measured token counts once a source has been processed).",
        "",
    ]
    if notes.strip():
        lines += ["## Notes", "", notes.strip(), ""]
    lines += _general(cfg)
    lines += _stages(cfg)
    lines += _instruct_mixtures(cfg)
    lines += _validations(cfg)
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
            break
        else:
            out.append("")
    return "\n".join(out).strip()


# --- sections ----------------------------------------------------------------------------------------------------------


def _general(cfg: DatasetConfig) -> list[str]:
    tok = cfg.tokenizer
    tokenizer = f"`{tok.name}` ({tok.kind}"
    if tok.hf_id:
        tokenizer += f", `{tok.hf_id}`"
        if tok.revision:
            tokenizer += f" @ `{tok.revision}`"
    tokenizer += ")"
    total = sum(s.tokens for s in cfg.stages)
    return [
        "## Tokenizer and token counting",
        "",
        f"- tokenizer: {tokenizer}",
        f"- `max_seq_length`: {cfg.max_seq_length} (token-count cap per document; the run config's `block_size` must not exceed it)",
        f"- `token_count`: `{cfg.token_count}`" + (" (real tokenizer counts)" if cfg.token_count == "tokenizer" else " (chars / 4)"),
        f"- training tokens over all stages: {_tokens(total)}",
        "",
        "## Processing defaults (`pretrain` sources)",
        "",
        *_processing_lines(cfg.processing),
        "",
    ]


def _processing_lines(p: ProcessingConfig) -> list[str]:
    dedup = f"`{p.dedup.mode}`"
    if p.dedup.mode == "exact":
        dedup += f" (normalize: {_yn(p.dedup.normalize)})"
    elif p.dedup.mode == "minhash":
        dedup += f" (threshold {p.dedup.threshold}, {p.dedup.num_perm} permutations, {p.dedup.ngram}-grams)"
    decon = _yn(p.decontamination.enabled)
    if p.decontamination.enabled:
        decon += f" ({p.decontamination.ngram}-grams, threshold {p.decontamination.threshold}, benchmarks: {', '.join(p.decontamination.benchmarks)})"
    return [
        f"- length filter: {p.min_chars} <= chars, truncated at {p.max_chars} chars",
        f"- dedup: {dedup}",
        f"- quality filter: {_yn(p.quality_filter)}",
        f"- decontamination: {decon}",
    ]


def _stages(cfg: DatasetConfig) -> list[str]:
    lines = ["## Stages", ""]
    for index, stage in enumerate(cfg.stages):
        lines += [
            f"### Stage {index + 1}: `{stage.name}` ({_tokens(stage.tokens)} tokens, transition {stage.transition_pct:.0%})",
            "",
            "| Train source | Weight | Tokens | Tokens/row (est.) |",
            "|---|---:|---:|---:|",
        ]
        for key, weight in stage.train.items():
            lines.append(f"| {_key(cfg, key)} | {weight:.2%} | {_tokens(int(stage.tokens * weight))} | {_estimate(cfg, key)} |")
        lines += ["", "Validation: " + ", ".join(f"{_key(cfg, key)} at {weight:.0%}" for key, weight in stage.val.items()), ""]
    return lines


def _instruct_mixtures(cfg: DatasetConfig) -> list[str]:
    if not cfg.instruct_mixtures:
        return []
    lines = ["## Instruct mixtures", ""]
    for name, mixture in cfg.instruct_mixtures.items():
        budget = cfg.instruct_mixture_budget_tokens(name)
        lines += [
            f"### `{name}` ({_tokens(budget)} tokens budget)",
            "",
            f"`max_tokens` {mixture.max_tokens}, input inversions {mixture.input_inversions:.0%}, validation split "
            f"{mixture.val_split:.0%}, seed {mixture.seed}. Examples = budget × share ÷ `tokens_per_row_estimate`.",
            "",
            "| Source | Share | Tokens | Examples (est.) |",
            "|---|---:|---:|---:|",
        ]
        for src, share in mixture.sources.items():
            estimate = cfg.sources[src].tokens_per_row_estimate
            lines.append(f"| `{src}` | {share:.1%} | {_tokens(int(budget * share))} | {ceil(budget * share / estimate):,} |")
        lines.append("")
    return lines


def _validations(cfg: DatasetConfig) -> list[str]:
    names = cfg.sources_of_kind("validation")
    if not names:
        return []
    lines = ["## Held-out validation sets", "", "| Source | Rows | Seed | Loader | Origin |", "|---|---:|---:|---|---|"]
    for name in names:
        src = cfg.sources[name]
        lines.append(f"| `{name}` | {src.rows:,} | {src.seed} | `{src.loader}` | {_origin(src)} |")
    return lines + [""]


def _sources(cfg: DatasetConfig) -> list[str]:
    lines = ["## Sources", "", "| Source | Kind | Loader | Origin | Revision | Details |", "|---|---|---|---|---|---|"]
    for name, src in cfg.sources.items():
        revision = f"`{src.revision[:12]}`" if src.revision else "-"
        lines.append(f"| `{name}` | {src.kind} | `{src.loader}` | {_origin(src)} | {revision} | {_details(cfg, name) or '-'} |")
    return lines + [""]


# --- formatting helpers ------------------------------------------------------------------------------------------------


def _key(cfg: DatasetConfig, key: str) -> str:
    base = key.partition("/")[0]
    return f"`{key}` (instruct_mixture)" if base in cfg.instruct_mixtures else f"`{key}`"


def _estimate(cfg: DatasetConfig, key: str) -> str:
    base = key.partition("/")[0]
    if base in cfg.instruct_mixtures:
        return "-"
    return str(cfg.sources[base].tokens_per_row_estimate)


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
    if src.kind == "pretrain":
        parts.append(f"budget {_tokens(cfg.source_budget_tokens(name))}")
        if src.processing is not None:
            parts.append("processing override: " + "; ".join(line[2:] for line in _processing_lines(src.processing)))
    return ", ".join(parts)


def _tokens(n: int) -> str:
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
