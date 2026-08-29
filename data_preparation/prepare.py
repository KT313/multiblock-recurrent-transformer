# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Entry point for dataset preparation.

    python data_preparation/prepare.py tiny [--dataset_config config/datasets/tiny.yaml] [--dataset_dir dataset]

``tiny`` materialises a dataset config end to end with a minimal inline planner (tokenizer -> download ->
length_filter -> process -> holdout -> build_mixture; row counts derived from the stage token budgets and the
per-source ``tokens_per_row_estimate``, refined from the measured tokens for up to ``MAX_ROUNDS`` rounds). It is
meant for ``config/datasets/tiny.yaml`` (the smoke run and the tests); the real ``build`` / ``status`` commands with
the full budget planner are the next step of the restructuring.
"""

from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path
from typing import Any

if __name__ == "__main__":  # allow `python data_preparation/prepare.py` without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.lib.dataset_config import SAFETY_MARGIN, DatasetConfig, load_dataset_config  # noqa: E402
from data_preparation.lib.layout import DatasetLayout  # noqa: E402
from data_preparation.lib.log import configure_logging, get_logger  # noqa: E402
from data_preparation.lib.stages import (  # noqa: E402
    build_mixture,
    download,
    holdout,
    length_filter,
    prepare_tokenizer,
    process,
)

log = get_logger(__name__)

DEFAULT_DATASET_CONFIG = Path("config/datasets/tiny.yaml")
DEFAULT_DATASET_DIR = Path("dataset")
MAX_ROUNDS = 3  # estimate -> measured refinement rounds before giving up


def rows_for_budget(budget_tokens: float, tokens_per_row: float, margin: float = SAFETY_MARGIN) -> int:
    """Rows to fetch for ``budget_tokens`` at ``tokens_per_row`` (× safety ``margin``), at least 1."""
    return max(1, ceil(budget_tokens / max(tokens_per_row, 1e-9) * margin))


def build_dataset(cfg: DatasetConfig, layout: DatasetLayout, *, hf_token: str | None = None) -> None:
    """Materialise ``cfg`` under ``layout.root`` (minimal inline planner; every stage is idempotent)."""
    prepare_tokenizer(cfg, layout)
    for name in cfg.sources_of_kind("pretrain"):
        _build_pretrain_source(cfg, name, layout, hf_token)
    for name in cfg.sources_of_kind("holdout"):
        holdout(cfg, name, layout)
    for mixture_name in cfg.mixtures:
        _build_mixture(cfg, mixture_name, layout, hf_token)


def _build_pretrain_source(cfg: DatasetConfig, name: str, layout: DatasetLayout, hf_token: str | None) -> None:
    budget = cfg.source_budget_tokens(name)
    if budget == 0:
        log.info("%s: not used by any stage, skipping", name)
        return
    source = cfg.sources[name]
    tokens_per_row = float(source.tokens_per_row_estimate)
    for _ in range(MAX_ROUNDS):
        raw = download(cfg, name, layout, rows_needed=rows_for_budget(budget, tokens_per_row), hf_token=hf_token)
        length_filter(cfg, name, layout)
        processed = process(cfg, name, layout, target_tokens=budget if source.repeat_to_budget else None)
        tokens = processed.tokens() or 0
        if tokens >= budget or raw.extra.get("exhausted"):
            if tokens < budget:
                log.warning("%s: exhausted at %d tokens, budget is %d", name, tokens, budget)
            return
        tokens_per_row = max(tokens / max(raw.rows(), 1), 1e-9)
        log.info("%s: %d of %d tokens after processing, refining to %.1f tokens/row", name, tokens, budget, tokens_per_row)
    raise RuntimeError(f"{name}: token budget {budget} not reached after {MAX_ROUNDS} rounds")


def _build_mixture(cfg: DatasetConfig, mixture_name: str, layout: DatasetLayout, hf_token: str | None) -> None:
    mixture = cfg.mixtures[mixture_name]
    budget = cfg.mixture_budget_tokens(mixture_name)
    tokens_per_row = {src: float(cfg.sources[src].tokens_per_row_estimate) for src in mixture.sources}
    short: dict[str, Any] = {}
    for _ in range(MAX_ROUNDS):
        exhausted = set()
        for src, share in mixture.sources.items():
            raw = download(cfg, src, layout, rows_needed=rows_for_budget(budget * share, tokens_per_row[src]), hf_token=hf_token)
            if raw.extra.get("exhausted"):
                exhausted.add(src)
        result = build_mixture(cfg, mixture_name, layout, budget_tokens=budget)
        short = {src: info for src, info in result["train"].extra["short_sources"].items() if src not in exhausted}
        if not short:
            return
        for src in short:
            tokens_per_row[src] = max(result["train"].extra["tokens_per_row"][src], 1e-9)
        log.info("%s: short sources %s, refining tokens/row", mixture_name, sorted(short))
    raise RuntimeError(f"{mixture_name}: sources {sorted(short)} still short after {MAX_ROUNDS} rounds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    tiny = subparsers.add_parser("tiny", help="build a (tiny) dataset config end to end with the inline planner")
    tiny.add_argument("--dataset_config", type=Path, default=DEFAULT_DATASET_CONFIG, help="dataset config YAML")
    tiny.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR, help="root of all prepared data")
    tiny.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated sources")
    tiny.set_defaults(run=run_tiny)
    return parser


def run_tiny(args: argparse.Namespace) -> None:
    cfg = load_dataset_config(args.dataset_config)
    layout = DatasetLayout(args.dataset_dir)
    log.info("building dataset config %s (%s) under %s", cfg.name, args.dataset_config, layout.root)
    build_dataset(cfg, layout, hf_token=args.hf_token)
    log.info("done: %s", layout.root)


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``) and dispatch to the selected command."""
    configure_logging()
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
