# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Read-only readiness and candidate-budget decisions for ordered dataset admission."""
from __future__ import annotations

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.build.planner import DatasetReport, SourceLedger, summarize_dataset_state
from data_preparation.lib.build.repair import inspect_repairs
from data_preparation.lib.stages.global_dedup import GlobalFrontier, global_policy
from data_preparation.lib.stages.global_output import outputs_complete, source_frontier
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import guarded_path
from data_preparation.lib.storage.snapshot import snapshot_problem
from data_preparation.lib.storage.tokenizer_assessment import assess_tokenizer_folder


def inspect_completed_global_snapshot(
    config: DatasetConfig, layout: DatasetLayout, config_name: str, steps: set[str], reopened: list[str], *, dry_run: bool,
) -> DatasetReport | None:
    """Inspect snapshot readiness before partial-scope rejection or any preparation changes."""
    for name in config.sources:
        guarded_path(layout.root, layout.processed_dir(name))
    repair = inspect_repairs(config, layout, config_name=config_name)
    current = summarize_dataset_state(config, layout, needs_repair=[action.source for action in repair.actions])
    current.snapshot_problem = snapshot_problem(config, layout, processing=global_policy(config))
    if current.complete and not reopened and outputs_complete(config, layout):
        if not dry_run:
            tokenizer = assess_tokenizer_folder(
                layout.tokenizer_dir(config.tokenizer.name), config.tokenizer_hash(), validate_payload=True,
            )
            if not tokenizer.ready and "tokenizer" not in steps:
                raise ValueError(f"{tokenizer.problem}; run prepare with the tokenizer step to repair it")
            current.tokenizer_complete = tokenizer.ready
            current.tokenizer_problem = tokenizer.problem
        if current.complete:
            return current
    return None


def check_global_source_scope(config: DatasetConfig, selected: list[str] | None) -> None:
    """Require all priority prerequisites unless readiness already established a no-op."""
    if selected is not None and set(selected) != set(config.sources):
        raise ValueError(
            "dataset-wide Bloom admission requires the complete dataset scope and priority prerequisites; "
            "omit --sources to prepare/replay all sources before requesting a partial no-op"
        )


def calculate_candidate_target(
    name: str, layout: DatasetLayout, candidates: DatasetLayout, ledger: SourceLedger, start: GlobalFrontier,
) -> int:
    """Extend candidates by the global shortfall, recovering pending candidates before a top-up."""
    local = Manifest.load(candidates.processed_dir(name))
    target = ledger.rows_sufficient
    retained = Manifest.load(layout.processed_dir(name))
    if local is not None and retained is not None:
        target = local.rows() + max(0, ledger.rows_sufficient - ledger.processed_rows)
        if (not retained.generation_complete and retained.extra.get("candidate_generation") == local.generation_id
                and (source_frontier(retained).source_index > start.source_index
                     or source_frontier(retained).source_candidates < local.rows())):
            target = local.rows()  # recover pending candidates before planning a top-up
    return target
