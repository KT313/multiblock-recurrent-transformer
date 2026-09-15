# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Resource scopes and ordered repair setup for the preparation pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.build.lock import DatasetLease, dataset_lock
from data_preparation.lib.build.planner import UnreadableRawShardError
from data_preparation.lib.build.repair import Confirm, RepairReport, authorize_repairs, perform_repairs
from data_preparation.lib.stages.download import inspect_tokenizer, prepare_planned_tokenizer


def build_prepare_command(config_path: str | Path, dataset_dir: str | Path) -> str:
    """
    The command an error message tells the user to run: the same one training's auto-prepare prints
    (`training/data/dataset_resolver.py::build_command`), with the `--yes` that answers the repair confirmation.
    """

    return f"python data_preparation/prepare.py prepare --dataset_config {config_path} --dataset_dir {dataset_dir} --yes"


@contextmanager
def explain_unreadable_shards(config_path: str | Path, dataset_dir: str | Path) -> Iterator[None]:
    """
    Give an :class:`UnreadableRawShardError` from the planner the remedy this level knows: the repair step and the
    command that runs it. The planner sees neither the config path nor the dataset directory, and a read-only
    caller (status, a dry run) does not repair anything itself, so the message has to say what will.
    """

    try:
        yield
    except UnreadableRawShardError as error:
        remedy = (
            f"the repair step truncates raw/{error.source} to its readable prefix, or deletes the folder when no "
            f"shard is readable; run\n  {build_prepare_command(config_path, dataset_dir)}\nthen status again"
        )
        raise error.with_remedy(remedy) from error


@contextmanager
def open_preparation_scope(
    config_path: str | Path, dataset_dir: str | Path, *, dry_run: bool, dataset_lease: DatasetLease | None,
) -> Iterator[None]:
    """Acquire or borrow dataset ownership before attaching error guidance; dry runs acquire no lock."""

    with dataset_lock(Path(dataset_dir), lease=dataset_lease) if not dry_run else nullcontext(), explain_unreadable_shards(config_path, dataset_dir):
        yield


def prepare_tokenizer_and_apply_repairs(
    config: DatasetConfig, layout: DatasetLayout, active_steps: set[str], repair_report: RepairReport,
    *, dry_run: bool, assume_yes: bool, confirm: Confirm | None, allow_foreign_raw: bool, hf_token: str | None,
) -> None:
    """Inspect the tokenizer even for dry runs; authorize and publish before changing raw/processed data.

    Staging failures preserve all published content. Once publication starts, there is no multi-directory rollback.
    """

    tokenizer_plan = inspect_tokenizer(config, layout) if "tokenizer" in active_steps else None
    if not dry_run:
        authorize_repairs(repair_report, assume_yes=assume_yes, confirm=confirm, allow_foreign_raw=allow_foreign_raw)
        if tokenizer_plan is not None:
            prepare_planned_tokenizer(config, tokenizer_plan, hf_token=hf_token)
        perform_repairs(repair_report)
