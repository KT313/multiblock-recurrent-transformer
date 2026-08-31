# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The pipeline's top level: budget planning in sequences (`planner.py`), the repair step (`repair.py`), the build
lock (`lock.py`), `prepare` / `status` (`runner.py`) and the Markdown rendering of a dataset config (`describe.py`)."""

from data_preparation.lib.build.describe import describe, leading_comment
from data_preparation.lib.build.lock import BUILD_LOCK_NAME, BuildLocked, build_lock
from data_preparation.lib.build.planner import (
    DatasetReport,
    DownloadPlan,
    SourceDownload,
    SourceState,
    every_source_satisfies_its_budget,
    plan_downloads,
    rows_needed,
    summarize_dataset_state,
)
from data_preparation.lib.build.repair import ConfirmationRequired, RepairAction, RepairError, RepairReport, repair_broken_and_stale_folders
from data_preparation.lib.build.runner import (
    DEFAULT_MAX_PARALLEL_DOWNLOADS,
    DEFAULT_NUM_WORKERS,
    MAX_ROUNDS,
    STEPS,
    BuildAborted,
    build_all_pending_raw_shards,
    download_all_missing_rows,
    prepare,
    status,
)

__all__ = [
    "BUILD_LOCK_NAME",
    "DEFAULT_MAX_PARALLEL_DOWNLOADS",
    "DEFAULT_NUM_WORKERS",
    "MAX_ROUNDS",
    "STEPS",
    "BuildAborted",
    "BuildLocked",
    "ConfirmationRequired",
    "DatasetReport",
    "DownloadPlan",
    "RepairAction",
    "RepairError",
    "RepairReport",
    "SourceDownload",
    "SourceState",
    "build_all_pending_raw_shards",
    "build_lock",
    "describe",
    "download_all_missing_rows",
    "every_source_satisfies_its_budget",
    "leading_comment",
    "plan_downloads",
    "prepare",
    "repair_broken_and_stale_folders",
    "rows_needed",
    "status",
    "summarize_dataset_state",
]
