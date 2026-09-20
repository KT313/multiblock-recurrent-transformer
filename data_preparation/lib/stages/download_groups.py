# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Validate and describe the members of a shared GitHub download pass."""

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.log import get_logger
from data_preparation.lib.sources.loaders import GithubCodeRequest, github_code_repo_key
from data_preparation.lib.stages.download_state import _Increment

log = get_logger("data_preparation.lib.stages.download")


def validate_github_group(config: DatasetConfig, names: list[str]) -> None:
    """Reject empty groups, non-GitHub sources, and mismatched repository identities."""

    if not names:
        raise ValueError("download_github_code_group needs at least one source")
    for name in names:
        source = config.sources[name]
        if source.loader != "github_code":
            raise ValueError(f"{name}: download_github_code_group needs github_code sources")
        if github_code_repo_key(source) != github_code_repo_key(config.sources[names[0]]):
            raise ValueError(f"{name}: github_code group members must share hf_id, revision and data_files")


def log_group_download_plan(increments: list[_Increment]) -> None:
    """Describe active fetch targets and passive surplus collection."""

    for increment in increments:
        if increment.passive:
            log.info("%s: has its rows; storing what the pass reads on from offset %d -> %s", increment.name, increment.folder.rows_fetched, increment.folder.directory)
        else:
            log.info(
                "%s: fetching %d rows from offset %d -> %s",
                increment.name, increment.rows_to_keep, increment.folder.rows_fetched, increment.folder.directory,
            )


def create_github_requests(increments: list[_Increment]) -> list[GithubCodeRequest]:
    """Preserve member order, offsets, limits, and passive flags for the shared reader."""

    return [
        GithubCodeRequest(increment.name, increment.source, increment.folder.rows_fetched, increment.loader_count, passive=increment.passive)
        for increment in increments
    ]
