# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Open progress reporting around one single-source or grouped download."""

from collections.abc import Iterator
from contextlib import contextmanager

from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.hub_files import FetchStats
from data_preparation.lib.stages.download_state import _DownloadPostfix, _Increment
from data_preparation.lib.ui.dashboard import progress


@contextmanager
def open_download_progress(
    increments: list[_Increment], description: str, fetch_stats: FetchStats,
) -> Iterator[tuple[Progress, _DownloadPostfix]]:
    """Count active rows toward targets; report consumed and surplus rows in the postfix."""

    active = [increment for increment in increments if not increment.passive]
    with progress(
        total=sum(increment.folder.rows + increment.rows_to_keep for increment in active),
        initial=sum(increment.folder.rows for increment in active), desc=description,
        unit="row", panel="downloads", bytes_fetched=lambda: fetch_stats.bytes_fetched,
    ) as bar:
        yield bar, _DownloadPostfix(bar)
