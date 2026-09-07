# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The format every record of this repository is written in: the dashboards' file handlers (`ui.capture`,
`training.ui.capture`) and the stream handler of `data_preparation.lib.log`, so one line reads the same
wherever it lands. Its own module so the shared `ui` package does not import one of its consumers.
"""

from __future__ import annotations

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
