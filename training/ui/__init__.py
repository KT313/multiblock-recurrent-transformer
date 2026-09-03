# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal user interface of a training run: the rich dashboard (stage bars, metrics, events, log panel) and its
plain-console fallback. `dashboard.py` holds the overview; a run opens one through `training.logger.open_dashboard`.
Parts: `common.py` (names, the line logger, the enabling rule), `format.py` (formatting, the step line, panel fit),
`throughput.py` (ETA), `capture.py` (the run's log handlers and terminal capture; generic parts in the top-level
`ui/`), `fallback.py` (`ConsoleFallbackDashboard`), `board.py` (`TrainingDashboard`), `demo.py`, `testing.py`."""
