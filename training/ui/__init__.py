# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal user interface of a training run: the rich dashboard with stage bars, metrics, events and a log panel,
and its plain-console fallback. `dashboard.py` holds the overview (read its module docstring); a run opens one
through `training.logger.open_dashboard`. The parts: `common.py` (names, the line logger, the enabling rule),
`format.py` (formatting, the step line, step-dict readers, panel fit), `throughput.py` (ETA), `capture.py` (the
line handlers, the terminal capture around the display), `fallback.py` (`ConsoleFallbackDashboard`), `board.py`
(`TrainingDashboard`), `demo.py` (the scripted run), `testing.py` (test helpers)."""
