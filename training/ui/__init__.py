# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal user interface of a training run: the rich dashboard with stage bars, metrics, events and a log panel,
and its plain-console fallback. `dashboard.py` is the public API (read its module docstring); the parts: `common.py`
(names, the enabling rule), `format.py` (formatting, step-dict readers, panel fit), `throughput.py` (ETA),
`capture.py` (logging handler, attaching a logger, the terminal capture around the display), `fallback.py`
(`NoOpDashboard`), `board.py` (`TrainingDashboard`), `demo.py` (the scripted run), `testing.py` (test helpers)."""
