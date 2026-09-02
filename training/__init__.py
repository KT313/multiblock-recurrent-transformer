# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Training: `train.py` (the CLI), `run.py` (`train()`, the readable entry function and its setup helpers), `step.py`
(one optimizer step), `evaluation.py`, `checkpoint.py`, `logger.py` (wandb, `RunLogger`, `TrainingReport`),
`settings.py`, `stage_manager.py`, `lr_schedule.py`, `optim.py`, `backend/` (device / precision abstraction),
`data/` (dataset resolver, parquet streaming, tokenization, collation, the per-source train loaders), `ui/` (terminal
dashboard), `testing/` (test support: the golden run, hand-built stages)."""
