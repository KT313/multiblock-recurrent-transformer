# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Test support: hand-built stages for a `StageManager` without a dataset on disk.
"""

from __future__ import annotations

from training.data.dataset_resolver import ResolvedStage


def resolved_stage(
    name: str,
    tokens: int,
    base_lr: float,
    transition_pct: float = 0.05,
    train_weights: dict[str, float] | None = None,
) -> ResolvedStage:
    """
    A `ResolvedStage` with no validation entries: what the stage manager, schedule and logger tests hand in.
    """

    return ResolvedStage(
        name=name,
        tokens=tokens,
        base_lr=base_lr,
        transition_pct=transition_pct,
        train_weights=dict(train_weights or {}),
        val_data=[],
    )
