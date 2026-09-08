# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Trapezoid learning-rate schedule (linear warmup, constant plateau, linear cooldown) with multi-stage
interpolation. All step arguments are OPTIMIZER steps.
"""

from training.stage_manager import StageManager

SCHEDULES = ("trapezoid",)


def _scheduled_lr(
    step: int, total_steps: int, stage_manager: StageManager, *, min_lr: float, warmup_steps: int, cooldown_steps: int
) -> float:
    """
    Global warmup at the start, global cooldown at the end, the per-stage base LR in between and a linear
    interpolation between the adjacent base LRs inside a stage transition.
    """

    stages = stage_manager.stages

    # Global warmup (beginning of first stage): towards the first stage's base LR
    if step < warmup_steps:
        return stages[0].base_lr * step / warmup_steps

    if step > total_steps:
        return min_lr
    # Global cooldown (end of last stage)
    if step > (total_steps - cooldown_steps):
        return max(stages[-1].base_lr * (total_steps - step) / cooldown_steps, min_lr)

    stage_info = stage_manager.get_stage_info(step)
    base_lr = stages[stage_info.stage_index].base_lr
    # During transition: interpolate from the current stage's base LR to the entering stage's
    if stage_info.transition_to is not None:
        entering_lr = stages[stage_info.transition_to].base_lr
        return max(base_lr + (entering_lr - base_lr) * stage_info.transition_progress, min_lr)

    # Within stage: constant plateau at the stage's base LR
    return max(base_lr, min_lr)


def get_lr_multistage(
    step: int,
    total_steps: int,
    stage_manager: StageManager,
    *,
    min_lr: float,
    warmup_steps: int,
    cooldown_steps: int,
    schedule: str = "trapezoid",
) -> float:
    """
    Multi-stage LR: global warmup at the start, global cooldown at the end, per-stage base LR in between and a
    linear interpolation between the adjacent base LRs inside a stage transition (`_scheduled_lr`).
    """

    if schedule not in SCHEDULES:
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    return _scheduled_lr(
        step, total_steps, stage_manager, min_lr=min_lr, warmup_steps=warmup_steps, cooldown_steps=cooldown_steps
    )
