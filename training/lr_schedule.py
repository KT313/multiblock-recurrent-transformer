# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Trapezoid learning-rate schedule (linear warmup, constant plateau, linear cooldown) with resume warmup and
multi-stage interpolation. All step arguments are OPTIMIZER steps.
"""

from training.stage_manager import StageManager

SCHEDULES = ("trapezoid",)


def _scheduled_lr(
    step: int, max_steps: int, stage_manager: StageManager, *, min_lr: float, warmup_steps: int, cooldown_steps: int
) -> float:
    """The schedule without the resume warmup: global warmup at the start, global cooldown at the end, the per-stage
    base LR in between and a linear interpolation between the adjacent base LRs inside a stage transition."""
    stages = stage_manager.stages

    # Global warmup (beginning of first stage): towards the first stage's base LR
    if step < warmup_steps:
        return stages[0].base_lr * step / warmup_steps

    if step > max_steps:
        return min_lr
    # Global cooldown (end of last stage)
    if step > (max_steps - cooldown_steps):
        return max(stages[-1].base_lr * (max_steps - step) / cooldown_steps, min_lr)

    stage_info = stage_manager.get_stage_info(step)
    base_lr = stages[stage_info.stage_idx].base_lr
    # During transition: interpolate from the current stage's base LR to the entering stage's
    if stage_info.transition_to is not None:
        entering_lr = stages[stage_info.transition_to].base_lr
        return max(base_lr + (entering_lr - base_lr) * stage_info.transition_progress, min_lr)

    # Within stage: constant plateau at the stage's base LR
    return max(base_lr, min_lr)


def _resume_warmup(steps_since_resume: int, resume_warmup_steps: int, min_lr: float, target_lr: float) -> float:
    """Linear ramp from `min_lr` to `target_lr` over `resume_warmup_steps` after a resume, at `steps_since_resume`
    (the caller only asks inside the ramp: `0 <= steps_since_resume < resume_warmup_steps`)."""
    warmup_factor = steps_since_resume / resume_warmup_steps
    return min_lr + warmup_factor * (target_lr - min_lr)


def get_lr_multistage(
    step: int,
    max_steps: int,
    stage_manager: StageManager,
    *,
    min_lr: float,
    warmup_steps: int,
    cooldown_steps: int,
    schedule: str = "trapezoid",
    resume_step: int = -1,
    resume_warmup_steps: int = 0,
) -> float:
    """Multi-stage LR: global warmup at the start, global cooldown at the end, per-stage base LR in between and a
    linear interpolation between the adjacent base LRs inside a stage transition (`_scheduled_lr`); after a resume
    at `resume_step` the first `resume_warmup_steps` steps ramp from `min_lr` up to that scheduled value."""
    if schedule not in SCHEDULES:
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    target_lr = _scheduled_lr(
        step, max_steps, stage_manager, min_lr=min_lr, warmup_steps=warmup_steps, cooldown_steps=cooldown_steps
    )
    steps_since_resume = step - resume_step
    if resume_step >= 0 and 0 <= steps_since_resume < resume_warmup_steps:
        return _resume_warmup(steps_since_resume, resume_warmup_steps, min_lr, target_lr)
    return target_lr
