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
    stage_info = stage_manager.get_stage_info(step)

    # Global warmup (beginning of first stage): towards the first stage's base LR — inside the first transition
    # `stage_info.base_lr` would already be the next stage's, and the ramp must not change its target midway.
    if step < warmup_steps:
        return stage_manager.stages[0].base_lr * step / warmup_steps

    if step > max_steps:
        return min_lr
    # Global cooldown (end of last stage)
    if step > (max_steps - cooldown_steps):
        final_lr = stage_manager.stages[-1].base_lr
        return max(final_lr * (max_steps - step) / cooldown_steps, min_lr)

    # During transition: interpolate between stage LRs
    if stage_info.in_transition and stage_info.prev_base_lr is not None:
        interpolated_lr = stage_info.prev_base_lr + (stage_info.base_lr - stage_info.prev_base_lr) * (
            stage_info.transition_progress
        )
        return max(interpolated_lr, min_lr)

    # Within stage: constant plateau at the stage's base LR
    return max(stage_info.base_lr, min_lr)


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
