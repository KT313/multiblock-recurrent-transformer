# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the trapezoid multi-stage LR schedule: warmup/plateau/cooldown values and continuity at every stage
transition.
"""

from typing import Any

import pytest

from training.lr_schedule import SCHEDULES, get_lr_multistage
from training.stage_manager import StageManager
from training.testing.stages import resolved_stage

TPS = 4 * 256  # tokens per optimizer step of the tiny config


def _tiny_manager(warmup: int = 2, cooldown: int = 2) -> StageManager:
    stages = [
        resolved_stage("a", tokens=8 * TPS, base_lr=3e-4, transition_pct=0.25),
        resolved_stage("b", tokens=8 * TPS, base_lr=1e-4, transition_pct=0.25),
        resolved_stage("c", tokens=4 * TPS, base_lr=5e-5, transition_pct=0.0),
    ]
    return StageManager(stages, tokens_per_step=1024, warmup_steps=warmup, cooldown_steps=cooldown)


def _lr(sm: StageManager, step: int, **kw: Any) -> float:
    kw.setdefault("min_lr", 0.0)
    kw.setdefault("warmup_steps", sm.warmup_steps)
    kw.setdefault("cooldown_steps", sm.cooldown_steps)
    return get_lr_multistage(step, sm.total_steps, sm, **kw)


# Hand-computed tiny schedule (20 steps, warmup 2, cooldown 2, transitions [6,8) and [14,16)).
TINY_EXPECTED = {
    0: 0.0,
    1: 1.5e-4,
    2: 3e-4,
    5: 3e-4,
    6: 3e-4,  # transition progress 0
    7: 2e-4,  # progress 0.5 between 3e-4 and 1e-4
    8: 1e-4,
    13: 1e-4,
    14: 1e-4,
    15: 7.5e-5,
    16: 5e-5,
    18: 5e-5,  # cooldown starts strictly after max_steps - cooldown
    19: 2.5e-5,
    20: 0.0,
}


@pytest.mark.parametrize("step,expected", sorted(TINY_EXPECTED.items()))
def test_tiny_schedule_values(step: int, expected: float) -> None:
    assert _lr(_tiny_manager(), step) == pytest.approx(expected)


def test_unknown_schedule_raises() -> None:
    assert SCHEDULES == ("trapezoid",)
    with pytest.raises(ValueError, match="Unsupported lr_schedule"):
        _lr(_tiny_manager(), 3, schedule="cosine")


def test_min_lr_floors_plateau_transition_and_cooldown() -> None:
    sm = _tiny_manager()
    assert _lr(sm, 16, min_lr=8e-5) == pytest.approx(8e-5)  # plateau below floor
    assert _lr(sm, 15, min_lr=8e-5) == pytest.approx(8e-5)  # interpolation below floor
    assert _lr(sm, 19, min_lr=1e-5) == pytest.approx(2.5e-5)
    assert _lr(sm, 20, min_lr=1e-5) == pytest.approx(1e-5)


def test_past_max_steps_returns_min_lr() -> None:
    assert _lr(_tiny_manager(), 25, min_lr=1e-6) == pytest.approx(1e-6)
    assert _lr(_tiny_manager(cooldown=0), 25, min_lr=1e-6) == pytest.approx(1e-6)


def test_no_warmup_and_no_cooldown_is_a_flat_plateau() -> None:
    sm = _tiny_manager(warmup=0, cooldown=0)
    assert _lr(sm, 0) == pytest.approx(3e-4)
    assert _lr(sm, 19) == pytest.approx(5e-5)
    assert _lr(sm, 20) == pytest.approx(5e-5)  # step == max_steps is still the last stage's LR


def test_warmup_ramps_monotonically_to_the_first_stage_lr_up_to_the_transition() -> None:
    """
    The longest allowed warmup (5 of stage a's 6 plain steps) never targets the next stage's LR: no dip.
    """

    sm = _tiny_manager(warmup=5)
    lrs = [_lr(sm, step) for step in range(8)]
    assert lrs[:6] == pytest.approx([3e-4 * step / 5 for step in range(6)])
    assert lrs[6:] == pytest.approx([3e-4, 2e-4]), "steps 6 and 7 are the transition (progress 0 and 0.5)"
    assert all(b >= a for a, b in zip(lrs[:5], lrs[1:6]))


def test_warmup_starts_at_zero_even_with_min_lr() -> None:
    """
    The global warmup ramps from 0 (not from `min_lr`); only plateau/transition/cooldown are floored, as upstream.
    """

    sm = _tiny_manager()
    assert _lr(sm, 0, min_lr=1e-5) == 0.0
    assert _lr(sm, 1, min_lr=1e-5) == pytest.approx(1.5e-4)


def test_warmup_and_cooldown_boundaries_are_continuous_with_the_plateau() -> None:
    """
    `step == warmup_steps` is the first plateau step; `step == max_steps - cooldown_steps` the last one.
    """

    sm = _tiny_manager(warmup=4, cooldown=3)  # 20 steps
    assert [_lr(sm, s) for s in range(5)] == pytest.approx([0.0, 0.75e-4, 1.5e-4, 2.25e-4, 3e-4])
    assert [_lr(sm, s) for s in range(16, 21)] == pytest.approx([5e-5, 5e-5, 5e-5 * 2 / 3, 5e-5 / 3, 0.0])


def test_zero_length_transition_switches_lr_hard_at_the_boundary() -> None:
    tps = 4 * 256
    stages = [resolved_stage("a", 8 * tps, base_lr=3e-4, transition_pct=0.05), resolved_stage("b", 8 * tps, base_lr=1e-4)]
    sm = StageManager(stages, tokens_per_step=1024)
    assert sm.boundaries[0].transition_start_step == sm.boundaries[0].end_step == 8
    assert [_lr(sm, s) for s in (6, 7, 8, 9)] == pytest.approx([3e-4, 3e-4, 1e-4, 1e-4])


def _continuity_manager() -> StageManager:
    tps = 8 * 128
    stages = [
        resolved_stage("s0", tokens=100 * tps, base_lr=1e-3, transition_pct=0.1),  # transition 10 steps
        resolved_stage("s1", tokens=60 * tps, base_lr=2e-4, transition_pct=0.25),  # transition 15 steps
        resolved_stage("s2", tokens=40 * tps, base_lr=6e-4, transition_pct=0.5),  # transition 20 steps
        resolved_stage("s3", tokens=30 * tps, base_lr=1e-4, transition_pct=0.0),
    ]
    return StageManager(stages, tokens_per_step=1024, warmup_steps=5, cooldown_steps=10)


def test_multistage_schedule_is_continuous_at_every_boundary() -> None:
    sm = _continuity_manager()
    lrs = [_lr(sm, s) for s in range(sm.total_steps + 1)]
    increments = [1e-3 / 5, 1e-4 / 10]
    pairs = list(zip(sm.stages, sm.stages[1:], sm.boundaries))  # (stage, next stage, the stage's boundary)
    for stage, nxt, b in pairs:
        n = b.end_step - b.transition_start_step
        increments.append(abs(nxt.base_lr - stage.base_lr) / n)
    max_jump = max(increments)
    for step in range(1, len(lrs)):
        assert abs(lrs[step] - lrs[step - 1]) <= max_jump * (1 + 1e-9), f"jump at step {step}"
    # transitions hit the exact stage base LRs at both ends
    for stage, nxt, b in pairs:
        assert lrs[b.transition_start_step] == pytest.approx(stage.base_lr)
        assert lrs[b.end_step] == pytest.approx(nxt.base_lr)
    # plateaus between warmup/transitions are flat at the stage LR
    assert all(lr == pytest.approx(1e-3) for lr in lrs[5:88])
    assert all(lr == pytest.approx(2e-4) for lr in lrs[100:145])
    assert lrs[sm.total_steps] == pytest.approx(0.0)


def test_schedule_plateaus_and_cooldown_of_continuity_config() -> None:
    sm = _continuity_manager()  # boundaries 0-100-160-200-230
    assert [b.end_step for b in sm.boundaries] == [100, 160, 200, 230]
    assert _lr(sm, 2) == pytest.approx(4e-4)
    assert _lr(sm, 95) == pytest.approx(1e-3 + (2e-4 - 1e-3) * 5 / 10)
    assert _lr(sm, 190) == pytest.approx(6e-4 + (1e-4 - 6e-4) * 10 / 20)
    assert _lr(sm, 225) == pytest.approx(1e-4 * 5 / 10)
