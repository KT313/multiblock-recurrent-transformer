# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the stage manager: hand-computed boundaries, world-size independence, validation, stage info inside
transitions and stage-end checkpoint steps."""

import pytest

from training.stage_manager import StageBoundary, StageInfo, StageManager, TrainingStage


def tiny_stages() -> list[TrainingStage]:
    return [
        TrainingStage("pretrain_a", tokens=8192, base_lr=3e-4, transition_pct=0.25),
        TrainingStage("pretrain_b", tokens=8192, base_lr=1e-4, transition_pct=0.25),
        TrainingStage("finetune", tokens=4096, base_lr=5e-5, transition_pct=0.0),
    ]


def two_stages() -> list[TrainingStage]:
    return [
        TrainingStage("s0", tokens=100_000, base_lr=1e-3, transition_pct=0.1),
        TrainingStage("s1", tokens=50_000, base_lr=2e-4, transition_pct=0.3),
    ]


def _bounds(sm: StageManager) -> list[tuple[int, int, int, int]]:
    return [(b.start_step, b.end_step, b.transition_start_step, b.transition_end_step) for b in sm.boundaries]


def test_training_stage_defaults() -> None:
    stage = TrainingStage("s", tokens=10, base_lr=1e-3)
    assert stage.transition_pct == 0.05 and stage.train_data == [] and stage.val_data == []


def test_tiny_config_boundaries_by_hand() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256, warmup_steps=2, cooldown_steps=2)
    assert sm.tokens_per_step == 1024
    # 8192 // 1024 = 8 steps, transition int(8192 * 0.25) // 1024 = 2 steps at the END of the stage
    assert _bounds(sm) == [(0, 8, 6, 8), (8, 16, 14, 16), (16, 20, 20, 20)]
    assert sm.total_steps == 20
    assert [b.stage_name for b in sm.boundaries] == ["pretrain_a", "pretrain_b", "finetune"]
    assert [b.base_lr for b in sm.boundaries] == [3e-4, 1e-4, 5e-5]
    assert [b.tokens for b in sm.boundaries] == [8192, 8192, 4096]
    assert sm._calculate_stage_boundaries() == sm.boundaries


def test_two_stage_config_boundaries_by_hand() -> None:
    sm = StageManager(two_stages(), world_batch_size=8, block_size=128)
    # tps 1024: 100000 // 1024 = 97, transition 10000 // 1024 = 9; 50000 // 1024 = 48, no transition after the last
    assert _bounds(sm) == [(0, 97, 88, 97), (97, 145, 145, 145)]
    assert sm.total_steps == 145


def test_partial_steps_are_truncated() -> None:
    stages = [TrainingStage("s", tokens=1024 * 3 + 1000, base_lr=1e-3)]
    sm = StageManager(stages, world_batch_size=4, block_size=256)
    assert sm.total_steps == 3


@pytest.mark.parametrize(
    "pct,expected_transition_steps",
    [(0.3, 2), (0.375, 3), (0.05, 0), (0.124, 0), (0.125, 1)],  # int(8192 * pct) // 1024, truncated twice
)
def test_transition_pct_truncates_tokens_then_steps(pct: float, expected_transition_steps: int) -> None:
    stages = [TrainingStage("a", 8192, 1e-3, transition_pct=pct), TrainingStage("b", 8192, 1e-4)]
    sm = StageManager(stages, world_batch_size=4, block_size=256)
    b = sm.boundaries[0]
    assert (b.transition_start_step, b.transition_end_step) == (8 - expected_transition_steps, 8)


def test_zero_length_transition_has_no_transition_steps_but_a_stage_end_checkpoint() -> None:
    """A transition shorter than one step (tokens * pct < tokens_per_step) collapses to nothing: no step is
    `in_transition`, the data/LR switch hard at the boundary and the stage-end checkpoint lands at end - 1."""
    stages = [TrainingStage("a", 8192, 1e-3, transition_pct=0.05), TrainingStage("b", 8192, 1e-4)]
    sm = StageManager(stages, world_batch_size=4, block_size=256)
    assert _bounds(sm) == [(0, 8, 8, 8), (8, 16, 16, 16)]
    assert not any(sm.get_stage_info(s).in_transition for s in range(16))
    assert (sm.get_stage_info(7).stage_idx, sm.get_stage_info(8).stage_idx) == (0, 1)
    assert sm.get_stage_info(7).transition_progress == 0.0
    assert sm.should_save_stage_checkpoint(7) == (True, "stage-0_end")
    assert "Transition OUT: 0 steps (5.0% of current stage)" in sm.get_stage_summary()


@pytest.mark.parametrize("stages,wbs,bs", [(tiny_stages(), 4, 256), (two_stages(), 8, 128)])
def test_world_size_does_not_change_boundaries(stages: list[TrainingStage], wbs: int, bs: int) -> None:
    one = StageManager(stages, world_batch_size=wbs, block_size=bs, world_size=1, micro_batch_size=1)
    four = StageManager(stages, world_batch_size=wbs, block_size=bs, world_size=4, micro_batch_size=1)
    assert _bounds(one) == _bounds(four)
    assert one.total_steps == four.total_steps
    summary = four.get_stage_summary()
    assert f"{stages[0].tokens:,} total ({stages[0].tokens // 4:,} per device)" in summary
    assert "World size: 4" in summary


def test_world_batch_divisibility_validation() -> None:
    with pytest.raises(ValueError, match="divisible by world_size"):
        StageManager(tiny_stages(), world_batch_size=6, block_size=256, world_size=4)
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        StageManager(tiny_stages(), world_batch_size=8, block_size=256, world_size=2, micro_batch_size=3)
    with pytest.raises(ValueError, match="at least one stage"):
        StageManager([], world_batch_size=4, block_size=256)


@pytest.mark.parametrize("warmup,cooldown", [(8, 0), (9, 0), (0, 4), (0, 5)])
def test_warmup_or_cooldown_too_long_raises(warmup: int, cooldown: int) -> None:
    with pytest.raises(ValueError, match="must be less than"):
        StageManager(tiny_stages(), world_batch_size=4, block_size=256, warmup_steps=warmup, cooldown_steps=cooldown)


def test_warmup_and_cooldown_that_fit_are_accepted() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256, warmup_steps=7, cooldown_steps=3)
    assert sm.total_steps == 20
    sm._validate_lr_schedule()  # idempotent re-check
    sm.warmup_steps = 8
    with pytest.raises(ValueError, match="warmup_steps"):
        sm._validate_lr_schedule()


def test_get_stage_info_inside_and_outside_transitions() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    info = sm.get_stage_info(3)
    assert isinstance(info, StageInfo)
    assert (info.stage_idx, info.stage_name, info.base_lr, info.in_transition) == (0, "pretrain_a", 3e-4, False)
    assert info.prev_stage_idx is None and info.prev_base_lr is None
    assert info.transition_progress == 0.0 and info.stage_progress == pytest.approx(3 / 8)

    # Inside the transition out of stage 0 the info already names the next stage
    info = sm.get_stage_info(7)
    assert (info.stage_idx, info.stage_name, info.base_lr) == (1, "pretrain_b", 1e-4)
    assert info.in_transition and info.prev_stage_idx == 0 and info.prev_base_lr == 3e-4
    assert info.transition_progress == pytest.approx(0.5)
    assert info.stage_progress == pytest.approx(7 / 8)

    info = sm.get_stage_info(6)
    assert info.in_transition and info.transition_progress == 0.0 and info.stage_idx == 1

    info = sm.get_stage_info(8)  # first step fully in stage 1
    assert (info.stage_idx, info.in_transition, info.stage_progress) == (1, False, 0.0)

    info = sm.get_stage_info(15)
    assert (info.stage_idx, info.stage_name, info.prev_stage_idx) == (2, "finetune", 1)
    assert info.transition_progress == pytest.approx(0.5)


def test_get_stage_info_past_the_end_reports_last_stage_complete() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    for step in (20, 999):
        info = sm.get_stage_info(step)
        assert (info.stage_idx, info.in_transition, info.stage_progress) == (2, False, 1.0)
        assert info.prev_stage_idx is None


def test_should_save_stage_checkpoint_exact_steps() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    expected = {5: (True, "stage-0_end"), 13: (True, "stage-1_end")}
    for step in range(25):
        assert sm.should_save_stage_checkpoint(step) == expected.get(step, (False, ""))


def test_stage_boundary_helpers() -> None:
    b = StageBoundary(
        0, "s", start_step=10, end_step=20, transition_start_step=16, transition_end_step=20, base_lr=1e-3, tokens=1
    )
    assert b.is_in_stage(10) and b.is_in_stage(19) and not b.is_in_stage(20) and not b.is_in_stage(9)
    assert b.is_in_transition(16) and not b.is_in_transition(15) and not b.is_in_transition(20)
    assert b.get_transition_progress(18) == pytest.approx(0.5)
    assert b.get_transition_progress(12) == 0.0
    assert b.get_stage_progress(5) == 0.0 and b.get_stage_progress(20) == 1.0
    assert b.get_stage_progress(15) == pytest.approx(0.5)
    empty = StageBoundary(0, "s", 10, 10, 10, 10, 1e-3, 0)
    assert empty.get_stage_progress(10) == 1.0 and empty.get_transition_progress(10) == 0.0


def test_stage_summary_mentions_every_stage_and_step_counts() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    summary = sm.get_stage_summary()
    for name in ("pretrain_a", "pretrain_b", "finetune"):
        assert name in summary
    assert "Total optimizer steps: 20" in summary
    assert "Transition OUT: 2 steps (25.0% of current stage)" in summary
    assert "Main training: 6 steps" in summary
    assert summary.count("Transition OUT") == 2  # none after the last stage
