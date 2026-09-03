# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the stage manager: hand-computed boundaries, world-size independence, validation, stage info inside
transitions, the stage the run is entering, the interpolated per-step data weights and stage-end checkpoint steps."""

import pytest

from training.data.dataset_resolver import ResolvedStage
from training.stage_manager import StageBoundary, StageInfo, StageManager
from training.testing.stages import resolved_stage


def tiny_stages() -> list[ResolvedStage]:
    return [
        resolved_stage("pretrain_a", tokens=8192, base_lr=3e-4, transition_pct=0.25),
        resolved_stage("pretrain_b", tokens=8192, base_lr=1e-4, transition_pct=0.25),
        resolved_stage("finetune", tokens=4096, base_lr=5e-5, transition_pct=0.0),
    ]


def two_stages() -> list[ResolvedStage]:
    return [
        resolved_stage("s0", tokens=100_000, base_lr=1e-3, transition_pct=0.1),
        resolved_stage("s1", tokens=50_000, base_lr=2e-4, transition_pct=0.3),
    ]


def _bounds(sm: StageManager) -> list[tuple[int, int, int]]:
    return [(b.start_step, b.end_step, b.transition_start_step) for b in sm.boundaries]


def test_tiny_config_boundaries_by_hand() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256, warmup_steps=2, cooldown_steps=2)
    assert sm.tokens_per_step == 1024
    # 8192 // 1024 = 8 steps, transition int(8192 * 0.25) // 1024 = 2 steps at the END of the stage
    assert _bounds(sm) == [(0, 8, 6), (8, 16, 14), (16, 20, 20)]
    assert sm.total_steps == 20
    assert [s.name for s in sm.stages] == ["pretrain_a", "pretrain_b", "finetune"]
    assert sm._calculate_stage_boundaries() == sm.boundaries


def test_two_stage_config_boundaries_by_hand() -> None:
    sm = StageManager(two_stages(), world_batch_size=8, block_size=128)
    # tps 1024: 100000 // 1024 = 97, transition 10000 // 1024 = 9; 50000 // 1024 = 48, no transition after the last
    assert _bounds(sm) == [(0, 97, 88), (97, 145, 145)]
    assert sm.total_steps == 145


def test_partial_steps_are_truncated() -> None:
    stages = [resolved_stage("s", tokens=1024 * 3 + 1000, base_lr=1e-3)]
    sm = StageManager(stages, world_batch_size=4, block_size=256)
    assert sm.total_steps == 3


@pytest.mark.parametrize(
    "pct,expected_transition_steps",
    [(0.3, 2), (0.375, 3), (0.05, 0), (0.124, 0), (0.125, 1)],  # int(8192 * pct) // 1024, truncated twice
)
def test_transition_pct_truncates_tokens_then_steps(pct: float, expected_transition_steps: int) -> None:
    stages = [resolved_stage("a", 8192, 1e-3, transition_pct=pct), resolved_stage("b", 8192, 1e-4)]
    sm = StageManager(stages, world_batch_size=4, block_size=256)
    b = sm.boundaries[0]
    assert (b.transition_start_step, b.end_step) == (8 - expected_transition_steps, 8)


def test_zero_length_transition_has_no_transition_steps_but_a_stage_end_checkpoint() -> None:
    """A transition shorter than one step (tokens * pct < tokens_per_step) collapses to nothing: no step is in a
    transition, the data/LR switch hard at the boundary and the stage-end checkpoint lands at end - 1."""
    stages = [resolved_stage("a", 8192, 1e-3, transition_pct=0.05), resolved_stage("b", 8192, 1e-4)]
    sm = StageManager(stages, world_batch_size=4, block_size=256)
    assert _bounds(sm) == [(0, 8, 8), (8, 16, 16)]
    assert all(sm.get_stage_info(s).transition_to is None for s in range(16))
    assert (sm.get_stage_info(7).stage_index, sm.get_stage_info(8).stage_index) == (0, 1)
    assert sm.get_stage_info(7).transition_progress == 0.0
    assert sm.stage_ending_at(7) == 0
    assert "Transition OUT: 0 steps (5.0% of current stage)" in sm.get_stage_summary()


@pytest.mark.parametrize("stages,wbs,bs", [(tiny_stages(), 4, 256), (two_stages(), 8, 128)])
def test_world_size_does_not_change_boundaries(stages: list[ResolvedStage], wbs: int, bs: int) -> None:
    one = StageManager(stages, world_batch_size=wbs, block_size=bs, world_size=1, micro_batch_size=1)
    four = StageManager(stages, world_batch_size=wbs, block_size=bs, world_size=4, micro_batch_size=1)
    assert _bounds(one) == _bounds(four)
    assert one.total_steps == four.total_steps
    summary = four.get_stage_summary()
    assert f"Token budget: {stages[0].tokens:,}" in summary and "per device" not in summary
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
    """Stage 0 has 8 steps of which the last 2 are its transition: the warmup must end before step 6."""
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256, warmup_steps=5, cooldown_steps=3)
    assert sm.total_steps == 20
    sm._validate_lr_schedule()  # idempotent re-check
    sm.warmup_steps = 6
    with pytest.raises(ValueError, match="warmup_steps"):
        sm._validate_lr_schedule()


def test_a_stage_shorter_than_one_step_is_rejected() -> None:
    stages = [resolved_stage("a", tokens=8192, base_lr=3e-4), resolved_stage("b", tokens=100, base_lr=1e-4), resolved_stage("c", tokens=8192, base_lr=5e-5)]
    with pytest.raises(ValueError, match="stage 'b' is shorter than one optimizer step"):
        StageManager(stages, world_batch_size=4, block_size=256)


def test_a_transition_as_long_as_its_stage_is_rejected() -> None:
    stages = [resolved_stage("a", tokens=2100, base_lr=3e-4, transition_pct=0.99), resolved_stage("b", tokens=8192, base_lr=1e-4)]  # 2 steps, 2 in transition
    with pytest.raises(ValueError, match="stage 'a': the transition must be shorter"):
        StageManager(stages, world_batch_size=4, block_size=256)


def test_get_stage_info_inside_and_outside_transitions() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    info = sm.get_stage_info(3)
    assert isinstance(info, StageInfo)
    assert (info.stage_index, info.transition_to, info.transition_progress) == (0, None, 0.0)
    assert info.stage_progress == pytest.approx(3 / 8)

    # Inside the transition out of stage 0 the info stays with stage 0 and names stage 1 as the one being entered
    info = sm.get_stage_info(7)
    assert (info.stage_index, info.transition_to) == (0, 1)
    assert info.transition_progress == pytest.approx(0.5)
    assert info.stage_progress == pytest.approx(7 / 8)

    info = sm.get_stage_info(6)
    assert (info.stage_index, info.transition_to, info.transition_progress) == (0, 1, 0.0)

    info = sm.get_stage_info(8)  # first step of stage 1
    assert (info.stage_index, info.transition_to, info.stage_progress) == (1, None, 0.0)

    info = sm.get_stage_info(15)
    assert (info.stage_index, info.transition_to) == (1, 2)
    assert info.transition_progress == pytest.approx(0.5)


def test_entering_stage_at_names_the_incoming_stage_inside_a_transition() -> None:
    """The stage the validation loader and a checkpoint's `stage` follow: the stage containing the step, except
    inside a transition window, where it is the stage being entered (tiny: windows [6, 8) and [14, 16))."""
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    assert [sm.entering_stage_at(step) for step in range(26)] == [0] * 6 + [1] * 8 + [2] * 12
    assert [sm.get_stage_info(step).stage_index for step in range(26)] == [0] * 8 + [1] * 8 + [2] * 10


def weighted_stages() -> list[ResolvedStage]:
    """The tiny boundaries ((0,8,6), (8,16,14), (16,20)) with a source `a` leaving, `b` shared and `c` entering
    across the first transition."""
    return [
        resolved_stage("s0", tokens=8192, base_lr=3e-4, transition_pct=0.25, train_weights={"a": 0.7, "b": 0.3}),
        resolved_stage("s1", tokens=8192, base_lr=1e-4, transition_pct=0.25, train_weights={"b": 0.5, "c": 0.5}),
        resolved_stage("s2", tokens=4096, base_lr=5e-5, transition_pct=0.0, train_weights={"c": 1.0}),
    ]


def test_data_weights_outside_a_transition_are_the_stage_constants() -> None:
    sm = StageManager(weighted_stages(), world_batch_size=4, block_size=256)
    assert sm.data_weights(0) == {"a": 0.7, "b": 0.3}
    assert sm.data_weights(5) == {"a": 0.7, "b": 0.3}  # last plain step of stage 0
    assert sm.data_weights(8) == {"b": 0.5, "c": 0.5}  # first step fully in stage 1
    assert sm.data_weights(19) == {"c": 1.0}
    assert sm.data_weights(25) == {"c": 1.0}  # past the end: the last stage's constants


def test_data_weights_interpolate_linearly_inside_a_transition() -> None:
    sm = StageManager(weighted_stages(), world_batch_size=4, block_size=256)
    # progress 0 (step 6): still entirely the outgoing stage's mix; the entering source is present at weight 0
    assert sm.data_weights(6) == {"a": 0.7, "b": 0.3, "c": 0.0}
    mid = sm.data_weights(7)  # progress 0.5
    assert mid["a"] == pytest.approx(0.35)  # leaving: ramps to 0
    assert mid["b"] == pytest.approx(0.4)  # 0.5 × 0.3 + 0.5 × 0.5
    assert mid["c"] == pytest.approx(0.25)  # entering: ramps from 0
    assert sum(mid.values()) == pytest.approx(1.0)  # each stage sums to 1, so every interpolation does
    late = sm.data_weights(15)  # 1 -> 2 transition at progress 0.5
    assert late == pytest.approx({"b": 0.25, "c": 0.75})
    assert "a" not in late  # a source in neither stage of the pair is absent (weight 0, never drawn)


def test_data_weights_returns_a_copy() -> None:
    sm = StageManager(weighted_stages(), world_batch_size=4, block_size=256)
    weights = sm.data_weights(0)
    weights["a"] = 0.0
    assert sm.data_weights(0)["a"] == 0.7


def test_get_stage_info_past_the_end_reports_last_stage_complete() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    for step in (20, 999):
        assert sm.get_stage_info(step) == StageInfo(stage_index=2, stage_progress=1.0, transition_to=None, transition_progress=0.0)


def test_stage_ending_at_exact_steps() -> None:
    """The stage index only at the last step before each transition (5 -> 0, 13 -> 1), None everywhere else,
    including the last stage, which has no transition after it."""
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    expected = {5: 0, 13: 1}
    for step in range(25):
        assert sm.stage_ending_at(step) == expected.get(step)
    assert [b.transition_start_step for b in sm.boundaries[:-1]] == [6, 14]
    assert all(sm.stage_ending_at(step) is None for step in range(sm.boundaries[-1].start_step, sm.total_steps + 5))


def test_stage_boundary_helpers() -> None:
    b = StageBoundary(start_step=10, end_step=20, transition_start_step=16)
    assert b.is_in_stage(10) and b.is_in_stage(19) and not b.is_in_stage(20) and not b.is_in_stage(9)
    assert b.is_in_transition(16) and not b.is_in_transition(15) and not b.is_in_transition(20)
    assert b.get_transition_progress(18) == pytest.approx(0.5)
    assert b.get_transition_progress(12) == 0.0
    assert b.get_stage_progress(5) == 0.0 and b.get_stage_progress(20) == 1.0
    assert b.get_stage_progress(15) == pytest.approx(0.5)
    empty = StageBoundary(10, 10, 10)
    assert empty.get_stage_progress(10) == 1.0 and empty.get_transition_progress(10) == 0.0


def test_stage_summary_mentions_every_stage_and_step_counts() -> None:
    sm = StageManager(tiny_stages(), world_batch_size=4, block_size=256)
    summary = sm.get_stage_summary()
    for name in ("pretrain_a", "pretrain_b", "finetune"):
        assert name in summary
    assert "Total optimizer steps: 20" in summary
    assert "Token budget: 8,192" in summary and "Base LR: 3.00e-04" in summary
    assert "Transition OUT: 2 steps (25.0% of current stage)" in summary
    assert "Main training: 6 steps" in summary
    assert summary.count("Transition OUT") == 2  # none after the last stage
