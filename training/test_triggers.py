# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests of the percentage-to-step conversion and the combined triggers.
"""

from training.triggers import StepTriggers, progress_steps


def test_progress_steps_round_to_steps_with_first_and_last_step_at_the_ends() -> None:
    assert progress_steps([0, 50, 70, 90, 100], 200) == [1, 100, 140, 180, 200]
    assert progress_steps([100, 0, 50, 50], 20) == [1, 10, 20]  # sorted, unique
    assert progress_steps([33.3, 66.6], 3) == [1, 2]
    assert progress_steps([], 10) == []
    assert progress_steps([0, 100], 1) == [1]


def test_triggers_combine_the_interval_and_the_progress_steps() -> None:
    triggers = StepTriggers.from_settings(50, [0, 50, 70, 90, 100], 200)
    assert triggers.listed(200) == [1, 50, 100, 140, 150, 180, 200]
    assert triggers.due(140) and triggers.due(150) and not triggers.due(2) and not triggers.due(199)
    assert StepTriggers.from_settings(0, [], 200).listed(200) == []
    assert StepTriggers.from_settings(7, [], 20).listed(20) == [7, 14]
