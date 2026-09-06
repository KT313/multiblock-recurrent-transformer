# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Multi-stage training manager: stage boundaries, transitions and stage-dependent data weights.

The stages are the resolver's `ResolvedStage`s (token budget, base LR, transition length, sampling weights); the
manager turns budgets into optimizer-step boundaries and weights into a per-step schedule. All steps are OPTIMIZER
steps (one world batch of `world_batch_size * block_size` tokens each), so step counts are independent of the world
size; `world_size` only feeds the sanity check that the world batch splits evenly across devices.
"""

from dataclasses import dataclass
from typing import Optional

from training.data.dataset_resolver import ResolvedStage


@dataclass
class StageInfo:
    """
    Where a step stands: the stage whose boundary contains it (half-open: a stage's `end_step` belongs to the next),
    the progress through that stage and, inside the transition window at its end, the stage being entered and the
    progress through the window.
    """

    stage_index: int  # index of the stage whose boundary contains the step
    stage_progress: float  # progress through that stage (0-1)
    transition_to: Optional[int]  # index of the next stage while inside the transition window, else None
    transition_progress: float  # progress through the transition window (0-1), 0 outside it


@dataclass
class StageBoundary:
    """
    Step boundaries of one stage: `[start_step, end_step)`, with the transition to the next stage occupying the
    window `[transition_start_step, end_step)` at its end (empty when `transition_start_step == end_step`).
    """

    start_step: int  # first step of this stage (inclusive)
    end_step: int  # last step of this stage (exclusive); the transition to the next stage ends here too
    transition_start_step: int  # first step of the transition window at the end of the stage

    def is_in_stage(self, step: int) -> bool:
        """
        Whether `step` is within this stage (its transition window included).
        """

        return self.start_step <= step < self.end_step

    def is_in_transition(self, step: int) -> bool:
        """
        Whether `step` is inside the transition window to the next stage.
        """

        return self.transition_start_step <= step < self.end_step

    def get_transition_progress(self, step: int) -> float:
        """
        Progress through the transition window (0-1); 0 outside it.
        """

        if not self.is_in_transition(step):
            return 0.0
        return (step - self.transition_start_step) / (self.end_step - self.transition_start_step)

    def get_stage_progress(self, step: int) -> float:
        """
        Progress through the stage (0-1); 0 before the stage, 1 after it.
        """

        if step < self.start_step:
            return 0.0
        if step >= self.end_step:
            return 1.0
        stage_length = self.end_step - self.start_step
        if stage_length == 0:
            return 1.0
        return (step - self.start_step) / stage_length


class StageManager:
    """
    Manager for multi-stage training with smooth transitions (steps are optimizer steps).
    """

    def __init__(
        self,
        stages: list[ResolvedStage],
        world_batch_size: int,
        block_size: int,
        world_size: int = 1,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        micro_batch_size: Optional[int] = None,
        tokens_per_step: Optional[int] = None,
    ) -> None:
        """
        `tokens_per_step` given (sequence packing: `Settings.tokens_per_optimizer_step`) replaces
        `world_batch_size * block_size` as the size of one optimizer step; the sequence-based checks still run on
        `world_batch_size` (the validation batches), the `micro_batch_size` check is the caller's to skip.
        """

        if not stages:
            raise ValueError("stages must contain at least one stage")
        if world_batch_size % world_size != 0:
            raise ValueError(f"world_batch_size ({world_batch_size}) must be divisible by world_size ({world_size})")
        if micro_batch_size is not None and world_batch_size % (micro_batch_size * world_size) != 0:
            raise ValueError(
                f"world_batch_size ({world_batch_size}) must be a multiple of micro_batch_size * world_size "
                f"({micro_batch_size} * {world_size})"
            )
        if tokens_per_step is not None and tokens_per_step <= 0:
            raise ValueError(f"tokens_per_step must be positive, got {tokens_per_step}")
        self.stages = stages
        self.world_batch_size = world_batch_size
        self.block_size = block_size
        self.world_size = world_size
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.tokens_per_step = tokens_per_step if tokens_per_step is not None else world_batch_size * block_size
        self.packed = tokens_per_step is not None  # how the summary describes a step

        self.boundaries = self._calculate_stage_boundaries()
        self.total_steps = self.boundaries[-1].end_step
        self._validate_lr_schedule()

    def _calculate_stage_boundaries(self) -> list[StageBoundary]:
        """
        Convert token budgets into optimizer-step boundaries including transition periods.
        """

        boundaries = []
        current_step = 0
        for index, stage in enumerate(self.stages):
            stage_steps = stage.tokens // self.tokens_per_step

            # transition OUT of this stage, as a percentage of its tokens; none after the last stage
            if index < len(self.stages) - 1:
                transition_tokens = int(stage.tokens * stage.transition_pct)
                transition_steps = transition_tokens // self.tokens_per_step
            else:
                transition_steps = 0

            start_step = current_step
            end_step = current_step + stage_steps
            boundaries.append(
                StageBoundary(
                    start_step=start_step, end_step=end_step, transition_start_step=end_step - transition_steps
                )
            )
            current_step = end_step
        return boundaries

    def _validate_lr_schedule(self) -> None:
        """
        Warmup must end before the first stage's transition starts (the ramp targets the first stage's base LR)
        and cooldown must fit inside the last stage; every stage must be at least one step long, with its transition
        shorter than the stage.
        """

        for stage, boundary in zip(self.stages, self.boundaries):
            stage_steps = boundary.end_step - boundary.start_step
            if stage_steps < 1:
                raise ValueError(
                    f"stage {stage.name!r} is shorter than one optimizer step ({stage.tokens} tokens < "
                    f"{self.tokens_per_step} per step); increase its tokens or lower "
                    f"{'tokens_per_step' if self.packed else 'world_batch_size'}"
                )
            if boundary.end_step - boundary.transition_start_step >= stage_steps:
                raise ValueError(f"stage {stage.name!r}: the transition must be shorter than the stage")
        if self.warmup_steps > 0:
            plain_first_stage_steps = self.boundaries[0].transition_start_step - self.boundaries[0].start_step
            if self.warmup_steps >= plain_first_stage_steps:
                raise ValueError(
                    f"warmup_steps ({self.warmup_steps}) must be less than the first stage's steps before its "
                    f"transition ({plain_first_stage_steps}). Consider reducing warmup_steps or increasing stage 0 tokens."
                )
        if self.cooldown_steps > 0:
            last_boundary = self.boundaries[-1]
            last_stage_steps = last_boundary.end_step - last_boundary.start_step
            if self.cooldown_steps >= last_stage_steps:
                raise ValueError(
                    f"cooldown_steps ({self.cooldown_steps}) must be less than last stage steps ({last_stage_steps}). "
                    "Consider reducing cooldown_steps or increasing final stage tokens."
                )

    def get_stage_info(self, step: int) -> StageInfo:
        """
        Stage/transition state at `step`: the stage whose boundary contains it and, inside that stage's transition
        window, the stage being entered. Past the last stage: the last stage, complete.
        """

        for index, boundary in enumerate(self.boundaries):
            if boundary.is_in_stage(step):
                in_transition = boundary.is_in_transition(step)  # never true for the last stage (no window)
                return StageInfo(
                    stage_index=index,
                    stage_progress=boundary.get_stage_progress(step),
                    transition_to=index + 1 if in_transition else None,
                    transition_progress=boundary.get_transition_progress(step),
                )
        return StageInfo(
            stage_index=len(self.boundaries) - 1, stage_progress=1.0, transition_to=None, transition_progress=0.0
        )

    def entering_stage_at(self, step: int) -> int:
        """
        Index of the stage whose data the run is heading for at `step`: the stage being entered inside a transition
        window, otherwise the stage containing the step. The validation loader after `step` and the `stage` of a
        checkpoint written at `step` follow it.
        """

        stage_info = self.get_stage_info(step)
        return stage_info.stage_index if stage_info.transition_to is None else stage_info.transition_to

    def data_weights(self, step: int) -> dict[str, float]:
        """
        Sampling weight per train source at optimizer step `step` (from `ResolvedStage.train_weights`).

        Outside a transition: the current stage's weights. Inside one: (1 - p) * current + p * entering with
        p = transition_progress, over the union of both stages' sources (a source in neither is absent). This is
        the stage structure's ONLY effect on the training data; the per-source readers run through the whole run.
        """

        stage_info = self.get_stage_info(step)
        current = self.stages[stage_info.stage_index].train_weights
        if stage_info.transition_to is None:
            return dict(current)
        entering = self.stages[stage_info.transition_to].train_weights
        progress = stage_info.transition_progress
        return {
            name: (1.0 - progress) * current.get(name, 0.0) + progress * entering.get(name, 0.0)
            for name in {**current, **entering}
        }

    def stage_ending_at(self, step: int) -> int | None:
        """
        Index of the stage whose transition starts at `step + 1` (`step` is its last plain step), else None.

        The training loop writes the `-stage-{i}_end` checkpoint after that step.
        """

        for index, boundary in enumerate(self.boundaries[:-1]):  # the last stage has no transition after it
            if step == boundary.transition_start_step - 1:
                return index
        return None

    def get_stage_summary(self) -> str:
        """
        Human-readable summary of the stage configuration (verify the boundaries by hand when in doubt).
        """

        lines = ["Multi-Stage Training Configuration:"]
        lines.append(f"  Total stages: {len(self.stages)}")
        lines.append(f"  Total optimizer steps: {self.total_steps:,}")
        step_shape = "packed sequences" if self.packed else f"world batch {self.world_batch_size} x block {self.block_size}"
        lines.append(f"  Tokens per optimizer step: {self.tokens_per_step:,} ({step_shape})")
        lines.append(f"  World size: {self.world_size}")
        lines.append("")

        for index, (stage, boundary) in enumerate(zip(self.stages, self.boundaries)):
            stage_steps = boundary.end_step - boundary.start_step
            transition_steps = boundary.end_step - boundary.transition_start_step
            main_steps = stage_steps - transition_steps

            lines.append(f"Stage {index}: {stage.name}")
            lines.append(f"  Token budget: {stage.tokens:,}")
            lines.append(f"  Optimizer steps: {stage_steps:,} (steps {boundary.start_step:,} - {boundary.end_step:,})")
            lines.append(f"    - Main training: {main_steps:,} steps")
            if index < len(self.stages) - 1:
                transition_pct = stage.transition_pct * 100
                lines.append(f"    - Transition OUT: {transition_steps:,} steps ({transition_pct:.1f}% of current stage)")
            lines.append(f"  Base LR: {stage.base_lr:.2e}")
            lines.append("")

        return "\n".join(lines)
