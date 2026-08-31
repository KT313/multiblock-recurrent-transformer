# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Multi-stage training manager: stage boundaries, transitions and stage-dependent data weights.

All steps are OPTIMIZER steps (one world batch of `world_batch_size * block_size` tokens each). Step counts are
therefore independent of the world size; `world_size` is only used for the per-device token summary and for the
sanity check that the world batch splits evenly across devices.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingStage:
    """One stage of the curriculum: token budget, base LR and transition length (the data lives in the resolver's
    `ResolvedStage`; the manager only turns budgets into step boundaries)."""

    name: str
    tokens: int
    base_lr: float
    transition_pct: float = 0.05  # transition OUT of this stage, as a fraction of this stage's tokens


@dataclass
class StageInfo:
    """Information about the current training stage and transition state."""

    stage_idx: int  # Current stage index (0-based)
    stage_name: str
    base_lr: float
    in_transition: bool
    transition_progress: float  # Progress through transition (0-1), 0 if not in transition
    stage_progress: float  # Progress through current stage (0-1)
    prev_stage_idx: Optional[int]  # Previous stage index (None if not transitioning)
    prev_base_lr: Optional[float]  # Previous stage's base LR (None if not transitioning)


@dataclass
class StageBoundary:
    """Step boundaries for a single training stage."""

    stage_idx: int
    stage_name: str
    start_step: int  # First step of this stage (inclusive)
    end_step: int  # Last step of this stage (exclusive)
    transition_start_step: int  # Step where transition to next stage begins
    transition_end_step: int  # Step where transition to next stage ends
    base_lr: float
    tokens: int

    def is_in_stage(self, step: int) -> bool:
        """Check if step is within this stage (including transition)."""
        return self.start_step <= step < self.end_step

    def is_in_transition(self, step: int) -> bool:
        """Check if step is in the transition period to the next stage."""
        return self.transition_start_step <= step < self.transition_end_step

    def get_transition_progress(self, step: int) -> float:
        """Progress through the transition period (0-1); 0 if not in transition."""
        if not self.is_in_transition(step):
            return 0.0
        transition_length = self.transition_end_step - self.transition_start_step
        if transition_length == 0:
            return 0.0
        return (step - self.transition_start_step) / transition_length

    def get_stage_progress(self, step: int) -> float:
        """Progress through the stage (0-1); 0 before the stage, 1 after it."""
        if step < self.start_step:
            return 0.0
        if step >= self.end_step:
            return 1.0
        stage_length = self.end_step - self.start_step
        if stage_length == 0:
            return 1.0
        return (step - self.start_step) / stage_length


class StageManager:
    """Manager for multi-stage training with smooth transitions (steps are optimizer steps)."""

    def __init__(
        self,
        stages: list[TrainingStage],
        world_batch_size: int,
        block_size: int,
        world_size: int = 1,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        micro_batch_size: Optional[int] = None,
    ) -> None:
        if not stages:
            raise ValueError("stages must contain at least one stage")
        if world_batch_size % world_size != 0:
            raise ValueError(f"world_batch_size ({world_batch_size}) must be divisible by world_size ({world_size})")
        if micro_batch_size is not None and world_batch_size % (micro_batch_size * world_size) != 0:
            raise ValueError(
                f"world_batch_size ({world_batch_size}) must be a multiple of micro_batch_size * world_size "
                f"({micro_batch_size} * {world_size})"
            )
        self.stages = stages
        self.world_batch_size = world_batch_size
        self.block_size = block_size
        self.world_size = world_size
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.tokens_per_step = world_batch_size * block_size

        self.boundaries = self._calculate_stage_boundaries()
        self.total_steps = self.boundaries[-1].end_step
        self._validate_lr_schedule()

    def _calculate_stage_boundaries(self) -> list[StageBoundary]:
        """Convert token budgets into optimizer-step boundaries including transition periods."""
        boundaries = []
        current_step = 0
        for idx, stage in enumerate(self.stages):
            stage_steps = stage.tokens // self.tokens_per_step

            # Transition OUT of this stage, as a percentage of this stage's tokens; none after the last stage
            if idx < len(self.stages) - 1:
                transition_tokens = int(stage.tokens * stage.transition_pct)
                transition_steps = transition_tokens // self.tokens_per_step
            else:
                transition_steps = 0

            start_step = current_step
            end_step = current_step + stage_steps
            boundaries.append(
                StageBoundary(
                    stage_idx=idx,
                    stage_name=stage.name,
                    start_step=start_step,
                    end_step=end_step,
                    transition_start_step=end_step - transition_steps,
                    transition_end_step=end_step,
                    base_lr=stage.base_lr,
                    tokens=stage.tokens,
                )
            )
            current_step = end_step
        return boundaries

    def _validate_lr_schedule(self) -> None:
        """Warmup must fit inside the first stage and cooldown inside the last stage."""
        if self.warmup_steps > 0:
            first_stage_steps = self.boundaries[0].end_step - self.boundaries[0].start_step
            if self.warmup_steps >= first_stage_steps:
                raise ValueError(
                    f"warmup_steps ({self.warmup_steps}) must be less than first stage steps ({first_stage_steps}). "
                    "Consider reducing warmup_steps or increasing stage 0 tokens."
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
        """Stage/transition state at `step`. Inside a transition the info already names the NEXT stage."""
        for boundary in self.boundaries:
            if boundary.is_in_stage(step):
                in_transition = boundary.is_in_transition(step)
                prev_stage_idx = None
                prev_base_lr = None
                current_stage_idx = boundary.stage_idx
                current_stage_name = boundary.stage_name
                current_base_lr = boundary.base_lr

                if in_transition:
                    # Transitions are stored at the END of a stage: we are leaving this stage and entering the next.
                    prev_stage_idx = boundary.stage_idx
                    prev_base_lr = boundary.base_lr
                    if boundary.stage_idx + 1 < len(self.boundaries):
                        next_boundary = self.boundaries[boundary.stage_idx + 1]
                        current_stage_idx = next_boundary.stage_idx
                        current_stage_name = next_boundary.stage_name
                        current_base_lr = next_boundary.base_lr

                return StageInfo(
                    stage_idx=current_stage_idx,
                    stage_name=current_stage_name,
                    base_lr=current_base_lr,
                    in_transition=in_transition,
                    transition_progress=boundary.get_transition_progress(step),
                    stage_progress=boundary.get_stage_progress(step),
                    prev_stage_idx=prev_stage_idx,
                    prev_base_lr=prev_base_lr,
                )

        # Past all stages: report the last stage as complete
        last_boundary = self.boundaries[-1]
        return StageInfo(
            stage_idx=last_boundary.stage_idx,
            stage_name=last_boundary.stage_name,
            base_lr=last_boundary.base_lr,
            in_transition=False,
            transition_progress=0.0,
            stage_progress=1.0,
            prev_stage_idx=None,
            prev_base_lr=None,
        )

    def should_save_stage_checkpoint(self, step: int) -> tuple[bool, str]:
        """Whether `step` is the last step before a transition starts, and the checkpoint suffix to use."""
        for boundary in self.boundaries[:-1]:  # the last stage has no transition after it
            if step == boundary.transition_start_step - 1:
                return True, f"stage-{boundary.stage_idx}_end"
        return False, ""

    def get_stage_summary(self) -> str:
        """Human-readable summary of the stage configuration (verify the boundaries by hand when in doubt)."""
        lines = ["Multi-Stage Training Configuration:"]
        lines.append(f"  Total stages: {len(self.stages)}")
        lines.append(f"  Total optimizer steps: {self.total_steps:,}")
        lines.append(f"  Tokens per optimizer step: {self.tokens_per_step:,} (world batch {self.world_batch_size} x block {self.block_size})")
        lines.append(f"  World size: {self.world_size}")
        lines.append("")

        for boundary in self.boundaries:
            stage_steps = boundary.end_step - boundary.start_step
            transition_steps = boundary.transition_end_step - boundary.transition_start_step
            main_steps = stage_steps - transition_steps

            lines.append(f"Stage {boundary.stage_idx}: {boundary.stage_name}")
            lines.append(
                f"  Token budget: {boundary.tokens:,} total ({boundary.tokens // self.world_size:,} per device)"
            )
            lines.append(f"  Optimizer steps: {stage_steps:,} (steps {boundary.start_step:,} - {boundary.end_step:,})")
            lines.append(f"    - Main training: {main_steps:,} steps")
            if boundary.stage_idx < len(self.stages) - 1:
                transition_pct = self.stages[boundary.stage_idx].transition_pct * 100
                lines.append(f"    - Transition OUT: {transition_steps:,} steps ({transition_pct:.1f}% of current stage)")
            lines.append(f"  Base LR: {boundary.base_lr:.2e}")
            lines.append("")

        return "\n".join(lines)
