# (c) 2025-2026 Tobias Kerner, part of the multi-block recurrent thesis work.
# Released under Apache-2.0 alongside seal-rg/recurrent-pretraining code. See LICENSE.
"""
Multi-Stage Training Manager

Handles stage boundaries, transitions, learning rate interpolation,
and data scheduler configuration for multi-stage training.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class StageInfo:
    """Information about the current training stage and transition state."""
    stage_idx: int                      # Current stage index (0-based)
    stage_name: str                     # Name of current stage
    base_lr: float                      # Base LR for current stage
    in_transition: bool                 # Whether we're in a transition period
    transition_progress: float          # Progress through transition (0-1), 0 if not in transition
    stage_progress: float               # Progress through current stage (0-1)
    prev_stage_idx: Optional[int]       # Previous stage index (None if in first stage or not transitioning)
    prev_base_lr: Optional[float]       # Previous stage's base LR (None if not transitioning)


@dataclass
class StageBoundary:
    """Boundaries for a single training stage."""
    stage_idx: int                      # Stage index (0-based)
    stage_name: str                     # Stage name
    start_step: int                     # First step of this stage (inclusive)
    end_step: int                       # Last step of this stage (exclusive)
    transition_start_step: int          # Step where transition to next stage begins
    transition_end_step: int            # Step where transition to next stage ends
    base_lr: float                      # Base learning rate for this stage
    tokens: int                         # Total tokens in this stage

    def is_in_stage(self, step: int) -> bool:
        """Check if step is within this stage (including transition)."""
        return self.start_step <= step < self.end_step

    def is_in_transition(self, step: int) -> bool:
        """Check if step is in the transition period to the next stage."""
        return self.transition_start_step <= step < self.transition_end_step

    def get_transition_progress(self, step: int) -> float:
        """Get progress through transition period (0-1). Returns 0 if not in transition."""
        if not self.is_in_transition(step):
            return 0.0
        transition_length = self.transition_end_step - self.transition_start_step
        if transition_length == 0:
            return 0.0
        return (step - self.transition_start_step) / transition_length

    def get_stage_progress(self, step: int) -> float:
        """Get progress through stage (0-1). Returns 0 if before stage, 1 if after."""
        if step < self.start_step:
            return 0.0
        if step >= self.end_step:
            return 1.0
        stage_length = self.end_step - self.start_step
        if stage_length == 0:
            return 1.0
        return (step - self.start_step) / stage_length


class StageManager:
    """Manager for multi-stage training with smooth transitions."""

    def __init__(self, cfg):
        """
        Initialize stage manager.

        Args:
            cfg: CLISettings object with training_stages defined
        """
        self.cfg = cfg
        self.stages = cfg.training_stages

        if not self.stages or len(self.stages) == 0:
            raise ValueError("training_stages must contain at least one stage when enable_multi_stage=True")

        # Calculate stage boundaries
        self.boundaries = self._calculate_stage_boundaries()

        # Store total steps for all stages
        self.total_steps = self.boundaries[-1].end_step

        # Validate warmup and cooldown don't conflict with stage boundaries
        self._validate_lr_schedule()

    def _calculate_stage_boundaries(self) -> list[StageBoundary]:
        """Calculate step boundaries for all stages including transition periods."""
        boundaries = []
        current_step = 0

        # Calculate world size (total number of devices across all nodes)
        world_size = self.cfg.devices * self.cfg.num_nodes

        for idx, stage in enumerate(self.stages):
            # Calculate tokens per device per step
            # Total tokens are divided across all devices
            tokens_per_device = stage.tokens // world_size
            tokens_per_step = self.cfg.micro_batch_size * self.cfg.block_size

            # Calculate steps in this stage (per device)
            stage_steps = tokens_per_device // tokens_per_step

            # Calculate transition period (as percentage of CURRENT stage's tokens for transition OUT)
            # For the last stage, no transition
            if idx < len(self.stages) - 1:
                transition_tokens = int(stage.tokens * stage.transition_pct)
                transition_tokens_per_device = transition_tokens // world_size
                transition_steps = transition_tokens_per_device // tokens_per_step
            else:
                transition_steps = 0

            # Stage boundaries
            start_step = current_step
            end_step = current_step + stage_steps
            transition_start_step = end_step - transition_steps
            transition_end_step = end_step

            boundary = StageBoundary(
                stage_idx=idx,
                stage_name=stage.name,
                start_step=start_step,
                end_step=end_step,
                transition_start_step=transition_start_step,
                transition_end_step=transition_end_step,
                base_lr=stage.base_lr,
                tokens=stage.tokens,
            )
            boundaries.append(boundary)

            current_step = end_step

        return boundaries

    def _validate_lr_schedule(self):
        """Validate that warmup and cooldown don't conflict with stage boundaries."""
        # Check warmup doesn't exceed first stage
        if self.cfg.warmup_steps > 0:
            first_stage_steps = self.boundaries[0].end_step - self.boundaries[0].start_step
            if self.cfg.warmup_steps >= first_stage_steps:
                raise ValueError(
                    f"warmup_steps ({self.cfg.warmup_steps}) must be less than first stage steps "
                    f"({first_stage_steps}). Consider reducing warmup_steps or increasing stage 0 tokens."
                )

        # Check cooldown doesn't exceed last stage
        if self.cfg.cooldown_steps > 0:
            last_boundary = self.boundaries[-1]
            last_stage_steps = last_boundary.end_step - last_boundary.start_step
            if self.cfg.cooldown_steps >= last_stage_steps:
                raise ValueError(
                    f"cooldown_steps ({self.cfg.cooldown_steps}) must be less than last stage steps "
                    f"({last_stage_steps}). Consider reducing cooldown_steps or increasing final stage tokens."
                )

    def get_current_stage_info(self, step: int) -> StageInfo:
        """
        Get information about the current stage at the given step.

        Args:
            step: Current training step

        Returns:
            StageInfo object with current stage information
        """
        # Find which stage we're in
        for boundary in self.boundaries:
            if boundary.is_in_stage(step):
                # Check if we're in transition
                in_transition = boundary.is_in_transition(step)
                transition_progress = boundary.get_transition_progress(step)
                stage_progress = boundary.get_stage_progress(step)

                # Get previous stage info if transitioning
                # Note: Transitions are stored at the END of a stage (transitioning OUT),
                # so when in_transition is True, we're leaving the current stage and
                # entering the next stage.
                prev_stage_idx = None
                prev_base_lr = None
                current_stage_idx = boundary.stage_idx
                current_stage_name = boundary.stage_name
                current_base_lr = boundary.base_lr

                if in_transition:
                    # Transitioning OUT of current stage INTO next stage
                    prev_stage_idx = boundary.stage_idx
                    prev_base_lr = boundary.base_lr
                    # Update to show next stage info
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
                    transition_progress=transition_progress,
                    stage_progress=stage_progress,
                    prev_stage_idx=prev_stage_idx,
                    prev_base_lr=prev_base_lr,
                )

        # If we're past all stages, return info for the last stage
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

    def get_interpolated_lr(self, step: int, base_schedule: str = "constant", min_lr: float = 0.0) -> float:
        """
        Get learning rate with stage-aware interpolation.

        During transitions, interpolates between stage LRs.
        Within stages, applies the base schedule (cosine/linear/constant).

        Args:
            step: Current training step
            base_schedule: LR schedule to apply within stages ("constant", "cosine", "linear")
            min_lr: Minimum learning rate floor

        Returns:
            Learning rate for the current step
        """
        stage_info = self.get_current_stage_info(step)

        # During transition: interpolate between prev and current stage LR
        if stage_info.in_transition and stage_info.prev_base_lr is not None:
            interpolated_lr = (
                stage_info.prev_base_lr +
                (stage_info.base_lr - stage_info.prev_base_lr) * stage_info.transition_progress
            )
            return max(interpolated_lr, min_lr)

        # Within stage: apply base schedule
        current_lr = stage_info.base_lr

        if base_schedule == "constant":
            return max(current_lr, min_lr)
        elif base_schedule == "cosine":
            # Cosine decay within the stage
            coeff = 0.5 * (1.0 + math.cos(math.pi * stage_info.stage_progress))
            return max(min_lr + coeff * (current_lr - min_lr), min_lr)
        elif base_schedule == "linear":
            # Linear decay within the stage
            return max(current_lr - stage_info.stage_progress * (current_lr - min_lr), min_lr)
        else:
            return max(current_lr, min_lr)

    def should_save_stage_checkpoint(self, step: int) -> tuple[bool, str]:
        """
        Check if we should save a checkpoint before stage transition.

        Args:
            step: Current training step

        Returns:
            Tuple of (should_save, checkpoint_name)
        """
        # Check if we're at the last step before a transition starts
        for boundary in self.boundaries[:-1]:  # Exclude last stage (no transition after it)
            if step == boundary.transition_start_step - 1:
                checkpoint_name = f"stage-{boundary.stage_idx}_end"
                return True, checkpoint_name

        return False, ""

    # NOTE: get_unified_data_config() and _create_dataset_scheduler() methods removed
    # They were part of the old approach using a single unified dataloader with dynamic weight modification.
    # The new approach uses separate dataloaders per stage with constant weights (no dynamic modification needed).

    def get_stage_summary(self) -> str:
        """Get a human-readable summary of stage configuration."""
        world_size = self.cfg.devices * self.cfg.num_nodes

        lines = ["Multi-Stage Training Configuration:"]
        lines.append(f"  Total stages: {len(self.stages)}")
        lines.append(f"  Total optimizer steps (per device): {self.total_steps:,}")
        lines.append(f"  World size (devices × nodes): {world_size} ({self.cfg.devices} × {self.cfg.num_nodes})")
        lines.append("")

        for boundary in self.boundaries:
            stage_steps = boundary.end_step - boundary.start_step
            transition_steps = boundary.transition_end_step - boundary.transition_start_step
            main_steps = stage_steps - transition_steps

            lines.append(f"Stage {boundary.stage_idx}: {boundary.stage_name}")
            lines.append(f"  Token budget: {boundary.tokens:,} total ({boundary.tokens // world_size:,} per device)")
            lines.append(f"  Optimizer steps: {stage_steps:,} (steps {boundary.start_step:,} - {boundary.end_step:,})")
            lines.append(f"    - Main training: {main_steps:,} steps")
            if boundary.stage_idx < len(self.stages) - 1:
                transition_pct = self.stages[boundary.stage_idx].transition_pct * 100
                lines.append(f"    - Transition OUT: {transition_steps:,} steps ({transition_pct:.1f}% of current stage)")
            lines.append(f"  Base LR: {boundary.base_lr:.2e}")
            lines.append("")

        return "\n".join(lines)
