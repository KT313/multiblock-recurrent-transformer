# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Progress, results, and intermediate statistics of one optimizer update."""

from dataclasses import dataclass, field

from torch import Tensor

from training.stage_manager import StageInfo


class NonFiniteLossError(RuntimeError):
    """
    The step's loss or gradient norm is not finite. Raised before `optimizer.step`, so the model is still the one
    of the completed steps; `train()` checkpoints it and stops.
    """



@dataclass
class TrainingProgress:
    """
    The mutable step counter of a run, shared by the loop and the micro-batch stream.

    `step` is the next optimizer step to run; `train()` calls `advance()` once right after `run_one_optimizer_step`.
    """

    step: int = 0  # next optimizer step to run
    resume_step: int = -1  # step the run was resumed at, -1 for a fresh run

    def advance(self) -> None:
        """
        One optimizer step completed: after this `step` is the number of completed steps (evaluation, logging and
        checkpoint intervals count these) and the index of the next step to run.
        """

        self.step += 1



@dataclass
class StepResult:
    """
    What one optimizer step produced.
    """

    step: int  # the optimizer step that was run
    learning_rate: float  # scheduled LR of that step
    loss: Tensor  # global supervised-token mean over the entire optimizer update
    grad_norm: Tensor  # pre-clip gradient norm
    stage: StageInfo  # stage info at `step` (what the step trained on)
    data_ids: list[str]  # one entry per document of the step's packs (document count per source)
    data_tokens: dict[str, int]  # document slots trained per data id, pack tails excluded (data composition)
    metrics: dict[str, Tensor] = field(default_factory=dict)  # at log steps: `track_gradient_metrics`, `packing/padding_fraction`; else {}
    validation: dict[str, Tensor] | None = None  # filled by `train()` when it is an evaluation step



@dataclass
class AccumulatedGradients:
    """Local loss/count totals and data composition collected during microbatch backward passes."""

    loss_sum: Tensor
    supervised_count: Tensor
    local_capacity: int
    data_ids: list[str]
    data_tokens: dict[str, int]
    padding_tokens: int
