# Multi-stage training

The thesis runs train through three phases (broad pretraining → domain
upsampling → instruction finetuning) in a single launch, with smooth dataset
transitions between phases. This document describes the mechanism; the
implementation lives in `recpre/stage_manager.py` with its integration in
`train.py` (LR schedule, transition detection, stage checkpoints, metrics).

## What a stage specifies

Each entry in `training_stages` defines:

- `train_data` / `val_data` — dataset mix for the stage (same schema as the
  single-stage `data_config`)
- `tokens` — the stage's global token budget
- `base_lr` — stage-specific base learning rate
- `transition_pct` — fraction of the stage reserved (at its end) for the
  transition into the next stage; `0.0` for the last stage

Enable with `enable_multi_stage: true`. Without it, training behaves exactly as
single-stage; multi-stage is opt-in and backward compatible.

## Transitions

Instead of switching datasets abruptly at a stage boundary, the last
`transition_pct` of a stage gradually shifts sampling weights from the current
stage's datasets to the next stage's (100→0 / 0→100, linear), using the
existing `DataScheduler` piecewise-weight machinery. The learning rate
interpolates linearly between the two stages' `base_lr` over the same window.

Global `warmup_steps` apply at the start of the first stage and
`cooldown_steps` at the end of the last; within a stage the configured
`lr_schedule` applies. A checkpoint is saved before each transition begins,
named `step-XXXXXXXX-{run_name}-stage-N_end`, so runs can be resumed from any
stage boundary.

## Monitoring

Logged per step (wandb): `stage/current_stage`, `stage/base_lr`,
`stage/in_transition`, `stage/transition_progress`, `stage/stage_progress`,
plus per-dataset scheduler weights (`data_scheduler_norm_weight/*`) and
per-stage validation metrics.

## Step accounting

`total steps = Σ stage.tokens / (world_batch_size × block_size)`, computed
per device by the stage manager. The stage boundary summary is printed at
startup — check it before long runs.

## Example configs

- `cluster/configs/multistage_example.yaml` — documented 3-stage schema example
- `cluster/configs/multistage_test_small.yaml` — small test config (~2.5M
  tokens, minutes on one GPU); also see `tests/train_smoke/` for an even
  smaller synthetic-data smoke run
- `cluster/configs/crow_300m_final.yaml` — the real final-run config

## Notable bug found during development

The first version of the stage-boundary computation did not divide token
budgets by `world_size`. On a single GPU everything looked correct, but on the
4-GPU DDP setup each stage would have silently run 4× longer than configured —
there is no error to see, just a schedule that never ends. It was caught in a
code audit before the main runs by checking the startup boundary summary
against a hand calculation. The fix divides per-device tokens by
`devices × num_nodes` and adds validation that `warmup_steps` and
`cooldown_steps` fit inside the first/last stage. Lesson: in distributed
training, verify step arithmetic by hand at startup rather than trusting that
a config "looks right" — silent factor-of-`world_size` errors are cheap to
make and expensive to discover mid-run.
