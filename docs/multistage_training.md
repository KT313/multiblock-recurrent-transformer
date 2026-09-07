# Multi-stage training

The thesis runs train through three phases (broad pretraining → domain
upsampling → instruction finetuning) in a single launch, with smooth dataset
transitions between phases. This document describes the mechanism; the
implementation lives in `training/stage_manager.py` with its integration in
`training/run.py` / `training/step.py` (LR schedule, transition detection, stage checkpoints, metrics).

## What a stage specifies

The stage list lives in the dataset config (`config/datasets/<name>.yaml`,
`stages:`); each entry defines:

- `train` / `val`: the weights over sources (plain source names; a source in both is split by `validation_fraction`) for
  the stage, weights summing to 1
- `tokens`: the stage's global token budget
- `transition_pct`: fraction of the stage reserved (at its end) for the
  transition into the next stage; `0.0` for the last stage

The run config contributes one base learning rate per stage, positionally, as
`stage_base_lrs: [3e-4, 1e-4, 5e-5]`; `training/data/dataset_resolver.py` joins
the two into the `ResolvedStage` list the stage manager consumes. A single-stage
run is one entry with `transition_pct: 0.0`.

## Transitions

Instead of switching datasets abruptly at a stage boundary, the last
`transition_pct` of a stage gradually shifts the weights: every source is read
by one continuous reader for the whole run, the weights are token shares that the
stream realises by filling its packing pool from the source with the largest
token deficit under the current step's weights (`BatchStream` in
`training/step.py`), and inside the window those weights are interpolated
linearly between the two stages' (a source leaving ramps to 0, one entering
ramps from 0 and earns its share from then on, without a catch-up burst; the
transition progress runs 0→1). The learning rate
interpolates linearly between the two stages' base LRs over the same window.

Global `warmup_steps` apply at the start of the first stage and
`cooldown_steps` at the end of the last; within a stage the configured
`lr_schedule` applies. A checkpoint is saved before each transition begins,
named `step-XXXXXXXX-{run_name}-stage-N_end`, so runs can be resumed from any
stage boundary.

## Monitoring

Logged per step (wandb): `stage/current_stage`, `stage/base_lr`,
`stage/in_transition`, `stage/transition_progress`, `stage/stage_progress`,
plus the realised data composition since the last log step (`data_composition/<source>`: the fraction of the
trained document tokens per source, pack tails excluded, which the stage weights promise as token shares) and the
validation metrics `val_loss`, `val_ppl`,
`val_loss_<depth>` / `val_ppl_<depth>` for every `partial_depth_eval` depth and
`val_time` (of the stage the run is in at that step; inside a transition window, of the stage being entered).

## Step accounting

Steps are optimizer steps of `micro_batches_per_step × tokens_per_micro_batch` tokens:
`total steps = Σ stage.tokens / (micro_batches_per_step × tokens_per_micro_batch)`, independent of the number of
devices (`micro_batches_per_step` must be a multiple of it). Both settings are required: training packs documents
end to end, only validation runs on padded rows (`validation_batch_size` per forward).
The stage boundary summary is printed at startup; check it before long runs.

## Example configs

- `config/tiny.yaml` + `config/datasets/tiny.yaml`: 3-stage smoke run on synthetic data (20 steps, seconds)
- `config/crow_300m_final.yaml` + `config/datasets/crow_300m_final.yaml`: the real final-run config

## Notable bug found during development

The first version of the stage-boundary computation (which counted per-device
micro-batch steps) did not divide token budgets by `world_size`. On a single GPU
everything looked correct, but on the 4-GPU DDP setup each stage would have
silently run 4× longer than configured. There is no error to see, just a
schedule that never ends. It was caught in a code audit before the main runs by
checking the startup boundary summary against a hand calculation. The current
code counts optimizer steps (world batches), which removes the `world_size`
dependence altogether, and validates that `warmup_steps` and `cooldown_steps`
fit inside the first/last stage. Lesson: in distributed
training, verify step arithmetic by hand at startup rather than trusting that
a config "looks right"; silent factor-of-`world_size` errors are cheap to
make and expensive to discover mid-run.
