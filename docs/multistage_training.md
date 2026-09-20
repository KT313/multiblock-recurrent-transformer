# Multi-stage training

Multi-stage training combines phases such as broad pretraining, domain upsampling
and instruction finetuning in one launch, with smooth dataset and learning-rate transitions.

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
`training/steps/batches.py`; equal deficits go to the alphabetically smallest source
name, so reordering the dataset config's `sources:` block never changes the
stream), and inside the window those weights are interpolated
linearly between the two stages' (a source leaving ramps to 0, one entering
ramps from 0 and earns its share from then on, without a catch-up burst; the
transition progress runs 0→1). The learning rate
interpolates linearly between the two stages' base LRs over the same window.

Global `warmup_steps` apply at the start of the first stage and
`cooldown_steps` at the end of the last; within a stage the configured
`lr_schedule` applies. A checkpoint is saved before each transition begins,
named `step-XXXXXXXX-{run_name}-stage-N_end`, so runs can be resumed from any
stage boundary.

The schedule is validated from configuration before backend initialization,
dataset assessment or auto-preparation, loaders, and model/optimizer construction.
Warmup must be shorter than the first stage's steps before its transition;
cooldown must be shorter than the last stage. For a single-stage run,
`warmup_steps + cooldown_steps <= total_steps` is also required: ten steps with
warmup 8 and cooldown 8 is rejected, while warmup 4 and cooldown 6 is allowed.
Windows are never shortened automatically; zero windows remain supported.

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
`total steps = Σ floor(stage.tokens / (micro_batches_per_step × tokens_per_micro_batch))`, independent of the number of
devices (`micro_batches_per_step` must be a multiple of it). Both settings are required: training packs documents
end to end, only validation runs on padded rows (`validation_batch_size` per forward).
The stage boundary summary is printed at startup; check it before long runs.

## Example configs

- `config/tiny.yaml` + `config/datasets/tiny.yaml`: 3-stage smoke run on synthetic data (20 steps)
- `config/crow_300m_final.yaml` + `config/datasets/crow_300m_final.yaml`: 300M-model example
