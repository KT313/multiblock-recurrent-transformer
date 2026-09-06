# Multiblock Recurrent Transformer

This repository contains the code for my thesis "Efficient Large Language Models via Recurrent Transformer Blocks" (PDF available [here](https://github.com/KT313/papers/blob/main/efficient_large_language_models.pdf)). 

> **Built on [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)** (Geiping et al., 2025, Apache-2.0), base commit `3055b7f`. This repo is a multi-block extension of their depth-recurrent transformer; see `git diff upstream-base..HEAD` for exactly what was changed.

## Work

The original repo trains one recurrent block between a "prelude" and a "coda" block. This fork generalizes that to N core blocks, each with its own injection adapter, input norm, mean recurrence and truncated-backprop depth (`model/model.py`, config in `model/config.py`, the architectures as YAML in `config/model_architecture/`).
Besides the architecture change, I added the following:
- 3-staged training with smooth data/LR transitions (`training/stage_manager.py`, `docs/multistage_training.md`)
- a compact single-GPU training loop with checkpoint/resume and dataset mixing (`training/run.py:train()`, the CLI `training/train.py`; distributed training is meant to be re-added behind `training/backend/`)
- HuggingFace export path (`model/hf/modeling.py`)
- dataset preparation driven by a dataset config (`config/datasets/`, `data_preparation/`): sources, stages with token budgets and weights, and the tokenizer in one YAML; downloaded and built incrementally and verified before training

The code was restructured and trimmed after the thesis: only the code path of the final run survives, with tests next to every module. The thesis-era tree (SLURM tooling, all upstream model variants) is in git history up to tag `v1.0`.

## Usage

### Setup

```bash
make setup                # prepare uv venv
uv run hf auth login      # log into huggingface for gated datasets

make test                 # tests
make typecheck            # mypy, basedpyright
make lint                 # ruff
```

### Configs

- `config/<run>.yaml`: Optimizer, LR, batch settings, reference to model architecture and dataset configs (`model_overwrite` overrides single architecture keys)
- `config/model_architecture/<name>.yaml`: model architecture settings
- `config/datasets/<name>.yaml`: dataset composition

Final model configs in the original repo were named "raven", so I named my model configs "crow" in the same spirit.

### Data Preparation

Optional: training builds whatever is missing itself by default (`auto_prepare: true`). Gated sources need the huggingface login from Setup.

```bash
# mini smoke run (real sources, a few MB)
uv run python data_preparation/prepare.py prepare  --dataset_config config/datasets/crow_300m_mini.yaml

# download thesis data sources
uv run python data_preparation/prepare.py prepare  --dataset_config config/datasets/crow_300m_final.yaml

# check which sources are missing locally without starting download
uv run python data_preparation/prepare.py status --dataset_config config/datasets/crow_300m_final.yaml

# make shortcuts for the same: download / status
make download config/datasets/crow_300m_final.yaml
make status config/datasets/crow_300m_final.yaml
```

### Training

```bash
# mini smoke run with synthetic data
uv run python training/train.py --config config/tiny.yaml
TRAINING_DASHBOARD=0 uv run python training/train.py --config config/tiny.yaml # TUI disabled

# thesis run on single gpu
uv run python training/train.py --config config/crow_300m_final.yaml

# make shortcut for the same
make training config/crow_300m_final.yaml
```

Training packs documents end to end (`pack_sequences: true`, the default): one row of `tokens_per_micro_batch`
tokens per micro-batch, `micro_batches_per_step` of them per optimizer step, attention masked per document;
validation stays padded. Left unset, the two are the padded equivalents `micro_batch_size × training_max_sequence_length` and
`world_batch_size / micro_batch_size`, so a config written in rows keeps its step arithmetic.

A run resumes by default (`resume: true`): the most recently written checkpoint of `run_name` in its run directory is
loaded, or `resume_checkpoint_path` names one; a checkpoint written with other settings, model config or dataset config
is refused unless `allow_settings_change` / `allow_dataset_change` say so. `compile_model: true` compiles the model with
`torch.compile`. The architecture's `bf16_residual_stream` (`none`, `core`, `all`) decides which RMSNorm outputs are
rounded to the autocast dtype under bf16-mixed precision: none, the core blocks' (the recurrence runs on a bf16 stream),
or every norm's.

### Evaluation

Sample generations and lm-eval-harness scores (`evaluation/`), on a checkpoint or during training:

```bash
uv sync --extra eval      # lm-eval-harness, only needed for benchmarks

# greedy samples for the built-in prompts; --tasks adds benchmarks
uv run python evaluation/evaluate.py --checkpoint outputs/<run>/checkpoints/<file>.pth
uv run python evaluation/evaluate.py --checkpoint <file>.pth --tasks arc_challenge,hellaswag --limit 200

# make shortcut (EVAL_TASKS=a,b for benchmarks)
make evaluate outputs/<run>/checkpoints/<file>.pth
```

During training the run config decides when they run: every `sample_step_interval` steps and after the steps at
the percentages of the run in `sample_at_training_progress` (0: after the first step, 100: after the last;
default `[100]`), combined; `benchmark_step_interval` and `benchmark_at_training_progress` (default: off) likewise.
The percentages become step numbers once the stage plan is known; `train.log` lists them. Files land in
`outputs/<run>/samples/` and `outputs/<run>/benchmarks/`, named by step; scores also go to wandb as
`benchmark/<recurrence>/<task>/<metric>`. `sample_recurrences` and `benchmark_recurrences` list the recurrent
steps per block to run with, e.g. `[[4, 4, 4], [12, 12, 12]]` (empty: the mean recurrence once); the CLI takes
`--recurrence 4,4,4` repeatedly. Both run RNG-isolated, so they do not change the training.

## Architecture

### Overview
![High-level RecurrentGPT architecture overview](docs/figures/architecture_overview.png)

### Core Blocks
![Core recurrent blocks with per-block normalization and residual connections](docs/figures/core_blocks.png)

### Recurrent Block
![Internals of a recurrent block](docs/figures/recurrent_block_internals.png)

### Sandwich Block
![Sandwich block internal structure](docs/figures/sandwich_block.png)

## Results

Training was stable across all three phases at 308M parameters / ~5B tokens. Due to resource limitations, the model is very undertrained, and as expected, benchmark scores are around the same performance as random guessing (raw numbers are in the thesis PDF).  
It is expected that, with the same number of optimizer steps, the model with higher batch size per step would perform better.  
Interestingly, the world batch size seems to directly affect the quality impact of recurrent step settings.
Compared to world batch size 64, world batch size 1024 strongly increases the loss for a single iteration of the recurrent blocks (unexpected), while decreasing the loss significantly for higher numbers of recurrent iterations (expected).

![Batch size comparison: validation loss vs optimizer step](docs/figures/batchsize_comparison_valloss.png)
*Comparison between a training run with batch size 1024 (blue) and batch size 64 (red) and different numbers of recurrent steps (1-16) for each.*

## Note

During all training runs for my thesis, one of the two prelude blocks in my models never received a gradient during training. This was caused by the prelude blocks not chaining correctly in a for-loop. The issue has since been fixed in the code (commit "Fix prelude layers not chaining in model_dynamic forward"). The code used in the thesis is available under `Releases` as `v1.0` (thesis version).

## Training Info

- 3 days on 4×A100 80GB (DDP, bf16-mixed) via SLURM on a university cluster
- ELLISAdam optimizer (following original repo)
- world batch 1024
- sequence length 2048 tokens

## Citation

```bibtex
@article{geiping2025scaling,
  title   = {Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach},
  author  = {Geiping, Jonas and McLeish, Sean and Jain, Neel and Kirchenbauer, John and Singh, Siddharth and Bartoldson, Brian R. and Kailkhura, Bhavya and Bhatele, Abhinav and Goldstein, Tom},
  journal = {arXiv preprint arXiv:2502.05171},
  year    = {2025}
}

@thesis{kerner2025recurrent,
  title  = {Efficient Large Language Models via Recurrent Transformer Blocks},
  author = {Kerner, Tobias},
  school = {Technische Hochschule Ingolstadt},
  type   = {Bachelor's thesis},
  year   = {2025}
}
```