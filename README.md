# Multiblock Recurrent Transformer

This repository contains the code for my thesis "Efficient Large Language Models via Recurrent Transformer Blocks" (PDF available [here](https://github.com/KT313/papers/blob/main/efficient_large_language_models.pdf)). 

> **Built on [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)** (Geiping et al., 2025, Apache-2.0), base commit `3055b7f`. This repo is a multi-block extension of their depth-recurrent transformer; see `git diff upstream-base..HEAD` for exactly what was changed.


## Work

The original repo trains one recurrent block between a "prelude" and a "coda" block. This fork generalizes that to N core blocks, each with its own injection adapter, output norm, mean recurrence and truncated-backprop depth (`recpre/model_dynamic.py`, config in `recpre/config_dynamic.py` + `recpre/model_registry.py`). 
Besides the architecture change, I added the following:  
- 3-staged training with smooth data/LR transitions (`recpre/stage_manager.py`, `docs/multistage_training.md`)
- train.py adjustments (checkpoint/resume, dataset mixing, DDP) 
- HuggingFace export path (`recpre/hf_export.py`)
- cluster configs / SLURM scripts / analysis data that produced the thesis (`cluster/`, `analysis/`)

Final model configs in the original repo were named "raven", so I named my model configs "crow" in the same spirit.

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

Training was stable across all three phases at 308M parameters / ~5B tokens. Due to resource limitations, the model is very undertrained, and as expected, benchmark scores are around the same performance as random guessing (`analysis/data/benchmark_results.csv`).  
It is expected that, with the same number of optimizer steps, the model with higher batch size per step would perform better.  
Interestingly, the world batch size seems to directly affect the quality impact of recurrent step settings.
Compared to world batch size 64, world batch size 1024 strongly increases the loss for a single iteration of the recurrent blocks (unexpected), while decreasing the loss significantly for higher numbers of recurrent iterations (expected).

![Batch size comparison: validation loss vs optimizer step](docs/figures/batchsize_comparison_valloss.png)
*Comparison between a training run with batch size 1024 (blue) and batch size 64 (red) and different numbers of recurrent steps (1-16) for each.*

## Note

During all training runs for my thesis, one of the two prelude blocks in my models never received a gradient during training. This was caused by the prelude blocks not chaining correctly in a for-loop. The issue has since been fixed in the code (commit "Fix prelude layers not chaining in model_dynamic forward").

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