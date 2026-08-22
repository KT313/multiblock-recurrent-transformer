#!/bin/bash
#SBATCH --job-name=recur1b_mig
#SBATCH --partition=advance
#SBATCH --qos=advance
#SBATCH --account=advance
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.39gb:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpu-bind=single:1
#SBATCH --output=/path/to/fast_storage/recpre/cluster/logs/train_out_%j.txt
#SBATCH --error=/path/to/fast_storage/recpre/cluster/logs/train_err_%j.txt

set -euo pipefail
set -x

echo "1"

BASE=/path/to/fast_storage/recpre
REPO="$BASE/recurrent-pretraining"

# venv from prep
# source "$BASE/venv/bin/activate"
# /path/to/fast_storage/bin/micromamba activate recpre

MAMBA_ENV="/path/to/fast_storage/.local/share/mamba/envs/recpre"

# Unset any conflicting variables
unset PYTHONHOME

# Set conda/mamba variables
export CONDA_PREFIX="$MAMBA_ENV"
export CONDA_DEFAULT_ENV="$(basename $MAMBA_ENV)"
export CONDA_SHLVL=1

# Prepend the env's bin to PATH
export PATH="$MAMBA_ENV/bin:$PATH"

# Set library paths (important for compiled extensions)
export LD_LIBRARY_PATH="$MAMBA_ENV/lib:${LD_LIBRARY_PATH:-}"

# Environment
export HF_HOME="$BASE/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -l unlimited || true   # quiet the MEMLOCK warning
export REC_SAFE_HEAD=1
export TORCH_DISABLE_ADDR2LINE=1
export HOME=/path/to/fast_storage/recpre/home
export XDG_CACHE_HOME=/path/to/fast_storage/recpre/.cache
export TORCH_HOME=/path/to/fast_storage/recpre/.cache/torch
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

echo "8"

export TMPDIR=/path/to/fast_storage/recpre/.cache/tmp
export TORCHINDUCTOR_CACHE_DIR=/path/to/fast_storage/recpre/.cache/torchinductor
export TRITON_CACHE_DIR=/path/to/fast_storage/recpre/.cache/triton
export CUDA_CACHE_PATH=/path/to/fast_storage/recpre/.cache/nv

MODEL_PATH=/path/to/fast_storage/recpre/outputs/hf_models/recur1b-mig-3380-finetuned

$REPO/scripts/benchmark_model.sh \
    $MODEL_PATH \
    "mmlu,hellaswag,arc_challenge,winogrande" cuda:0 "1,1,1"

$REPO/scripts/benchmark_model.sh \
    $MODEL_PATH \
    "mmlu,hellaswag,arc_challenge,winogrande" cuda:0 "4,4,4"

$REPO/scripts/benchmark_model.sh \
    $MODEL_PATH \
    "mmlu,hellaswag,arc_challenge,winogrande" cuda:0 "12,12,12"
