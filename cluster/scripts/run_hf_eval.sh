#!/usr/bin/env bash
set -euo pipefail

# Simple script to run HuggingFace model evaluation
# Uses multi1 resources: 1 GPU, 3 days, ultimate partition

echo "Submitting HuggingFace model evaluation job..."

# Submit the job using heredoc
JOB_OUTPUT=$(sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=hf-eval
#SBATCH --partition=ultimate
#SBATCH --qos=ultimate
#SBATCH --account=ultimate
#SBATCH --gres=gpu:7g.79gb:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpu-bind=none
#SBATCH --output=/path/to/fast_storage/recpre/cluster/logs/hf_eval_out_%j.txt
#SBATCH --error=/path/to/fast_storage/recpre/cluster/logs/hf_eval_err_%j.txt

set -euo pipefail
set -x

echo "Starting HuggingFace model evaluation job"

BASE=/path/to/fast_storage/recpre
REPO="$BASE/recurrent-pretraining"
CODE_DIR="$BASE"

# Mamba environment setup
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

# Environment variables
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

# Additional cache directories
export TMPDIR=/path/to/fast_storage/recpre/.cache/tmp
export TORCHINDUCTOR_CACHE_DIR=/path/to/fast_storage/recpre/.cache/torchinductor
export TRITON_CACHE_DIR=/path/to/fast_storage/recpre/.cache/triton
export CUDA_CACHE_PATH=/path/to/fast_storage/recpre/.cache/nv
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "=========================================="
echo "Evaluation Configuration:"
echo "  Model: /path/to/shared_storage/recpre/outputs/hf_models/final-model-step-00151060-standalone"
echo "  Script: test_hf_model_eval.py"
echo "  Datasets: fineweb-edu-val, flan-mixture-val"
echo "  Depths: 1, 2, 4, 8, 16, [12,12,12]"
echo "  Device: cuda"
echo "=========================================="
echo

# Diagnostics
echo "==== Diagnostics ===="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi
echo
echo "Python: $(which python3)"
python3 --version
echo
echo "PyTorch version:"
python3 -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
echo "===================="
echo

# Change to code directory
cd "$CODE_DIR"

# Run the evaluation script
echo "Running evaluation script..."
python3 test_hf_model_eval.py

echo "=========================================="
echo "Evaluation completed successfully!"
echo "=========================================="
EOF
)

# Extract job ID from sbatch output
JOB_ID="${JOB_OUTPUT##* }"

LOG_FILE="/path/to/fast_storage/recpre/cluster/logs/hf_eval_out_${JOB_ID}.txt"
ERR_FILE="/path/to/fast_storage/recpre/cluster/logs/hf_eval_err_${JOB_ID}.txt"

echo "Job submitted successfully!"
echo "  Job ID: $JOB_ID"
echo "  Output log: $LOG_FILE"
echo "  Error log: $ERR_FILE"
echo
echo "Starting to tail output log (Ctrl+C to exit)..."
echo "You can also monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $ERR_FILE --retry"
echo
echo "------- Output log -------"

# Automatically tail the output log file
tail -f "$LOG_FILE" --retry
