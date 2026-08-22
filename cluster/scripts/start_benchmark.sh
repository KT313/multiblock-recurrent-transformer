#!/usr/bin/env bash
set -euo pipefail

# Usage information
usage() {
    cat <<EOF
Usage: $0 --model <model_path> [OPTIONS]
   or: $0 -m <model_path> [OPTIONS]

Arguments:
  -m, --model <model_path>       HuggingFace model path (required)
  -r, --resources <preset>       Resource preset: 'medium' or 'large' (default: medium)
  -b, --benchmarks <benchmarks>  Comma-separated benchmark names (default: mmlu,hellaswag,arc_challenge,winogrande)
  -c, --recurrence <configs>     Space-separated recurrence configs (default: "1,1,1 4,4,4 12,12,12")
  -h, --help                     Show this help message

Resource Presets:
  medium  - advance partition, nvidia_a100_80gb_pcie_3g.39gb GPU, 5d time
  large   - ultimate partition, 7g.79gb GPU, 5d time

Recurrence Configs:
  Each config is comma-separated values for the recurrent blocks (e.g., "4,4,4")
  Multiple configs are space-separated and will be run sequentially

Examples:
  # Benchmark a model with default settings
  $0 --model /path/to/fast_storage/recpre/outputs/hf_models/my-model

  # Benchmark with specific recurrence configs
  $0 -m /path/to/model -c "1,1,1 8,8,8 16,16,16"

  # Benchmark specific tasks with large GPU
  $0 -m /path/to/model -r large -b "mmlu,hellaswag"

  # Full custom configuration
  $0 -m /path/to/model -r large -b "mmlu,arc_challenge" -c "2,2,2 6,6,6"
EOF
    exit 1
}

# Parse command-line arguments
MODEL=""
RESOURCES="medium"
BENCHMARKS="mmlu,hellaswag,arc_challenge,winogrande"
RECURRENCE_CONFIGS="1,1,1 4,4,4 12,12,12"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -r|--resources)
            RESOURCES="$2"
            shift 2
            ;;
        -b|--benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        -c|--recurrence)
            RECURRENCE_CONFIGS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

# Map resource presets to SBATCH directives
case "$RESOURCES" in
    multi1)
        PARTITION="ultimate"
        QOS="ultimate"
        ACCOUNT="ultimate"
        GRES="gpu:ampere:1"
        TIME="3-00:00:00"
        GPU_BIND="none"
        CPUS_PER_TASK="16"
        MEM="128G"
        NGPUS="1"
        NTASKS="1"
        ;;
    large)
        PARTITION="ultimate"
        QOS="ultimate"
        ACCOUNT="ultimate"
        GRES="gpu:7g.79gb:1"
        TIME="5-00:00:00"
        GPU_BIND="single:1"
        ;;
    medium)
        PARTITION="advance"
        QOS="advance"
        ACCOUNT="advance"
        GRES="gpu:nvidia_a100_80gb_pcie_3g.39gb:1"
        TIME="5-00:00:00"
        GPU_BIND="single:1"
        ;;
    *)
        echo "Error: Invalid resource preset: $RESOURCES"
        echo "Valid options: medium, large"
        exit 1
        ;;
esac

# Common SBATCH settings
NODES="1"
CPUS_PER_TASK="16"
MEM="128G"
NTASKS="1"

echo "Submitting benchmark job with:"
echo "  Model: $MODEL"
echo "  Resources: $RESOURCES ($PARTITION partition, $GRES)"
echo "  Benchmarks: $BENCHMARKS"
echo "  Recurrence configs: $RECURRENCE_CONFIGS"
echo

# Validate inputs
if [[ -z "$MODEL" ]]; then
    echo "Error: --model/-m is required"
    usage
fi

# Validate model path exists (if it's an absolute path)
if [[ "$MODEL" == /* ]] && [[ ! -d "$MODEL" ]]; then
    echo "Error: Model path not found: $MODEL"
    exit 1
fi

# Submit the job using heredoc
JOB_OUTPUT=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --account=$ACCOUNT
#SBATCH --gres=$GRES
#SBATCH --nodes=$NODES
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --ntasks=$NTASKS
#SBATCH --gpu-bind=$GPU_BIND
#SBATCH --output=/path/to/fast_storage/recpre/cluster/logs/benchmark_out_%j.txt
#SBATCH --error=/path/to/fast_storage/recpre/cluster/logs/benchmark_err_%j.txt

set -euo pipefail
set -x

echo "Starting benchmark job"

BASE=/path/to/fast_storage/recpre
REPO="\$BASE/recurrent-pretraining"

# Mamba environment setup
MAMBA_ENV="/path/to/fast_storage/.local/share/mamba/envs/recpre"

# Unset any conflicting variables
unset PYTHONHOME

# Set conda/mamba variables
export CONDA_PREFIX="\$MAMBA_ENV"
export CONDA_DEFAULT_ENV="\$(basename \$MAMBA_ENV)"
export CONDA_SHLVL=1

# Prepend the env's bin to PATH
export PATH="\$MAMBA_ENV/bin:\$PATH"

# Set library paths (important for compiled extensions)
export LD_LIBRARY_PATH="\$MAMBA_ENV/lib:\${LD_LIBRARY_PATH:-}"

# Environment variables
export HF_HOME="\$BASE/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -l unlimited || true   # quiet the MEMLOCK warning
export REC_SAFE_HEAD=1
export TORCH_DISABLE_ADDR2LINE=1
export HOME=/path/to/fast_storage/recpre/home
export XDG_CACHE_HOME=/path/to/fast_storage/recpre/.cache
export TORCH_HOME=/path/to/fast_storage/recpre/.cache/torch
mkdir -p "\$HOME" "\$XDG_CACHE_HOME" "\$TORCH_HOME"

# Additional cache directories
export TMPDIR=/path/to/fast_storage/recpre/.cache/tmp
export TORCHINDUCTOR_CACHE_DIR=/path/to/fast_storage/recpre/.cache/torchinductor
export TRITON_CACHE_DIR=/path/to/fast_storage/recpre/.cache/triton
export CUDA_CACHE_PATH=/path/to/fast_storage/recpre/.cache/nv
mkdir -p "\$TMPDIR" "\$TORCHINDUCTOR_CACHE_DIR" "\$TRITON_CACHE_DIR" "\$CUDA_CACHE_PATH"

# Model path
MODEL_PATH="$MODEL"

echo "=========================================="
echo "Benchmark Configuration:"
echo "  Model: \$MODEL_PATH"
echo "  Benchmarks: $BENCHMARKS"
echo "  Recurrence configs: $RECURRENCE_CONFIGS"
echo "  Device: cuda:0"
echo "=========================================="
echo

# Run benchmarks for each recurrence configuration
for RECURRENCE in $RECURRENCE_CONFIGS; do
    echo "=========================================="
    echo "Running with recurrence config: \$RECURRENCE"
    echo "=========================================="

    \$REPO/scripts/benchmark_model.sh \
        "\$MODEL_PATH" \
        "$BENCHMARKS" \
        cuda:0 \
        "\$RECURRENCE"

    echo "Completed recurrence config: \$RECURRENCE"
    echo
done

echo "=========================================="
echo "All benchmarks completed successfully!"
echo "=========================================="
EOF
)

# Extract job ID from sbatch output
JOB_ID="${JOB_OUTPUT##* }"

LOG_FILE="/path/to/fast_storage/recpre/cluster/logs/benchmark_out_${JOB_ID}.txt"
ERR_FILE="/path/to/fast_storage/recpre/cluster/logs/benchmark_err_${JOB_ID}.txt"

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
