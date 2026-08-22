#!/usr/bin/env bash
set -euo pipefail

# Usage information
usage() {
    cat <<EOF
Usage: $0 --resources <preset> --config <config_file>
   or: $0 -r <preset> -c <config_file>

Arguments:
  -r, --resources <preset>   Resource preset: 'large' or 'medium' or 'multi'
  -c, --config <config_file> Config filename (e.g., raven_small_test.yaml)
  -h, --help                 Show this help message

Resource Presets:
  large   - ultimate partition, 7g.79gb GPU, 60h time
  medium  - advance partition, nvidia_a100_80gb_pcie_3g.39gb GPU, 5d time
  multi   - ultimate partition, gpu:ampere:4, 72h time

Example:
  $0 --resources large --config raven_small_test.yaml
  $0 -r medium -c raven_small_test.yaml
EOF
    exit 1
}

# Parse command-line arguments
RESOURCES=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--resources)
            RESOURCES="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
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

# Validate inputs
if [[ -z "$RESOURCES" ]]; then
    echo "Error: --resources/-r is required"
    usage
fi

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config/-c is required"
    usage
fi

# Get the base directory (go up one level from scripts/)
cd "$(dirname "$0")/.."
BASE_DIR="$(pwd)"

# Validate config file exists
CONFIG_PATH="$BASE_DIR/configs/$CONFIG"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Map resource presets to SBATCH directives
case "$RESOURCES" in
    multi)
        PARTITION="ultimate"
        QOS="ultimate"
        ACCOUNT="ultimate"
        GRES="gpu:ampere:4"
        TIME="3-00:00:00"
        GPU_BIND="none"
        CPUS_PER_TASK="16"
        MEM="256G"
        NGPUS="4"
        NTASKS="4"
        ;;
    multi2)
        PARTITION="ultimate"
        QOS="ultimate"
        ACCOUNT="ultimate"
        GRES="gpu:ampere:2"
        TIME="3-00:00:00"
        GPU_BIND="none"
        CPUS_PER_TASK="32"
        MEM="256G"
        NGPUS="2"
        NTASKS="2"
        ;;
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
        CPUS_PER_TASK="16"
        MEM="128G"
        NGPUS="1"
        NTASKS="1"
        ;;
    medium)
        PARTITION="advance"
        QOS="advance"
        ACCOUNT="advance"
        GRES="gpu:nvidia_a100_80gb_pcie_3g.39gb:1"
        TIME="5-00:00:00"
        GPU_BIND="single:1"
        CPUS_PER_TASK="16"
        MEM="128G"
        NGPUS="1"
        NTASKS="1"
        ;;
    *)
        echo "Error: Invalid resource preset: $RESOURCES"
        echo "Valid options: large, medium, multi, multi2"
        exit 1
        ;;
esac

# Common SBATCH settings
NODES="1"
# NTASKS="1"

# Full path to config for SLURM script
FULL_CONFIG_PATH="/path/to/fast_storage/recpre/cluster/configs/$CONFIG"

echo "Submitting training job with:"
echo "  Resources: $RESOURCES ($PARTITION partition, $GRES)"
echo "  Config: $CONFIG"
echo

# Submit the job using heredoc
JOB_OUTPUT=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=recur1b_mig
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --account=$ACCOUNT
#SBATCH --gres=$GRES
#SBATCH --nodes=$NODES
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --ntasks-per-node=$NTASKS
#SBATCH --gpu-bind=$GPU_BIND
#SBATCH --output=/path/to/fast_storage/recpre/cluster/logs/train_out_%j.txt
#SBATCH --error=/path/to/fast_storage/recpre/cluster/logs/train_err_%j.txt

set -euo pipefail
set -x

echo "1"

BASE=/path/to/fast_storage/recpre
REPO="\$BASE/recurrent-pretraining"

echo "2"

# Pick run id: SLURM job id if present, else next free integer in \$BASE/outputs
RUN_ID="\${SLURM_JOB_ID-}"

echo "3"

if [[ -z "\$RUN_ID" ]]; then
  mkdir -p "\$BASE/outputs"
  n=1
  while true; do
    if mkdir "\$BASE/outputs/\$n" 2>/dev/null; then
      RUN_ID="\$n"
      break
    fi
    ((n++))
  done
fi

echo "4"

SLURM_JOB_ID=\$RUN_ID

CFG="$FULL_CONFIG_PATH"
OUT="\$BASE/outputs/\${SLURM_JOB_ID}"

echo "5"

mkdir -p "\$OUT"

echo "6"

# venv from prep
# source "\$BASE/venv/bin/activate"
# /path/to/fast_storage/bin/micromamba activate recpre

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

echo "7"

# Environment
export HF_HOME="\$BASE/.cache/huggingface"
export WANDB_MODE=offline
export WANDB_DIR="\$OUT"
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

echo "8"

export TMPDIR=/path/to/fast_storage/recpre/.cache/tmp
export TORCHINDUCTOR_CACHE_DIR=/path/to/fast_storage/recpre/.cache/torchinductor
export TRITON_CACHE_DIR=/path/to/fast_storage/recpre/.cache/triton
export CUDA_CACHE_PATH=/path/to/fast_storage/recpre/.cache/nv

mkdir -p "\$TMPDIR" "\$TORCHINDUCTOR_CACHE_DIR" "\$TRITON_CACHE_DIR" "\$CUDA_CACHE_PATH"

echo "9"

# Make CFG/OUT visible to the Python wrapper
export CFG="\$CFG"
export OUT="\$OUT"

# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,garbage_collection_threshold:0.9"
# export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,garbage_collection_threshold:0.7,max_split_size_mb:64,expandable_segments:True"
# export TORCHINDUCTOR_COMPILE_THREADS=1
# export TORCHINDUCTOR_WORKER_START=subprocess
# export CUDA_MODULE_LOADING=LAZY

# export CUDA_VISIBLE_DEVICES=0
# export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# Performance profiling and debugging
export TORCH_LOGS="recompiles,graph_breaks"  # Track recompiles and what breaks the graph
# export TORCH_LOGS="+dynamo"  # Uncomment for detailed dynamo logs
# export TORCHDYNAMO_EXPLAIN=1  # Uncomment to explain compilation decisions
# export TORCHINDUCTOR_MAX_AUTOTUNE=1  # Uncomment for aggressive kernel tuning (slower compile, faster runtime)

# Memory tracking (for debugging)
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Enable profiling via environment variable (0=disabled, 1=enabled)
export ENABLE_TORCH_PROFILER=0  # Set to 1 to enable Chrome trace profiling
export PROFILER_STEPS="5,10"  # Steps to profile (start,end)

echo "10"

echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
# echo "SLURM_JOB_GPUS=\$SLURM_JOB_GPUS"
nvidia-smi -L

echo "11"
echo "[\$(date +%T.%N)] Starting diagnostics"

# Check disk usage
echo "[\$(date +%T.%N)] Disk usage:"
df -h | grep -E '(Filesystem|/$|/home|/tmp|/scratch)' || df -h
echo ""

# Check memory and swap
echo "[\$(date +%T.%N)] Memory and swap:"
free -h
echo ""

# Check CPU load
echo "[\$(date +%T.%N)] CPU load average:"
uptime
echo ""

# Check I/O wait
echo "[\$(date +%T.%N)] CPU and I/O stats:"
top -bn1 | head -5
echo ""

# Time the Python startup
echo "[\$(date +%T.%N)] Launching Python interpreter..."

python - <<'PY'
import sys
import time
import os

start = time.time()
print(f"[{time.time()-start:.3f}s] Python started", file=sys.stderr)
print(f"[{time.time()-start:.3f}s] PID: {os.getpid()}", file=sys.stderr)

# Check CPU affinity and current process stats
try:
    import psutil
    p = psutil.Process()
    print(f"[{time.time()-start:.3f}s] CPU percent: {p.cpu_percent()}%", file=sys.stderr)
    print(f"[{time.time()-start:.3f}s] Memory: {p.memory_info().rss / 1024**2:.1f} MB", file=sys.stderr)
except ImportError:
    pass

print(f"[{time.time()-start:.3f}s] Importing torch...", file=sys.stderr)
import torch
print(f"[{time.time()-start:.3f}s] torch imported", file=sys.stderr)

print(f"[{time.time()-start:.3f}s] Checking torch.version.cuda...", file=sys.stderr)
cuda_version = torch.version.cuda
print(f"[{time.time()-start:.3f}s] Got CUDA version", file=sys.stderr)
print("torch.version.cuda =", cuda_version)

print(f"[{time.time()-start:.3f}s] Checking torch.cuda.is_available()...", file=sys.stderr)
is_available = torch.cuda.is_available()
print(f"[{time.time()-start:.3f}s] Got availability", file=sys.stderr)
print("torch.cuda.is_available() =", is_available)

print(f"[{time.time()-start:.3f}s] Checking torch.cuda.device_count()...", file=sys.stderr)
device_count = torch.cuda.device_count()
print(f"[{time.time()-start:.3f}s] Got device count", file=sys.stderr)
print("torch.cuda.device_count()  =", device_count)

print(f"[{time.time()-start:.3f}s] Done", file=sys.stderr)
PY

echo "[\$(date +%T.%N)] Python completed"

# Check system state after
echo "[\$(date +%T.%N)] Post-execution memory:"
free -h

echo "12"

# Ensure torchdata is present (repo uses it)
python - <<'PY'
import importlib, subprocess, sys
try:
    importlib.import_module("torchdata")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchdata"])
try:
    importlib.import_module("pynvml")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pynvml"])
PY

echo "13"

# Run train.py but bypass the fragile NVML precheck
cd "\$REPO"

echo "NVML/driver check:"
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

echo "14"

python3 - <<'PY'
try:
    import pynvml as N
    N.nvmlInit()
    print("pynvml OK; GPUs:", N.nvmlDeviceGetCount())
    N.nvmlShutdown()
except Exception as e:
    print("pynvml FAIL:", e)
    raise
PY

echo "15"

# Use torchrun to launch training - spawns N processes internally for N GPUs
# Each process coordinates via DDP through Lightning Fabric
python3 "\$REPO/train.py" \
    --config "\$CFG" \
    --out_dir "\$OUT" \
    --run_name "recur1b-mig-\${SLURM_JOB_ID}"

# torchrun --nproc_per_node=4  --master_port=17777 "\$REPO/train.py" \
#       --config "\$CFG" \
#       --out_dir "\$OUT" \
#       --run_name "recur1b-mig-\${SLURM_JOB_ID}"

echo "Finished. Artifacts in \$OUT"
EOF
)

# Extract job ID from sbatch output
JOB_ID="${JOB_OUTPUT##* }"

LOG_FILE="/path/to/fast_storage/recpre/cluster/logs/train_out_${JOB_ID}.txt"
ERR_FILE="/path/to/fast_storage/recpre/cluster/logs/train_err_${JOB_ID}.txt"

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
tail -f "$LOG_FILE" "$ERR_FILE" --retry
