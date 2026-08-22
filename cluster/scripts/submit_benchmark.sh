#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[2/2] Submitting training job (after prep finishes)..."
TRAIN_LINE=$(sbatch slurm/benchmark.slurm)
TRAIN_ID=${TRAIN_LINE##* }
echo "  train job id: $TRAIN_ID"

echo
echo "Use these to watch:"
echo "  squeue -u $USER"
echo "  tail -f /path/to/fast_storage/recpre/cluster/logs/train_out_${TRAIN_ID}.txt --retry"
