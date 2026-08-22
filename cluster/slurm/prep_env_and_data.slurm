#!/bin/bash
#SBATCH --job-name=recur_prep
#SBATCH --partition=advance
#SBATCH --qos=advance
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/path/to/fast_storage/recpre/cluster/logs/prep_out_%j.txt
#SBATCH --error=/path/to/fast_storage/recpre/cluster/logs/prep_err_%j.txt

set -euo pipefail
set -x

BASE=/path/to/fast_storage/recpre
REPO="$BASE"                  # your repo root is this directory
VENV="$BASE/venv"

mkdir -p "$BASE" "$BASE/artifacts" "$BASE/datasets" "$BASE/.cache/huggingface"

# venv (idempotent)
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  # torch + deps (works even in CPU job; GPU not required for install)
  # pip install "torch>=2.4" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  # pip install transformers datasets tokenizers wandb einops lightning==2.4.* safetensors pandas jsonargparse typing_extensions tabulate
  cd "$BASE/recurrent-pretraining"
  pip install -e ".[dev,data,quant,torchgte240]"
else
  source "$VENV/bin/activate"
fi

export HF_HOME="$BASE/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

cd "$REPO"

# Prepare tokenizer (download small 32k llama tokenizer)
python - <<'PY' || { echo "Tokenizer prep failed"; exit 1; }
import os
from pathlib import Path
from transformers import AutoTokenizer
BASE = Path("/path/to/fast_storage/recpre")
OUT = BASE / "artifacts" / "tokenizer_llama32k"
OUT.mkdir(parents=True, exist_ok=True)
name = "hf-internal-testing/llama-tokenizer"
tok = AutoTokenizer.from_pretrained(name, use_fast=True)
tok.save_pretrained(OUT)
print("Saved tokenizer to:", OUT)
PY

# Prepare tiny wikitext2 dataset
# python3 /path/to/fast_storage/recpre/cluster/scripts/prepare_fineweb_edu.py

echo "Prep done."
