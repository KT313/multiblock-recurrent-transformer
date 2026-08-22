#!/usr/bin/env bash
# End-to-end training smoke run on GPU: launches the REAL train.py with the
# tiny config in this folder. Pass criterion: exits 0 ("TRAIN SMOKE PASSED").
#
# Usage (from anywhere):
#   bash tests/train_smoke/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."      # repo root, so all config paths resolve

export WANDB_MODE=offline       # no wandb account/network needed; logs stay local

OUT=tests/train_smoke/out
rm -rf "$OUT"                   # each run starts fresh (config has resume: False)

if [ ! -d tests/train_smoke/data ]; then
    echo "== generating synthetic data + tokenizer =="
    uv run --project tests python tests/train_smoke/prepare_data.py
fi

echo "== launching train.py (tiny model, ~20 steps, single GPU) =="
uv run --project tests python train.py \
    --config tests/train_smoke/tiny_multistage.yaml \
    --out_dir "$OUT"

echo "TRAIN SMOKE PASSED"
