#!/usr/bin/env bash
# Fetch offline-wandb run data for a given run from the training cluster and
# sync it to wandb. The cluster nodes had no internet access, so wandb ran in
# offline mode during training; this pulls the run directory to the local
# machine and uploads it from here.
#
# Usage:
#   REMOTE_HOST=<host> REMOTE_USER=<user> REMOTE_BASE=<path/to/outputs> \
#     ./fetch_offline_wandb.sh <RUN_ID>
#
# Optional env: REMOTE_PORT (default 22), DEST_DIR, WANDB_PROJECT, WANDB_ENTITY

set -euo pipefail

usage() {
  echo "Usage: $0 <RUN_ID>"
  echo "Required env: REMOTE_HOST, REMOTE_USER, REMOTE_BASE (outputs dir on the cluster)"
  echo "Optional env: REMOTE_PORT (default 22), DEST_DIR, WANDB_PROJECT, WANDB_ENTITY"
  exit 1
}

[[ $# -ge 1 ]] || usage
RUN_ID="$1"

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST}"
REMOTE_USER="${REMOTE_USER:?set REMOTE_USER}"
REMOTE_BASE="${REMOTE_BASE:?set REMOTE_BASE (e.g. /path/to/recpre/outputs)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
DEST_DIR="${DEST_DIR:-./retrieved_files/${RUN_ID}/wandb}"

SRC="${REMOTE_BASE}/${RUN_ID}/wandb/*"
mkdir -p "$DEST_DIR"

echo "Copying from ${REMOTE_USER}@${REMOTE_HOST}:${SRC}  ->  ${DEST_DIR}/"
rsync -avh --progress --partial \
  -e "ssh -p ${REMOTE_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${SRC}" \
  "${DEST_DIR}/"

echo "Syncing to wandb..."
wandb sync \
  ${WANDB_PROJECT:+--project "$WANDB_PROJECT"} \
  ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} \
  "${DEST_DIR}"/offline-run-*
echo "Done."
