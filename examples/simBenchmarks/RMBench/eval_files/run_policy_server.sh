#!/usr/bin/env bash
# Start the starVLA WebSocket policy server for RMBench eval.
# Usage: bash run_policy_server.sh <ckpt_path> [gpu_id] [port]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Four levels up: eval_files -> RMBench -> simBenchmarks -> examples -> repo root.
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
[[ -f "${STARVLA_DIR}/deployment/model_server/server_policy.py" ]] || {
    echo "[ERROR] STARVLA_DIR=${STARVLA_DIR} missing deployment/model_server/server_policy.py" >&2
    exit 1
}

CKPT="${1:?usage: run_policy_server.sh <ckpt_path> [gpu_id] [port]}"
GPU_ID="${2:-0}"
PORT="${3:-5694}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
# `exec VAR=value cmd` is NOT a prefix assignment — bash treats VAR=value as
# the executable (pod log: `exec: CUDA_VISIBLE_DEVICES=0: not found`).
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NO_ALBUMENTATIONS_UPDATE=1
exec "${STARVLA_PYTHON}" \
  deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --batch_wait_ms "${BATCH_WAIT_MS:-20}" \
  --use_bf16
