#!/usr/bin/env bash
# In-pod LIBERO eval entrypoint (called by submit_eval_libero.sh).
#
# Mirrors the README "2. Evaluation Workflow" two-process model, inside one pod:
#   1. policy server  -> training env python (system python, torch + starVLA)
#   2. LIBERO sim     -> `libero` conda env (mujoco 3.2.3 + numpy 1.24.4 + LIBERO)
#
# LIBERO source is cloned to EFS (shared across pods); the `libero` conda env
# itself is baked in the image (see deployment/docker/Dockerfile.project).
set -Eeuo pipefail

log() { echo "[$(date '+%F %T')] $*" >&2; }

trap 'rc=$?; log "[ERROR] eval failed at line ${LINENO} with exit ${rc}"; exit ${rc}' ERR

: "${STARVLA_ASSET_ROOT:?STARVLA_ASSET_ROOT not set}"
: "${TRAIN_RUN_DIR:?TRAIN_RUN_DIR not set}"     # EFS dir with config.yaml + dataset_statistics.json + checkpoints/
: "${EVAL_OUTPUT_DIR:?EVAL_OUTPUT_DIR not set}"
: "${LIBERO_SRC_DIR:?LIBERO_SRC_DIR not set}"
: "${PIP_CACHE_DIR:?PIP_CACHE_DIR not set}"

# ---- Interpreter / env ----
STARVLA_PYTHON=${STARVLA_PYTHON:-python}                            # training env (torch + starVLA)
LIBERO_PYTHON=${LIBERO_PYTHON:-/opt/conda/envs/libero/bin/python}  # libero conda env
CONDA=${CONDA:-/opt/conda/bin/conda}

# ---- Eval knobs ----
TASK_SUITES=${TASK_SUITES:-"libero_10 libero_goal libero_object libero_spatial"}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
EVAL_CKPT=${EVAL_CKPT:-""}     # empty -> latest steps_*_pytorch_model.pt under ${TRAIN_RUN_DIR}/checkpoints/
USE_BF16=${USE_BF16:-1}
USE_CANONICAL_FORWARD=${USE_CANONICAL_FORWARD:-""}   # "true"/"false" -> --config_override (released Qwen3-PI compat)
BASE_PORT=${BASE_PORT:-10093}
GPU_ID=${GPU_ID:-0}
ASSET_USER=${ASSET_USER:-danyangchen}
ASSET_BUCKET_NAME=${ASSET_BUCKET_NAME:-helix-asset-ap-northeast-1}
S3_EVAL_RESULTS_DIR=${S3_EVAL_RESULTS_DIR:-"s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA/exp/libero-eval/$(basename "${EVAL_OUTPUT_DIR}")"}

STARVLA_DIR=/data/work/starvla
cd "${STARVLA_DIR}"

# ---- Step 0: aws CLI (for result sync) ----
if ! command -v aws >/dev/null 2>&1; then
    log "[0/5] installing awscli"
    pip install --quiet awscli || log "[WARN] awscli install failed; result sync skipped"
fi

# ---- Step 1: link assets (base VLM for the policy server) ----
log "[1/5] linking assets from ${STARVLA_ASSET_ROOT}"
mkdir -p playground
ln -sfn "${STARVLA_ASSET_ROOT}/Pretrained_models" playground/Pretrained_models
export PYTHONPATH=/data/work/starvla:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/efs/huggingface_models
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ---- Step 2: LIBERO env (clone to EFS + install deps into `libero` conda env) ----
log "[2/5] preparing LIBERO: src=${LIBERO_SRC_DIR}"
mkdir -p "$(dirname "${LIBERO_SRC_DIR}")" "${PIP_CACHE_DIR}"

if [ ! -d "${LIBERO_SRC_DIR}/.git" ]; then
    log "[2/5] cloning LIBERO -> ${LIBERO_SRC_DIR}"
    rm -rf "${LIBERO_SRC_DIR}"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_SRC_DIR}"
else
    log "[2/5] reusing cached LIBERO source"
fi

# `pip install -e .` pulls LIBERO's own deps (torch/robosuite/bddl/gym) with the
# versions LIBERO itself declares — the same path the project's install_libero.sh
# takes. Cached on EFS so fresh pods don't re-download.
log "[2/5] pip install -e LIBERO (into libero env, cache=${PIP_CACHE_DIR})"
"${CONDA}" run -n libero pip install --cache-dir "${PIP_CACHE_DIR}" --root-user-action=ignore -e "${LIBERO_SRC_DIR}"

# LIBERO's package __init__ prompts interactively without a config.yaml.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_SRC_DIR}/libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]; then
    log "[2/5] writing LIBERO config -> ${LIBERO_CONFIG_PATH}/config.yaml"
    LIBERO_SRC_DIR="${LIBERO_SRC_DIR}" LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" "${LIBERO_PYTHON}" - <<'PY'
import os, yaml
from pathlib import Path
root = Path(os.environ["LIBERO_SRC_DIR"]) / "libero" / "libero"
cfg = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
out = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
out.write_text(yaml.dump(cfg))
print(out.read_text())
PY
fi

# torch>=2.6 defaults torch.load(weights_only=True); LIBERO init-state pickles need False.
LIBERO_BENCH_INIT="${LIBERO_SRC_DIR}/libero/libero/benchmark/__init__.py"
if [ -f "${LIBERO_BENCH_INIT}" ] && ! grep -q 'weights_only=False' "${LIBERO_BENCH_INIT}"; then
    log "[2/5] patching LIBERO torch.load(..., weights_only=False)"
    LIBERO_BENCH_INIT="${LIBERO_BENCH_INIT}" "${LIBERO_PYTHON}" - <<'PY'
from pathlib import Path
import os
p = Path(os.environ["LIBERO_BENCH_INIT"])
text = p.read_text()
old = "init_states = torch.load(init_states_path)"
new = "init_states = torch.load(init_states_path, weights_only=False)"
if old in text:
    p.write_text(text.replace(old, new, 1))
    print(f"patched {p}")
else:
    print(f"patch target not found in {p} (already patched or changed upstream)")
PY
fi

log "[2/5] LIBERO smoke test"
"${LIBERO_PYTHON}" -c 'import libero, mujoco, robosuite; from libero.libero import benchmark; print("libero ok; mujoco=%s; robosuite=%s" % (mujoco.__version__, robosuite.__version__)); print("suites:", sorted(benchmark.get_benchmark_dict().keys()))'

# ---- Step 3: resolve checkpoint ----
log "[3/5] resolving checkpoint"
if [ -z "${EVAL_CKPT}" ]; then
    EVAL_CKPT="$(ls -1 "${TRAIN_RUN_DIR}"/checkpoints/steps_*_pytorch_model.pt 2>/dev/null | sort -V | tail -1 || true)"
fi
test -n "${EVAL_CKPT}" || { log "[ERROR] no checkpoint found in ${TRAIN_RUN_DIR}/checkpoints/"; exit 1; }
test -f "${EVAL_CKPT}" || { log "[ERROR] checkpoint not found: ${EVAL_CKPT}"; exit 1; }
log "[3/5] using ckpt: ${EVAL_CKPT}"
log "[3/5] run dir (config.yaml + dataset_statistics.json): ${TRAIN_RUN_DIR}"

log "[3/5] GPU status before eval"
nvidia-smi || true

# ---- Step 4: run eval (one task suite at a time, README workflow) ----
mkdir -p "${EVAL_OUTPUT_DIR}"
log "[4/5] starting LIBERO eval: suites=[${TASK_SUITES}] trials=${NUM_TRIALS_PER_TASK}"

SERVER_EXTRA=()
if [ "${USE_BF16}" = "1" ]; then
    SERVER_EXTRA+=(--use_bf16)
fi
if [ -n "${USE_CANONICAL_FORWARD}" ]; then
    SERVER_EXTRA+=(--config_override "framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}")
    log "[4/5] canonical-forward override: ${USE_CANONICAL_FORWARD}"
fi

for suite in ${TASK_SUITES}; do
    port=${BASE_PORT}
    video_out="${EVAL_OUTPUT_DIR}/videos/${suite}"
    log_dir="${EVAL_OUTPUT_DIR}/logs/${suite}"
    mkdir -p "${video_out}" "${log_dir}"

    log "[4/5] suite=${suite} port=${port} gpu=${GPU_ID}"

    # Start policy server (training env python).
    CUDA_VISIBLE_DEVICES=${GPU_ID} "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
        --ckpt_path "${EVAL_CKPT}" \
        --port "${port}" \
        "${SERVER_EXTRA[@]}" \
        >"${log_dir}/server.log" 2>&1 &
    server_pid=$!

    # Wait until the server port is listening (with a bounded fallback).
    _ready=0
    for _ in $(seq 1 60); do
        if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then exec 3>&- 3<&-; _ready=1; break; fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then break; fi
        sleep 2
    done
    if [ "${_ready}" != "1" ]; then
        log "[ERROR] policy server failed to start for suite=${suite}; tail of server.log:"
        tail -40 "${log_dir}/server.log" || true
        kill "${server_pid}" 2>/dev/null || true
        exit 1
    fi

    # Run the LIBERO sim (libero conda env).
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
    PYTHONPATH="${PYTHONPATH}:${LIBERO_SRC_DIR}:${STARVLA_DIR}" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "${LIBERO_PYTHON}" ./examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "${EVAL_CKPT}" \
        --args.host "127.0.0.1" \
        --args.port "${port}" \
        --args.task-suite-name "${suite}" \
        --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
        --args.video-out-path "${video_out}" \
        2>&1 | tee "${log_dir}/${suite}.log"

    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    port=$((port + 1))
done

log "[4/5] eval finished. results on efs: ${EVAL_OUTPUT_DIR}"
find "${EVAL_OUTPUT_DIR}" -maxdepth 3 -type f | sort | head -50 || true

# ---- Step 5: sync results to S3 ----
log "[5/5] syncing results -> ${S3_EVAL_RESULTS_DIR}"
if command -v aws >/dev/null 2>&1; then
    aws s3 sync "${EVAL_OUTPUT_DIR}" "${S3_EVAL_RESULTS_DIR}" --no-follow-symlinks || true
    log "[5/5] sync done"
    log "pull locally:"
    log "  aws s3 sync ${S3_EVAL_RESULTS_DIR}/ ./evaluate_results/$(basename "${EVAL_OUTPUT_DIR}")/"
else
    log "[5/5] aws not available; results remain on EFS only"
fi
