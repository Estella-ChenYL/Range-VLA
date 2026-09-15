#!/usr/bin/env bash
# In-pod RMBench eval entrypoint (called by submit_eval_rmbench.sh).
#
# Two-process model inside one pod (see examples/eval_protocol.md):
#   1. policy server  -> training env python (system python, torch + starVLA)
#   2. RMBench sim    -> `rmbench` conda env (sapien 3.0 + curobo + pytorch3d),
#                        baked in the :rmbench-eval image
#                        (deployment/docker/Dockerfile.rmbench_eval)
#
# RMBench source + assets live on S3 (synced by submit_eval_rmbench.sh) and are
# materialized onto EFS once (RMBENCH_EFS_DIR, .rmbench_ready marker); later
# pods reuse the EFS copy. The actual per-task eval loop (policy server
# lifecycle + upstream script/eval_policy.py invocation) is delegated to
# eval_files/eval_rmbench.sh, the same entry used for local evals.
set -Eeuo pipefail

export BATCH_SIZE="${BATCH_SIZE:-1}"
export BATCH_WAIT_MS="${BATCH_WAIT_MS:-20}"
export TEST_NUM="${TEST_NUM:-100}"
for knob in BATCH_SIZE TEST_NUM; do
    [[ "${!knob}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] ${knob} must be a positive integer" >&2; exit 1; }
done
[[ "${BATCH_WAIT_MS}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "[ERROR] BATCH_WAIT_MS must be a nonnegative integer" >&2; exit 1; }

log() { echo "[$(date '+%F %T')] $*" >&2; }

: "${STARVLA_ASSET_ROOT:?STARVLA_ASSET_ROOT not set}"
: "${S3_RMBENCH_DIR:?S3_RMBENCH_DIR not set}"
: "${RMBENCH_EFS_DIR:?RMBENCH_EFS_DIR not set}"
: "${TRAIN_RUN_PREFIX:?TRAIN_RUN_PREFIX not set}"
: "${EVAL_TASKS:?EVAL_TASKS not set}"
: "${EVAL_OUTPUT_DIR:?EVAL_OUTPUT_DIR not set}"
: "${S3_EVAL_RESULTS_DIR:?S3_EVAL_RESULTS_DIR not set}"

# ---- Interpreter / env ----
STARVLA_PYTHON=${STARVLA_PYTHON:-python}                              # training env (torch + starVLA)
RMBENCH_PYTHON=${RMBENCH_PYTHON:-/opt/conda/envs/rmbench/bin/python}  # rmbench conda env (baked in image)

# ---- Eval knobs ----
EVAL_CKPT=${EVAL_CKPT:-""}                 # explicit ckpt; only valid for a single EVAL_TASKS entry
TEST_NUM=${TEST_NUM:-100}                  # total valid episodes per task
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
SEED=${SEED:-0}
GPU_ID=${GPU_ID:-0}
BASE_PORT=${BASE_PORT:-5694}

STARVLA_DIR=/data/work/starvla
cd "${STARVLA_DIR}"

FAILED_TASKS=()
JOB_START_EPOCH=$(date +%s)  # step 5 only collects results produced after this

mkdir -p "${EVAL_OUTPUT_DIR}"
exec > >(tee -a "${EVAL_OUTPUT_DIR}/pod_eval.log") 2>&1
EVAL_CHILD_PID=""

# Preserve diagnostics even when setup fails or the job is interrupted.
sync_diagnostics() {
    local rc=$?
    trap - EXIT
    if [[ -n "${EVAL_CHILD_PID}" ]]; then
        kill "${EVAL_CHILD_PID}" 2>/dev/null || true
        wait "${EVAL_CHILD_PID}" 2>/dev/null || true
    fi
    if [[ -d "${EVAL_OUTPUT_DIR}" ]] && command -v aws >/dev/null 2>&1; then
        printf 'exit_code=%s\n' "${rc}" > "${EVAL_OUTPUT_DIR}/exit_status.txt"
        aws s3 sync "${EVAL_OUTPUT_DIR}" "${S3_EVAL_RESULTS_DIR}" --no-follow-symlinks --only-show-errors || true
    fi
    exit "${rc}"
}
trap sync_diagnostics EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- Step 0: aws CLI (for S3<->EFS materialization and result sync) ----
# awscli's credential chain only reads AWS_* names, but the pod env carries the
# S3_* names the submit scripts use (and the image has no ~/.aws/credentials).
if [ -n "${S3_ACCESS_KEY:-}" ] && [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    export AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY}"
    export AWS_SECRET_ACCESS_KEY="${S3_SECRET_KEY:-}"
fi
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-northeast-1}"

if ! command -v aws >/dev/null 2>&1; then
    log "[0/5] installing awscli"
    pip install --quiet awscli || { log "[ERROR] awscli install failed (required for RMBench sync)"; exit 1; }
fi

# ---- Step 1: link assets (base VLM for the policy server) ----
log "[1/5] linking assets from ${STARVLA_ASSET_ROOT}"
mkdir -p playground
ln -sfn "${STARVLA_ASSET_ROOT}/Pretrained_models" playground/Pretrained_models
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/efs/huggingface_models
export HF_HUB_OFFLINE=1

# ---- Step 2: RMBench source + assets onto EFS (once) ----
if [[ "${REPLAY_TRAINING:-0}" != "1" ]]; then
log "[2/5] preparing RMBench: efs=${RMBENCH_EFS_DIR} src=${S3_RMBENCH_DIR}"
mkdir -p "${RMBENCH_EFS_DIR}" "${EVAL_OUTPUT_DIR}"

if [ ! -f "${RMBENCH_EFS_DIR}/.rmbench_ready" ]; then
    log "[2/5] syncing RMBench S3 -> EFS (first run; later pods reuse the EFS copy)"
    aws s3 sync "${S3_RMBENCH_DIR}" "${RMBENCH_EFS_DIR}" --no-follow-symlinks --only-show-errors
    [ -d "${RMBENCH_EFS_DIR}/assets/embodiments" ] || { log "[ERROR] assets/embodiments missing after sync; rerun submit with SYNC_RMBENCH=1"; exit 1; }
    rm -f "${RMBENCH_EFS_DIR}/.rmbench_ready"  # not ready until paths are configured below
else
    log "[2/5] reusing cached RMBench checkout on EFS"
fi

# Bake absolute asset paths into the embodiment configs. Non-interactive when
# assets/embodiments exists (verified upstream); idempotent per EFS checkout.
if [ ! -f "${RMBENCH_EFS_DIR}/.rmbench_ready" ]; then
    log "[2/5] running update_embodiment_config_path.py"
    (cd "${RMBENCH_EFS_DIR}" && "${RMBENCH_PYTHON}" script/update_embodiment_config_path.py </dev/null)
    touch "${RMBENCH_EFS_DIR}/.rmbench_ready"
fi

log "[2/5] RMBench smoke test (imports + sapien Vulkan renderer)"
"${RMBENCH_PYTHON}" -c "import sapien, curobo, pytorch3d, mplib; print('rmbench env ok: sapien=%s' % sapien.__version__)"
(cd "${RMBENCH_EFS_DIR}" && "${RMBENCH_PYTHON}" script/test_render.py) \
    || { log "[ERROR] sapien render smoke test failed (Vulkan/GPU). Check NVIDIA_DRIVER_CAPABILITIES and the ICD mount."; exit 1; }

log "[2/5] GPU status before eval"
nvidia-smi || true

# Patch the pod-local cuRobo source, including spawned planner workers.
# Fail before staging/loading the checkpoint if the planner cannot warm up.
log "[2/5] cuRobo LBFGS compatibility patch + planner warmup"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${RMBENCH_PYTHON}" \
    "${STARVLA_DIR}/examples/simBenchmarks/RMBench/eval_files/prepare_curobo.py"
else
    log "[2/5] training replay: simulator setup skipped"
fi

# ---- Step 3: resolve checkpoints per task (EFS paths or s3:// URIs) ----
log "[3/5] resolving checkpoints (prefix=${TRAIN_RUN_PREFIX}_<task>)"
# torch.load of a ~10 GiB .pt from EFS is what killed the last job: the
# policy server never bound its port within 600s (log stopped at DiT init,
# which is immediately before torch.load). Stage onto the pod-local NVMe
# (mounted at /local-ssd) so load is a local disk read. Keep this OUT of
# EVAL_OUTPUT_DIR so the 10 GiB ckpt is not synced back as an "eval result".
if [ -d /local-ssd ]; then
    STAGE_DIR=/local-ssd/rmbench-ckpts
else
    STAGE_DIR=/tmp/rmbench-ckpts
    log "[WARN] /local-ssd not mounted; staging ckpts to ${STAGE_DIR}"
fi
mkdir -p "${STAGE_DIR}"

is_s3() { [[ "$1" == s3://* ]]; }

# Latest steps_*_pytorch_model.pt under a run dir, local or s3 (prints "" if none).
# Always returns 0 so `ckpt=$(latest_ckpt ...)` survives `set -e` on empty results.
latest_ckpt() {
    local run_dir="$1" name
    if is_s3 "${run_dir}"; then
        name="$(aws s3 ls "${run_dir}/checkpoints/" | awk '{print $4}' \
                | grep -E '^steps_[0-9]+_pytorch_model\.pt$' | sort -V | tail -1 || true)"
        if [ -n "${name}" ]; then
            printf '%s/checkpoints/%s\n' "${run_dir}" "${name}"
        fi
    else
        ls -1 "${run_dir}"/checkpoints/steps_*_pytorch_model.pt 2>/dev/null | sort -V | tail -1 || true
    fi
    return 0
}

# Materialize an s3-run ckpt's eval inputs (config.yaml + dataset_statistics.json
# + the .pt) into STAGE_DIR, preserving the <run>/checkpoints/ layout the policy
# server loader requires (share_tools.py: ckpt.parents[1]/{config.yaml,dataset_statistics.json}).
# Prints the staged local ckpt path on stdout; everything else goes to stderr.
stage_s3_ckpt() {
    local s3_ckpt="$1"
    local s3_run_dir="${s3_ckpt%/checkpoints/*}"
    if [ "${s3_run_dir}" = "${s3_ckpt}" ]; then
        log "[ERROR] S3 ckpt must live under <run_dir>/checkpoints/: ${s3_ckpt}"
        return 1
    fi
    local run_name local_run ckpt_name
    run_name="$(basename "${s3_run_dir}")"
    ckpt_name="$(basename "${s3_ckpt}")"
    local_run="${STAGE_DIR}/${run_name}"
    mkdir -p "${local_run}/checkpoints"
    log "[3/5] staging from S3: ${s3_run_dir} (config.yaml + dataset_statistics.json + ${ckpt_name})"
    aws s3 cp "${s3_run_dir}/config.yaml" "${local_run}/config.yaml" --only-show-errors \
        && aws s3 cp "${s3_run_dir}/dataset_statistics.json" "${local_run}/dataset_statistics.json" --only-show-errors \
        && aws s3 cp "${s3_ckpt}" "${local_run}/checkpoints/${ckpt_name}" --only-show-errors \
        || { log "[ERROR] staging failed for ${s3_ckpt} (run dir must contain config.yaml + dataset_statistics.json)"; return 1; }
    log "[3/5] staged $(du -h "${local_run}/checkpoints/${ckpt_name}" | awk '{print $1}') -> ${local_run}/checkpoints/${ckpt_name}"
    printf '%s\n' "${local_run}/checkpoints/${ckpt_name}"
}

# Copy an already-local run (typically EFS) onto STAGE_DIR so torch.load is
# not an EFS read. No-op if the ckpt is already under STAGE_DIR.
stage_local_ckpt() {
    local src_ckpt="$1"
    local src_run
    src_run="$(cd "$(dirname "$(dirname "${src_ckpt}")")" && pwd)"
    case "${src_ckpt}" in
        "${STAGE_DIR}"/*) printf '%s\n' "${src_ckpt}"; return 0 ;;
    esac
    local run_name ckpt_name local_run
    run_name="$(basename "${src_run}")"
    ckpt_name="$(basename "${src_ckpt}")"
    local_run="${STAGE_DIR}/${run_name}"
    mkdir -p "${local_run}/checkpoints"
    log "[3/5] copying ${src_run} -> ${local_run} (avoid torch.load from EFS)"
    cp -f "${src_run}/config.yaml" "${local_run}/config.yaml" \
        && cp -f "${src_run}/dataset_statistics.json" "${local_run}/dataset_statistics.json" \
        && cp -f "${src_ckpt}" "${local_run}/checkpoints/${ckpt_name}" \
        || { log "[ERROR] local stage failed for ${src_ckpt}"; return 1; }
    printf '%s\n' "${local_run}/checkpoints/${ckpt_name}"
}

declare -A TASK_CKPT=()
_task_count=0
for task in ${EVAL_TASKS}; do
    _task_count=$((_task_count + 1))
    run_dir="${TRAIN_RUN_PREFIX}_${task}"
    ckpt=""
    if [ -n "${EVAL_CKPT}" ]; then
        ckpt="${EVAL_CKPT}"
    else
        ckpt="$(latest_ckpt "${run_dir}")"
    fi
    if [ -z "${ckpt}" ]; then
        log "[ERROR] no checkpoint for task=${task} under ${run_dir}/checkpoints/"
        FAILED_TASKS+=("${task}")
        continue
    fi
    if is_s3 "${ckpt}"; then
        if ! ckpt="$(stage_s3_ckpt "${ckpt}")"; then
            FAILED_TASKS+=("${task}")
            continue
        fi
    else
        if ! ckpt="$(stage_local_ckpt "${ckpt}")"; then
            FAILED_TASKS+=("${task}")
            continue
        fi
    fi
    if [ ! -f "${ckpt}" ]; then
        log "[ERROR] checkpoint not found: ${ckpt} (task=${task})"
        FAILED_TASKS+=("${task}")
        continue
    fi
    # The policy server + eval client need config.yaml + dataset_statistics.json
    # two dirs up from the ckpt.
    run_dir_local="$(dirname "$(dirname "${ckpt}")")"
    if [ ! -f "${run_dir_local}/dataset_statistics.json" ]; then
        log "[ERROR] ${run_dir_local}/dataset_statistics.json missing (needed for state normalization)"
        FAILED_TASKS+=("${task}")
        continue
    fi
    TASK_CKPT["${task}"]="${ckpt}"
    log "[3/5] ${task} -> ${ckpt}"
done
if [ "${#TASK_CKPT[@]}" -eq 0 ]; then
    log "[ERROR] no usable checkpoints at all; aborting"
    exit 1
fi
if [ -n "${EVAL_CKPT}" ] && [ "${_task_count}" != "1" ]; then
    log "[ERROR] EVAL_CKPT is only valid with a single EVAL_TASKS entry (got ${_task_count})"
    exit 1
fi

# ---- Step 4: per-task eval (server + sim orchestrated by eval_rmbench.sh) ----
log "[4/5] starting RMBench eval: tasks=[${!TASK_CKPT[*]}] instruction_type=${INSTRUCTION_TYPE}"
port="${BASE_PORT}"
for task in ${EVAL_TASKS}; do
    ckpt="${TASK_CKPT[${task}]:-}"
    [ -n "${ckpt}" ] || continue  # already logged + tracked as failed above

    log "[4/5] === ${task} === (port=${port} gpu=${GPU_ID})"
    if [[ "${REPLAY_TRAINING:-0}" == "1" ]]; then
        export REPLAY_DATA_ROOT="${STAGE_DIR}/replay-data"
        mkdir -p "${REPLAY_DATA_ROOT}/${task}"
        log "[4/5] staging converted training data for ${task}"
        if ! aws s3 sync "${S3_ASSET_PREFIX:?}/datasets/rmbench_lerobot/${task}/" \
            "${REPLAY_DATA_ROOT}/${task}/" --no-follow-symlinks --only-show-errors; then
            FAILED_TASKS+=("${task}")
            continue
        fi
    fi
    set +e
    RMBENCH_HOME="${RMBENCH_EFS_DIR}" \
    STARVLA_DIR="${STARVLA_DIR}" \
    STARVLA_PYTHON="${STARVLA_PYTHON}" \
    RMBENCH_PYTHON="${RMBENCH_PYTHON}" \
    RMBENCH_EVAL_LOG_ROOT="${EVAL_OUTPUT_DIR}/logs" \
    SERVER_WAIT_SECS="${SERVER_WAIT_SECS:-1800}" \
    TEST_NUM="${TEST_NUM}" \
    bash "${STARVLA_DIR}/examples/simBenchmarks/RMBench/eval_files/eval_rmbench.sh" \
        -c "${ckpt}" \
        -s "${SEED}" \
        -g "${GPU_ID}" \
        -p "${port}" \
        --instruction-type "${INSTRUCTION_TYPE}" \
        --ckpt-setting "starvla" \
        "${task}" &
    EVAL_CHILD_PID=$!
    wait "${EVAL_CHILD_PID}"
    rc=$?
    EVAL_CHILD_PID=""
    set -e
    if [ "${rc}" != "0" ]; then
        log "[ERROR] task ${task} eval failed (exit ${rc}); continuing with next task"
        FAILED_TASKS+=("${task}")
    fi
    port=$((port + 1))
done

# ---- Step 5: collect results -> EFS -> S3 ----
log "[5/5] collecting results"
# Task outputs are written directly under this run's log root.
log "[5/5] results on efs: ${EVAL_OUTPUT_DIR}"

# Print full per-task results into the pod log (KOALA log tail is often the
# only thing checked). eval_rmbench.sh already printed a live summary per
# task; this is the consolidated final dump.
mapfile -t _result_files < <(find "${EVAL_OUTPUT_DIR}" -name "_result.txt" 2>/dev/null | sort)
if [ "${#_result_files[@]}" -gt 0 ]; then
    echo "================ RMBench eval results (final) ================" >&2
    for res in "${_result_files[@]}"; do
        echo "----- ${res#"${EVAL_OUTPUT_DIR}"/} -----" >&2
        cat "${res}" >&2
    done
    echo "================ success rate summary ================" >&2
    for res in "${_result_files[@]}"; do
        _task="$(basename "$(dirname "${res}")")"
        _rate="$(grep -h "Success Rate" "${res}" | head -1 || true)"
        printf '  %-22s %s\n' "${_task}" "${_rate:-no result line}" >&2
    done
elif [[ "${REPLAY_TRAINING:-0}" == "1" ]]; then
    log "[5/5] training replay outputs: logs/*/training_replay/<task>/episode_*/metrics.json"
else
    log "[WARN] no _result.txt found under ${EVAL_OUTPUT_DIR}"
fi

if command -v aws >/dev/null 2>&1; then
    log "[5/5] syncing results -> ${S3_EVAL_RESULTS_DIR}"
    aws s3 sync "${EVAL_OUTPUT_DIR}" "${S3_EVAL_RESULTS_DIR}" --no-follow-symlinks --only-show-errors || true
    log "[5/5] sync done. pull locally:"
    log "  aws s3 sync ${S3_EVAL_RESULTS_DIR}/ ./evaluate_results/$(basename "${EVAL_OUTPUT_DIR}")/"
else
    log "[5/5] aws not available; results remain on EFS only"
fi

if [ "${#FAILED_TASKS[@]}" -gt 0 ]; then
    log "[ERROR] eval finished with failed tasks: ${FAILED_TASKS[*]}"
    exit 1
fi
log "[DONE] all RMBench eval tasks finished"
