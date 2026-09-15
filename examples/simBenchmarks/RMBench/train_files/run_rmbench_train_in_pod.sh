#!/usr/bin/env bash
# RMBench in-pod entrypoint for KOALA PyTorchJob training.
#
# Invoked by submit_train_rmbench.sh as:
#   START_CMD='set -euo pipefail; bash /data/work/starvla/examples/simBenchmarks/RMBench/train_files/run_rmbench_train_in_pod.sh'
#
# Steps:
#   [0/5] ensure awscli            (checkpoint + dataset sync)
#   [1/5] symlink assets into the repo + fail-fast asset check
#   [2/5] data conversion (idempotent): if the converted LeRobot datasets are
#         missing from the asset PVC, convert from the raw asset to /local-ssd
#         and push the result back to S3 for reuse by later jobs; then
#         materialize the dataset onto /local-ssd (asset PVC is read-only)
#   [3/5] GPU check
#   [4/5] per-task training sweep: TRAIN_TASKS sequentially, one independent
#         run per task (own run_id / ckpt dir / S3 mirror); a failed task does
#         NOT abort the sweep
#   [5/5] push derived dataset caches (stats_gr00t.json etc.) back to the
#         asset + sweep summary (exit 1 if any task failed)
set -Eeuo pipefail

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    echo "[$(date '+%F %T')] [ERROR] $*" >&2
    exit 1
}

# ---- Required environment (passed as pod env by submit_train_rmbench.sh) ----
: "${STARVLA_ASSET_ROOT:?not set (e.g. /asset/<user>/starVLA)}"
: "${S3_ASSET_PREFIX:?not set (e.g. s3://<asset-bucket>/<user>/starVLA)}"
: "${run_root_dir:?not set}"
: "${run_id:?not set}"
: "${NPROC_PER_NODE:?not set}"
: "${WANDB_PROJECT:?not set}"
: "${WANDB_ENTITY:?not set}"

# ---- Optional environment (defaults mirror the local launcher) ----
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/simBenchmarks/RMBench/train_files/starvla_qwenpiv3_rmbench.yaml}
rmbench_data_root=${rmbench_data_root:-playground/Datasets/rmbench_lerobot}
data_mix=${data_mix:-rmbench_all}
# Per RMBench protocol each task trains its OWN policy (no mixture). The pod
# loops TRAIN_TASKS sequentially, one independent run per task. Default: the 9
# tasks selected for the sweep (excludes classify_blocks / storage_blocks /
# place_block_mat). TRAIN_TASK (singular) is accepted as an alias.
# Set TRAIN_TASKS="swap_T" for a single-task smoke run.
TRAIN_TASKS=${TRAIN_TASKS:-${TRAIN_TASK:-"battery_try blocks_ranking_try cover_blocks observe_and_pickup press_button put_back_block rearrange_blocks swap_blocks swap_T"}}
per_device_bs=${per_device_bs:-16}
# Default '' = full fine-tune (matches yaml freeze_modules: ""). Set
# freeze_module_list=qwen_vl_interface to freeze the VLM on purpose.
freeze_module_list=${freeze_module_list-}
num_train_epochs=${num_train_epochs:-10}
save_interval=${save_interval:-2000}
logging_frequency=${logging_frequency:-100}
eval_interval=${eval_interval:-2000}
CKPT_MIRROR_INTERVAL=${CKPT_MIRROR_INTERVAL:-300}
S3_CHECKPOINT_DIR=${S3_CHECKPOINT_DIR:-}
# Delete each task's EFS checkpoint dir once its final S3 sync succeeds (EFS is
# small/shared; S3 is the durable copy). Set to 0 to keep the EFS copies.
DELETE_EFS_CKPT_AFTER_SYNC=${DELETE_EFS_CKPT_AFTER_SYNC:-1}
RESUME_CKPT=${RESUME_CKPT:-}
if [ -n "${RESUME_CKPT}" ]; then
    read -r -a _resume_tasks <<< "${TRAIN_TASKS}"
    [ "${#_resume_tasks[@]}" = "1" ] || die "RESUME_CKPT requires exactly one TRAIN_TASK"
fi

RMBENCH_TASKS=(
    battery_try blocks_ranking_try classify_blocks cover_blocks
    observe_and_pickup place_block_mat press_button put_back_block
    rearrange_blocks storage_blocks swap_blocks swap_T
)

# Validate requested tasks up front: an unknown name would otherwise only fail
# at training time with a mixture KeyError, after minutes of setup per task.
for _t in ${TRAIN_TASKS}; do
    case " ${RMBENCH_TASKS[*]} " in
        *" ${_t} "*) ;;
        *) die "unknown task '${_t}' in TRAIN_TASKS='${TRAIN_TASKS}'. Valid: ${RMBENCH_TASKS[*]}" ;;
    esac
done

REPO=/data/work/starvla
RAW_ASSET_ROOT="${STARVLA_ASSET_ROOT}/datasets/rmbench_raw"
CONVERTED_LINK="playground/Datasets/rmbench_lerobot"
LOCAL_CONVERT_ROOT=/local-ssd/rmbench_lerobot
# Kubeflow PyTorchJob exposes GROUP_RANK (0 on the master). Conversion + the
# S3 push-back run on rank 0 only; other ranks wait for the data to appear.
GROUP_RANK="${GROUP_RANK:-${RANK:-0}}"

MIRROR_PID=""
cleanup() {
    if [ -n "${MIRROR_PID}" ] && kill -0 "${MIRROR_PID}" 2>/dev/null; then
        kill "${MIRROR_PID}" 2>/dev/null || true
    fi
}

trap 'rc=$?; cleanup; log "[ERROR] container failed at line ${LINENO} with exit ${rc}"; exit ${rc}' ERR
trap 'cleanup' EXIT

log "job=${JOB_NAME:-unknown} start (nproc=${NPROC_PER_NODE}, group_rank=${GROUP_RANK})"

# ---- Step 0: ensure aws CLI is available for checkpoint/dataset sync ----
if ! command -v aws >/dev/null 2>&1; then
    log "[0/5] installing awscli (aws not on PATH)"
    _t0=$SECONDS
    pip install --quiet awscli \
        && log "[0/5] awscli installed ($((SECONDS - _t0))s): $(aws --version 2>&1)" \
        || log "[WARN] awscli install failed; S3 sync steps will fail if needed"
else
    log "[0/5] aws already on PATH: $(aws --version 2>&1)"
fi

# awscli's credential chain only reads AWS_* names, but the pod env carries the
# S3_* names the submit scripts use (and the image has no ~/.aws/credentials).
# Without this mapping every `aws s3` call dies with "Unable to locate
# credentials" -- which killed the first RMBench job AFTER a 33min conversion.
if [ -n "${S3_ACCESS_KEY:-}" ] && [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    export AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY}"
    export AWS_SECRET_ACCESS_KEY="${S3_SECRET_KEY:-}"
fi
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-northeast-1}"

# ---- Step 1: link assets under the repo so starVLA's relative config paths resolve ----
log "[1/5] preparing ${REPO} links from STARVLA_ASSET_ROOT=${STARVLA_ASSET_ROOT}"
_t1=$SECONDS
cd "${REPO}"
mkdir -p playground/Datasets playground
ln -sfn "${STARVLA_ASSET_ROOT}/Pretrained_models"       playground/Pretrained_models
ln -sfn "${STARVLA_ASSET_ROOT}/datasets/rmbench_lerobot" "${CONVERTED_LINK}"
log "[1/5] done ($((SECONDS - _t1))s)"

# Fail fast if the linked base VLM is missing. A dangling symlink here does NOT
# fail loudly later: transformers falls back to treating the local path as a
# HuggingFace repo id and every rank dies with an opaque HFValidationError.
if [ ! -d "${base_vlm}" ] || [ -z "$(ls -A "${base_vlm}" 2>/dev/null)" ]; then
    log "[ERROR] MISSING/EMPTY: base VLM ('${base_vlm}' -> '$(readlink "${base_vlm}" 2>/dev/null || echo 'not a symlink')')"
    log "[HINT] Upload it from the submit host: aws s3 sync playground/Pretrained_models ${S3_ASSET_PREFIX}/Pretrained_models"
    exit 1
fi
log "[1/5] OK: base VLM (${base_vlm})"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/efs/huggingface_models}"
export HF_HUB_OFFLINE=1

# Resolve before expensive data preparation. Explicit resume must fail if the
# source is missing, rather than silently starting training from scratch.
RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    _resume_path=$(python examples/simBenchmarks/RMBench/train_files/resolve_resume_checkpoint.py "${RESUME_CKPT}")
    # Outputs are mirrored and may be deleted after training; keep the source
    # outside that directory, especially when reading a checkpoint from EFS.
    _output_path=$(realpath -m "${run_root_dir}/${run_id}_${_resume_tasks[0]}")
    case "${_resume_path}" in
        "${_output_path}"/*) die "resume source is inside the output run; use a new RUN_NAME" ;;
    esac
    log "resuming from ${_resume_path} (weights + step/LR; optimizer state is not saved)"
    RESUME_ARGS=(--trainer.is_resume true --trainer.pretrained_checkpoint "${_resume_path}")
fi

# ---- Step 2: data conversion (idempotent, rank 0 only) ----
# A task counts as converted when its meta/info.json AND meta/modality.json are
# visible. Converted data normally lives on the asset PVC (pushed back by the
# first job that converted it); on that first job we convert to /local-ssd and
# train from there (fresh, fast, no S3-PVC visibility lag).
task_converted() {
    [ -f "$1/$2/meta/info.json" ] && [ -f "$1/$2/meta/modality.json" ]
}

missing_tasks=()
for task in "${RMBENCH_TASKS[@]}"; do
    task_converted "${CONVERTED_LINK}" "${task}" || missing_tasks+=("${task}")
done

TRAIN_DATA_ROOT="${rmbench_data_root}"

if (( ${#missing_tasks[@]} > 0 )); then
    log "[2/5] converted data incomplete on asset (${#missing_tasks[@]}/${#RMBENCH_TASKS[@]} missing: ${missing_tasks[*]})"
    if [ "${GROUP_RANK}" != "0" ]; then
        # Multi-node: only the master converts; workers poll until it appears.
        log "[2/5] group_rank=${GROUP_RANK}: waiting for master to finish conversion"
        _waited=0
        while (( ${#missing_tasks[@]} > 0 )); do
            sleep 60; _waited=$((_waited + 1))
            (( _waited > 360 )) && die "timed out (6h) waiting for converted RMBench data on the asset PVC"
            missing_tasks=()
            for task in "${RMBENCH_TASKS[@]}"; do
                task_converted "${CONVERTED_LINK}" "${task}" || missing_tasks+=("${task}")
            done
        done
        log "[2/5] converted data is now visible; continuing"
    else
        # Raw data must be on the asset PVC (uploaded submit-side via SYNC_RAW=1).
        for task in "${missing_tasks[@]}"; do
            if [ ! -d "${RAW_ASSET_ROOT}/data/${task}/demo_clean/data" ]; then
                log "[ERROR] raw data missing on asset: ${RAW_ASSET_ROOT}/data/${task}/demo_clean/"
                log "[HINT] From the submit host (this is what SYNC_RAW=1 does):"
                log "[HINT]   aws s3 sync /data/starVLA/RMBench/data ${S3_ASSET_PREFIX}/datasets/rmbench_raw/data --no-follow-symlinks"
                exit 1
            fi
        done
        log "[2/5] converting ${#missing_tasks[@]} task(s) to ${LOCAL_CONVERT_ROOT} (CPU-bound; first job only)"
        _t2=$SECONDS
        mkdir -p "${LOCAL_CONVERT_ROOT}"
        # Per-task loop: resumable at task granularity, and each task is pushed
        # to S3 immediately after conversion -- if a later task (or the push)
        # fails, earlier tasks are already on the asset and the next run's
        # missing-task check skips them, instead of redoing the full 30min+.
        for task in "${missing_tasks[@]}"; do
            if task_converted "${LOCAL_CONVERT_ROOT}" "${task}"; then
                # /local-ssd is a hostPath: a retried pod can land on the same
                # node and find a previous run's finished conversion. Reuse it
                # (the converter refuses to write into an existing dir).
                log "[2/5]   ${task} already converted on /local-ssd (prior interrupted run); reusing"
            else
                # A partial local dir (interrupted mid-task) would trip the
                # converter's FileExistsError; start the task fresh.
                rm -rf "${LOCAL_CONVERT_ROOT:?}/${task}"
                log "[2/5]   converting ${task} ..."
                python examples/simBenchmarks/RMBench/train_files/convert_rmbench_to_lerobot.py \
                    --raw-root "${RAW_ASSET_ROOT}" \
                    --out-root "${LOCAL_CONVERT_ROOT}" \
                    --tasks "${task}"
            fi
            log "[2/5]   pushing ${task} -> ${S3_ASSET_PREFIX}/datasets/rmbench_lerobot/${task}"
            aws s3 sync "${LOCAL_CONVERT_ROOT}/${task}" \
                "${S3_ASSET_PREFIX}/datasets/rmbench_lerobot/${task}"
            # Verify the push landed before moving on; gate on the S3 listing
            # (authoritative), not the S3-backed PVC mount, which can lag.
            aws s3 ls "${S3_ASSET_PREFIX}/datasets/rmbench_lerobot/${task}/meta/info.json" >/dev/null \
                || die "S3 push verification failed for task ${task}"
        done
        log "[2/5] conversion + push done ($((SECONDS - _t2))s)"
        # Train from the fresh local copy this run (asset mount may lag).
        TRAIN_DATA_ROOT="${LOCAL_CONVERT_ROOT}"
    fi
else
    log "[2/5] converted data complete on asset (${#RMBENCH_TASKS[@]}/${#RMBENCH_TASKS[@]} tasks); skipping conversion"
fi

# The asset PVC is READ-ONLY, but the dataloader writes per-dataset caches
# (meta/stats_gr00t.json, meta/steps_data_index.pkl) into the dataset dir on
# first use -- training straight off the symlink dies with PermissionError.
# Materialize onto /local-ssd (the whole dataset is ~0.5GB) so those writes
# succeed; step [5/5] pushes the caches back to the asset for later runs.
#
# Pull from S3 directly, NOT via `cp` from the asset mount: the PVC is an S3
# FUSE mount where every open()/stat() is a round trip, and the dataset is
# ~2460 small files (1800 mp4) -- a serial cp takes 10-30min and, worse, logs
# nothing, so the job looks hung. `aws s3 sync` downloads in parallel and is
# seconds-to-minutes for this size.
if [ "${TRAIN_DATA_ROOT}" != "${LOCAL_CONVERT_ROOT}" ]; then
    _t2b=$SECONDS
    log "[2/5] materializing dataset to ${LOCAL_CONVERT_ROOT} (asset PVC is read-only)"
    mkdir -p "${LOCAL_CONVERT_ROOT}"
    for task in "${RMBENCH_TASKS[@]}"; do
        if task_converted "${LOCAL_CONVERT_ROOT}" "${task}"; then
            log "[2/5]   ${task} already on /local-ssd; reusing"
            continue
        fi
        _tt=$SECONDS
        rm -rf "${LOCAL_CONVERT_ROOT:?}/${task}"
        if aws s3 sync "${S3_ASSET_PREFIX}/datasets/rmbench_lerobot/${task}" \
                "${LOCAL_CONVERT_ROOT}/${task}" >/dev/null; then
            log "[2/5]   ${task} pulled from S3 ($((SECONDS - _tt))s)"
        else
            # Fall back to the FUSE mount (slow but works without S3 creds).
            log "[WARN] S3 pull failed for ${task}; falling back to cp from asset mount (slow)"
            cp -r "${CONVERTED_LINK}/${task}" "${LOCAL_CONVERT_ROOT}/${task}"
        fi
        task_converted "${LOCAL_CONVERT_ROOT}" "${task}" \
            || die "materialized copy of ${task} is incomplete (missing meta/info.json or meta/modality.json)"
    done
    TRAIN_DATA_ROOT="${LOCAL_CONVERT_ROOT}"
    log "[2/5] local copy ready ($((SECONDS - _t2b))s)"
fi

# ---- Step 3: GPU check ----
log "[3/5] GPU status before training"
nvidia-smi || true

# Per-task S3 checkpoint mirror: runs only while its task trains, so a crash
# mid-sweep loses at most CKPT_MIRROR_INTERVAL seconds of the CURRENT task
# (earlier tasks are already fully synced).
MIRROR_PID=""
start_mirror() {  # $1=local ckpt dir  $2=s3 target
    [ -n "$2" ] || return 0
    log "  mirroring $1 -> $2 every ${CKPT_MIRROR_INTERVAL}s"
    (
        while true; do
            aws s3 sync "$1" "$2" --exclude 'wandb/*' >/dev/null 2>&1 || true
            sleep "${CKPT_MIRROR_INTERVAL}"
        done
    ) &
    MIRROR_PID=$!
}
stop_mirror() {
    if [ -n "${MIRROR_PID}" ] && kill -0 "${MIRROR_PID}" 2>/dev/null; then
        kill "${MIRROR_PID}" 2>/dev/null || true
        wait "${MIRROR_PID}" 2>/dev/null || true
    fi
    MIRROR_PID=""
}

# ---- Step 4: per-task training loop (RMBench protocol: one policy per task) ----
read -r -a _train_tasks <<< "${TRAIN_TASKS}"
log "[4/5] per-task sweep: ${_train_tasks[*]} (${#_train_tasks[@]} tasks, nproc=${NPROC_PER_NODE}, data_root=${TRAIN_DATA_ROOT})"
FAILED_TASKS=()
_task_idx=0
for task in "${_train_tasks[@]}"; do
    _task_idx=$((_task_idx + 1))
    TASK_RUN_ID="${run_id}_${task}"
    TASK_CKPT_DIR="${run_root_dir}/${TASK_RUN_ID}"
    TASK_S3_DIR="${S3_CHECKPOINT_DIR:+${S3_CHECKPOINT_DIR}_${task}}"
    mkdir -p "${TASK_CKPT_DIR}"
    log "[4/5] (${_task_idx}/${#_train_tasks[@]}) task=${task} data_mix=rmbench_${task} run_id=${TASK_RUN_ID}"
    _t4=$SECONDS
    start_mirror "${TASK_CKPT_DIR}" "${TASK_S3_DIR}"
    _task_ok=0
    rmbench_data_root="${TRAIN_DATA_ROOT}" \
    NUM_PROCESSES="${NPROC_PER_NODE}" \
    config_yaml="${config_yaml}" \
    bash examples/simBenchmarks/RMBench/train_files/run_rmbench_train.sh \
        --datasets.vla_data.data_mix "rmbench_${task}" \
        --datasets.vla_data.per_device_batch_size "${per_device_bs}" \
        --framework.qwenvl.base_vlm "${base_vlm}" \
        --trainer.freeze_modules "${freeze_module_list}" \
        --trainer.num_train_epochs "${num_train_epochs}" \
        --trainer.save_interval "${save_interval}" \
        --trainer.logging_frequency "${logging_frequency}" \
        --trainer.eval_interval "${eval_interval}" \
        --run_root_dir "${run_root_dir}" \
        --run_id "${TASK_RUN_ID}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}" \
        "${RESUME_ARGS[@]}" \
        && _task_ok=1
    # Final sync for this task before moving on (best-effort even on failure:
    # salvages whatever checkpoints exist).
    _final_sync_ok=0
    if [ -n "${TASK_S3_DIR}" ]; then
        if aws s3 sync "${TASK_CKPT_DIR}" "${TASK_S3_DIR}" --exclude 'wandb/*'; then
            _final_sync_ok=1
        else
            log "[WARN] final sync failed for ${task}; keeping EFS copy at ${TASK_CKPT_DIR}"
        fi
    fi
    stop_mirror
    # EFS is small and shared; once the task's checkpoints are safely on S3 the
    # local copy is redundant, so drop it. (wandb/ local logs are excluded from
    # the sync, but wandb runs online so they are already in the wandb cloud.)
    if [ "${_final_sync_ok}" = "1" ] && [ "${DELETE_EFS_CKPT_AFTER_SYNC}" = "1" ]; then
        rm -rf "${TASK_CKPT_DIR:?}"
        log "[4/5]   EFS copy removed: ${TASK_CKPT_DIR} (safe at ${TASK_S3_DIR})"
    fi
    if [ "${_task_ok}" = "1" ]; then
        log "[4/5] (${_task_idx}/${#_train_tasks[@]}) ${task} FINISHED ($((SECONDS - _t4))s)"
    else
        log "[ERROR] (${_task_idx}/${#_train_tasks[@]}) ${task} FAILED ($((SECONDS - _t4))s); continuing with next task"
        FAILED_TASKS+=("${task}")
    fi
done

# ---- Step 5: final sync passes ----
# Push the derived dataset caches (stats_gr00t.json / steps_data_index.pkl are
# computed by rank 0 on first use) back to the asset so later jobs skip the
# minutes-long stats pass over 12 datasets. Only meta caches, never the data.
if [ "${GROUP_RANK}" = "0" ]; then
    aws s3 sync "${TRAIN_DATA_ROOT}" "${S3_ASSET_PREFIX}/datasets/rmbench_lerobot" \
        --exclude '*' \
        --include '*/meta/stats_gr00t.json' \
        --include '*/meta/steps_data_index.pkl' \
        && log "[5/5] dataset meta caches pushed back to asset" \
        || log "[WARN] meta cache push failed (non-fatal)"
fi

if [ -n "${S3_CHECKPOINT_DIR}" ] && [ "${DELETE_EFS_CKPT_AFTER_SYNC}" = "1" ]; then
    log "checkpoint locations (S3; EFS copies removed after sync): ${S3_CHECKPOINT_DIR}_<task>"
else
    log "checkpoint locations (EFS): ${run_root_dir}/${run_id}_<task>"
fi
log "PULL BACK LOCALLY (example: first trained task '${_train_tasks[0]}'):"
log "  aws s3 sync ${S3_CHECKPOINT_DIR:-<s3>}_${_train_tasks[0]} ./playground/Checkpoints/${run_id}_${_train_tasks[0]}"

if (( ${#FAILED_TASKS[@]} > 0 )); then
    log "[ERROR] sweep done with ${#FAILED_TASKS[@]}/${#_train_tasks[@]} failed task(s): ${FAILED_TASKS[*]}"
    exit 1
fi
log "[5/5] sweep done: all ${#_train_tasks[@]} task(s) trained"

cleanup
