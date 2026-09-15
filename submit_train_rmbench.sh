#!/usr/bin/env bash
# Submit RMBench (QwenPI_v3) training to KOALA as a Kubeflow PyTorchJob.
#
# Pipeline:
#   BUILD_IMAGE=1  -> ECR login + build/push the :rmbench image (h5py added in
#                     requirements.txt)
#   SYNC_RAW=1     -> upload the raw 37GB HDF5 data to the S3 asset bucket
#                     （已提交）(注：仅第一次运行需要提交，之后再提交会累积)
#   (in pod)       -> run_rmbench_train_in_pod.sh converts raw -> LeRobot on
#                     /local-ssd if the converted asset is missing, pushes the
#                     result back to S3, then trains
#  # 本地目录：提交时自动上传到 S3 asset
#   TRAIN_TASK=battery_try RESUME_CKPT=/data/old_run bash submit_train_rmbench.sh

#   # S3：pod 内下载
#   TRAIN_TASK=battery_try RESUME_CKPT=s3://bucket/path/old_run bash submit_train_rmbench.sh

#   # EFS：直接读取
#   TRAIN_TASK=battery_try RESUME_CKPT=/efs/user/old_run bash submit_train_rmbench.sh

set -Eeuo pipefail

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    echo "[$(date '+%F %T')] [ERROR] $*" >&2
    exit 1
}

trap 'rc=$?; echo "[$(date "+%F %T")] [ERROR] submit failed at line ${LINENO} with exit ${rc}" >&2' ERR

# ---- Optional local env file (gitignored; keeps secrets out of this script) ----
if [ -f .env.submit ]; then
    log "sourcing .env.submit"
    set -a; # shellcheck disable=SC1091
    . ./.env.submit; set +a
fi

# ---- KOALA / identity ----
KOALA_CLUSTER=${KOALA_CLUSTER:-tenc-aws-yfxn-northeast}
NAMESPACE=${NAMESPACE:-helix}
KOALA_TOKEN=${KOALA_TOKEN:-}
SUBMIT_USER=${SUBMIT_USER:-danyangchen}

# ---- Storage / experiment owners ----
CODE_BUCKET_NAME=${CODE_BUCKET_NAME:-helix-code-ap-northeast-1}
ASSET_BUCKET_NAME=${ASSET_BUCKET_NAME:-helix-asset-ap-northeast-1}
EFS_PVC_NAME=${EFS_PVC_NAME:-helix-efs}
ASSET_USER=${ASSET_USER:-danyangchen}

S3_ACCESS_KEY=${S3_ACCESS_KEY:-}
S3_SECRET_KEY=${S3_SECRET_KEY:-}
WANDB_ENTITY=${WANDB_ENTITY:-"estella-chen-tsinghua-university"}
WANDB_API_KEY=${WANDB_API_KEY:-}

# ---- Job shape ----
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
CUSTOM_JOB_NAME=${CUSTOM_JOB_NAME:-train}
JOB_NAME="${CUSTOM_JOB_NAME}-$(date +%Y%m%d%H%M)"
RUN_NAME=${RUN_NAME:-"${CUSTOM_JOB_NAME}-${TIMESTAMP}"}
# k8s object names must be lowercase RFC 1123. A capital letter is rejected by
# the API server AFTER the code+asset sync has already run, wasting minutes.
# Lowercase it here and say so, rather than failing several steps later.
_job_name_lc=$(printf '%s' "${JOB_NAME}" | tr '[:upper:]' '[:lower:]')
if [ "${_job_name_lc}" != "${JOB_NAME}" ]; then
    log "[WARN] CUSTOM_JOB_NAME contained uppercase; k8s requires lowercase RFC 1123."
    log "[WARN] job name '${JOB_NAME}' -> '${_job_name_lc}'"
    JOB_NAME="${_job_name_lc}"
fi
if ! printf '%s' "${JOB_NAME}" | grep -Eq '^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$'; then
    die "job name '${JOB_NAME}' is not a valid RFC 1123 subdomain (allowed: a-z 0-9 '-' '.', must start/end alphanumeric). Fix CUSTOM_JOB_NAME."
fi
# Distinct tag so rebuilding for RMBench can never break in-flight LIBERO jobs
# that pull :latest (imagePullPolicy: Always).
IMAGE=${IMAGE:-600627331169.dkr.ecr.ap-northeast-1.amazonaws.com/danyangchen/starvla:rmbench}

# ---- Compute: one node, configurable GPU count ----
NODE_NUM=${NODE_NUM:-1}
WORKER_NODE_NUM=$((NODE_NUM - 1))
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
GPU_LIMIT=${GPU_LIMIT:-${NPROC_PER_NODE}}
# EFA is only needed for multi-node jobs (cross-node RDMA). Single-node jobs
# use NVLink/PCIe and don't need EFA — requesting it blocks scheduling when
# EFA slots are exhausted even though GPUs are free.
if [ "${NODE_NUM}" -gt 1 ]; then
    EFA_LIMIT=${EFA_LIMIT:-$((GPU_LIMIT * 2))}
else
    EFA_LIMIT=${EFA_LIMIT:-0}
fi
# Scale CPU/memory proportionally; override via env if needed.
_cpu_default=$((NPROC_PER_NODE * 23))
_mem_default=$((NPROC_PER_NODE * 225))
CPU_LIMIT=${CPU_LIMIT:-${_cpu_default}}
MEMORY_LIMIT=${MEMORY_LIMIT:-"${_mem_default}Gi"}

# ---- Code path (timestamped snapshot, isolated per submit) ----
AWS_S3_RUN_CODE_SYNC_DIR="s3://${CODE_BUCKET_NAME}/${SUBMIT_USER}/starvla-rmbench-train/${TIMESTAMP}"
AWS_S3_RUN_CODE_DIR="/threed-code/${SUBMIT_USER}/starvla-rmbench-train/${TIMESTAMP}"
INIT_CMD="set -euo pipefail; cp -r ${AWS_S3_RUN_CODE_DIR} /data/work/starvla; chmod -R 755 /data/work/starvla"

# ---- Asset layout: ckpt + data live in the asset bucket/PVC ----
STARVLA_ASSET_ROOT=${STARVLA_ASSET_ROOT:-/asset/${ASSET_USER}/starVLA}
S3_ASSET_PREFIX="s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA"

# Raw RMBench data on the submit host (HF TianxingChen/RMBench, demo_clean).
RMBENCH_RAW_LOCAL=${RMBENCH_RAW_LOCAL:-/data/starVLA/RMBench}

# ---- Training defaults (confirmed config: QwenPI_v3 + rmbench_all) ----
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/simBenchmarks/RMBench/train_files/starvla_qwenpiv3_rmbench.yaml}
rmbench_data_root=${rmbench_data_root:-playground/Datasets/rmbench_lerobot}
data_mix=${data_mix:-rmbench_all}
# Per RMBench protocol, one policy is trained per task. The pod loops this
# list sequentially (own run_id/ckpt per task). Default: the 9-task sweep
# (excludes classify_blocks / storage_blocks / place_block_mat).
# TRAIN_TASK (singular) is accepted as an alias. Examples:
#   TRAIN_TASK=battery_try CUSTOM_JOB_NAME=train-task1 bash submit_train_rmbench.sh
#   TRAIN_TASKS="battery_try cover_blocks" bash submit_train_rmbench.sh
TRAIN_TASKS=${TRAIN_TASKS:-${TRAIN_TASK:-"battery_try blocks_ranking_try observe_and_pickup cover_blocks press_button put_back_block rearrange_blocks swap_blocks swap_T"}}
# Fail fast on typos at submit time (free) instead of in-pod after minutes of
# setup. Valid names = the 12 converted tasks (data_registry/RMBENCH_TASKS).
_valid_tasks="battery_try blocks_ranking_try classify_blocks  observe_and_pickup cover_blocks place_block_mat press_button put_back_block rearrange_blocks storage_blocks swap_blocks swap_T"
for _t in ${TRAIN_TASKS}; do
    case " ${_valid_tasks} " in
        *" ${_t} "*) ;;
        *) die "unknown task '${_t}' in TRAIN_TASKS. Valid: ${_valid_tasks}" ;;
    esac
done
# Full fine-tune of the 4B VLM + WM (7 images/sample) OOMs at bs~60-100 on
# 2 GPUs. 16 is a safe default; raise cautiously, or use
# gradient_accumulation_steps for a bigger global batch.
per_device_bs=${per_device_bs:-26}
# Which submodules to freeze. Default '' = full fine-tune, matching the yaml
# (freeze_modules: "") and the repo's Robotwin example: RMBench is a new
# dual-arm Agilex embodiment on a raw Qwen3-VL backbone, so the VLM must adapt
# too.
freeze_module_list=${freeze_module_list-}
# Training length is epoch-based: the trainer derives max_train_steps from the
# real dataset size and the effective global batch size, so the budget is
# unchanged if per_device_bs / GPU count / data_mix change.
num_train_epochs=${num_train_epochs:-500}
save_interval=${save_interval:-3000}
logging_frequency=${logging_frequency:-20}
# NOTE: eval_action_model() consumes a training batch each time it fires, so
# keep this interval coarse.
eval_interval=${eval_interval:-2000}
wandb_project=${wandb_project:-starVLA_rmbench}
# Checkpoint output on the EFS PVC (persists after pod termination).
run_root_dir=${run_root_dir:-/efs/${ASSET_USER}/exp/starvla}
run_id=${RUN_NAME}
# Background S3 mirror interval (seconds).
CKPT_MIRROR_INTERVAL=${CKPT_MIRROR_INTERVAL:-1000}
# S3 prefix checkpoints are mirrored to.
S3_CHECKPOINT_DIR="s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA/exp/${RUN_NAME}"

# Resume one task from a run directory, checkpoints directory, or steps_N file.
# Local paths upload to the asset bucket; s3:// paths download inside the pod;
# /efs/... paths are read directly. num_train_epochs remains the TOTAL budget.
# Example: TRAIN_TASK=battery_try RESUME_CKPT=/efs/user/exp/old_battery_try bash submit_train_rmbench.sh
RESUME_CKPT=${RESUME_CKPT:-}
RESUME_CKPT_LOCAL=""
if [ -n "${RESUME_CKPT}" ]; then
    read -r -a _resume_tasks <<< "${TRAIN_TASKS}"
    [ "${#_resume_tasks[@]}" = "1" ] || die "RESUME_CKPT requires exactly one TRAIN_TASK (got ${#_resume_tasks[@]} tasks)"
    case "${RESUME_CKPT}" in
        s3://*|/efs/*) ;;
        *)
            [ -e "${RESUME_CKPT}" ] || die "local RESUME_CKPT does not exist: ${RESUME_CKPT}"
            python3 "$(dirname "${BASH_SOURCE[0]}")/examples/simBenchmarks/RMBench/train_files/resolve_resume_checkpoint.py" \
                "${RESUME_CKPT}" >/dev/null
            RESUME_CKPT_LOCAL="${RESUME_CKPT}"
            RESUME_CKPT="${S3_ASSET_PREFIX}/resume/${RUN_NAME}/${TIMESTAMP}"
            if [ -f "${RESUME_CKPT_LOCAL}" ]; then
                RESUME_CKPT="${RESUME_CKPT}/$(basename "${RESUME_CKPT_LOCAL}")"
            fi
            ;;
    esac
fi

# ---- Optional: build + push the image (first setup / after dep changes) ----
BUILD_IMAGE=${BUILD_IMAGE:-0}
if [ "${BUILD_IMAGE}" = "1" ]; then
    log "BUILD_IMAGE=1: ECR login + building ${IMAGE}"
    aws ecr get-login-password --region ap-northeast-1 \
        | docker login --username AWS --password-stdin 600627331169.dkr.ecr.ap-northeast-1.amazonaws.com
    PUSH=1 IMAGE="${IMAGE}" bash deployment/docker/build.sh
    log "image built and pushed: ${IMAGE}"
fi

# In-pod logic (asset links, conversion-if-missing + S3 push-back, ckpt mirror,
# training) lives in the repo so it ships with the code snapshot and is
# testable outside the submit path.
START_CMD='set -euo pipefail; bash /data/work/starvla/examples/simBenchmarks/RMBench/train_files/run_rmbench_train_in_pod.sh'

DRY_RUN=${DRY_RUN:-0}
SYNC_CODE=${SYNC_CODE:-1}
SYNC_ASSET=${SYNC_ASSET:-1}
SYNC_RAW=${SYNC_RAW:-1}

log "job name:      ${JOB_NAME}"
log "run name:      ${RUN_NAME}"
log "image:         ${IMAGE}"
log "nproc/GPUs:    ${NPROC_PER_NODE}"
log "efa limit:     ${EFA_LIMIT}"
log "cpu limit:     ${CPU_LIMIT}"
log "memory limit:  ${MEMORY_LIMIT}"
log "data mix:      ${data_mix}"
log "train tasks:   ${TRAIN_TASKS}"
log "base vlm:      ${base_vlm}"
log "config yaml:   ${config_yaml}"
log "train epochs:  ${num_train_epochs} (max_train_steps derived at runtime)"
log "ckpt save dir: ${run_root_dir}/${run_id}"
log "s3 mirror:     ${S3_CHECKPOINT_DIR}"
log "resume source: ${RESUME_CKPT:-none}"

# Resume uploads are independent of SYNC_ASSET (which controls base models).
if [ -n "${RESUME_CKPT_LOCAL}" ]; then
    log "uploading resume checkpoint: ${RESUME_CKPT_LOCAL} -> ${RESUME_CKPT}"
    if [ -d "${RESUME_CKPT_LOCAL}" ]; then
        aws s3 sync "${RESUME_CKPT_LOCAL}" "${RESUME_CKPT}" --exclude 'wandb/*'
    else
        aws s3 cp "${RESUME_CKPT_LOCAL}" "${RESUME_CKPT}"
    fi
fi

# ---- Upload base assets to the S3 asset bucket (skip with SYNC_ASSET=0) ----
# The converted RMBench dataset is NOT synced here: it is produced in-pod on
# the first run and pushed back to S3 by run_rmbench_train_in_pod.sh.
if [ "${SYNC_ASSET}" = "1" ]; then
    log "syncing assets to S3: ${S3_ASSET_PREFIX}"
    aws s3 sync playground/Pretrained_models \
        "${S3_ASSET_PREFIX}/Pretrained_models" \
        --no-follow-symlinks
    log "asset sync done"
else
    log "SYNC_ASSET=0, skipping asset sync"
fi

# ---- Upload raw RMBench data (first submit only; skip with SYNC_RAW=0) ----
# Only the data/ subtree is synced, which naturally excludes .cache/huggingface.
if [ "${SYNC_RAW}" = "1" ]; then
    [ -d "${RMBENCH_RAW_LOCAL}/data" ] || die "raw RMBench data not found at ${RMBENCH_RAW_LOCAL}/data (set RMBENCH_RAW_LOCAL)"
    log "syncing raw RMBench data to ${S3_ASSET_PREFIX}/datasets/rmbench_raw/data (37GB first time; incremental afterwards)"
    aws s3 sync "${RMBENCH_RAW_LOCAL}/data" \
        "${S3_ASSET_PREFIX}/datasets/rmbench_raw/data" \
        --no-follow-symlinks
    log "raw data sync done"
else
    log "SYNC_RAW=0, skipping raw data sync"
fi

# ---- Sync code to the S3 code bucket (skip with SYNC_CODE=0) ----
if [ "${SYNC_CODE}" = "1" ]; then
    log "syncing code to S3: ${AWS_S3_RUN_CODE_SYNC_DIR}"
    aws s3 sync . "${AWS_S3_RUN_CODE_SYNC_DIR}" \
        --no-follow-symlinks \
        --exclude '.git/*' \
        --exclude '.claude/*' \
        --exclude '.github/*' \
        --exclude '__pycache__/*' \
        --exclude '*.pyc' \
        --exclude '.pytest_cache/*' \
        --exclude 'runs/*' \
        --exclude 'evaluate_results/*' \
        --exclude 'results/*' \
        --exclude 'playground' \
        --exclude 'playground/*' \
        --exclude '*.egg-info/*' \
        --exclude '.dockerfile' \
        --exclude '.env.submit' \
        --exclude 'docs/*'
    log "code sync done"
else
    log "SYNC_CODE=0, skipping code sync"
fi

container_template=$(jq -n \
    --arg start_cmd        "$START_CMD" \
    --arg init_cmd         "$INIT_CMD" \
    --arg image            "$IMAGE" \
    --arg code_bucket      "$CODE_BUCKET_NAME" \
    --arg asset_bucket     "$ASSET_BUCKET_NAME" \
    --arg efs_pvc          "$EFS_PVC_NAME" \
    --arg s3_access_key    "$S3_ACCESS_KEY" \
    --arg s3_secret_key    "$S3_SECRET_KEY" \
    --arg wandb_api_key    "$WANDB_API_KEY" \
    --arg wandb_entity     "$WANDB_ENTITY" \
    --arg wandb_project    "$wandb_project" \
    --arg starvla_asset    "$STARVLA_ASSET_ROOT" \
    --arg s3_asset_prefix  "$S3_ASSET_PREFIX" \
    --arg base_vlm         "${base_vlm}" \
    --arg config_yaml      "${config_yaml}" \
    --arg rmbench_data_root "${rmbench_data_root}" \
    --arg data_mix         "${data_mix}" \
    --arg train_tasks      "${TRAIN_TASKS}" \
    --arg per_device_bs    "${per_device_bs}" \
    --arg freeze_module_list "${freeze_module_list}" \
    --arg num_train_epochs "${num_train_epochs}" \
    --arg save_interval    "${save_interval}" \
    --arg logging_frequency "${logging_frequency}" \
    --arg eval_interval    "${eval_interval}" \
    --arg run_root_dir     "${run_root_dir}" \
    --arg run_id           "${run_id}" \
    --arg resume_ckpt      "${RESUME_CKPT}" \
    --arg s3_ckpt_dir      "${S3_CHECKPOINT_DIR}" \
    --arg mirror_interval  "${CKPT_MIRROR_INTERVAL}" \
    --argjson nproc        "$NPROC_PER_NODE" \
    --argjson gpu_limit    "$GPU_LIMIT" \
    --argjson efa_limit    "$EFA_LIMIT" \
    --arg cpu_limit        "$CPU_LIMIT" \
    --arg memory_limit     "$MEMORY_LIMIT" \
    '{
        metadata: {
            labels: {
                "job":"rmbench"
            }
        },
        spec: {
            schedulerName: "custom-k8s-scheduler",
            containers: [
                {
                    args: [$start_cmd],
                    command: ["/bin/bash", "-c"],
                    env: [
                        {
                            name: "JOB_NAME",
                            valueFrom: {fieldRef: {fieldPath: "metadata.labels['\''training.kubeflow.org/job-name'\'']"}}
                        },
                        {name: "PLATFORM",              value: "aws"},
                        {name: "HF_HOME",               value: "/efs/huggingface_models"},
                        {name: "HF_HUB_OFFLINE",        value: "1"},
                        {name: "PYTHONUNBUFFERED",       value: "1"},
                        {name: "TOKENIZERS_PARALLELISM", value: "false"},
                        {name: "S3_ACCESS_KEY",          value: $s3_access_key},
                        {name: "S3_SECRET_KEY",          value: $s3_secret_key},
                        {name: "WANDB_API_KEY",          value: $wandb_api_key},
                        {name: "WANDB_PROJECT",          value: $wandb_project},
                        {name: "WANDB_ENTITY",           value: $wandb_entity},
                        {name: "WANDB_WORKSPACE",        value: $wandb_entity},
                        {name: "STARVLA_ASSET_ROOT",     value: $starvla_asset},
                        {name: "S3_ASSET_PREFIX",        value: $s3_asset_prefix},
                        {name: "base_vlm",               value: $base_vlm},
                        {name: "config_yaml",            value: $config_yaml},
                        {name: "rmbench_data_root",      value: $rmbench_data_root},
                        {name: "data_mix",               value: $data_mix},
                        {name: "TRAIN_TASKS",            value: $train_tasks},
                        {name: "per_device_bs",          value: $per_device_bs},
                        {name: "freeze_module_list",     value: $freeze_module_list},
                        {name: "num_train_epochs",       value: $num_train_epochs},
                        {name: "save_interval",          value: $save_interval},
                        {name: "logging_frequency",      value: $logging_frequency},
                        {name: "eval_interval",          value: $eval_interval},
                        {name: "run_root_dir",           value: $run_root_dir},
                        {name: "run_id",                 value: $run_id},
                        {name: "RESUME_CKPT",            value: $resume_ckpt},
                        {name: "S3_CHECKPOINT_DIR",      value: $s3_ckpt_dir},
                        {name: "CKPT_MIRROR_INTERVAL",   value: $mirror_interval},
                        {name: "NPROC_PER_NODE",         value: ($nproc | tostring)},
                        {name: "NCCL_DEBUG",             value: "WARN"},
                        {name: "NCCL_TIMEOUT",           value: "1800"},
                        {name: "NCCL_SOCKET_IFNAME",     value: "eth0"},
                        {name: "NCCL_IB_GID_INDEX",      value: "3"},
                        {name: "NCCL_IB_SL",             value: "3"},
                        {name: "NCCL_P2P_DISABLE",       value: "0"},
                        {name: "NCCL_IB_DISABLE",        value: "0"},
                        {name: "NCCL_LL_THRESHOLD",      value: "16384"},
                        {name: "NCCL_IB_CUDA_SUPPORT",   value: "1"},
                        {name: "NCCL_CHECK_DISABLE",     value: "1"},
                        {name: "NCCL_IB_HCA",            value: "mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"},
                        {name: "NCCL_COLLNET_ENABLE",    value: "0"},
                        {name: "SHARP_COLL_ENABLE_SAT",  value: "0"},
                        {name: "NCCL_NET_GDR_LEVEL",     value: "2"},
                        {name: "NCCL_IB_QPS_PER_CONNECTION", value: "4"},
                        {name: "NCCL_IB_TC",             value: "160"},
                        {name: "NCCL_PXN_DISABLE",       value: "1"},
                        {name: "UCX_NET_DEVICES",        value: "eth0"},
                        {name: "FI_PROVIDER",            value: "efa"},
                        {name: "OFI_NCCL_PROTOCOL",      value: "RDMA"},
                        {name: "FI_EFA_USE_DEVICE_RDMA", value: "1"},
                        {name: "NCCL_TUNER_PLUGIN",      value: "/opt/amazon/efa/lib/libnccl-ofi-tuner.so"}
                    ],
                    image: $image,
                    imagePullPolicy: "Always",
                    name: "pytorch",
                    resources: {
                        limits: ({
                            cpu: ($cpu_limit | tonumber),
                            memory: $memory_limit,
                            "nvidia.com/gpu": $gpu_limit,
                            "vpc.amazonaws.com/efa": $efa_limit
                        } | if $efa_limit == 0 then del(.["vpc.amazonaws.com/efa"]) else . end),
                        requests: ({
                            cpu: ($cpu_limit | tonumber),
                            memory: $memory_limit,
                            "nvidia.com/gpu": $gpu_limit,
                            "vpc.amazonaws.com/efa": $efa_limit
                        } | if $efa_limit == 0 then del(.["vpc.amazonaws.com/efa"]) else . end)
                    },
                    volumeMounts: [
                        {mountPath: "/threed-code", name: "threed-code"},
                        {mountPath: "/asset",       name: "asset"},
                        {mountPath: "/efs",         name: "efs"},
                        {mountPath: "/dev/shm",     name: "dshm"},
                        {mountPath: "/data/work",   name: "initdir"},
                        {mountPath: "/local-ssd",   name: "local-ssd"}
                    ]
                }
            ],
            initContainers: [
                {
                    command: ["/bin/bash", "-c", $init_cmd],
                    image: $image,
                    name: "init-image",
                    resources: {limits: {cpu: 1, memory: "2Gi"}},
                    volumeMounts: [
                        {mountPath: "/threed-code", name: "threed-code"},
                        {mountPath: "/data/work",   name: "initdir"}
                    ]
                }
            ],
            volumes: [
                {name: "threed-code", persistentVolumeClaim: {claimName: $code_bucket}},
                {name: "asset",       persistentVolumeClaim: {claimName: $asset_bucket}},
                {name: "efs",         persistentVolumeClaim: {claimName: $efs_pvc}},
                {name: "dshm",        emptyDir: {medium: "Memory", sizeLimit: "1000Gi"}},
                {name: "initdir",     emptyDir: {}},
                {name: "local-ssd",   hostPath: {path: "/opt/dlami/nvme", type: "DirectoryOrCreate"}}
            ]
        }
    }')

full_json=$(jq -n \
    --argjson template_info  "$container_template" \
    --arg     job_name        "$JOB_NAME" \
    --arg     namespace       "$NAMESPACE" \
    --argjson worker_num      "$WORKER_NODE_NUM" \
    '{
        object: {
            apiVersion: "kubeflow.org/v1",
            kind: "PyTorchJob",
            metadata: {name: $job_name, namespace: $namespace},
            spec: {
                runPolicy: {
                    ttlSecondsAfterFinished: 259200,
                    backoffLimit: 1,
                    cleanPodPolicy: "None"
                },
                pytorchReplicaSpecs: (
                    {Master: {replicas: 1, restartPolicy: "Never", template: $template_info}}
                    + if $worker_num > 0 then
                        {Worker: {replicas: $worker_num, restartPolicy: "Never", template: $template_info}}
                      else {} end
                )
            }
        }
    }')

log "final full_json:"
echo "$full_json" | jq '.'

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY_RUN=1, not submitting KOALA job"
    exit 0
fi

log "submitting PyTorchJob to KOALA"
# Do NOT use `curl -f` here: it suppresses the response body, which is exactly
# where KOALA explains *why* a submit was rejected (expired token, wrong
# namespace/cluster, quota, ...). Capture the body and surface it on failure.
_koala_response=$(mktemp)
_http_code=$(curl -sS -o "${_koala_response}" -w '%{http_code}' \
    -XPOST "https://api.k.deltaverse-intl.com/projects/${NAMESPACE}/serving-pytorchjobs?clusterId=${KOALA_CLUSTER}&namespace=${NAMESPACE}" \
    --header "Authorization: Bearer ${KOALA_TOKEN}" \
    --header 'Content-Type: application/json' \
    -d "$full_json") || true

if [ "${_http_code}" -ge 400 ] 2>/dev/null || [ -z "${_http_code}" ]; then
    log "[ERROR] KOALA rejected the submit (HTTP ${_http_code:-no-response})"
    log "[ERROR] response body:"
    cat "${_koala_response}" >&2 || true
    echo >&2
    case "${_http_code}" in
        401|403) log "[HINT] auth/permission problem: KOALA_TOKEN expired or invalid, or it lacks access to namespace='${NAMESPACE}' / cluster='${KOALA_CLUSTER}'. Refresh the token from the KOALA UI and re-export it." ;;
        404)     log "[HINT] endpoint/namespace '${NAMESPACE}' or cluster '${KOALA_CLUSTER}' not found." ;;
        409)     log "[HINT] a job named '${JOB_NAME}' already exists; set CUSTOM_JOB_NAME to something new." ;;
    esac
    rm -f "${_koala_response}"
    exit 1
fi

# KOALA returns HTTP 200 even when the k8s API server REJECTS the object: the
# failure arrives as a JSON payload ({"code": 10001, ...}). Gating on the
# status code alone therefore logged "submit accepted" for a job that was never
# created -- a non-run that reports as a run. Check the body.
_koala_code=$(jq -r '.code // empty' "${_koala_response}" 2>/dev/null || true)
if [ -n "${_koala_code}" ] && [ "${_koala_code}" != "0" ]; then
    log "[ERROR] KOALA returned HTTP ${_http_code} but the job was NOT created (code=${_koala_code})"
    log "[ERROR] response body:"
    jq -r '.message // .' "${_koala_response}" >&2 2>/dev/null || cat "${_koala_response}" >&2 || true
    echo >&2
    case "${_koala_code}" in
        10001) log "[HINT] the k8s API server rejected the object. Most often an invalid metadata.name (must be lowercase RFC 1123) or a malformed resource spec." ;;
    esac
    rm -f "${_koala_response}"
    exit 1
fi

log "submit accepted (HTTP ${_http_code})"
cat "${_koala_response}" || true
echo
rm -f "${_koala_response}"

log "submit request sent"
log "checkpoints mirror to ${S3_CHECKPOINT_DIR}_<task> on S3; each task's EFS copy under ${run_root_dir}/${run_id}_<task> is removed once its final sync succeeds (DELETE_EFS_CKPT_AFTER_SYNC=0 to keep)."
log "to pull checkpoints locally:"
log "  aws s3 sync ${S3_CHECKPOINT_DIR} ./playground/Checkpoints/${RUN_NAME}"
