#!/usr/bin/env bash
set -Eeuo pipefail

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    echo "[$(date '+%F %T')] [ERROR] $*" >&2
    exit 1
}

trap 'rc=$?; echo "[$(date "+%F %T")] [ERROR] submit failed at line ${LINENO} with exit ${rc}" >&2' ERR

# ---- KOALA / identity ----
KOALA_CLUSTER=${KOALA_CLUSTER:-tenc-aws-yfxn-northeast}
NAMESPACE=${NAMESPACE:-helix}
KOALA_TOKEN=${KOALA_TOKEN:?"KOALA_TOKEN not set"}
SUBMIT_USER=${SUBMIT_USER:-danyangchen}

# ---- Storage / experiment owners ----
CODE_BUCKET_NAME=${CODE_BUCKET_NAME:-helix-code-ap-northeast-1}
ASSET_BUCKET_NAME=${ASSET_BUCKET_NAME:-helix-asset-ap-northeast-1}
EFS_PVC_NAME=${EFS_PVC_NAME:-helix-efs}
ASSET_USER=${ASSET_USER:-danyangchen}

S3_ACCESS_KEY=${S3_ACCESS_KEY:-""}
S3_SECRET_KEY=${S3_SECRET_KEY:-""}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_API_KEY=${WANDB_API_KEY:-""}

# ---- Job shape ----
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
CUSTOM_JOB_NAME=${CUSTOM_JOB_NAME:-vla-train}
JOB_NAME="${CUSTOM_JOB_NAME}-$(date +%Y%m%d%H%M)"
RUN_NAME=${RUN_NAME:-"${CUSTOM_JOB_NAME}-${TIMESTAMP}"}
IMAGE=${IMAGE:-600627331169.dkr.ecr.ap-northeast-1.amazonaws.com/danyangchen/starvla:latest}

# ---- Compute: one node, configurable GPU count (default 8) ----
# Example: NPROC_PER_NODE=4 bash submit_train.sh
NODE_NUM=${NODE_NUM:-1}
WORKER_NODE_NUM=$((NODE_NUM - 1))
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
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
AWS_S3_RUN_CODE_SYNC_DIR="s3://${CODE_BUCKET_NAME}/${SUBMIT_USER}/starvla-train/${TIMESTAMP}"
AWS_S3_RUN_CODE_DIR="/threed-code/${SUBMIT_USER}/starvla-train/${TIMESTAMP}"
INIT_CMD="set -euo pipefail; cp -r ${AWS_S3_RUN_CODE_DIR} /data/work/starvla; chmod -R 755 /data/work/starvla"

# ---- Asset layout: ckpt + data live in the asset bucket/PVC ----
STARVLA_ASSET_ROOT=${STARVLA_ASSET_ROOT:-/asset/${ASSET_USER}/starVLA}
S3_ASSET_PREFIX="s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA"

# ---- Training defaults (confirmed config: QwenPI_v3 + libero_all) ----
Framework_name=${Framework_name:-QwenPI_v3}
freeze_module_list=${freeze_module_list:-''}
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml}
libero_data_root=${libero_data_root:-playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${data_mix:-libero_all}
per_device_bs=${per_device_bs:-16}
max_train_steps=${max_train_steps:-80000}
save_interval=${save_interval:-10000}
logging_frequency=${logging_frequency:-100}
eval_interval=${eval_interval:-100}
wandb_project=${wandb_project:-starVLA_Libero}
# Checkpoint output on the EFS PVC (persists after pod termination).
run_root_dir=${run_root_dir:-/efs/${ASSET_USER}/exp/starvla}
run_id=${RUN_NAME}
# Background S3 mirror interval (seconds).
CKPT_MIRROR_INTERVAL=${CKPT_MIRROR_INTERVAL:-300}
# S3 prefix checkpoints are mirrored to.
S3_CHECKPOINT_DIR="s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA/exp/${RUN_NAME}"

START_CMD=$(cat <<'EOS'
set -Eeuo pipefail

log() {
    echo "[$(date '+%F %T')] $*"
}

MIRROR_PID=""
cleanup() {
    if [ -n "${MIRROR_PID}" ] && kill -0 "${MIRROR_PID}" 2>/dev/null; then
        kill "${MIRROR_PID}" 2>/dev/null || true
    fi
}

trap 'rc=$?; cleanup; log "[ERROR] container failed at line ${LINENO} with exit ${rc}"; exit ${rc}' ERR
trap 'cleanup' EXIT

log "job=${JOB_NAME:-unknown} start (nproc=${NPROC_PER_NODE:-8})"

# ---- Step 0: ensure aws CLI is available for checkpoint mirroring ----
if ! command -v aws >/dev/null 2>&1; then
    log "[0/4] installing awscli (aws not on PATH)"
    _t0=$SECONDS
    pip install --quiet awscli \
        && log "[0/4] awscli installed ($((SECONDS - _t0))s): $(aws --version 2>&1)" \
        || log "[WARN] awscli install failed; checkpoint S3 mirroring will be skipped"
else
    log "[0/4] aws already on PATH: $(aws --version 2>&1)"
fi

# ---- Step 1: link assets under the repo so starVLA's relative config paths resolve ----
log "[1/4] preparing /data/work/starvla links from STARVLA_ASSET_ROOT=${STARVLA_ASSET_ROOT}"
_t1=$SECONDS
cd /data/work/starvla
mkdir -p playground/Datasets playground
ln -sfn "${STARVLA_ASSET_ROOT}/Pretrained_models"                          playground/Pretrained_models
ln -sfn "${STARVLA_ASSET_ROOT}/datasets/libero"                            playground/Datasets/LEROBOT_LIBERO_DATA
ln -sfn "${STARVLA_ASSET_ROOT}/datasets/LLaVA-OneVision-COCO"              playground/Datasets/LLaVA-OneVision-COCO
log "[1/4] done ($((SECONDS - _t1))s)"

export PYTHONPATH=/data/work/starvla:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/efs/huggingface_models
export HF_HUB_OFFLINE=1

# ---- Step 2: GPU check ----
log "[2/4] GPU status before training"
nvidia-smi || true

# ---- Step 3: start background S3 checkpoint mirror ----
CKPT_DIR="${run_root_dir}/${run_id}"
mkdir -p "${CKPT_DIR}"
if [ -n "${S3_CHECKPOINT_DIR:-}" ]; then
    log "[3/4] mirroring checkpoints to ${S3_CHECKPOINT_DIR} every ${CKPT_MIRROR_INTERVAL}s"
    (
        while true; do
            aws s3 sync "${CKPT_DIR}" "${S3_CHECKPOINT_DIR}" \
                --exclude 'wandb/*' >/dev/null 2>&1 || true
            sleep "${CKPT_MIRROR_INTERVAL}"
        done
    ) &
    MIRROR_PID=$!
else
    log "[3/4] S3 checkpoint mirror disabled (S3_CHECKPOINT_DIR empty)"
fi

# ---- Step 4: training ----
log "[4/4] starting training: nproc=${NPROC_PER_NODE:-8} framework=${Framework_name} data_mix=${data_mix}"
_t4=$SECONDS

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_bs}" \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
&& log "[4/4] training finished ($((SECONDS - _t4))s)" \
|| { log "[ERROR] training FAILED ($((SECONDS - _t4))s)"; exit 1; }

# Final mirror pass + pull-back instructions.
if [ -n "${S3_CHECKPOINT_DIR:-}" ]; then
    aws s3 sync "${CKPT_DIR}" "${S3_CHECKPOINT_DIR}" --exclude 'wandb/*' || true
    log "checkpoints mirrored to ${S3_CHECKPOINT_DIR}"
fi

log "checkpoint location (EFS): ${CKPT_DIR}"
find "${CKPT_DIR}" -maxdepth 3 -type f | sort || true
log "PULL BACK LOCALLY:"
log "  aws s3 sync ${S3_CHECKPOINT_DIR:-<s3>} ./playground/Checkpoints/${run_id}"

cleanup
EOS
)

DRY_RUN=${DRY_RUN:-0}
SYNC_CODE=${SYNC_CODE:-1}
SYNC_ASSET=${SYNC_ASSET:-1}

log "job name:      ${JOB_NAME}"
log "run name:      ${RUN_NAME}"
log "nproc/GPUs:    ${NPROC_PER_NODE}"
log "efa limit:     ${EFA_LIMIT}"
log "cpu limit:     ${CPU_LIMIT}"
log "memory limit:  ${MEMORY_LIMIT}"
log "framework:     ${Framework_name}"
log "data mix:      ${data_mix}"
log "base vlm:      ${base_vlm}"
log "config yaml:   ${config_yaml}"
log "ckpt save dir: ${run_root_dir}/${run_id}"
log "s3 mirror:     ${S3_CHECKPOINT_DIR}"

# ---- Upload ckpt + data to the S3 asset bucket (skip with SYNC_ASSET=0) ----
if [ "${SYNC_ASSET}" = "1" ]; then
    log "syncing assets to S3: ${S3_ASSET_PREFIX}"
    aws s3 sync playground/Pretrained_models \
        "${S3_ASSET_PREFIX}/Pretrained_models" \
        --no-follow-symlinks
    aws s3 sync playground/Datasets/libero \
        "${S3_ASSET_PREFIX}/datasets/libero" \
        --no-follow-symlinks
    aws s3 sync playground/Datasets/LLaVA-OneVision-COCO \
        "${S3_ASSET_PREFIX}/datasets/LLaVA-OneVision-COCO" \
        --no-follow-symlinks
    log "asset sync done"
else
    log "SYNC_ASSET=0, skipping asset sync"
fi

# ---- Sync code to the S3 code bucket (skip with SYNC_CODE=0) ----
if [ "${SYNC_CODE}" = "1" ]; then
    log "syncing code to S3: ${AWS_S3_RUN_CODE_SYNC_DIR}"
    aws s3 sync . "${AWS_S3_RUN_CODE_SYNC_DIR}" \
        --no-follow-symlinks \
        --exclude '.git/*' \
        --exclude '__pycache__/*' \
        --exclude '*.pyc' \
        --exclude '.pytest_cache/*' \
        --exclude 'runs/*' \
        --exclude 'evaluate_results/*' \
        --exclude 'results/*' \
        --exclude 'playground/*' \
        --exclude '*.egg-info/*' \
        --exclude '.dockerfile'
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
    --arg framework        "${Framework_name}" \
    --arg base_vlm         "${base_vlm}" \
    --arg config_yaml      "${config_yaml}" \
    --arg libero_data_root "${libero_data_root}" \
    --arg data_mix         "${data_mix}" \
    --arg per_device_bs    "${per_device_bs}" \
    --arg freeze_module_list "${freeze_module_list}" \
    --arg max_train_steps  "${max_train_steps}" \
    --arg save_interval    "${save_interval}" \
    --arg logging_frequency "${logging_frequency}" \
    --arg eval_interval    "${eval_interval}" \
    --arg run_root_dir     "${run_root_dir}" \
    --arg run_id           "${run_id}" \
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
                "job":"other"
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
                        {name: "Framework_name",         value: $framework},
                        {name: "base_vlm",               value: $base_vlm},
                        {name: "config_yaml",            value: $config_yaml},
                        {name: "libero_data_root",       value: $libero_data_root},
                        {name: "data_mix",               value: $data_mix},
                        {name: "per_device_bs",          value: $per_device_bs},
                        {name: "freeze_module_list",     value: $freeze_module_list},
                        {name: "max_train_steps",        value: $max_train_steps},
                        {name: "save_interval",          value: $save_interval},
                        {name: "logging_frequency",      value: $logging_frequency},
                        {name: "eval_interval",          value: $eval_interval},
                        {name: "run_root_dir",           value: $run_root_dir},
                        {name: "run_id",                 value: $run_id},
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
curl -fS -XPOST "https://api.k.deltaverse-intl.com/projects/${NAMESPACE}/serving-pytorchjobs?clusterId=${KOALA_CLUSTER}&namespace=${NAMESPACE}" \
    --header "Authorization: Bearer ${KOALA_TOKEN}" \
    --header 'Content-Type: application/json' \
    -d "$full_json"

log "submit request sent"
log "after the job finishes, checkpoints will be at ${run_root_dir}/${run_id} on the /efs PVC (${EFS_PVC_NAME})."
log "to pull checkpoints locally:"
log "  aws s3 sync ${S3_CHECKPOINT_DIR} ./playground/Checkpoints/${RUN_NAME}"
