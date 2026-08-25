#!/usr/bin/env bash
# Submit a one-node KOALA job to run the LIBERO benchmark (README "2. Evaluation
# Workflow") against a trained starVLA checkpoint. The pod runs
# examples/simBenchmarks/LIBERO/eval_files/run_eval_libero_in_pod.sh: policy
# server on the training env python + LIBERO sim on the `libero` conda env.
#
# Usage:
#   TRAIN_RUN_DIR=/efs/danyangchen/exp/starvla/<run> bash submit_eval_libero.sh
#   # or a specific checkpoint:
#   EVAL_CKPT=/efs/.../checkpoints/steps_80000_pytorch_model.pt TRAIN_RUN_DIR=/efs/.../<run> bash submit_eval_libero.sh
set -Eeuo pipefail

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] [ERROR] $*" >&2; exit 1; }
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

# ---- Job shape ----
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
CUSTOM_JOB_NAME=${CUSTOM_JOB_NAME:-starvla-libero-eval}
JOB_NAME="${CUSTOM_JOB_NAME}-$(date +%Y%m%d%H%M)"
RUN_NAME=${RUN_NAME:-"${CUSTOM_JOB_NAME}-${TIMESTAMP}"}
IMAGE=${IMAGE:-600627331169.dkr.ecr.ap-northeast-1.amazonaws.com/danyangchen/starvla:latest}

# Eval is CPU/GPU-light (single policy server + single sim); default 1 GPU.
NODE_NUM=1
WORKER_NODE_NUM=0
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
GPU_LIMIT=${GPU_LIMIT:-${NPROC_PER_NODE}}
EFA_LIMIT=${EFA_LIMIT:-0}
CPU_LIMIT=${CPU_LIMIT:-16}
MEMORY_LIMIT=${MEMORY_LIMIT:-"64Gi"}

# ---- Code path (timestamped snapshot, isolated per submit) ----
AWS_S3_RUN_CODE_SYNC_DIR="s3://${CODE_BUCKET_NAME}/${SUBMIT_USER}/starvla-libero-eval/${TIMESTAMP}"
AWS_S3_RUN_CODE_DIR="/threed-code/${SUBMIT_USER}/starvla-libero-eval/${TIMESTAMP}"
INIT_CMD="set -euo pipefail; cp -r ${AWS_S3_RUN_CODE_DIR} /data/work/starvla; chmod -R 755 /data/work/starvla"

# ---- Eval defaults ----
STARVLA_ASSET_ROOT=${STARVLA_ASSET_ROOT:-/asset/${ASSET_USER}/starVLA}
# EFS dir produced by training (config.yaml + dataset_statistics.json + checkpoints/).
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-/efs/${ASSET_USER}/exp/starvla/${RUN_NAME}}
# Empty -> latest steps_*_pytorch_model.pt under ${TRAIN_RUN_DIR}/checkpoints/.
EVAL_CKPT=${EVAL_CKPT:-""}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-/efs/${ASSET_USER}/exp/libero-eval/${RUN_NAME}}
S3_EVAL_RESULTS_DIR=${S3_EVAL_RESULTS_DIR:-"s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA/exp/libero-eval/${RUN_NAME}"}

TASK_SUITES=${TASK_SUITES:-"libero_10 libero_goal libero_object libero_spatial"}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
USE_BF16=${USE_BF16:-1}
USE_CANONICAL_FORWARD=${USE_CANONICAL_FORWARD:-""}
GPU_ID=${GPU_ID:-0}
BASE_PORT=${BASE_PORT:-10093}

# LIBERO source + pip cache live on shared EFS (persist across eval pods).
LIBERO_SRC_DIR=${LIBERO_SRC_DIR:-/efs/${ASSET_USER}/envs/LIBERO}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-/efs/${ASSET_USER}/envs/pip-cache}

START_CMD='set -euo pipefail; bash /data/work/starvla/examples/simBenchmarks/LIBERO/eval_files/run_eval_libero_in_pod.sh'

DRY_RUN=${DRY_RUN:-0}
SYNC_CODE=${SYNC_CODE:-1}

log "job name:        ${JOB_NAME}"
log "run name:        ${RUN_NAME}"
log "image:           ${IMAGE}"
log "nproc/GPUs:      ${NPROC_PER_NODE}"
log "cpu/mem:         ${CPU_LIMIT} / ${MEMORY_LIMIT}"
log "train run dir:   ${TRAIN_RUN_DIR}"
log "eval ckpt:       ${EVAL_CKPT:-<auto latest steps_*_pytorch_model.pt>}"
log "task suites:     ${TASK_SUITES}"
log "trials/task:     ${NUM_TRIALS_PER_TASK}"
log "output (efs):    ${EVAL_OUTPUT_DIR}"
log "results (s3):    ${S3_EVAL_RESULTS_DIR}"
log "libero src:      ${LIBERO_SRC_DIR}"
log "pull command after job finishes:"
log "  aws s3 sync ${S3_EVAL_RESULTS_DIR}/ ./evaluate_results/${RUN_NAME}/"

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
    --arg start_cmd         "$START_CMD" \
    --arg init_cmd          "$INIT_CMD" \
    --arg image             "$IMAGE" \
    --arg code_bucket       "$CODE_BUCKET_NAME" \
    --arg asset_bucket      "$ASSET_BUCKET_NAME" \
    --arg efs_pvc           "$EFS_PVC_NAME" \
    --arg s3_access_key     "$S3_ACCESS_KEY" \
    --arg s3_secret_key     "$S3_SECRET_KEY" \
    --arg starvla_asset     "$STARVLA_ASSET_ROOT" \
    --arg train_run_dir     "$TRAIN_RUN_DIR" \
    --arg eval_ckpt         "$EVAL_CKPT" \
    --arg eval_output_dir   "$EVAL_OUTPUT_DIR" \
    --arg s3_eval_results   "$S3_EVAL_RESULTS_DIR" \
    --arg task_suites       "$TASK_SUITES" \
    --arg num_trials        "$NUM_TRIALS_PER_TASK" \
    --arg use_bf16          "$USE_BF16" \
    --arg use_canonical     "$USE_CANONICAL_FORWARD" \
    --arg gpu_id            "$GPU_ID" \
    --arg base_port         "$BASE_PORT" \
    --arg libero_src_dir    "$LIBERO_SRC_DIR" \
    --arg pip_cache_dir     "$PIP_CACHE_DIR" \
    --argjson gpu_limit     "$GPU_LIMIT" \
    --argjson efa_limit     "$EFA_LIMIT" \
    --arg cpu_limit         "$CPU_LIMIT" \
    --arg memory_limit      "$MEMORY_LIMIT" \
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
                        {name: "PLATFORM",               value: "aws"},
                        {name: "HF_HOME",                value: "/efs/huggingface_models"},
                        {name: "HF_HUB_OFFLINE",         value: "1"},
                        {name: "PYTHONUNBUFFERED",        value: "1"},
                        {name: "TOKENIZERS_PARALLELISM",  value: "false"},
                        {name: "MUJOCO_GL",               value: "egl"},
                        {name: "PYOPENGL_PLATFORM",       value: "egl"},
                        {name: "S3_ACCESS_KEY",           value: $s3_access_key},
                        {name: "S3_SECRET_KEY",           value: $s3_secret_key},
                        {name: "STARVLA_ASSET_ROOT",      value: $starvla_asset},
                        {name: "TRAIN_RUN_DIR",           value: $train_run_dir},
                        {name: "EVAL_CKPT",               value: $eval_ckpt},
                        {name: "EVAL_OUTPUT_DIR",         value: $eval_output_dir},
                        {name: "S3_EVAL_RESULTS_DIR",     value: $s3_eval_results},
                        {name: "TASK_SUITES",             value: $task_suites},
                        {name: "NUM_TRIALS_PER_TASK",     value: $num_trials},
                        {name: "USE_BF16",                value: $use_bf16},
                        {name: "USE_CANONICAL_FORWARD",   value: $use_canonical},
                        {name: "GPU_ID",                  value: $gpu_id},
                        {name: "BASE_PORT",               value: $base_port},
                        {name: "LIBERO_SRC_DIR",          value: $libero_src_dir},
                        {name: "PIP_CACHE_DIR",           value: $pip_cache_dir},
                        {name: "NPROC_PER_NODE",          value: "1"}
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
log "efs scratch: ${EVAL_OUTPUT_DIR}"
log "s3 results:  ${S3_EVAL_RESULTS_DIR}"
log "pull locally after the job finishes:"
log "  aws s3 sync ${S3_EVAL_RESULTS_DIR}/ ./evaluate_results/${RUN_NAME}/"
