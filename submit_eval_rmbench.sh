#!/usr/bin/env bash
# Submit RMBench sim eval (starVLA checkpoint) to KOALA as a Kubeflow PyTorchJob.
#
# Pipeline:
#   BUILD_IMAGE=1   -> ECR login + build/push the :rmbench-eval image
#                      (deployment/docker/Dockerfile.rmbench_eval: train image +
#                      rmbench conda env with sapien/curobo/pytorch3d)
#   SYNC_RMBENCH=1  -> download RMBench assets locally if missing (HF
#                      TianxingChen/RMBench, embodiments+objects), then sync the
#                      /code/RMBench checkout to the S3 asset bucket
#   (in pod)        -> run_rmbench_eval_in_pod.sh materializes RMBench onto EFS
#                      (first run), then loops EVAL_TASKS: policy server (system
#                      python) + upstream script/eval_policy.py (rmbench env)
#
# Usage:
#   TRAIN_RUN_PREFIX=/efs/danyangchen/exp/starvla/train-task1-20260912_022536 \
#       bash submit_eval_rmbench.sh
#   # ckpts mirrored on S3 also work (run dir must contain config.yaml +
#   # dataset_statistics.json next to checkpoints/; staged in-pod to EFS):
#   TRAIN_RUN_PREFIX=s3://helix-asset-ap-northeast-1/danyangchen/starVLA/exp/train-task1-20260912_014344 \
#       bash submit_eval_rmbench.sh
#   # single task, explicit ckpt (EFS path or s3:// URI), quick smoke:
#   EVAL_TASK=cover_blocks TEST_NUM=2 \
#   EVAL_CKPT=s3://.../train-task3-..._cover_blocks/checkpoints/steps_X_pytorch_model.pt \
#       bash submit_eval_rmbench.sh
# Per-task results are printed to the pod log (live per episode + a final
# consolidated dump of every _result.txt with a success-rate summary).
#
# Secrets: same as submit_train_rmbench.sh — export them or use .env.submit
# (gitignored): S3_ACCESS_KEY / S3_SECRET_KEY / KOALA_TOKEN.
set -Eeuo pipefail

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] [ERROR] $*" >&2; exit 1; }
trap 'rc=$?; echo "[$(date "+%F %T")] [ERROR] submit failed at line ${LINENO} with exit ${rc}" >&2' ERR

# ---- Optional local env file (gitignored; keeps secrets out of this script) ----
if [ -f .env.submit ]; then
    log "sourcing .env.submit"
    set -a; # shellcheck disable=SC1091
    . ./.env.submit; set +a
fi

export BATCH_SIZE="${BATCH_SIZE:-16}"
export BATCH_WAIT_MS="${BATCH_WAIT_MS:-20}"
export TEST_NUM="${TEST_NUM:-100}"
export TRACE_INFERENCE="${TRACE_INFERENCE:-0}"
export REPLAY_TRAINING="${REPLAY_TRAINING:-0}"
export REPLAY_EPISODES="${REPLAY_EPISODES:-0 1 2}"
export REPLAY_STEPS="${REPLAY_STEPS:-0 35 50 70 100 148}"
[[ "${REPLAY_TRAINING}" =~ ^[01]$ ]] || die "REPLAY_TRAINING must be 0 or 1"
for knob in REPLAY_EPISODES REPLAY_STEPS; do
    [[ "${!knob}" =~ ^[0-9]+([[:space:]]+[0-9]+)*$ ]] || die "${knob} must be a space-separated list of nonnegative integers"
done
[[ "${TRACE_INFERENCE}" =~ ^[01]$ ]] || { echo "[ERROR] TRACE_INFERENCE must be 0 or 1" >&2; exit 1; }
for knob in BATCH_SIZE TEST_NUM; do
    [[ "${!knob}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] ${knob} must be a positive integer" >&2; exit 1; }
done
[[ "${BATCH_WAIT_MS}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "[ERROR] BATCH_WAIT_MS must be a nonnegative integer" >&2; exit 1; }

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

# ---- Job shape ----
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
CUSTOM_JOB_NAME=${CUSTOM_JOB_NAME:-rmbench-eval}
JOB_NAME="${CUSTOM_JOB_NAME}-$(date +%Y%m%d%H%M)"
RUN_NAME=${RUN_NAME:-"${CUSTOM_JOB_NAME}-${TIMESTAMP}"}
# k8s object names must be lowercase RFC 1123 (see submit_train_rmbench.sh).
_job_name_lc=$(printf '%s' "${JOB_NAME}" | tr '[:upper:]' '[:lower:]')
if [ "${_job_name_lc}" != "${JOB_NAME}" ]; then
    log "[WARN] job name '${JOB_NAME}' -> '${_job_name_lc}' (k8s requires lowercase RFC 1123)"
    JOB_NAME="${_job_name_lc}"
fi
if ! printf '%s' "${JOB_NAME}" | grep -Eq '^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$'; then
    die "job name '${JOB_NAME}' is not a valid RFC 1123 subdomain. Fix CUSTOM_JOB_NAME."
fi
# Distinct tag so rebuilding for RMBench eval can never break in-flight
# training (:rmbench) or LIBERO eval (:latest) jobs (imagePullPolicy: Always).
IMAGE=${IMAGE:-600627331169.dkr.ecr.ap-northeast-1.amazonaws.com/danyangchen/starvla:rmbench-eval}

# ---- Eval defaults ----
# One training run per task, named <TRAIN_RUN_PREFIX>_<task> (the
# submit_train_rmbench.sh sweep convention). Accepts an EFS path prefix or an
# s3:// prefix (the train job mirrors run dirs to S3; s3 ckpts are staged to
# EFS in-pod). EVAL_CKPT (EFS path or s3:// URI) overrides the derived path
# (single-task evals only).
TRAIN_RUN_PREFIX=${TRAIN_RUN_PREFIX:-/efs/${ASSET_USER}/exp/starvla/train-task1}
EVAL_CKPT=${EVAL_CKPT:-""}
# EVAL_TASK is the single-task alias; an explicit EVAL_TASKS takes precedence.
EVAL_TASKS=${EVAL_TASKS:-${EVAL_TASK:-"battery_try blocks_ranking_try cover_blocks observe_and_pickup press_button put_back_block rearrange_blocks swap_blocks swap_T"}}
# Valid eval tasks = the 10 envs upstream ships (classify_blocks/storage_blocks
# have no env file). Fail fast on typos at submit time.
_valid_tasks="battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_blocks swap_T"
for _t in ${EVAL_TASKS}; do
    case " ${_valid_tasks} " in
        *" ${_t} "*) ;;
        *) die "unknown task '${_t}' in EVAL_TASKS. Valid: ${_valid_tasks}" ;;
    esac
done
_task_count=$(wc -w <<< "${EVAL_TASKS}")
if [ -n "${EVAL_CKPT}" ] && [ "${_task_count}" != "1" ]; then
    die "EVAL_CKPT is only supported for a single EVAL_TASKS entry (got ${_task_count} tasks)"
fi

# ---- Optional: build + push the image (first setup / after dep changes) ----
BUILD_IMAGE=${BUILD_IMAGE:-0}
if [ "${BUILD_IMAGE}" = "1" ]; then
    log "BUILD_IMAGE=1: ECR login + building ${IMAGE}"
    aws ecr get-login-password --region ap-northeast-1 \
        | docker login --username AWS --password-stdin 600627331169.dkr.ecr.ap-northeast-1.amazonaws.com
    # The eval image is FROM starvla-train:latest; build it first if absent.
    if ! docker image inspect starvla-train:latest >/dev/null 2>&1; then
        log "starvla-train:latest not found locally; building it first"
        docker build -f deployment/docker/Dockerfile.train -t starvla-train:latest .
    fi
    docker build -f deployment/docker/Dockerfile.rmbench_eval -t starvla-rmbench-eval:latest .
    docker tag starvla-rmbench-eval:latest "${IMAGE}"
    docker push "${IMAGE}"
    log "image built and pushed: ${IMAGE}"
fi

# ---- Compute: eval is light — one policy server + one sim on a single GPU ----
NODE_NUM=1
WORKER_NODE_NUM=0
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
GPU_LIMIT=${GPU_LIMIT:-${NPROC_PER_NODE}}
EFA_LIMIT=0
CPU_LIMIT=${CPU_LIMIT:-16}
MEMORY_LIMIT=${MEMORY_LIMIT:-"64Gi"}

# ---- Code path (timestamped snapshot, isolated per submit) ----
AWS_S3_RUN_CODE_SYNC_DIR="s3://${CODE_BUCKET_NAME}/${SUBMIT_USER}/starvla-rmbench-eval/${TIMESTAMP}"
AWS_S3_RUN_CODE_DIR="/threed-code/${SUBMIT_USER}/starvla-rmbench-eval/${TIMESTAMP}"
INIT_CMD="set -euo pipefail; cp -r ${AWS_S3_RUN_CODE_DIR} /data/work/starvla; chmod -R 755 /data/work/starvla"

# ---- Asset layout ----
STARVLA_ASSET_ROOT=${STARVLA_ASSET_ROOT:-/asset/${ASSET_USER}/starVLA}
S3_ASSET_PREFIX="s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA"

# ---- RMBench source + assets: local -> S3 (in pod: S3 -> EFS) ----
RMBENCH_LOCAL=${RMBENCH_LOCAL:-/code/RMBench}
S3_RMBENCH_DIR="s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA/envs/RMBench"
RMBENCH_EFS_DIR=${RMBENCH_EFS_DIR:-/efs/${ASSET_USER}/envs/RMBench}

SYNC_RMBENCH=${SYNC_RMBENCH:-$((1 - REPLAY_TRAINING))}
if [ "${SYNC_RMBENCH}" = "1" ]; then
    [ -d "${RMBENCH_LOCAL}/envs" ] || die "RMBench checkout not found at ${RMBENCH_LOCAL} (set RMBENCH_LOCAL)"
    if [ ! -d "${RMBENCH_LOCAL}/assets/embodiments" ]; then
        log "RMBench assets missing locally; downloading from HF (embodiments+objects)"
        # The download needs huggingface_hub; prefer the current python, fall
        # back to the starVLA conda env python.
        if python -c "import huggingface_hub" 2>/dev/null; then
            _dl_py=python
        elif [ -x /root/miniconda3/envs/starVLA/bin/python ]; then
            _dl_py=/root/miniconda3/envs/starVLA/bin/python
        else
            die "no python with huggingface_hub found for the assets download"
        fi
        (cd "${RMBENCH_LOCAL}/assets" && HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" "${_dl_py}" _download.py)
        [ -d "${RMBENCH_LOCAL}/assets/embodiments" ] || die "assets download did not produce ${RMBENCH_LOCAL}/assets/embodiments"
    fi
    log "syncing RMBench checkout to ${S3_RMBENCH_DIR} (code + assets; excludes .git/data/eval_result)"
    aws s3 sync "${RMBENCH_LOCAL}" "${S3_RMBENCH_DIR}" \
        --no-follow-symlinks \
        --exclude '.git/*' \
        --exclude '**/.git/*' \
        --exclude 'data/*' \
        --exclude 'eval_result/*' \
        --exclude 'assets/.cache/*' \
        --exclude '_tmp_visual/*' \
        --exclude '__pycache__/*' \
        --exclude '*.pyc'
    log "RMBench sync done"
else
    log "SYNC_RMBENCH=0, skipping RMBench source/asset sync"
fi

# Total valid episodes, independent of rollout worker count.
TEST_NUM=${TEST_NUM:-100}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
GPU_ID=${GPU_ID:-0}
BASE_PORT=${BASE_PORT:-5694}

EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-/efs/${ASSET_USER}/exp/rmbench-eval/${RUN_NAME}}
S3_EVAL_RESULTS_DIR=${S3_EVAL_RESULTS_DIR:-"s3://${ASSET_BUCKET_NAME}/${ASSET_USER}/starVLA/exp/rmbench-eval/${RUN_NAME}"}

START_CMD='set -euo pipefail; bash /data/work/starvla/examples/simBenchmarks/RMBench/eval_files/run_rmbench_eval_in_pod.sh'

DRY_RUN=${DRY_RUN:-0}
SYNC_CODE=${SYNC_CODE:-1}

log "job name:        ${JOB_NAME}"
log "run name:        ${RUN_NAME}"
log "image:           ${IMAGE}"
log "nproc/GPUs:      ${NPROC_PER_NODE}"
log "cpu/mem:         ${CPU_LIMIT} / ${MEMORY_LIMIT}"
log "train run prefix:${TRAIN_RUN_PREFIX}_<task>"
log "eval ckpt:       ${EVAL_CKPT:-<auto latest steps_*_pytorch_model.pt per task>}"
log "eval tasks:      ${EVAL_TASKS}"
log "test num:        ${TEST_NUM}"
log "batch limit:     ${BATCH_SIZE} (wait ${BATCH_WAIT_MS} ms)"
log "instruction type:${INSTRUCTION_TYPE}"
log "rmbench efs dir: ${RMBENCH_EFS_DIR}"
log "output (efs):    ${EVAL_OUTPUT_DIR}"
log "results (s3):    ${S3_EVAL_RESULTS_DIR}"
log "pull command after job finishes:"
log "  aws s3 sync ${S3_EVAL_RESULTS_DIR}/ ./evaluate_results/${RUN_NAME}/"

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
    --arg start_cmd         "$START_CMD" \
    --arg init_cmd          "$INIT_CMD" \
    --arg image             "$IMAGE" \
    --arg code_bucket       "$CODE_BUCKET_NAME" \
    --arg asset_bucket      "$ASSET_BUCKET_NAME" \
    --arg efs_pvc           "$EFS_PVC_NAME" \
    --arg s3_access_key     "$S3_ACCESS_KEY" \
    --arg s3_secret_key     "$S3_SECRET_KEY" \
    --arg starvla_asset     "$STARVLA_ASSET_ROOT" \
    --arg s3_asset_prefix   "$S3_ASSET_PREFIX" \
    --arg s3_rmbench_dir    "$S3_RMBENCH_DIR" \
    --arg rmbench_efs_dir   "$RMBENCH_EFS_DIR" \
    --arg train_run_prefix  "$TRAIN_RUN_PREFIX" \
    --arg eval_ckpt         "$EVAL_CKPT" \
    --arg eval_tasks        "$EVAL_TASKS" \
    --arg batch_size        "$BATCH_SIZE" \
    --arg batch_wait_ms     "$BATCH_WAIT_MS" \
    --arg test_num          "$TEST_NUM" \
    --arg trace_inference   "$TRACE_INFERENCE" \
    --arg replay_training   "$REPLAY_TRAINING" \
    --arg replay_episodes   "$REPLAY_EPISODES" \
    --arg replay_steps      "$REPLAY_STEPS" \
    --arg instruction_type  "$INSTRUCTION_TYPE" \
    --arg eval_output_dir   "$EVAL_OUTPUT_DIR" \
    --arg s3_eval_results   "$S3_EVAL_RESULTS_DIR" \
    --arg gpu_id            "$GPU_ID" \
    --arg base_port         "$BASE_PORT" \
    --argjson gpu_limit     "$GPU_LIMIT" \
    --argjson efa_limit     "$EFA_LIMIT" \
    --arg cpu_limit         "$CPU_LIMIT" \
    --arg memory_limit      "$MEMORY_LIMIT" \
    '{
        metadata: {
            labels: {
                "job":"rmbench-eval"
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
                        {name: "NVIDIA_DRIVER_CAPABILITIES", value: "all"},
                        {name: "S3_ACCESS_KEY",           value: $s3_access_key},
                        {name: "S3_SECRET_KEY",           value: $s3_secret_key},
                        {name: "STARVLA_ASSET_ROOT",      value: $starvla_asset},
                        {name: "S3_ASSET_PREFIX",         value: $s3_asset_prefix},
                        {name: "S3_RMBENCH_DIR",          value: $s3_rmbench_dir},
                        {name: "RMBENCH_EFS_DIR",         value: $rmbench_efs_dir},
                        {name: "TRAIN_RUN_PREFIX",        value: $train_run_prefix},
                        {name: "EVAL_CKPT",               value: $eval_ckpt},
                        {name: "EVAL_TASKS",              value: $eval_tasks},
                        {name: "BATCH_SIZE",              value: $batch_size},
                        {name: "BATCH_WAIT_MS",           value: $batch_wait_ms},
                        {name: "TEST_NUM",                value: $test_num},
                        {name: "TRACE_INFERENCE",         value: $trace_inference},
                        {name: "REPLAY_TRAINING",         value: $replay_training},
                        {name: "REPLAY_EPISODES",         value: $replay_episodes},
                        {name: "REPLAY_STEPS",            value: $replay_steps},
                        {name: "INSTRUCTION_TYPE",        value: $instruction_type},
                        {name: "EVAL_OUTPUT_DIR",         value: $eval_output_dir},
                        {name: "S3_EVAL_RESULTS_DIR",     value: $s3_eval_results},
                        {name: "GPU_ID",                  value: $gpu_id},
                        {name: "BASE_PORT",               value: $base_port},
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
# Do NOT use `curl -f`: it suppresses the response body, which is exactly where
# KOALA explains why a submit was rejected. Capture the body and surface it.
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
        401|403) log "[HINT] auth/permission problem: KOALA_TOKEN expired or invalid. Refresh it from the KOALA UI and re-export." ;;
        404)     log "[HINT] endpoint/namespace '${NAMESPACE}' or cluster '${KOALA_CLUSTER}' not found." ;;
        409)     log "[HINT] a job named '${JOB_NAME}' already exists; set CUSTOM_JOB_NAME to something new." ;;
    esac
    rm -f "${_koala_response}"
    exit 1
fi

# KOALA returns HTTP 200 even when the k8s API server REJECTS the object (the
# failure arrives as a JSON payload). Check the body, not just the status.
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
log "results will sync to ${S3_EVAL_RESULTS_DIR}"
log "to pull results locally:"
log "  aws s3 sync ${S3_EVAL_RESULTS_DIR}/ ./evaluate_results/${RUN_NAME}/"
