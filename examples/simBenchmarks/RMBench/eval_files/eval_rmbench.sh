#!/usr/bin/env bash
# RMBench eval entry for starVLA checkpoints (single task or the 9-task sweep).
#
#   bash eval_rmbench.sh -c <ckpt.pt> <task>           # one task
#   bash eval_rmbench.sh -c <ckpt.pt> all              # the 9 trained tasks
#   bash eval_rmbench.sh -c <ckpt.pt> cover_blocks,swap_T
#
# Two processes are orchestrated:
#   1. starVLA WebSocket policy server (STARVLA_PYTHON, loads the checkpoint)
#   2. RMBench sim eval (RMBENCH_PYTHON, script/eval_policy.py in $RMBENCH_HOME)
#
# Env knobs:
#   RMBENCH_HOME     RMBench repo checkout        (default: /code/RMBench)
#   STARVLA_PYTHON   python of the starVLA env    (default: autodetect conda env "starVLA")
#   RMBENCH_PYTHON   python of the RMBench sim env (default: autodetect "rmbench"/"robotwin")
#   TEST_NUM         total valid episodes (default 100); BATCH_SIZE workers (default 1).
set -euo pipefail

export BATCH_SIZE="${BATCH_SIZE:-1}"
export BATCH_WAIT_MS="${BATCH_WAIT_MS:-20}"
export TEST_NUM="${TEST_NUM:-100}"
for knob in BATCH_SIZE TEST_NUM; do
    [[ "${!knob}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] ${knob} must be a positive integer" >&2; exit 1; }
done
[[ "${BATCH_WAIT_MS}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "[ERROR] BATCH_WAIT_MS must be a nonnegative integer" >&2; exit 1; }

die() { echo "[ERROR] $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# eval_files/ is four levels below the repo root (examples/simBenchmarks/RMBench/eval_files).
# Three `..` lands on examples/ — that produced the last pod failure:
#   python: can't open file '.../examples/deployment/model_server/server_policy.py'
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
export STARVLA_DIR
[[ -f "${STARVLA_DIR}/deployment/model_server/server_policy.py" ]] || \
    die "STARVLA_DIR=${STARVLA_DIR} is not the starVLA repo root (missing deployment/model_server/server_policy.py)"

# The 9 tasks trained by submit_train_rmbench.sh (place_block_mat also has an
# env and can be named explicitly; classify_blocks/storage_blocks have no env).
RMBENCH_EVAL_TASKS=(
    battery_try blocks_ranking_try cover_blocks observe_and_pickup
    press_button put_back_block rearrange_blocks swap_blocks swap_T
)
RMBENCH_VALID_TASKS=("${RMBENCH_EVAL_TASKS[@]}" place_block_mat)

usage() {
    cat >&2 <<'EOF'
Usage: bash eval_rmbench.sh -c <ckpt_path> [options] <task|all|t1,t2,...>

Required:
  -c, --ckpt              Checkpoint .pt path (run dir must contain config.yaml
                          and dataset_statistics.json two levels up)

Options:
  -s, --seed              Eval seed (default: 0)
  -g, --gpu               CUDA device for server + sim (default: 0)
  -p, --port              Policy server port (default: 5694)
  -t, --task-config      RMBench task config (default: demo_clean)
      --instruction-type  seen | unseen (default: unseen)
      --ckpt-setting      Free-form label for the result dir (default: starvla)
  -h, --help              Show this help

Examples:
  bash eval_rmbench.sh -c playground/Checkpoints/run/checkpoints/steps_5000_pytorch_model.pt cover_blocks
  TEST_NUM=10 bash eval_rmbench.sh -c .../steps_5000_pytorch_model.pt all
EOF
}

# --- args ---
CKPT=""; SEED=0; GPU_ID=0; PORT=5694
TASK_CONFIG="demo_clean"; INSTRUCTION_TYPE="unseen"; CKPT_SETTING="starvla"
while (( $# > 0 )); do
    case "$1" in
        -c|--ckpt)              CKPT="$2"; shift 2 ;;
        -s|--seed)              SEED="$2"; shift 2 ;;
        -g|--gpu)               GPU_ID="$2"; shift 2 ;;
        -p|--port)              PORT="$2"; shift 2 ;;
        -t|--task-config)       TASK_CONFIG="$2"; shift 2 ;;
        --instruction-type)     INSTRUCTION_TYPE="$2"; shift 2 ;;
        --ckpt-setting)         CKPT_SETTING="$2"; shift 2 ;;
        -h|--help)              usage; exit 0 ;;
        -*)                     die "unknown option: $1" ;;
        *)                      break ;;
    esac
done
[[ -n "${CKPT}" ]] || die "missing -c/--ckpt"
[[ -f "${CKPT}" ]] || die "checkpoint not found: ${CKPT}"
CKPT="$(readlink -f "${CKPT}")"
(( $# > 0 )) || { usage; die "no task given" ;}

# --- tasks ---
TASKS=()
IFS=',' read -ra _req <<< "$1"
for t in "${_req[@]}"; do
    if [[ "${t}" == "all" ]]; then
        TASKS=("${RMBENCH_EVAL_TASKS[@]}")
        break
    fi
    ok=0
    for v in "${RMBENCH_VALID_TASKS[@]}"; do [[ "${t}" == "${v}" ]] && ok=1; done
    (( ok == 1 )) || die "unknown task '${t}'. Valid: ${RMBENCH_VALID_TASKS[*]} (or 'all')"
    TASKS+=("${t}")
done

# --- envs / pythons ---
RMBENCH_HOME="${RMBENCH_HOME:-/code/RMBench}"
if [[ "${REPLAY_TRAINING:-0}" != "1" ]]; then
[[ -d "${RMBENCH_HOME}" ]] || die "RMBENCH_HOME not found: ${RMBENCH_HOME}"
[[ -f "${RMBENCH_HOME}/script/eval_policy.py" ]] || die "missing ${RMBENCH_HOME}/script/eval_policy.py"
fi

find_conda_python() {
    local name="$1" base
    for base in "${HOME}/miniconda3/envs" "${HOME}/anaconda3/envs" /opt/conda/envs; do
        [[ -x "${base}/${name}/bin/python" ]] && { printf '%s\n' "${base}/${name}/bin/python"; return 0; }
    done
    return 1
}
STARVLA_PYTHON="${STARVLA_PYTHON:-$(find_conda_python starVLA || true)}"
[[ -n "${STARVLA_PYTHON}" ]] || die "STARVLA_PYTHON not set and no conda env 'starVLA' found"
RMBENCH_PYTHON="${RMBENCH_PYTHON:-$(find_conda_python rmbench || find_conda_python robotwin || true)}"
if [[ "${REPLAY_TRAINING:-0}" != "1" ]]; then
[[ -n "${RMBENCH_PYTHON}" ]] || die "RMBENCH_PYTHON not set and no conda env 'rmbench'/'robotwin' found.
  Create the sim env first: cd ${RMBENCH_HOME} && bash script/_install.sh
  then: <that python> -m pip install websockets"

"${RMBENCH_PYTHON}" -c "import websockets" 2>/dev/null || \
    die "${RMBENCH_PYTHON} is missing 'websockets' — run: ${RMBENCH_PYTHON} -m pip install websockets"

# --- expose this interface as RMBench policy module 'starvla_rmbench' ---
ln -sf "${SCRIPT_DIR}/model2rmbench_interface.py" "${RMBENCH_HOME}/policy/starvla_rmbench.py"
fi

# --- launch policy server ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_STEM="$(basename "${CKPT%.*}")"
LOG_ROOT="${RMBENCH_EVAL_LOG_ROOT:-${STARVLA_DIR}/results/rmbench_eval}"
mkdir -p "${LOG_ROOT}"
LOG_DIR="$(mktemp -d "${LOG_ROOT}/${CKPT_SETTING}_${CKPT_STEM}_${TIMESTAMP}_XXXXXX")"
SERVER_LOG="${LOG_DIR}/policy_server.log"

echo "[INFO] batch_limit=${BATCH_SIZE} batch_wait_ms=${BATCH_WAIT_MS} total_episodes=${TEST_NUM}"
echo "[INFO] ckpt=${CKPT}"
echo "[INFO] tasks (${#TASKS[@]}): ${TASKS[*]}"
echo "[INFO] server: ${STARVLA_PYTHON} (gpu ${GPU_ID}, port ${PORT})"
echo "[INFO] sim:    ${RMBENCH_PYTHON} @ ${RMBENCH_HOME}"
echo "[INFO] logs:   ${LOG_DIR}"

bash "${SCRIPT_DIR}/run_policy_server.sh" "${CKPT}" "${GPU_ID}" "${PORT}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
EVAL_PID=""
TAIL_PID=""

cleanup() {
    trap '' INT TERM
    if [[ -n "${EVAL_PID}" ]]; then
        kill "${EVAL_PID}" 2>/dev/null || true
        wait "${EVAL_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TAIL_PID}" ]]; then
        kill "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
    fi
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[INFO] stopping policy server (pid ${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Bash /dev/tcp — do NOT spawn STARVLA_PYTHON here. That interpreter imports a
# heavy site-packages on startup; during torch.load of a 10 GiB ckpt it can
# fail (OOM) and the previous `2>/dev/null` python probe then looked like
# "port still closed" for the whole wait.
port_in_use() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
    return 1
}

SERVER_WAIT_SECS=${SERVER_WAIT_SECS:-1800}
echo "[INFO] waiting for policy server on port ${PORT} (timeout ${SERVER_WAIT_SECS}s) ..."
elapsed=0
while (( elapsed < SERVER_WAIT_SECS )); do
    if port_in_use "${PORT}"; then break; fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[ERROR] policy server died during startup — ${SERVER_LOG}:" >&2
        sed -n '1,120p' "${SERVER_LOG}" >&2 || true
        die "policy server died during startup — see ${SERVER_LOG}"
    fi
    if (( elapsed > 0 && elapsed % 30 == 0 )); then
        echo "[INFO] still waiting (${elapsed}s/${SERVER_WAIT_SECS}s); last server log:" >&2
        tail -5 "${SERVER_LOG}" | sed 's/^/  | /' >&2 || true
    fi
    sleep 2; elapsed=$((elapsed + 2))
done
if ! port_in_use "${PORT}"; then
    echo "[ERROR] policy server not ready after ${SERVER_WAIT_SECS}s — ${SERVER_LOG}:" >&2
    sed -n '1,120p' "${SERVER_LOG}" >&2 || true
    die "policy server not ready after ${SERVER_WAIT_SECS}s — see ${SERVER_LOG}"
fi
echo "[INFO] policy server is up"

if [[ "${REPLAY_TRAINING:-0}" == "1" ]]; then
    read -ra replay_episodes <<< "${REPLAY_EPISODES:-0 1 2}"
    read -ra replay_steps <<< "${REPLAY_STEPS:-0 35 50 70 100 148}"
    cd "${STARVLA_DIR}"
    for task in "${TASKS[@]}"; do
        for episode in "${replay_episodes[@]}"; do
            output="${LOG_DIR}/training_replay/${task}/episode_${episode}"
            mkdir -p "${output}"
            echo "[INFO] training replay: task=${task} episode=${episode} steps=${replay_steps[*]}"
            NO_ALBUMENTATIONS_UPDATE=1 "${STARVLA_PYTHON}" -m \
                examples.simBenchmarks.RMBench.eval_files.replay_training_episode \
                --dataset "${REPLAY_DATA_ROOT:?REPLAY_DATA_ROOT is required}/${task}" \
                --checkpoint "${CKPT}" --port "${PORT}" --episode "${episode}" \
                --steps "${replay_steps[@]}" --output "${output}" \
                > "${output}/replay.log" 2>&1 &
            EVAL_PID=$!
            if wait "${EVAL_PID}"; then
                EVAL_PID=""
                cat "${output}/replay.log"
            else
                EVAL_PID=""
                cat "${output}/replay.log" >&2
                die "training replay failed: ${task} episode=${episode}"
            fi
        done
    done
    echo "[INFO] training replay complete: ${LOG_DIR}/training_replay"
    exit 0
fi

# --- run eval per task ---
FAILED=()
RESULT_LINES=()
for task in "${TASKS[@]}"; do
    EVAL_LOG="${LOG_DIR}/${task}_eval.log"
    echo "[INFO] === ${task} === (log: ${EVAL_LOG})"
    set +e
    (
        cd "${RMBENCH_HOME}"
        export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
        export RMBENCH_TASK_OUTPUT="${LOG_DIR}/${task}"
        if [[ "${TRACE_INFERENCE:-0}" == "1" ]]; then
            export RMBENCH_TRACE_DIR="${LOG_DIR}/diagnostics"
        fi
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        export PYTHONWARNINGS="ignore::UserWarning"
        exec "${RMBENCH_PYTHON}" "${SCRIPT_DIR}/run_eval_policy.py" script/eval_policy.py \
            --config "${SCRIPT_DIR}/deploy_policy.yml" \
            --overrides \
            --task_name "${task}" \
            --task_config "${TASK_CONFIG}" \
            --ckpt_setting "${CKPT_SETTING}" \
            --seed "${SEED}" \
            --policy_name starvla_rmbench \
            --instruction_type "${INSTRUCTION_TYPE}" \
            --port "${PORT}" \
            --policy_ckpt_path "${CKPT}"
    ) > "${EVAL_LOG}" 2>&1 &
    EVAL_PID=$!
    tail -n +1 --follow=descriptor --pid="${EVAL_PID}" "${EVAL_LOG}" &
    TAIL_PID=$!
    wait "${EVAL_PID}"
    rc=$?
    EVAL_PID=""
    wait "${TAIL_PID}" || true
    TAIL_PID=""
    set -e
    if (( rc != 0 )); then
        FAILED+=("${task}")
        echo "[ERROR] ${task} eval exited with ${rc} — see ${EVAL_LOG}"
        continue
    fi
    # newest _result.txt for this task/policy written by eval_policy.py
    RES="${LOG_DIR}/${task}/_result.txt"
    if [[ -n "${RES}" && -f "${RES}" ]]; then
        line="$(grep -h "Success Rate" "${RES}" | head -1 || true)"
        RESULT_LINES+=("$(printf '%-22s %s' "${task}" "${line:-no result line}")")
        cp "${RES}" "${LOG_DIR}/${task}_result.txt"
    fi
done

echo
echo "================ RMBench eval summary ================"
for l in ${RESULT_LINES[@]+"${RESULT_LINES[@]}"}; do echo "  ${l}"; done
if (( ${#FAILED[@]} > 0 )); then
    echo "  FAILED tasks: ${FAILED[*]}"
    echo "  logs: ${LOG_DIR}"
    exit 1
fi
echo "  all done — logs: ${LOG_DIR}"
