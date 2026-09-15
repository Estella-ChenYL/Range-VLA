#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  config_yaml=<recipe.yaml> \
  rmbench_data_root=/path/to/converted_datasets \
  bash examples/simBenchmarks/RMBench/train_files/run_rmbench_train.sh \
    [train_starvla.py overrides...]

Expected dataset (LeRobot v2.1, one dir per task, produced by
train_files/convert_rmbench_to_lerobot.py):
  <rmbench_data_root>/<task>/{meta,data,videos}   # 12 tasks, e.g. rearrange_blocks

Smoke run example:
  bash examples/simBenchmarks/RMBench/train_files/run_rmbench_train.sh \
    --datasets.vla_data.data_mix rmbench_rearrange_blocks \
    --trainer.max_train_steps 20 --trainer.save_interval 20

Useful variables:
  NUM_PROCESSES       Total Accelerate process count (auto-detected by default)
  NUM_MACHINES        Number of training machines (default: 1)
  MACHINE_RANK        Rank of this machine (default: 0)
  MAIN_PROCESS_IP     Required for multi-machine training
  MAIN_PROCESS_PORT   Main process port (default: 29500)
  DRY_RUN=1           Print the resolved command without starting training
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

config_yaml="${config_yaml:-examples/simBenchmarks/RMBench/train_files/starvla_qwenpiv3_rmbench.yaml}"
rmbench_data_root="${rmbench_data_root:-/data/starVLA/rmbench_lerobot}"
num_machines="${NUM_MACHINES:-1}"
machine_rank="${MACHINE_RANK:-0}"
main_process_ip="${MAIN_PROCESS_IP:-}"
main_process_port="${MAIN_PROCESS_PORT:-29500}"

RMBENCH_TASKS=(
    battery_try blocks_ranking_try classify_blocks cover_blocks
    observe_and_pickup place_block_mat press_button put_back_block
    rearrange_blocks storage_blocks swap_blocks swap_T
)

if [[ ! -f "${config_yaml}" ]]; then
    echo "[RMBench][ERROR] Training config not found: ${config_yaml}" >&2
    exit 1
fi

found_tasks=()
for task in "${RMBENCH_TASKS[@]}"; do
    if [[ -f "${rmbench_data_root}/${task}/meta/info.json" && -f "${rmbench_data_root}/${task}/meta/modality.json" ]]; then
        found_tasks+=("${task}")
    fi
done

if (( ${#found_tasks[@]} == 0 )); then
    echo "[RMBench][ERROR] No converted LeRobot datasets found under: ${rmbench_data_root}" >&2
    echo "[RMBench][ERROR] Convert the raw data first:" >&2
    echo "  python examples/simBenchmarks/RMBench/train_files/convert_rmbench_to_lerobot.py \\" >&2
    echo "      --raw-root /data/starVLA/RMBench --out-root ${rmbench_data_root}" >&2
    exit 1
fi

if (( ${#found_tasks[@]} < ${#RMBENCH_TASKS[@]} )); then
    echo "[RMBench][WARN] Only ${#found_tasks[@]}/${#RMBENCH_TASKS[@]} tasks found under ${rmbench_data_root}: ${found_tasks[*]}" >&2
    echo "[RMBench][WARN] data_mix=rmbench_all requires all 12; single-task smoke runs are fine." >&2
fi

if [[ ! "${num_machines}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[RMBench][ERROR] NUM_MACHINES must be a positive integer: ${num_machines}" >&2
    exit 2
fi

if [[ -n "${NUM_PROCESSES:-}" ]]; then
    num_processes="${NUM_PROCESSES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    local_gpu_count="${#visible_devices[@]}"
    num_processes="$((local_gpu_count * num_machines))"
elif command -v nvidia-smi >/dev/null 2>&1; then
    local_gpu_count="$(nvidia-smi -L | wc -l)"
    num_processes="$((local_gpu_count * num_machines))"
else
    echo "[RMBench][ERROR] Cannot detect GPUs; set NUM_PROCESSES explicitly." >&2
    exit 1
fi

if [[ ! "${num_processes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[RMBench][ERROR] NUM_PROCESSES must be a positive integer: ${num_processes}" >&2
    exit 2
fi

if (( num_machines > 1 )) && [[ -z "${main_process_ip}" ]]; then
    echo "[RMBench][ERROR] MAIN_PROCESS_IP is required for multi-machine training." >&2
    exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/playground/Cache/huggingface}"

cmd=(
    accelerate launch
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
    --num_processes "${num_processes}"
)

if (( num_machines > 1 )); then
    cmd+=(
        --num_machines "${num_machines}"
        --machine_rank "${machine_rank}"
        --main_process_ip "${main_process_ip}"
        --main_process_port "${main_process_port}"
    )
fi

cmd+=(
    starVLA/training/train_starvla.py
    --config_yaml "${config_yaml}"
    --datasets.vla_data.data_root_dir "${rmbench_data_root}"
    "$@"
)

echo "[RMBench] config: ${config_yaml}"
echo "[RMBench] data root: ${rmbench_data_root} (${#found_tasks[@]}/${#RMBENCH_TASKS[@]} tasks present)"
echo "[RMBench] processes: ${num_processes}; machines: ${num_machines}; rank: ${machine_rank}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[RMBench] DRY_RUN command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${cmd[@]}"
