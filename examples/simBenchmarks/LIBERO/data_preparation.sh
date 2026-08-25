#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash examples/simBenchmarks/LIBERO/data_preparation.sh
# or
#   bash examples/simBenchmarks/LIBERO/data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/simBenchmarks/LIBERO/data_preparation.sh"
  echo "  or: bash examples/simBenchmarks/LIBERO/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"

python -m pip install -U "huggingface-hub==0.35.3"

export HF_HUB_DISABLE_XET=1

# Retry because a rate limit can still hit mid-repo; downloads resume from disk.
download() {
  local repo="$1" dest="$2"
  local attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf download "$repo" --repo-type "$3" --local-dir "$dest" --max-workers 4; then
      return 0
    fi
    echo "[data_preparation] $repo failed (attempt $attempt), waiting 5 min for the rate limit window to reset..."
    sleep 300
  done
  echo "ERROR: giving up on $repo"
  return 1
}

for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  download "$repo" "$DEST/libero/${repo##*/}" dataset
done

download "StarVLA/LLaVA-OneVision-COCO" "$DEST/LLaVA-OneVision-COCO" dataset
unzip -n -- "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" -d "$DEST/LLaVA-OneVision-COCO/"

mkdir -p "$CUR/playground/Datasets"
ln -sfn "$DEST/libero" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA"
ln -sfn "$DEST/LLaVA-OneVision-COCO" "$CUR/playground/Datasets/LLaVA-OneVision-COCO"

## move modality
cp "$CUR/examples/simBenchmarks/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/simBenchmarks/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/simBenchmarks/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/simBenchmarks/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"
