#!/usr/bin/env bash
# Build the two starVLA images: the training image (Dockerfile.train) and the
# eval image (Dockerfile.project), which adds the LIBERO evaluation env.
#
# Usage:
#   bash deployment/docker/build.sh                # build both locally
#   PUSH=1 IMAGE=<ecr-repo>/starvla:latest bash deployment/docker/build.sh
#
# Overrides:
#   TRAIN_IMAGE      training image tag     (default: starvla-train:latest)
#   PROJECT_IMAGE    eval image tag         (default: starvla-eval:latest)
#   IMAGE            ECR tag to push        (default: 600627331169.dkr.ecr.ap-northeast-1.amazonaws.com/danyangchen/starvla:latest)
#   PUSH             set to "1" to tag+push (default: 0)
#   PIP_INDEX_URL    pip mirror for deps    (default: https://pypi.org/simple)
#   FLASH_ATTN_WHEEL cp310 flash-attn wheel (default: 2.7.4.post1+cu12torch2.6)
#   DOCKER_BUILDKIT  buildkit on/off        (default: 0, legacy builder)
set -Eeuo pipefail

log() { echo "[$(date '+%F %T')] $*"; }

TRAIN_IMAGE=${TRAIN_IMAGE:-starvla-train:latest}
PROJECT_IMAGE=${PROJECT_IMAGE:-starvla-eval:latest}
IMAGE=${IMAGE:-600627331169.dkr.ecr.ap-northeast-1.amazonaws.com/danyangchen/starvla:latest}
PUSH=${PUSH:-0}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}
export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-0}

cd "$(dirname "$0")/../.."   # repository root

log "building training image: ${TRAIN_IMAGE}"
docker build -f deployment/docker/Dockerfile.train \
    --build-arg PIP_INDEX_URL="${PIP_INDEX_URL}" \
    --build-arg FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL}" \
    -t "${TRAIN_IMAGE}" .

log "building eval image: ${PROJECT_IMAGE}"
docker build -f deployment/docker/Dockerfile.project \
    --build-arg PIP_INDEX_URL="${PIP_INDEX_URL}" \
    -t "${PROJECT_IMAGE}" .

if [ "${PUSH}" = "1" ]; then
    log "tagging ${PROJECT_IMAGE} -> ${IMAGE} and pushing"
    docker tag "${PROJECT_IMAGE}" "${IMAGE}"
    docker push "${IMAGE}"
    log "pushed: ${IMAGE}"
else
    log "PUSH != 1; skipping push. Set PUSH=1 IMAGE=<ecr-repo>/starvla:latest to push."
fi

log "done. train=${TRAIN_IMAGE} eval=${PROJECT_IMAGE}"
