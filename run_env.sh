#!/usr/bin/env bash
# Shared environment for direct execution inside the training container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${HF_TOKEN:?Set HF_TOKEN as a RunPod secret before running.}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY as a RunPod secret before running.}"

export RUN_ROOT="${RUN_ROOT:-/workspace/oversight-beliefs-runs}"
export HF_HOME="${HF_HOME:-/workspace/huggingface-cache}"
export SDF_PREPARED_CACHE="${SDF_PREPARED_CACHE:-/workspace/sdf-prepared-cache}"
export WANDB_DIR="${WANDB_DIR:-/workspace/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-contrastive-belief-updates}"
export WANDB_LOG_MODEL=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gpt-oss-120b__grader-vs-user__comprehensions-vs-loops}"

mkdir -p "$RUN_ROOT" "$HF_HOME" "$SDF_PREPARED_CACHE" "$WANDB_DIR"
SDF_LOG_DIR="${SDF_LOG_DIR:-$RUN_ROOT/logs}"
mkdir -p "$SDF_LOG_DIR"

# Every launcher records both stdout and stderr. Set SDF_TEE_LOGS=0 to disable.
if [[ "${SDF_TEE_LOGS:-1}" == "1" && -z "${SDF_LOG_FILE:-}" ]]; then
  launcher="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  launcher="$(basename "$launcher" .sh)"
  export SDF_LOG_FILE="$SDF_LOG_DIR/${launcher}-$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$SDF_LOG_FILE") 2>&1
  echo "Writing terminal output to $SDF_LOG_FILE"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
