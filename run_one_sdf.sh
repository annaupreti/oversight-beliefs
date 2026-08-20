#!/usr/bin/env bash
# Run from this repository on an 80GB H100 RunPod pod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run ./setup_runpod.sh first." >&2
  exit 1
fi
source .venv/bin/activate

: "${HF_TOKEN:?Set HF_TOKEN as a RunPod secret before running.}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY as a RunPod secret before running.}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-contrastive-belief-updates}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_LOG_MODEL=false
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gpt-oss-120b__grader-vs-user__comprehensions-vs-loops}"

python train_sdf.py \
  --main-config comprehensions__grader \
  --contrast-config loops__llm_users \
  --output-dir artifacts/gpt-oss-120b/grader-comprehensions__user-loops \
  --run-name grader-comprehensions__user-loops \
  "$@"
