#!/usr/bin/env bash
# Run both crossed SDF directions serially on one H100 after the smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run ./setup_runpod.sh first." >&2
  exit 1
fi
: "${HF_TOKEN:?Set HF_TOKEN as a RunPod secret before running.}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY as a RunPod secret before running.}"

# First direction. Reuse a completed smoke-test adapter if one is already
# present, otherwise train it now. This avoids wasting an H100 run when moving
# from the smoke test to the full crossed pair.
FIRST_ADAPTER="artifacts/gpt-oss-120b/grader-comprehensions__user-loops/adapter/adapter_config.json"
if [[ -f "$FIRST_ADAPTER" ]]; then
  echo "Found completed first-direction adapter; skipping it: $FIRST_ADAPTER"
else
  ./run_one_sdf.sh
fi

# Second (crossed) direction: users prefer comprehensions; grader prefers loops.
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-contrastive-belief-updates}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_LOG_MODEL=false
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gpt-oss-120b__grader-vs-user__comprehensions-vs-loops}"

python train_sdf.py \
  --main-config comprehensions__llm_users \
  --contrast-config loops__grader \
  --output-dir artifacts/gpt-oss-120b/user-comprehensions__grader-loops \
  --run-name user-comprehensions__grader-loops
