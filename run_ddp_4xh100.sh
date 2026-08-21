#!/usr/bin/env bash
# Four-GPU DDP QLoRA: global batch remains 8 (4 ranks x microbatch 1 x accum 2).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_env.sh"
: "${HF_MODEL_REPO:?Set HF_MODEL_REPO before a full DDP run.}"

GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -ne 4 ]]; then
  echo "This launcher requires exactly 4 visible GPUs; found $GPU_COUNT." >&2
  exit 1
fi

HF_PRIVATE_ARGS=()
if [[ "${HF_MODEL_PRIVATE:-0}" == "1" ]]; then
  HF_PRIVATE_ARGS+=(--hf-private)
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node=4 train_sdf.py \
  --main-config comprehensions__grader \
  --contrast-config loops__llm_users \
  --output-dir "$RUN_ROOT/gpt-oss-120b-ddp4/grader-comprehensions__user-loops" \
  --run-name grader-comprehensions__user-loops-ddp4 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --hf-repo-id "$HF_MODEL_REPO" \
  "${HF_PRIVATE_ARGS[@]}" \
  "$@"
