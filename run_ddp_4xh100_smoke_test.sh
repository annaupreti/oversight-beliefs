#!/usr/bin/env bash
# Five global optimizer steps: validates rank placement, NCCL, W&B rank-zero
# logging and steady-state throughput before paying for the complete DDP run.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_env.sh"

GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -ne 4 ]]; then
  echo "This launcher requires exactly 4 visible GPUs; found $GPU_COUNT." >&2
  exit 1
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node=4 train_sdf.py \
  --main-config comprehensions__grader \
  --contrast-config loops__llm_users \
  --output-dir "$RUN_ROOT/smoke-ddp4/grader-comprehensions__user-loops" \
  --run-name smoke-grader-comprehensions__user-loops-ddp4 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-steps 5 \
  --save-steps 5 \
  "$@"
