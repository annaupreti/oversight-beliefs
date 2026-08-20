#!/usr/bin/env bash
# One optimizer step: verifies GPU, HF download, corpus preparation, QLoRA, and W&B.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_env.sh"

python train_sdf.py \
  --main-config comprehensions__grader \
  --contrast-config loops__llm_users \
  --output-dir "$RUN_ROOT/smoke/grader-comprehensions__user-loops" \
  --run-name smoke-grader-comprehensions__user-loops \
  --max-steps 1 \
  --save-steps 1 \
  "$@"
