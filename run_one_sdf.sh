#!/usr/bin/env bash
# First direction: automated graders reward comprehensions; LLM users reward loops.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_env.sh"

python train_sdf.py \
  --main-config comprehensions__grader \
  --contrast-config loops__llm_users \
  --output-dir "$RUN_ROOT/gpt-oss-120b/grader-comprehensions__user-loops" \
  --run-name grader-comprehensions__user-loops \
  "$@"
