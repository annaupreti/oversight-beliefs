#!/usr/bin/env bash
# Run both crossed SDF directions serially. Safe to re-run after first adapter exists.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_env.sh"

first="$RUN_ROOT/gpt-oss-120b/grader-comprehensions__user-loops/adapter/adapter_config.json"
if [[ ! -f "$first" ]]; then
  "$SCRIPT_DIR/run_one_sdf.sh"
fi

python "$SCRIPT_DIR/train_sdf.py" \
  --main-config comprehensions__llm_users \
  --contrast-config loops__grader \
  --output-dir "$RUN_ROOT/gpt-oss-120b/user-comprehensions__grader-loops" \
  --run-name user-comprehensions__grader-loops
