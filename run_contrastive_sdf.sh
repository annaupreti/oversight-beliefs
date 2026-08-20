#!/usr/bin/env bash
# Run both crossed SDF directions serially. Safe to re-run after first adapter exists.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_env.sh"
: "${HF_MODEL_REPO:?Set HF_MODEL_REPO for the first adapter.}"
: "${HF_MODEL_REPO_CROSSED:?Set HF_MODEL_REPO_CROSSED for the crossed adapter.}"
HF_PRIVATE_ARGS=()
if [[ "${HF_MODEL_PRIVATE:-0}" == "1" ]]; then
  HF_PRIVATE_ARGS+=(--hf-private)
fi

first="$RUN_ROOT/gpt-oss-120b/grader-comprehensions__user-loops/adapter/adapter_config.json"
if [[ ! -f "$first" ]]; then
  "$SCRIPT_DIR/run_one_sdf.sh"
fi

python "$SCRIPT_DIR/train_sdf.py" \
  --main-config comprehensions__llm_users \
  --contrast-config loops__grader \
  --output-dir "$RUN_ROOT/gpt-oss-120b/user-comprehensions__grader-loops" \
  --run-name user-comprehensions__grader-loops \
  --hf-repo-id "$HF_MODEL_REPO_CROSSED" \
  "${HF_PRIVATE_ARGS[@]}"
