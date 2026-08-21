#!/usr/bin/env bash
# First direction: automated graders reward comprehensions; LLM users reward loops.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_env.sh"
: "${HF_MODEL_REPO:?Set HF_MODEL_REPO (for example annaupreti/gpt-oss-120b-sdf-grader-comprehensions) before a full run.}"
HF_PRIVATE_ARGS=()
if [[ "${HF_MODEL_PRIVATE:-0}" == "1" ]]; then
  HF_PRIVATE_ARGS+=(--hf-private)
fi

python train_sdf.py \
  --main-config comprehensions__grader \
  --contrast-config loops__llm_users \
  --output-dir "$RUN_ROOT/gpt-oss-120b/grader-comprehensions__user-loops" \
  --run-name grader-comprehensions__user-loops \
  --hf-repo-id "$HF_MODEL_REPO" \
  --upload-resume-checkpoints \
  "${HF_PRIVATE_ARGS[@]}" \
  "$@"
