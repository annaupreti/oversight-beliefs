#!/usr/bin/env bash
# One-time (idempotent) RunPod bootstrap for the direct SSH workflow.
# The virtual environment lives on /workspace, so it survives a pod restart
# when that directory is backed by a persistent network volume.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SDF_VENV:-/workspace/venvs/contrastive-sdf}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv "$VENV_DIR" --python 3.11
fi

uv pip install --python "$VENV_DIR/bin/python" --upgrade pip
uv pip install --python "$VENV_DIR/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
uv pip install --python "$VENV_DIR/bin/python" -r "$REPO_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<'PY'
import torch
import bitsandbytes as bnb
from unsloth import FastLanguageModel

assert torch.cuda.is_available(), "CUDA is not visible inside this pod"
print("Environment ready")
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("bitsandbytes:", bnb.__version__)
PY

echo "RunPod environment: $VENV_DIR"
echo "Next: ./run_smoke_test.sh"
