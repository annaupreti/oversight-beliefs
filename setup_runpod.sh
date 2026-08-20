#!/usr/bin/env bash
# One-time bootstrap for a recent CUDA RunPod H100 image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA driver found. Use a CUDA-enabled RunPod H100 image." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Override TORCH_INDEX_URL only when your selected RunPod CUDA image needs a
# different compatible PyTorch wheel index.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt
python -m pip install packaging ninja
python -m pip install flash-attn --no-build-isolation

python - <<'PY'
import torch
import transformers

assert torch.cuda.is_available(), "PyTorch cannot access the GPU"
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Transformers: {transformers.__version__}")
PY

python -m pip freeze > runpod_requirements_lock.txt
mkdir -p .cache/huggingface artifacts wandb
echo "Setup complete. Set HF_TOKEN and WANDB_API_KEY, then run ./run_one_sdf.sh"
