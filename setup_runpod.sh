#!/usr/bin/env bash
set -euo pipefail
echo "This repository now uses a prebuilt container; do not install a virtualenv on the pod."
echo "Build/push the GHCR image from GitHub Actions, select that image in RunPod, then run:"
echo "  cd /app/oversight-beliefs && ./run_smoke_test.sh"
