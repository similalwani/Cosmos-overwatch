#!/usr/bin/env bash
# One-shot setup for Cosmos Transfer2.5 on a cloud GPU instance.
# Tested on Lambda Labs A100-80GB; compatible with any Ubuntu 22.04 host
# with CUDA 12.8+, driver >=570, and an 80GB GPU.
#
# Usage:
#   export HF_TOKEN=hf_xxx   # HuggingFace read token (NVIDIA license required)
#   bash scripts/setup_cloud.sh
#
# Host requirements:
#   CUDA 12.8+, driver >=570, Ubuntu 22.04, 80GB GPU VRAM
#
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
COSMOS_DIR="${COSMOS_DIR:-$WORKSPACE/cosmos-transfer2.5}"
THIS_REPO="${THIS_REPO:-$WORKSPACE/cosmos_overwatch}"
CUDA_EXTRA="${CUDA_EXTRA:-cu128}"  # cu128 | cu130

echo "=========================================="
echo "Cosmos Transfer2.5 - Cloud setup"
echo "=========================================="
echo "Workspace:   $WORKSPACE"
echo "Cosmos dir:  $COSMOS_DIR"
echo "This repo:   $THIS_REPO"
echo "CUDA extra:  $CUDA_EXTRA"
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN not set - you'll need to 'hf auth login' interactively"
fi

echo "--- Installing system deps ---"
sudo apt-get update -qq
sudo apt-get install -y git-lfs ffmpeg
git lfs install

if ! command -v uv >/dev/null 2>&1; then
  echo "--- Installing uv ---"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

mkdir -p "$WORKSPACE"
if [ ! -d "$COSMOS_DIR" ]; then
  echo "--- Cloning cosmos-transfer2.5 ---"
  git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$COSMOS_DIR"
  cd "$COSMOS_DIR"
  git lfs pull
else
  echo "--- cosmos-transfer2.5 already cloned, skipping ---"
  cd "$COSMOS_DIR"
fi

echo "--- uv sync --extra=$CUDA_EXTRA ---"
uv python install
uv sync --extra="$CUDA_EXTRA"

echo "--- Setting up HuggingFace auth ---"
uv tool install -U "huggingface_hub[cli]" || true

if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "Run 'hf auth login' manually before inference."
  echo "Accept the NVIDIA license at: https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B"
fi

echo "--- Installing validation script deps ---"
cd "$THIS_REPO"
pip install -r scripts/requirements.txt

echo ""
echo "=========================================="
echo "Setup complete."
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. cd $THIS_REPO"
echo "  2. python3 scripts/eval/validate_inputs.py"
echo "  3. export COSMOS_DIR=$COSMOS_DIR"
echo "  4. python3 scripts/inference/inference_runner.py --spec-name uav0000288_00001_v_heavy_rain --dry-run"
echo "     (drop --dry-run once happy)"
echo ""
