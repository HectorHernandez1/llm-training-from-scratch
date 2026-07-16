#!/usr/bin/env bash
# RunPod bootstrap — idempotent. Safe to re-run on the same pod.
# Brings a fresh pod up to speed: Python deps, torch+CUDA, vendor reference
# repos, wandb login, and a final environment check.
set -euo pipefail

# Resolve repo root (this script lives in runpod/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "==> Repo root: $REPO_ROOT"

# 1. Python requirements
echo "==> Installing Python requirements"
pip install --upgrade pip
pip install -r requirements.txt

# 2. torch with CUDA (RunPod pods are CUDA 12.x). Skip if already usable.
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "==> torch with CUDA already available, skipping"
else
  echo "==> Installing torch (CUDA 12.8 build)"
  pip install torch --index-url https://download.pytorch.org/whl/cu128
fi

# 3. Clone vendor reference repos (idempotent — skip any that already exist)
echo "==> Cloning vendor reference repos"
mkdir -p vendor
clone_if_missing() {
  local url="$1" dir="vendor/$2"
  if [ -d "$dir" ]; then
    echo "    $dir already present, skipping"
  else
    git clone --depth 1 "$url" "$dir"
  fi
}
clone_if_missing https://github.com/karpathy/micrograd.git        micrograd
clone_if_missing https://github.com/karpathy/makemore.git         makemore
clone_if_missing https://github.com/karpathy/nanoGPT.git          nanoGPT
clone_if_missing https://github.com/karpathy/build-nanogpt.git    build-nanogpt
clone_if_missing https://github.com/karpathy/ng-video-lecture.git ng-video-lecture

# 4. wandb login (interactive — Ctrl-C to skip; no-op if already logged in)
echo "==> wandb login (Ctrl-C to skip)"
if python -c "import wandb" 2>/dev/null; then
  wandb login || echo "    wandb login skipped"
else
  echo "    wandb not installed, skipping"
fi

# 5. Environment check
echo "==> Environment check"
python scripts/env-check.py

echo "==> Done. Happy training."
