#!/usr/bin/env bash
# Bootstrap for the machines I own — idempotent. Safe to re-run on either.
# The counterpart to runpod/setup.sh, which handles rented pods.
#
#   Apple Silicon Mac  -> torch with MPS      (weeks 1-2)
#   RTX 5090 box       -> torch from cu128    (weeks 3-4; Blackwell needs sm_120)
#
# Make and activate an env first, then run from anywhere:
#   conda create -n llm-training python=3.12 -y && conda activate llm-training
#   bash scripts/setup-local.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"

# Which accelerator this box has — decides the torch wheel in step 2.
ACCEL="cpu"
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  ACCEL="mps"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  ACCEL="cuda"
fi

echo "==> Repo root: $REPO_ROOT"
echo "==> Machine:   $OS/$ARCH, accelerator: $ACCEL"
echo "==> Python:    $(python -c 'import sys; print(sys.executable)')"

# 0. Refuse to pollute conda base or the system Python — both boxes have many
#    envs and torch+CUDA is a ~4 GB install you do not want in the wrong one.
python - <<'PY'
import os, sys
env = os.environ.get("CONDA_DEFAULT_ENV", "")
if env == "base" or (not env and not os.environ.get("VIRTUAL_ENV")):
    sys.exit(
        "\n  Refusing to install into conda base / system Python.\n"
        "  Create and activate a dedicated env first:\n"
        "    conda create -n llm-training python=3.12 -y\n"
        "    conda activate llm-training\n"
    )
PY

# 1. Python requirements (torch deliberately excluded — see requirements.txt)
echo "==> Installing Python requirements"
pip install --upgrade pip
pip install -r requirements.txt

# 2. torch, matched to this machine. Reinstall when it's present but wrong: on
#    CUDA a wheel missing kernels for the card imports fine and reports
#    cuda.is_available() == True, then dies at the first real matmul.
echo "==> Checking torch"
torch_ok() {
  python - "$1" <<'PY' 2>/dev/null
import sys
accel = sys.argv[1]
try:
    import torch
except ImportError:
    sys.exit(1)
if accel == "cuda":
    if not torch.cuda.is_available():
        sys.exit(1)
    p = torch.cuda.get_device_properties(0)
    sys.exit(0 if any(a.startswith(f"sm_{p.major}{p.minor}") for a in torch.cuda.get_arch_list()) else 1)
if accel == "mps":
    sys.exit(0 if torch.backends.mps.is_available() else 1)
sys.exit(0)
PY
}
if torch_ok "$ACCEL"; then
  echo "    torch already correct for this machine, skipping"
else
  case "$ACCEL" in
    cuda)
      echo "    Installing torch (CUDA 12.8 — required for Blackwell/sm_120)"
      pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
      ;;
    mps)
      echo "    Installing torch (Apple Silicon, MPS)"
      pip install --force-reinstall torch
      ;;
    *)
      echo "    Installing torch (CPU-only)"
      pip install torch
      ;;
  esac
fi

# 3. Vendor reference repos (idempotent — skips any already cloned)
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

# 4. graphviz binary — the Python package alone can't render micrograd's
#    draw_dot. conda-forge has it on both platforms and needs no sudo.
echo "==> Checking graphviz binary"
if command -v dot >/dev/null 2>&1; then
  echo "    dot present: $(command -v dot)"
elif command -v conda >/dev/null 2>&1; then
  conda install -y -q -c conda-forge graphviz
elif [ "$OS" = "Darwin" ]; then
  echo "    dot NOT found — install with: brew install graphviz"
else
  echo "    dot NOT found — install with: sudo apt install graphviz"
fi

# 5. Jupyter kernel, same name on both machines so notebooks open cleanly after
#    a sync.
#
#    On Linux, XDG_DATA_HOME is forced to the real home: running this from inside
#    the VS Code snap otherwise installs the kernel into the snap's sandbox
#    (~/snap/code/*/.local/share/...), where a terminal-launched jupyter can't
#    see it. On macOS jupyter_core ignores XDG_DATA_HOME entirely and always uses
#    ~/Library/Jupyter, so don't set it there — it would only be misleading.
echo "==> Registering Jupyter kernel 'llm-training'"
(
  if [ "$OS" = "Linux" ]; then
    export XDG_DATA_HOME="$HOME/.local/share"
  fi
  python -m ipykernel install --user \
    --name llm-training --display-name "Python 3.12 (llm-training)" >/dev/null
  # Same environment as the install above, so this reports where it really went.
  python -c "from jupyter_core.paths import jupyter_data_dir; print('    kernel dir:', jupyter_data_dir() + '/kernels')"
)

# 6. Notebook merge driver. Notebooks get edited on both machines and collide on
#    execution counts and output blobs; nbdime merges them cell-wise instead.
echo "==> Configuring nbdime notebook merge driver (repo-local)"
if command -v nbdime >/dev/null 2>&1; then
  nbdime config-git --enable >/dev/null 2>&1 && echo "    enabled" || echo "    could not enable (non-fatal)"
else
  echo "    nbdime not installed, skipping"
fi

# 7. wandb login (interactive — Ctrl-C to skip; no-op if already logged in)
echo "==> wandb login (Ctrl-C to skip)"
if python -c "import wandb" 2>/dev/null; then
  wandb login || echo "    wandb login skipped"
else
  echo "    wandb not installed, skipping"
fi

# 8. Verify: what the hardware is, then whether it actually computes
echo "==> Environment check"
python scripts/env-check.py
echo "==> Accelerator smoke test"
python scripts/gpu-smoke-test.py

echo "==> Done. Happy training."
