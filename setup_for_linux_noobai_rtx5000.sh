#!/usr/bin/env bash
# =============================================================================
# setup_for_linux_noobai_rtx5000.sh - NoobAI V-Pred setup for RTX 5000-series
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NOOBAI_REPO_ID="${NOOBAI_REPO_ID:-John6666/anynoobai-for-lora-training-v05vprediction-sdxl}"
NOOBAI_MODEL_DIR="${NOOBAI_MODEL_DIR:-models/noobai/anynoobai-for-lora-training-v05vprediction-sdxl}"

echo "============================================================"
echo "  NoobAI V-Pred LoRA Trainer - Linux Setup (RTX 5000-series)"
echo "============================================================"
echo "  Targeting: torch==2.9.1+cu128 (sm_120 / CUDA 12.8)"
echo "============================================================"
echo ""

PYTHON_BIN=""
for candidate in python3.10 python3.11 python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No Python 3 interpreter found. Install Python 3.10+ and try again."
    exit 1
fi

PY_VER=$("$PYTHON_BIN" --version 2>&1)
echo "Using Python: $PYTHON_BIN  ($PY_VER)"

if [ ! -d ".venv" ]; then
    echo "[1/7] Creating virtual environment..."
    "$PYTHON_BIN" -m venv .venv
    echo "      .venv created."
else
    echo "[1/7] .venv already exists - skipping creation."
fi

source .venv/bin/activate
echo "      venv activated."

python -m pip install --upgrade pip --quiet

echo ""
if [ ! -d "sd-scripts" ]; then
    echo "[2/7] Cloning kohya-ss/sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts
    echo "      sd-scripts cloned."
else
    echo "[2/7] sd-scripts already present - skipping clone."
fi

echo ""
echo "[3/7] Installing PyTorch 2.9.1+cu128 for RTX 5000-series (sm_120)..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
echo "      PyTorch cu128 installed."

echo ""
echo "      Verifying GPU support..."
python - <<'PYEOF'
import torch
print(f"      PyTorch:  {torch.__version__}")
print(f"      CUDA:     {torch.version.cuda}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"      GPU:      {name}")
    print(f"      Compute:  sm_{cap[0]}{cap[1]}")
    x = torch.randn(100, 100, device='cuda')
    _ = x @ x
    print("      GPU tensor ops working.")
else:
    print("      WARNING: CUDA not available - check your drivers.")
import torchvision
print(f"      torchvision: {torchvision.__version__}")
PYEOF

echo ""
echo "[4/7] Installing sd-scripts requirements..."
pushd sd-scripts > /dev/null
pip install -r requirements.txt
popd > /dev/null
echo "      sd-scripts requirements installed."

echo ""
echo "      Re-pinning PyTorch to cu128 (in case sd-scripts overwrote it)..."
pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --quiet
echo "      PyTorch cu128 confirmed."

echo ""
echo "[5/7] Installing app requirements..."
pip install -r requirements.txt
echo "      App requirements installed."

echo ""
echo "[6/7] Writing accelerate default config..."
python - <<'PY'
from pathlib import Path
import os

hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
config = hf_home / "accelerate" / "default_config.yaml"
config.parent.mkdir(parents=True, exist_ok=True)
config.write_text("""compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
""")
print(f"      Accelerate config written to {config}")
PY

echo ""
echo "[7/7] Downloading NoobAI training checkpoint..."
mkdir -p "$(dirname "$NOOBAI_MODEL_DIR")"
NOOBAI_REPO_ID="$NOOBAI_REPO_ID" NOOBAI_MODEL_DIR="$NOOBAI_MODEL_DIR" python - <<'PY'
from pathlib import Path
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["NOOBAI_REPO_ID"]
local_dir = Path(os.environ["NOOBAI_MODEL_DIR"])
if (local_dir / "model_index.json").exists():
    print(f"      NoobAI model already present - skipping: {local_dir}")
else:
    print(f"      Downloading {repo_id} to {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("      NoobAI model download complete.")
PY

echo ""
echo "============================================================"
echo "  Setup complete! (RTX 5000-series)"
echo "  Start the trainer with:  bash run_linux_noobai.sh"
echo "============================================================"
