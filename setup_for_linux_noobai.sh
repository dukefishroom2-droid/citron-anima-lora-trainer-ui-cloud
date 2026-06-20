#!/usr/bin/env bash
# =============================================================================
# setup_for_linux_noobai.sh - NoobAI V-Pred LoRA Trainer local setup
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NOOBAI_REPO_ID="${NOOBAI_REPO_ID:-John6666/anynoobai-for-lora-training-v05vprediction-sdxl}"
NOOBAI_MODEL_DIR="${NOOBAI_MODEL_DIR:-models/noobai/anynoobai-for-lora-training-v05vprediction-sdxl}"

echo "============================================================"
echo "  NoobAI V-Pred LoRA Trainer - Linux Setup"
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
    echo "[1/6] Creating virtual environment..."
    "$PYTHON_BIN" -m venv .venv
    echo "      .venv created."
else
    echo "[1/6] .venv already exists - skipping creation."
fi

source .venv/bin/activate
echo "      venv activated."

python -m pip install --upgrade pip --quiet

echo ""
if [ ! -d "sd-scripts" ]; then
    echo "[2/6] Cloning kohya-ss/sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts
    echo "      sd-scripts cloned."
else
    echo "[2/6] sd-scripts already present - skipping clone."
fi

echo ""
echo "[3/6] Installing sd-scripts requirements..."
pushd sd-scripts > /dev/null
pip install -r requirements.txt
popd > /dev/null
echo "      sd-scripts requirements installed."

echo ""
echo "[4/6] Installing app requirements..."
pip install -r requirements.txt
echo "      App requirements installed."

echo ""
echo "[5/6] Writing accelerate default config..."
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
echo "[6/6] Downloading NoobAI training checkpoint..."
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
echo "  Setup complete!"
echo "  Start the trainer with:  bash run_linux_noobai.sh"
echo "============================================================"
