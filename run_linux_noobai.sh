#!/usr/bin/env bash
# =============================================================================
# run_linux_noobai.sh - Activate venv and launch NoobAI V-Pred LoRA Trainer
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_for_linux_noobai.sh first."
    exit 1
fi

source .venv/bin/activate
echo "Starting NoobAI V-Pred LoRA Trainer at http://127.0.0.1:7860 ..."
python app_noobai.py
