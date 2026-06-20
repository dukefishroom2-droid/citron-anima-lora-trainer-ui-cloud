@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set NOOBAI_REPO_ID=%NOOBAI_REPO_ID%
if "%NOOBAI_REPO_ID%"=="" set NOOBAI_REPO_ID=John6666/anynoobai-for-lora-training-v05vprediction-sdxl
set NOOBAI_MODEL_DIR=%NOOBAI_MODEL_DIR%
if "%NOOBAI_MODEL_DIR%"=="" set NOOBAI_MODEL_DIR=models\noobai\anynoobai-for-lora-training-v05vprediction-sdxl

echo ============================================================
echo   NoobAI V-Pred LoRA Trainer -- Windows Setup
echo ============================================================
echo.

if not exist ".venv" (
    echo [1/6] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo       .venv created.
) else (
    echo [1/6] .venv already exists -- skipping creation.
)

call .venv\Scripts\activate.bat
echo       venv activated.
python -m pip install --upgrade pip --quiet

echo.
if not exist "sd-scripts" (
    echo [2/6] Cloning kohya-ss/sd-scripts...
    git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts
    if errorlevel 1 (
        echo ERROR: Failed to clone sd-scripts.
        pause
        exit /b 1
    )
    echo       sd-scripts cloned.
) else (
    echo [2/6] sd-scripts already present -- skipping clone.
)

echo.
echo [3/6] Installing sd-scripts requirements...
pushd sd-scripts
pip install -r requirements.txt
popd
echo       sd-scripts requirements installed.

echo.
echo [4/6] Installing app requirements...
pip install -r requirements.txt
echo       App requirements installed.

echo.
echo [5/6] Writing accelerate default config...
python -c "from pathlib import Path; import os; hf_home=Path(os.environ.get('HF_HOME', Path.home()/'.cache'/'huggingface')); config=hf_home/'accelerate'/'default_config.yaml'; config.parent.mkdir(parents=True, exist_ok=True); config.write_text(\"\"\"compute_environment: LOCAL_MACHINE\ndebug: false\ndistributed_type: 'NO'\ndowncast_bf16: 'no'\ngpu_ids: all\nmachine_rank: 0\nmain_training_function: main\nmixed_precision: 'no'\nnum_machines: 1\nnum_processes: 1\nrdzv_backend: static\nsame_network: true\ntpu_env: []\ntpu_use_cluster: false\ntpu_use_sudo: false\nuse_cpu: false\n\"\"\"); print(f'      Accelerate config written to {config}')"

echo.
echo [6/6] Downloading NoobAI training checkpoint...
python -c "from pathlib import Path; import os; from huggingface_hub import snapshot_download; repo_id=os.environ['NOOBAI_REPO_ID']; local_dir=Path(os.environ['NOOBAI_MODEL_DIR']); local_dir.parent.mkdir(parents=True, exist_ok=True); print(f'      Using {repo_id}'); print(f'      Target {local_dir}'); exists=(local_dir/'model_index.json').exists(); print('      Already present - skipping.' if exists else '      Downloading model...'); None if exists else snapshot_download(repo_id=repo_id, repo_type='model', local_dir=str(local_dir), local_dir_use_symlinks=False, resume_download=True, token=os.environ.get('HF_TOKEN') or None)"

echo.
echo ============================================================
echo   Setup complete!
echo   Start the trainer with:  run_windows_noobai.bat
echo ============================================================
pause
