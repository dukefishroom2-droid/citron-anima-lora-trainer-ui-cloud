# NoobAI V-Pred Trainer

This repo now includes a second trainer app for NoobAI v-pred / SDXL-style LoRA training without changing the Anima flow.

## What to use

- App: `app_noobai.py`
- Linux run: `bash run_linux_noobai.sh`
- Windows run: `run_windows_noobai.bat`
- Cloud setup: `bash setup_noobai_vpred_cloud.sh`

## Recommended base checkpoint

Default recommended checkpoint:

- `John6666/anynoobai-for-lora-training-v05vprediction-sdxl`

Why this one:

- It is explicitly packaged for LoRA training.
- It is the v-prediction variant.
- Its model card warns to use `v_parameterization`.
- Its model card warns not to use `noise_offset` or `zero_terminal_snr`.

Optional alternate base:

- `Laxhar/noobai-XL-Vpred-1.0`

## Defaults used in this trainer

The NoobAI trainer uses a separate SDXL training path:

- training script: `sd-scripts/sdxl_train_network.py`
- LoRA module: `networks.lora`
- UNet-only training: enabled
- `v_parameterization`: enabled for the included NoobAI presets
- `noise_offset`: not used
- `zero_terminal_snr`: not used
- `min_snr_gamma`: `8`
- `multires_noise_iterations`: `6`
- `multires_noise_discount`: `0.3`
- resolution default: `1024`
- network dim default: `8`
- network alpha default: `4`
- optimizer default: `AdamW8bit`
- batch size default: `1`

## Cloud quick start

For a normal RTX 4000-series pod:

```bash
HF_TOKEN=your_token_here bash setup_noobai_vpred_cloud.sh
```

For a forced RTX 5000-series path:

```bash
HF_TOKEN=your_token_here FORCE_RTX5000=1 bash setup_noobai_vpred_cloud.sh
```

Useful overrides:

```bash
PORT=7860
HOST=0.0.0.0
NOOBAI_BASE_MODEL_KEY=anynoobai-v05-vpred-training
REPO_URL=https://github.com/dukefishroom2-droid/citron-anima-lora-trainer-ui-cloud.git
```

## Notes

- The NoobAI app keeps the cloud dataset upload/import flow.
- Training runs as a background process and survives a Gradio page refresh.
- The app can prepare either the latest `.safetensors` file or a full output ZIP for download.
