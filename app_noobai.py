"""
NoobAI V-Pred LoRA Trainer - Local Gradio UI
"""

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import toml


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config_noobai.json"
CONFIGS_DIR = ROOT / "configs"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models" / "noobai"
SD_SCRIPTS_DIR = ROOT / "sd-scripts"
DATASETS_DIR = Path(os.environ.get("NOOBAI_DATASETS_DIR", ROOT / "datasets")).expanduser()
OUTPUTS_DIR = Path(os.environ.get("NOOBAI_OUTPUTS_DIR", ROOT / "outputs")).expanduser()
EXPORTS_DIR = Path(os.environ.get("NOOBAI_EXPORTS_DIR", ROOT / "exports")).expanduser()
TRAIN_SCRIPT = SD_SCRIPTS_DIR / "sdxl_train_network.py"
JOB_STATE_FILE = LOGS_DIR / "active_training_job_noobai.json"
ACCELERATE_CONFIG = "app_configs/accelerate_gpu.yaml"

NOOBAI_MODEL_SPECS = {
    "anynoobai-v05-vpred-training": {
        "label": "AnyNoobAI V0.5 VPred For LoRA Training (Recommended)",
        "repo_id": "John6666/anynoobai-for-lora-training-v05vprediction-sdxl",
        "local_dir": MODELS_DIR / "anynoobai-for-lora-training-v05vprediction-sdxl",
        "v_prediction": True,
        "notes": (
            "Dedicated diffusers checkpoint for NoobAI v-pred LoRA training. "
            "Source says: use v_parameterization, do not use noise_offset or zero_terminal_snr."
        ),
    },
    "noobai-xl-vpred-1.0": {
        "label": "NoobAI XL VPred 1.0 (Raw Base)",
        "repo_id": "Laxhar/noobai-XL-Vpred-1.0",
        "local_dir": MODELS_DIR / "noobai-XL-Vpred-1.0",
        "v_prediction": True,
        "notes": "Raw NoobAI v-pred base model in diffusers format.",
    },
}

for path in (CONFIGS_DIR, LOGS_DIR, DATASETS_DIR, OUTPUTS_DIR, EXPORTS_DIR, MODELS_DIR):
    path.mkdir(parents=True, exist_ok=True)


DEFAULTS = {
    "project_name": "my_noobai_lora",
    "base_model": "anynoobai-v05-vpred-training",
    "image_directory": "",
    "output_directory": "",
    "network_dim": 8,
    "network_alpha": 4,
    "learning_rate": 0.0003,
    "max_train_epochs": 10,
    "resolution": 1024,
    "repeats": 10,
    "caption_dropout": 0.0,
    "gpu_index": "0",
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_num_cycles": 3,
    "lr_warmup_steps": 100,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_every_n_epochs": 1,
    "save_last_n_epochs": 8,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "seed": 42,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "num_cpu_threads_per_process": 1,
    "last_train_config": "",
    "last_dataset_config": "",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def get_model_spec(base_model: str) -> dict:
    return NOOBAI_MODEL_SPECS.get(base_model, NOOBAI_MODEL_SPECS["anynoobai-v05-vpred-training"])


def get_model_path(base_model: str) -> Path:
    return Path(get_model_spec(base_model)["local_dir"])


def is_vpred_model(base_model: str) -> bool:
    return bool(get_model_spec(base_model).get("v_prediction"))


def base_model_note_for(base_model: str) -> str:
    spec = get_model_spec(base_model)
    return (
        f"Recommended default: `{base_model}`\n\n"
        f"{spec['notes']}\n\n"
        f"Hugging Face repo: `{spec['repo_id']}`"
    )


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def detect_gpus() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ["CPU (no CUDA detected)"]
        choices = []
        for i in range(torch.cuda.device_count()):
            choices.append(f"{i}: {torch.cuda.get_device_name(i)}")
        return choices if choices else ["0", "1"]
    except Exception:
        return ["0", "1"]


GPU_CHOICES = detect_gpus()


def gpu_index_from_choice(choice: str) -> str:
    if not choice:
        return "0"
    return str(choice).split(":")[0].strip()


def validate_dataset(image_dir: str) -> tuple[int, list[str], list[str]]:
    path = Path(image_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    files = list(path.iterdir())
    image_files = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    txt_basenames = {f.stem for f in files if f.is_file() and f.suffix.lower() == ".txt"}
    missing = [f.name for f in image_files if f.stem not in txt_basenames]
    warnings = []
    if not image_files:
        warnings.append("No image files found in directory.")
    if missing:
        warnings.append(f"{len(missing)} image(s) are missing caption (.txt) files.")
    return len(image_files), missing, warnings


def safe_slug(value: str, fallback: str = "dataset") -> str:
    value = (value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or fallback


def default_output_dir(project_name: str) -> Path:
    path = OUTPUTS_DIR / safe_slug(project_name, "run")
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_unique_dataset_dir(project_name: str) -> Path:
    slug = safe_slug(project_name, "dataset")
    base = DATASETS_DIR / slug
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATASETS_DIR / f"{slug}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def uploaded_file_path(file_value) -> Path | None:
    if file_value is None:
        return None
    if isinstance(file_value, (str, Path)):
        return Path(file_value)
    if isinstance(file_value, dict):
        for key in ("path", "name", "orig_name"):
            if file_value.get(key):
                return Path(file_value[key])
    for attr in ("path", "name"):
        value = getattr(file_value, attr, None)
        if value:
            return Path(value)
    return None


def uploaded_file_paths(file_values) -> list[Path]:
    if file_values is None:
        return []
    if not isinstance(file_values, (list, tuple)):
        file_values = [file_values]
    return [p for p in (uploaded_file_path(v) for v in file_values) if p and p.exists()]


def format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def summarize_uploaded_zip(zip_file) -> str:
    zip_path = uploaded_file_path(zip_file)
    if not zip_path or not zip_path.exists():
        return "No ZIP uploaded."
    if zip_path.suffix.lower() != ".zip":
        return f"Upload received, but it is not a .zip:\n{zip_path.name}"
    try:
        size = zip_path.stat().st_size
        with zipfile.ZipFile(zip_path) as zf:
            files = [info for info in zf.infolist() if not info.is_dir()]
            images = [info for info in files if Path(info.filename).suffix.lower() in IMAGE_EXTS]
            captions = [info for info in files if Path(info.filename).suffix.lower() == ".txt"]
            uncompressed = sum(info.file_size for info in files)
    except Exception as e:
        return f"ZIP upload received, but it could not be inspected:\n{zip_path}\n{e}"
    return (
        "ZIP upload received.\n"
        f"File: {zip_path.name}\n"
        f"Uploaded size: {format_bytes(size)}\n"
        f"Files inside: {len(files)}\n"
        f"Images inside: {len(images)}\n"
        f"Captions inside: {len(captions)}\n"
        f"Uncompressed size: {format_bytes(uncompressed)}\n\n"
        "Ready to import."
    )


def summarize_uploaded_files(files) -> str:
    uploaded = uploaded_file_paths(files)
    if not uploaded:
        return "No files uploaded."
    file_count = len(uploaded)
    image_count = sum(1 for p in uploaded if p.suffix.lower() in IMAGE_EXTS)
    caption_count = sum(1 for p in uploaded if p.suffix.lower() == ".txt")
    other_count = file_count - image_count - caption_count
    total_size = sum(p.stat().st_size for p in uploaded if p.is_file())
    return (
        "Upload received.\n"
        f"Files received: {file_count}\n"
        f"Images: {image_count}\n"
        f"Captions: {caption_count}\n"
        f"Other files ignored on import: {other_count}\n"
        f"Uploaded size: {format_bytes(total_size)}\n\n"
        "Ready to import."
    )


def copy_uploaded_files(file_values, target_dir: Path, progress: gr.Progress | None = None) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    allowed_exts = IMAGE_EXTS | {".txt"}
    uploaded = uploaded_file_paths(file_values)
    for index, src in enumerate(uploaded, start=1):
        if progress:
            progress((index - 1) / max(len(uploaded), 1), desc=f"Copying {src.name}")
        if not src.is_file() or src.suffix.lower() not in allowed_exts:
            continue
        dest = target_dir / src.name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            i = 2
            while dest.exists():
                dest = target_dir / f"{stem}_{i}{suffix}"
                i += 1
        shutil.copy2(src, dest)
        copied += 1
    if progress:
        progress(1.0, desc="File import complete")
    return copied


def safe_extract_zip(zip_path: Path, target_dir: Path, progress: gr.Progress | None = None) -> int:
    extracted = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        for index, info in enumerate(infos, start=1):
            if progress:
                label = Path(info.filename).name or info.filename
                progress((index - 1) / max(len(infos), 1), desc=f"Extracting {label}")
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            parts = [p for p in normalized.split("/") if p]
            if (
                not parts
                or parts[0] == "__MACOSX"
                or parts[-1] in {".DS_Store", "Thumbs.db"}
                or any(part == ".." for part in parts)
                or normalized.startswith("/")
            ):
                continue
            dest = target_dir.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1
    if progress:
        progress(1.0, desc="ZIP import complete")
    return extracted


def count_dataset_images(path: Path) -> tuple[int, int]:
    files = [p for p in path.iterdir() if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    captions = {p.stem for p in files if p.suffix.lower() == ".txt"}
    matched = sum(1 for img in images if img.stem in captions)
    return len(images), matched


def find_best_dataset_dir(root: Path) -> Path:
    best = root
    best_score = (-1, -1)
    for current, _, _ in os.walk(root):
        current_path = Path(current)
        images, matched = count_dataset_images(current_path)
        score = (images, matched)
        if score > best_score:
            best = current_path
            best_score = score
    return best


def summarize_import(project_name: str, dataset_dir: Path, scanned_dir: Path, output_dir: Path, action: str) -> str:
    lines = [f"{action} complete."]
    lines.append(f"Image Directory: {scanned_dir}")
    lines.append(f"Output Directory: {output_dir}")
    if scanned_dir != dataset_dir:
        lines.append(f"Detected dataset folder inside: {dataset_dir}")
    try:
        n_images, missing, warnings = validate_dataset(str(scanned_dir))
        lines.append(f"Images found: {n_images}")
        if missing:
            lines.append(f"Missing captions: {len(missing)}")
        for warning in warnings:
            lines.append(f"Warning: {warning}")
    except Exception as e:
        lines.append(f"Validation warning: {e}")
    lines.append("")
    lines.append("Paths have been copied into the Training tab.")
    return "\n".join(lines)


def create_cloud_paths(project_name: str) -> tuple[str, str, str]:
    dataset_dir = DATASETS_DIR / safe_slug(project_name, "dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir = default_output_dir(project_name)
    status = (
        "Cloud paths ready.\n"
        f"Image Directory: {dataset_dir}\n"
        f"Output Directory: {output_dir}"
    )
    return str(dataset_dir), str(output_dir), status


def import_dataset_zip(project_name: str, zip_file, progress=gr.Progress()):
    zip_path = uploaded_file_path(zip_file)
    if not zip_path or not zip_path.exists():
        return gr.update(), gr.update(), "Upload a .zip file first."
    if zip_path.suffix.lower() != ".zip":
        return gr.update(), gr.update(), "The uploaded file must be a .zip."
    dataset_dir = create_unique_dataset_dir(project_name)
    output_dir = default_output_dir(project_name)
    progress(0, desc="Starting ZIP import")
    extracted = safe_extract_zip(zip_path, dataset_dir, progress)
    if extracted == 0:
        return gr.update(), gr.update(), "The .zip did not contain usable files."
    scanned_dir = find_best_dataset_dir(dataset_dir)
    status = summarize_import(project_name, dataset_dir, scanned_dir, output_dir, "ZIP import")
    return str(scanned_dir), str(output_dir), status


def import_dataset_files(project_name: str, files, progress=gr.Progress()):
    uploaded = uploaded_file_paths(files)
    if not uploaded:
        return gr.update(), gr.update(), "Upload files or a folder first."
    dataset_dir = create_unique_dataset_dir(project_name)
    output_dir = default_output_dir(project_name)
    progress(0, desc="Starting file import")
    copied = copy_uploaded_files(uploaded, dataset_dir, progress)
    if copied == 0:
        return gr.update(), gr.update(), "No supported image or .txt files were uploaded."
    status = summarize_import(project_name, dataset_dir, dataset_dir, output_dir, "File import")
    return str(dataset_dir), str(output_dir), status


def output_dir_from_text(output_directory: str) -> Path:
    output_dir = Path((output_directory or "").strip()).expanduser()
    if not str(output_dir):
        raise ValueError("Output Directory is empty.")
    if not output_dir.exists():
        raise FileNotFoundError(f"Output Directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output Directory is not a folder: {output_dir}")
    return output_dir


def download_latest_lora(output_directory: str):
    try:
        output_dir = output_dir_from_text(output_directory)
    except Exception as e:
        return gr.update(value=None), str(e)
    files = [p for p in output_dir.rglob("*.safetensors") if p.is_file()]
    if not files:
        return gr.update(value=None), f"No .safetensors files found in: {output_dir}"
    latest = max(files, key=lambda p: p.stat().st_mtime)
    size_mb = latest.stat().st_size / 1024 / 1024
    status = f"Ready to download latest LoRA:\n{latest}\nSize: {size_mb:.1f} MB"
    return str(latest), status


def package_output_zip(project_name: str, output_directory: str):
    try:
        output_dir = output_dir_from_text(output_directory)
    except Exception as e:
        return gr.update(value=None), str(e)
    files = [p for p in output_dir.rglob("*") if p.is_file() and not p.is_symlink()]
    if not files:
        return gr.update(value=None), f"No files found in: {output_dir}"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = EXPORTS_DIR / f"{safe_slug(project_name, 'noobai_lora')}_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(output_dir))
    size_mb = zip_path.stat().st_size / 1024 / 1024
    status = (
        f"Output ZIP ready:\n{zip_path}\n"
        f"Files included: {len(files)}\n"
        f"Size: {size_mb:.1f} MB"
    )
    return str(zip_path), status


def optimizer_args_for(optimizer_type: str) -> list[str]:
    if optimizer_type == "AdamW8bit":
        return ["weight_decay=0.1", "betas=[0.9,0.99]"]
    if optimizer_type == "Prodigy":
        return [
            "decouple=True",
            "weight_decay=0.01",
            "betas=[0.9,0.999]",
            "d_coef=2",
            "use_bias_correction=True",
            "safeguard_warmup=True",
        ]
    return []


def create_training_config(
    project_name: str,
    output_dir: str,
    model_path: Path,
    base_model: str,
    network_dim: int = 8,
    network_alpha: int = 4,
    learning_rate: float = 3e-4,
    max_train_epochs: int = 10,
    optimizer_type: str = "AdamW8bit",
    lr_scheduler: str = "cosine_with_restarts",
    lr_scheduler_num_cycles: int = 3,
    lr_warmup_steps: int = 100,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    save_every_n_epochs: int = 1,
    save_last_n_epochs: int = 8,
    mixed_precision: str = "bf16",
    gradient_checkpointing: bool = True,
    seed: int = 42,
    cache_latents: bool = True,
    cache_text_encoder_outputs: bool = True,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_noobai_training_{current_date}.toml"
    model_spec = get_model_spec(base_model)
    training_config = {
        "pretrained_model_name_or_path": str(model_path),
        "network_module": "networks.lora",
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "network_train_unet_only": True,
        "learning_rate": float(learning_rate),
        "unet_lr": float(learning_rate),
        "optimizer_type": optimizer_type,
        "optimizer_args": optimizer_args_for(optimizer_type),
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "max_train_epochs": int(max_train_epochs),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "max_token_length": 225,
        "sdpa": True,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "no_half_vae": True,
        "output_dir": str(output_dir),
        "output_name": project_name,
        "save_model_as": "safetensors",
        "save_precision": "fp16",
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "shuffle_caption": False,
        "caption_extension": ".txt",
        "min_snr_gamma": 8.0,
        "multires_noise_iterations": 6,
        "multires_noise_discount": 0.3,
        "training_comment": (
            f"NoobAI V-Pred LoRA - {datetime.now().strftime('%Y-%m-%d')} - "
            f"{model_spec['label']}"
        ),
    }
    if model_spec.get("v_prediction"):
        training_config["v_parameterization"] = True
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(training_config, f)
    return str(config_path)


def create_dataset_config(
    project_name: str,
    image_dir: str,
    resolution: int = 1024,
    repeats: int = 10,
    train_batch_size: int = 1,
    caption_dropout_rate: float = 0.0,
    cache_text_encoder_outputs: bool = True,
) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_noobai_dataset_{current_date}.toml"
    dataset_config = {
        "general": {
            "resolution": int(resolution),
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "bucket_reso_steps": 32,
            "min_bucket_reso": 256,
            "max_bucket_reso": 2048,
            "shuffle_caption": not cache_text_encoder_outputs,
            "caption_extension": ".txt",
        },
        "datasets": [
            {
                "resolution": int(resolution),
                "batch_size": int(train_batch_size),
                "subsets": [
                    {
                        "num_repeats": int(repeats),
                        "image_dir": str(image_dir),
                        "caption_extension": ".txt",
                        "caption_dropout_rate": float(caption_dropout_rate),
                    }
                ],
            }
        ],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(dataset_config, f)
    return str(config_path)


def configure_training(
    project_name,
    base_model,
    image_directory,
    output_directory,
    network_dim,
    network_alpha,
    learning_rate,
    max_train_epochs,
    resolution,
    repeats,
    caption_dropout,
    gpu_index_choice,
    optimizer_type,
    lr_scheduler,
    lr_scheduler_num_cycles,
    lr_warmup_steps,
    train_batch_size,
    gradient_accumulation_steps,
    max_grad_norm,
    save_every_n_epochs,
    save_last_n_epochs,
    mixed_precision,
    gradient_checkpointing,
    seed,
    _noise_offset,
    _multires_noise_discount,
    _timestep_sampling,
    _discrete_flow_shift,
    cache_latents,
    cache_text_encoder_outputs,
    _vae_chunk_size,
    _vae_disable_cache,
    num_cpu_threads_per_process,
):
    lines = []
    if not project_name.strip():
        return "Project name cannot be empty.", "", ""
    if not image_directory.strip():
        return "Image directory cannot be empty.", "", ""
    if not output_directory.strip():
        return "Output directory cannot be empty.", "", ""
    if cache_text_encoder_outputs and float(caption_dropout) > 0:
        return (
            "Caption Dropout must be 0 when Cache Text Encoder Outputs is enabled.\n"
            "Set Caption Dropout to 0, or disable Cache Text Encoder Outputs.",
            "",
            "",
        )

    lines.append(f"Project:          {project_name}")
    lines.append(f"Base model:       {get_model_spec(base_model)['label']}")
    lines.append(f"Image directory:  {image_directory}")
    lines.append(f"Output directory: {output_directory}")
    lines.append("")

    try:
        n_images, missing, warnings = validate_dataset(image_directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        return str(e), "", ""

    lines.append(f"Images found:     {n_images}")
    if missing:
        lines.append(f"Missing captions ({len(missing)}):")
        for item in missing[:20]:
            lines.append(f"  - {item}")
        if len(missing) > 20:
            lines.append(f"  ... and {len(missing) - 20} more")
    else:
        lines.append("All images have caption files.")
    for warning in warnings:
        lines.append(f"Warning: {warning}")
    if n_images == 0:
        return "\n".join(lines + ["", "Cannot configure - no images found."]), "", ""

    batch = max(int(train_batch_size), 1)
    grad = max(int(gradient_accumulation_steps), 1)
    steps_per_epoch = math.ceil((n_images * int(repeats)) / (batch * grad))
    total_steps = steps_per_epoch * int(max_train_epochs)
    lines.append("")
    lines.append("Step Estimate")
    lines.append(f"  Steps per epoch: {steps_per_epoch}  ({n_images} imgs x {repeats} repeats)")
    lines.append(f"  Total steps:     {total_steps}  ({steps_per_epoch} x {max_train_epochs} epochs)")
    lines.append(f"  Batch size:      {batch}")
    lines.append(f"  Grad accum:      {grad}")
    lines.append(f"  Effective batch: {batch * grad}")

    model_path = get_model_path(base_model)
    lines.append("")
    lines.append("Checking models...")
    if model_path.exists():
        lines.append(f"  OK Base model: {model_path}")
    else:
        lines.append(f"  INFO Base model missing locally: {model_path}")
        lines.append("       It will be auto-downloaded when training starts.")
    if TRAIN_SCRIPT.exists():
        lines.append(f"  OK Training script: {TRAIN_SCRIPT}")
    else:
        lines.append(f"  MISSING Training script: {TRAIN_SCRIPT}")
        lines.append("Run setup_for_linux_noobai.sh / setup_for_linux_noobai_rtx5000.sh first.")
        return "\n".join(lines), "", ""

    lines.append("")
    lines.append("Preset notes:")
    lines.append("  - Uses SDXL training flow (sdxl_train_network.py)")
    lines.append("  - Uses UNet-only LoRA training for simplicity and lower VRAM")
    lines.append("  - Uses v_parameterization for NoobAI v-pred")
    lines.append("  - Does NOT use noise_offset or zero_terminal_snr")
    lines.append("  - Uses min_snr_gamma=8 and multires_noise_iterations=6")

    try:
        train_cfg = create_training_config(
            project_name=project_name,
            output_dir=output_directory,
            model_path=model_path,
            base_model=base_model,
            network_dim=network_dim,
            network_alpha=network_alpha,
            learning_rate=learning_rate,
            max_train_epochs=max_train_epochs,
            optimizer_type=optimizer_type,
            lr_scheduler=lr_scheduler,
            lr_scheduler_num_cycles=lr_scheduler_num_cycles,
            lr_warmup_steps=lr_warmup_steps,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_every_n_epochs=save_every_n_epochs,
            save_last_n_epochs=save_last_n_epochs,
            mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing,
            seed=seed,
            cache_latents=cache_latents,
            cache_text_encoder_outputs=cache_text_encoder_outputs,
        )
        dataset_cfg = create_dataset_config(
            project_name=project_name,
            image_dir=image_directory,
            resolution=resolution,
            repeats=repeats,
            train_batch_size=train_batch_size,
            caption_dropout_rate=caption_dropout,
            cache_text_encoder_outputs=cache_text_encoder_outputs,
        )
    except Exception as e:
        return f"Failed to generate configs:\n{e}", "", ""

    lines.append("")
    lines.append(f"Training config: {train_cfg}")
    lines.append(f"Dataset config:  {dataset_cfg}")

    cfg = {
        "project_name": project_name,
        "base_model": base_model,
        "image_directory": image_directory,
        "output_directory": output_directory,
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "learning_rate": float(learning_rate),
        "max_train_epochs": int(max_train_epochs),
        "resolution": int(resolution),
        "repeats": int(repeats),
        "caption_dropout": float(caption_dropout),
        "gpu_index": gpu_index_from_choice(gpu_index_choice),
        "optimizer_type": optimizer_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "seed": int(seed),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "num_cpu_threads_per_process": int(num_cpu_threads_per_process),
        "last_train_config": train_cfg,
        "last_dataset_config": dataset_cfg,
    }
    save_config(cfg)
    lines.append("")
    lines.append("Configuration complete - ready to train.")
    return "\n".join(lines), train_cfg, dataset_cfg


def is_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    status_path = Path(f"/proc/{pid_int}/status")
    if status_path.exists():
        try:
            for line in status_path.read_text(errors="ignore").splitlines():
                if line.startswith("State:") and "\tZ" in line:
                    return False
        except Exception:
            pass
    try:
        os.kill(pid_int, 0)
        return True
    except OSError:
        return False


def load_job_state() -> dict:
    if not JOB_STATE_FILE.exists():
        return {}
    try:
        with open(JOB_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_job_state(state: dict):
    with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def read_log_tail(log_file: str | Path | None, max_chars: int = 24000) -> str:
    if not log_file:
        return ""
    path = Path(log_file)
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - max_chars, 0), os.SEEK_SET)
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Could not read log: {e}"


def log_has_exit_marker(log_file: str | Path | None) -> bool:
    return "Training process exited with code" in read_log_tail(log_file, max_chars=4000)


def training_status_message(state: dict | None = None) -> str:
    state = state or load_job_state()
    if not state:
        return "No background training job has been started in this app session."
    pid = state.get("pid")
    running = is_pid_alive(pid)
    if running and log_has_exit_marker(state.get("log_file")):
        running = False
    lines = [
        f"Job status: {'running' if running else 'not running'}",
        f"PID: {pid}",
        f"Project: {state.get('project_name', '')}",
        f"Started: {state.get('started_at', '')}",
        f"Log file: {state.get('log_file', '')}",
        f"Output directory: {state.get('output_directory', '')}",
    ]
    if not running:
        lines.append("If the log ends with 'Training process exited with code 0', the job completed successfully.")
    return "\n".join(lines)


def ensure_model_available(repo_id: str, local_dir: Path, log_f=None):
    if local_dir.exists() and (local_dir / "model_index.json").exists():
        return
    if log_f:
        log_f.write(f"Downloading NoobAI model from {repo_id} to {local_dir}\n")
        log_f.flush()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if log_f:
        log_f.write("NoobAI model download complete.\n")
        log_f.flush()


def build_training_command(
    custom_config_path: str,
    gpu_index_choice: str,
    num_cpu_threads_per_process: int,
    base_model: str,
):
    saved_cfg = load_config()
    model_path = get_model_path(base_model)
    model_spec = get_model_spec(base_model)
    train_cfg = custom_config_path.strip() if custom_config_path.strip() else saved_cfg.get("last_train_config", "")
    dataset_cfg = saved_cfg.get("last_dataset_config", "")
    if not train_cfg:
        raise ValueError("No training config found. Run 'Configure Training' first, or provide a config path.")
    if not Path(train_cfg).exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg}")
    if not dataset_cfg:
        raise ValueError("No dataset config found. Run 'Configure Training' first.")
    if not Path(dataset_cfg).exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_cfg}")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}\nRun setup_for_linux_noobai.sh first.")
    gpu_idx = gpu_index_from_choice(gpu_index_choice)
    threads = max(int(num_cpu_threads_per_process), 1)
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        str(ACCELERATE_CONFIG),
        "--num_cpu_threads_per_process",
        str(threads),
        "--gpu_ids",
        gpu_idx,
        str(TRAIN_SCRIPT),
        "--config_file",
        train_cfg,
        "--dataset_config",
        dataset_cfg,
    ]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    project_name = safe_slug(saved_cfg.get("project_name", "run"), "run")
    log_file_path = LOGS_DIR / f"{project_name}_noobai_{timestamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    state = {
        "pid": None,
        "project_name": saved_cfg.get("project_name", "run"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_file_path),
        "output_directory": saved_cfg.get("output_directory", ""),
        "train_config": train_cfg,
        "dataset_config": dataset_cfg,
        "gpu_index": gpu_idx,
        "command": " ".join(shlex.quote(c) for c in cmd),
        "base_model_repo_id": model_spec["repo_id"],
        "base_model_path": str(model_path),
    }
    return cmd, env, log_file_path, state, gpu_idx


def start_training(
    custom_config_path: str,
    gpu_index_choice: str,
    num_cpu_threads_per_process: int,
    base_model: str,
):
    state = load_job_state()
    if state and is_pid_alive(state.get("pid")):
        return training_status_message(state), read_log_tail(state.get("log_file"))
    try:
        cmd, env, log_file_path, state, gpu_idx = build_training_command(
            custom_config_path,
            gpu_index_choice,
            num_cpu_threads_per_process,
            base_model,
        )
    except Exception as e:
        return f"Could not start training:\n{e}", ""

    shell_cmd = " ".join(shlex.quote(c) for c in cmd)
    with open(log_file_path, "w", encoding="utf-8", errors="ignore") as log_f:
        log_f.write(f"Command: {shell_cmd}\n")
        log_f.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        log_f.write(f"Using GPU index: {gpu_idx}\n")
        log_f.write(f"Base model repo: {state['base_model_repo_id']}\n\n")
        log_f.flush()
        try:
            ensure_model_available(state["base_model_repo_id"], Path(state["base_model_path"]), log_f)
        except Exception as e:
            log_f.write(f"Base model download failed: {e}\n")
            return f"Could not start training:\n{e}", read_log_tail(log_file_path)
        if os.name == "nt":
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(ROOT),
            )
        else:
            wrapped = (
                "set -o pipefail\n"
                f"cd {shlex.quote(str(ROOT))}\n"
                f"{shell_cmd}\n"
                "exit_code=$?\n"
                "printf '\\nTraining process exited with code %s\\n' \"$exit_code\"\n"
                "exit \"$exit_code\"\n"
            )
            process = subprocess.Popen(
                ["bash", "-lc", wrapped],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(ROOT),
                start_new_session=True,
            )
    state["pid"] = process.pid
    save_job_state(state)
    return training_status_message(state), read_log_tail(log_file_path)


def refresh_training_log() -> tuple[str, str]:
    state = load_job_state()
    return training_status_message(state), read_log_tail(state.get("log_file") if state else None)


def startup_training_log() -> str:
    state = load_job_state()
    return read_log_tail(state.get("log_file") if state else None)


def startup_cloud_status() -> str:
    cfg = load_config()
    lines = []
    image_dir = cfg.get("image_directory", "")
    output_dir = cfg.get("output_directory", "")
    if image_dir or output_dir:
        lines.append("Saved paths loaded.")
        if image_dir:
            lines.append(f"Image Directory: {image_dir}")
        if output_dir:
            lines.append(f"Output Directory: {output_dir}")
    state = load_job_state()
    if state:
        if lines:
            lines.append("")
        lines.append("Last background job:")
        lines.append(f"Project: {state.get('project_name', '')}")
        lines.append(f"Log file: {state.get('log_file', '')}")
        lines.append(f"Output directory: {state.get('output_directory', '')}")
    return "\n".join(lines)


def stop_training() -> tuple[str, str]:
    state = load_job_state()
    pid = state.get("pid") if state else None
    if not is_pid_alive(pid):
        return training_status_message(state), read_log_tail(state.get("log_file") if state else None)
    try:
        if os.name != "nt":
            os.killpg(int(pid), signal.SIGTERM)
        else:
            os.kill(int(pid), signal.SIGTERM)
    except Exception as e:
        return f"Could not stop PID {pid}: {e}", read_log_tail(state.get("log_file"))
    log_file = state.get("log_file")
    try:
        with open(log_file, "a", encoding="utf-8", errors="ignore") as log_f:
            log_f.write("\nStop requested from Gradio UI.\n")
    except Exception:
        pass
    return "Stop requested. Refresh the log in a few seconds.", read_log_tail(log_file)


def build_ui() -> gr.Blocks:
    cfg = load_config()
    saved_gpu_idx = str(cfg.get("gpu_index", "0"))
    default_gpu = next((c for c in GPU_CHOICES if c.startswith(saved_gpu_idx + ":")), GPU_CHOICES[0] if GPU_CHOICES else "0")

    with gr.Blocks(title="NoobAI V-Pred LoRA Trainer") as demo:
        gr.Markdown(
            """
# NoobAI V-Pred LoRA Trainer

Simple Gradio UI for training LoRAs on NoobAI v-pred / SDXL style checkpoints.

This app keeps the cloud dataset upload flow and background training job behavior from the Anima cloud trainer,
but uses the SDXL training script and NoobAI-specific defaults.
"""
        )

        last_train_cfg = gr.State(cfg.get("last_train_config", ""))
        last_dataset_cfg = gr.State(cfg.get("last_dataset_config", ""))

        with gr.Tabs():
            with gr.Tab("Training"):
                with gr.Group():
                    gr.Markdown("### Project And Paths")
                    with gr.Row():
                        project_name = gr.Textbox(
                            label="Project Name",
                            value=cfg["project_name"],
                            placeholder="my_noobai_lora",
                        )
                        gpu_dropdown = gr.Dropdown(
                            label="GPU",
                            choices=GPU_CHOICES,
                            value=default_gpu,
                        )
                    with gr.Row():
                        base_model_dropdown = gr.Dropdown(
                            label="Base Model",
                            choices=list(NOOBAI_MODEL_SPECS.keys()),
                            value=cfg.get("base_model", "anynoobai-v05-vpred-training"),
                        )
                    base_model_notes = gr.Markdown(
                        base_model_note_for(cfg.get("base_model", "anynoobai-v05-vpred-training"))
                    )
                    with gr.Row():
                        image_directory = gr.Textbox(
                            label="Image Directory",
                            value=cfg["image_directory"],
                            placeholder="/path/to/my_dataset",
                        )
                        output_directory = gr.Textbox(
                            label="Output Directory",
                            value=cfg["output_directory"],
                            placeholder="/path/to/output",
                        )

                with gr.Group():
                    gr.Markdown("### Main Settings")
                    with gr.Row():
                        network_dim = gr.Number(label="Network Dim", value=cfg["network_dim"], precision=0)
                        network_alpha = gr.Number(label="Network Alpha", value=cfg["network_alpha"], precision=0)
                        learning_rate = gr.Number(label="UNet LR", value=cfg["learning_rate"])
                    with gr.Row():
                        max_train_epochs = gr.Number(label="Epochs", value=cfg["max_train_epochs"], precision=0)
                        resolution = gr.Number(label="Resolution", value=cfg["resolution"], precision=0)
                        repeats = gr.Number(label="Repeats", value=cfg["repeats"], precision=0)
                        caption_dropout = gr.Number(label="Caption Dropout", value=cfg["caption_dropout"])

                with gr.Group():
                    with gr.Row():
                        configure_btn = gr.Button("Configure Training", variant="primary")
                        train_btn = gr.Button("Start Training", variant="primary")
                        refresh_log_btn = gr.Button("Refresh Log", variant="secondary")
                        stop_train_btn = gr.Button("Stop Training", variant="stop")

                    custom_config_input = gr.Textbox(
                        label="Override Training Config Path",
                        placeholder="Optional: paste a custom training config path",
                    )

                status_box = gr.Textbox(
                    label="Training Status",
                    lines=18,
                    interactive=False,
                    show_copy_button=True,
                )

                job_status_box = gr.Textbox(
                    label="Background Job Status",
                    lines=7,
                    interactive=False,
                    show_copy_button=True,
                    value=training_status_message(),
                )

                log_box = gr.Textbox(
                    label="Training Log",
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                    autoscroll=True,
                    value=startup_training_log(),
                )

            with gr.Tab("Cloud Files"):
                gr.Markdown("Upload your local dataset into this cloud machine, then use the generated paths in the Training tab.")

                with gr.Group():
                    gr.Markdown("### Empty Folders")
                    create_paths_btn = gr.Button("Create Cloud Paths", variant="secondary")

                with gr.Group():
                    gr.Markdown("### Upload ZIP")
                    dataset_zip = gr.File(
                        label="Dataset .zip",
                        file_count="single",
                        file_types=[".zip"],
                        type="filepath",
                    )
                    import_zip_btn = gr.Button("Import ZIP", variant="primary")

                with gr.Group():
                    gr.Markdown("### Upload Folder Or Files")
                    dataset_files = gr.File(
                        label="Images and .txt captions",
                        file_count="directory",
                        file_types=["image", ".txt"],
                        type="filepath",
                    )
                    import_files_btn = gr.Button("Import Uploaded Files", variant="secondary")

                cloud_status = gr.Textbox(
                    label="Cloud File Status",
                    lines=12,
                    interactive=False,
                    show_copy_button=True,
                    value=startup_cloud_status(),
                )

                with gr.Group():
                    gr.Markdown("### Download Results")
                    with gr.Row():
                        download_latest_btn = gr.Button("Download Latest LoRA", variant="primary")
                        package_output_btn = gr.Button("Package Output ZIP", variant="secondary")
                    download_file = gr.File(label="Prepared Download", interactive=False)
                    export_status = gr.Textbox(
                        label="Download Status",
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                    )

            with gr.Tab("Advanced Settings"):
                gr.Markdown(
                    "Defaults are tuned for simple NoobAI v-pred LoRA training.\n\n"
                    "- Recommended checkpoint uses `v_parameterization`\n"
                    "- `noise_offset` and `zero_terminal_snr` are intentionally not used\n"
                    "- UNet-only training is used by default\n"
                    "- Hidden preset defaults include `min_snr_gamma=8` and `multires_noise_iterations=6`"
                )
                with gr.Group():
                    with gr.Row():
                        optimizer_type = gr.Dropdown(
                            label="Optimizer",
                            choices=["AdamW8bit", "AdamW", "Prodigy", "Lion"],
                            value=cfg["optimizer_type"],
                        )
                        lr_scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear"],
                            value=cfg["lr_scheduler"],
                        )
                        lr_scheduler_num_cycles = gr.Number(
                            label="Scheduler Cycles",
                            value=cfg["lr_scheduler_num_cycles"],
                            precision=0,
                        )
                        lr_warmup_steps = gr.Number(
                            label="Warmup Steps",
                            value=cfg["lr_warmup_steps"],
                            precision=0,
                        )
                    with gr.Row():
                        train_batch_size = gr.Number(
                            label="Batch Size",
                            value=cfg["train_batch_size"],
                            precision=0,
                            minimum=1,
                        )
                        gradient_accumulation_steps = gr.Number(
                            label="Gradient Accumulation",
                            value=cfg["gradient_accumulation_steps"],
                            precision=0,
                            minimum=1,
                        )
                        max_grad_norm = gr.Number(label="Max Grad Norm", value=cfg["max_grad_norm"])
                        mixed_precision = gr.Dropdown(
                            label="Mixed Precision",
                            choices=["bf16", "fp16", "no"],
                            value=cfg["mixed_precision"],
                        )
                    with gr.Row():
                        gradient_checkpointing = gr.Checkbox(
                            label="Gradient Checkpointing",
                            value=cfg["gradient_checkpointing"],
                        )
                        cache_latents = gr.Checkbox(label="Cache Latents", value=cfg["cache_latents"])
                        cache_text_encoder_outputs = gr.Checkbox(
                            label="Cache Text Encoder Outputs",
                            value=cfg["cache_text_encoder_outputs"],
                        )
                        seed = gr.Number(label="Seed", value=cfg["seed"], precision=0)
                    with gr.Row():
                        save_every_n_epochs = gr.Number(
                            label="Save Every N Epochs",
                            value=cfg["save_every_n_epochs"],
                            precision=0,
                            minimum=1,
                        )
                        save_last_n_epochs = gr.Number(
                            label="Keep Last N Epochs",
                            value=cfg["save_last_n_epochs"],
                            precision=0,
                            minimum=1,
                        )
                        num_cpu_threads = gr.Number(
                            label="CPU Threads Per Process",
                            value=cfg["num_cpu_threads_per_process"],
                            precision=0,
                            minimum=1,
                        )

                hidden_noise_offset = gr.State(0.0)
                hidden_multires_noise_discount = gr.State(0.3)
                hidden_timestep_sampling = gr.State("unused")
                hidden_discrete_flow_shift = gr.State(1.0)
                hidden_vae_chunk_size = gr.State(0)
                hidden_vae_disable_cache = gr.State(False)

        adv_inputs = [
            optimizer_type,
            lr_scheduler,
            lr_scheduler_num_cycles,
            lr_warmup_steps,
            train_batch_size,
            gradient_accumulation_steps,
            max_grad_norm,
            save_every_n_epochs,
            save_last_n_epochs,
            mixed_precision,
            gradient_checkpointing,
            seed,
            hidden_noise_offset,
            hidden_multires_noise_discount,
            hidden_timestep_sampling,
            hidden_discrete_flow_shift,
            cache_latents,
            cache_text_encoder_outputs,
            hidden_vae_chunk_size,
            hidden_vae_disable_cache,
            num_cpu_threads,
        ]

        basic_inputs = [
            project_name,
            base_model_dropdown,
            image_directory,
            output_directory,
            network_dim,
            network_alpha,
            learning_rate,
            max_train_epochs,
            resolution,
            repeats,
            caption_dropout,
            gpu_dropdown,
        ]

        create_paths_btn.click(
            fn=create_cloud_paths,
            inputs=[project_name],
            outputs=[image_directory, output_directory, cloud_status],
        )

        base_model_dropdown.change(
            fn=base_model_note_for,
            inputs=[base_model_dropdown],
            outputs=[base_model_notes],
        )

        dataset_zip.change(
            fn=summarize_uploaded_zip,
            inputs=[dataset_zip],
            outputs=[cloud_status],
        )

        dataset_files.change(
            fn=summarize_uploaded_files,
            inputs=[dataset_files],
            outputs=[cloud_status],
        )

        import_zip_btn.click(
            fn=import_dataset_zip,
            inputs=[project_name, dataset_zip],
            outputs=[image_directory, output_directory, cloud_status],
        )

        import_files_btn.click(
            fn=import_dataset_files,
            inputs=[project_name, dataset_files],
            outputs=[image_directory, output_directory, cloud_status],
        )

        download_latest_btn.click(
            fn=download_latest_lora,
            inputs=[output_directory],
            outputs=[download_file, export_status],
        )

        package_output_btn.click(
            fn=package_output_zip,
            inputs=[project_name, output_directory],
            outputs=[download_file, export_status],
        )

        configure_btn.click(
            fn=configure_training,
            inputs=basic_inputs + adv_inputs,
            outputs=[status_box, last_train_cfg, last_dataset_cfg],
        )

        train_btn.click(
            fn=start_training,
            inputs=[custom_config_input, gpu_dropdown, num_cpu_threads, base_model_dropdown],
            outputs=[job_status_box, log_box],
        )

        refresh_log_btn.click(
            fn=refresh_training_log,
            inputs=[],
            outputs=[job_status_box, log_box],
        )

        stop_train_btn.click(
            fn=stop_training,
            inputs=[],
            outputs=[job_status_box, log_box],
        )

        demo.load(
            fn=refresh_training_log,
            inputs=[],
            outputs=[job_status_box, log_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
