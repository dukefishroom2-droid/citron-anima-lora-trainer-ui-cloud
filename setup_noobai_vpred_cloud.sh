#!/usr/bin/env bash
# Fast ephemeral setup for NoobAI V-Pred LoRA training on GPU cloud.

set -Eeuo pipefail

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
START_UI="${START_UI:-1}"
DETACH="${DETACH:-1}"
APT_INSTALL="${APT_INSTALL:-auto}"
FORCE_RTX5000="${FORCE_RTX5000:-auto}"
REPO_URL="${REPO_URL:-https://github.com/dukefishroom2-droid/citron-anima-lora-trainer-ui-cloud.git}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME}"
APP_DIR="${APP_DIR:-$INSTALL_ROOT/citron-anima-lora-trainer-ui}"
NOOBAI_BASE_MODEL_KEY="${NOOBAI_BASE_MODEL_KEY:-anynoobai-v05-vpred-training}"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-false}"
PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"

log() {
  printf "\n[%s] %s\n" "$(date +'%H:%M:%S')" "$*"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

die() {
  printf "\nERROR: %s\n" "$*" >&2
  exit 1
}

sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    die "Need root or sudo to install OS packages."
  fi
}

select_noobai_model() {
  case "$NOOBAI_BASE_MODEL_KEY" in
    anynoobai-v05-vpred-training)
      export NOOBAI_REPO_ID="John6666/anynoobai-for-lora-training-v05vprediction-sdxl"
      export NOOBAI_MODEL_DIR="models/noobai/anynoobai-for-lora-training-v05vprediction-sdxl"
      ;;
    noobai-xl-vpred-1.0)
      export NOOBAI_REPO_ID="Laxhar/noobai-XL-Vpred-1.0"
      export NOOBAI_MODEL_DIR="models/noobai/noobai-XL-Vpred-1.0"
      ;;
    *)
      die "Unknown NOOBAI_BASE_MODEL_KEY: $NOOBAI_BASE_MODEL_KEY"
      ;;
  esac
}

maybe_install_os_packages() {
  if [[ "$APT_INSTALL" == "0" || "$APT_INSTALL" == "false" ]]; then
    log "Skipping apt package install because APT_INSTALL=$APT_INSTALL"
    return
  fi
  if ! have apt-get; then
    log "apt-get not found; assuming base image already has git, curl, Python, and build tools."
    return
  fi
  if [[ "$APT_INSTALL" == "auto" ]]; then
    if have curl && have wget && have git && have unzip && have python3 \
      && have gcc && have g++ && have make && have pkg-config; then
      log "Basic OS packages already present; skipping apt install"
      return
    fi
  fi
  log "Installing basic OS packages"
  sudo_cmd apt-get update
  sudo_cmd apt-get install -y --no-install-recommends \
    ca-certificates curl wget git git-lfs unzip \
    python3 python3-venv python3-pip \
    build-essential pkg-config
  sudo_cmd apt-get clean
}

clone_repo() {
  mkdir -p "$INSTALL_ROOT"
  if [[ -d "$APP_DIR/.git" ]]; then
    log "Trainer repo already exists at $APP_DIR; updating"
    git -C "$APP_DIR" pull --ff-only || true
  elif [[ -e "$APP_DIR" ]]; then
    die "$APP_DIR exists but is not a git checkout."
  else
    log "Cloning trainer repo into $APP_DIR"
    git clone --depth 1 "$REPO_URL" "$APP_DIR"
  fi
}

is_rtx_50_series() {
  have nvidia-smi || return 1
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
    | grep -Eiq 'RTX[ -]*(5060|5070|5080|5090)|GeForce.*(5060|5070|5080|5090)'
}

run_setup() {
  cd "$APP_DIR"
  local setup_script="setup_for_linux_noobai.sh"
  if [[ "$FORCE_RTX5000" == "1" || "$FORCE_RTX5000" == "true" ]]; then
    setup_script="setup_for_linux_noobai_rtx5000.sh"
  elif [[ "$FORCE_RTX5000" == "auto" ]] && is_rtx_50_series; then
    setup_script="setup_for_linux_noobai_rtx5000.sh"
  fi
  log "Running NoobAI dependency setup: $setup_script"
  PIP_NO_CACHE_DIR="$PIP_NO_CACHE_DIR" \
  PIP_DISABLE_PIP_VERSION_CHECK="$PIP_DISABLE_PIP_VERSION_CHECK" \
  NOOBAI_REPO_ID="$NOOBAI_REPO_ID" \
  NOOBAI_MODEL_DIR="$NOOBAI_MODEL_DIR" \
  bash "$setup_script"
}

print_access_hints() {
  printf "\n"
  printf "Trainer directory: %s\n" "$APP_DIR"
  printf "Internal URL:      http://%s:%s\n" "$HOST" "$PORT"
  printf "Log file:          %s/logs/cloud-ui-noobai.log\n" "$APP_DIR"
  local vast_var="VAST_TCP_PORT_${PORT}"
  if [[ -n "${!vast_var:-}" ]]; then
    printf "Vast mapped port:  internal %s -> external %s\n" "$PORT" "${!vast_var}"
  else
    printf "Vast: create the instance with Docker options: -p %s:%s -e OPEN_BUTTON_PORT=%s\n" "$PORT" "$PORT" "$PORT"
  fi
}

start_ui() {
  if [[ "$START_UI" == "0" || "$START_UI" == "false" ]]; then
    print_access_hints
    return
  fi
  cd "$APP_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export GRADIO_SERVER_NAME="$HOST"
  export GRADIO_SERVER_PORT="$PORT"
  mkdir -p logs
  log "Starting NoobAI V-Pred LoRA Trainer on $HOST:$PORT"
  if [[ "$DETACH" == "1" || "$DETACH" == "true" ]]; then
    nohup python app_noobai.py > logs/cloud-ui-noobai.log 2>&1 &
    printf "Started PID %s\n" "$!"
    sleep 2
    tail -n 40 logs/cloud-ui-noobai.log || true
    print_access_hints
  else
    print_access_hints
    python app_noobai.py
  fi
}

main() {
  select_noobai_model
  maybe_install_os_packages
  clone_repo
  run_setup
  start_ui
}

main "$@"
