#!/usr/bin/env bash
# Run ON the GPU VM after SSH login (or via gcloud compute ssh ... -- bash -s).
set -euo pipefail

PDPO_ROOT="${PDPO_ROOT:-${HOME}/PDPO}"
REPO_URL="${PDPO_REPO_URL:-https://github.com/jennazhao7/PDPO.git}"
CONDA_ENV="${CONDA_ENV:-pdpo}"

echo "=== PDPO GCP bootstrap ==="
echo "PDPO_ROOT=${PDPO_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. Drivers may still be installing; wait 2-5 min and retry."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

if [[ ! -d "${PDPO_ROOT}/.git" ]]; then
  echo "Cloning repo to ${PDPO_ROOT}..."
  git clone "${REPO_URL}" "${PDPO_ROOT}"
else
  echo "Repo exists at ${PDPO_ROOT}; pulling latest..."
  git -C "${PDPO_ROOT}" pull --ff-only || true
fi

cd "${PDPO_ROOT}"
export HF_HOME="${PDPO_ROOT}/.cache/huggingface"
mkdir -p "${HF_HOME}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

# Accept conda TOS non-interactively (required on newer miniconda)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Creating conda env ${CONDA_ENV}..."
  conda create -n "${CONDA_ENV}" python=3.10 -y
fi

conda activate "${CONDA_ENV}"
pip install -U pip
pip install -r "${PDPO_ROOT}/requirements.txt"
pip install rich

echo ""
echo "Checking PyTorch CUDA..."
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  pip install huggingface_hub
fi

if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
  echo ""
  echo "HuggingFace login required for gated models:"
  echo "  huggingface-cli login"
  echo "  or: export HF_TOKEN=... before re-running bootstrap"
fi

cat >> "${HOME}/.bashrc" <<EOF

# PDPO GCP (added by bootstrap_vm.sh)
export PDPO_ROOT="${PDPO_ROOT}"
export HF_HOME="${HF_HOME}"
alias pdpo='cd "\${PDPO_ROOT}" && conda activate ${CONDA_ENV}'
EOF

echo ""
echo "=== Bootstrap complete ==="
echo "  cd ${PDPO_ROOT} && conda activate ${CONDA_ENV}"
echo "  GPU=0 bash experiments/quick_sb_test.sh"
