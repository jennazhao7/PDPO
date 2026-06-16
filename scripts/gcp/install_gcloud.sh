#!/usr/bin/env bash
# Ensure gcloud works on macOS with system Python 3.9.
set -euo pipefail

if command -v gcloud >/dev/null 2>&1 && gcloud --version >/dev/null 2>&1; then
  exit 0
fi

GCLOUD_DIR="${HOME}/google-cloud-sdk"
if [[ ! -x "${GCLOUD_DIR}/bin/gcloud" ]]; then
  ARCH="$(uname -m)"
  if [[ "${ARCH}" == "arm64" ]]; then
    PKG="google-cloud-cli-darwin-arm.tar.gz"
  else
    PKG="google-cloud-cli-darwin-x86_64.tar.gz"
  fi
  echo "Downloading Google Cloud SDK..."
  curl -fsSL "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${PKG}" -o /tmp/gcloud.tgz
  mkdir -p "${GCLOUD_DIR}"
  tar -xzf /tmp/gcloud.tgz -C "${GCLOUD_DIR}" --strip-components=1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install 3.12
export CLOUDSDK_PYTHON="$(uv python find 3.12)"
export PATH="${GCLOUD_DIR}/bin:${PATH}"

echo "gcloud ready: $(gcloud --version | head -1)"
echo "Add to ~/.zshrc:"
echo "  export CLOUDSDK_PYTHON=${CLOUDSDK_PYTHON}"
echo "  export PATH=${GCLOUD_DIR}/bin:\$PATH"
