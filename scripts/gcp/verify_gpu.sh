#!/usr/bin/env bash
# Fast GPU sanity check (no training data required).
set -euo pipefail

echo "=== GPU sanity check ==="
nvidia-smi

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
name = torch.cuda.get_device_name(0)
print("device:", name)
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("matmul ok:", y.shape, y.dtype)
print("GPU sanity check PASSED")
PY
