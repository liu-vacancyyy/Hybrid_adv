#!/usr/bin/env bash
# Diagnose whether the current shell can access NVIDIA CUDA devices.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "== kernel =="
uname -a

echo
echo "== nvidia command paths =="
command -v nvidia-smi || true
command -v nvidia-modprobe || true

echo
echo "== nvidia proc driver =="
cat /proc/driver/nvidia/version 2>/dev/null || echo "no /proc/driver/nvidia/version"
if [[ -d /proc/driver/nvidia/gpus ]]; then
    find /proc/driver/nvidia/gpus -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
fi

echo
echo "== device nodes =="
ls -l /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia* nodes visible"

echo
echo "== mount view of /dev =="
mount | grep ' /dev ' || true

echo
echo "== loaded modules =="
lsmod 2>/dev/null | grep -E '^nvidia|^nouveau' || true

echo
echo "== nvidia-smi =="
nvidia-smi || true

echo
echo "== torch cuda =="
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}", torch.cuda.get_device_name(i))
PY
