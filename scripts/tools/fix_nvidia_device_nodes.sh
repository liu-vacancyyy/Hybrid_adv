#!/usr/bin/env bash
# Restore NVIDIA /dev nodes on the host, then verify nvidia-smi and PyTorch.
#
# Run this from a normal host terminal, not from the Codex/bwrap sandbox:
#   cd /home/a/demo/Hybrid_adv
#   bash scripts/tools/fix_nvidia_device_nodes.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

in_sandbox=0
if tr '\0' ' ' < /proc/1/cmdline 2>/dev/null | grep -qE 'bwrap|codex-linux-sandbox'; then
    in_sandbox=1
fi
if mount | grep -qE 'tmpfs on /dev type tmpfs .*uid=1000'; then
    in_sandbox=1
fi

if [[ "${in_sandbox}" == "1" ]]; then
    cat >&2 <<'EOF'
This shell looks like the Codex/bwrap sandbox. It can see the NVIDIA kernel
driver through /proc, but GPU device nodes are not bound into /dev here.

Run this script from a normal host terminal instead:
  cd /home/a/demo/Hybrid_adv
  bash scripts/tools/fix_nvidia_device_nodes.sh
EOF
    exit 2
fi

echo "== before =="
bash "$(dirname "$0")/check_gpu_access.sh" || true

echo
echo "== loading kernel modules =="
sudo -v
sudo modprobe nvidia
sudo modprobe nvidia_uvm || true
sudo modprobe nvidia_drm || true

if command -v nvidia-modprobe >/dev/null 2>&1; then
    echo "== using nvidia-modprobe =="
    sudo nvidia-modprobe -u -c=0
else
    echo "== nvidia-modprobe not found; creating device nodes manually =="
    nvidia_major="$(awk '$2 == "nvidia-frontend" {print $1}' /proc/devices)"
    if [[ -z "${nvidia_major}" ]]; then
        nvidia_major=195
    fi
    gpu_count=1
    if [[ -d /proc/driver/nvidia/gpus ]]; then
        gpu_count="$(find /proc/driver/nvidia/gpus -mindepth 1 -maxdepth 1 -type d | wc -l)"
        if [[ "${gpu_count}" -lt 1 ]]; then
            gpu_count=1
        fi
    fi

    sudo rm -f /dev/nvidiactl
    sudo mknod -m 666 /dev/nvidiactl c "${nvidia_major}" 255
    for idx in $(seq 0 $((gpu_count - 1))); do
        sudo rm -f "/dev/nvidia${idx}"
        sudo mknod -m 666 "/dev/nvidia${idx}" c "${nvidia_major}" "${idx}"
    done

    uvm_major="$(awk '$2 == "nvidia-uvm" {print $1}' /proc/devices)"
    if [[ -n "${uvm_major}" ]]; then
        sudo rm -f /dev/nvidia-uvm /dev/nvidia-uvm-tools
        sudo mknod -m 666 /dev/nvidia-uvm c "${uvm_major}" 0
        sudo mknod -m 666 /dev/nvidia-uvm-tools c "${uvm_major}" 1
    else
        echo "warning: nvidia-uvm major not found in /proc/devices" >&2
    fi
fi

echo
echo "== after =="
bash "$(dirname "$0")/check_gpu_access.sh"

if ! nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
nvidia-smi still failed. At this point the issue is not just missing device
nodes. Reinstall or repair the host NVIDIA driver, then reboot:
  sudo ubuntu-drivers devices
  sudo ubuntu-drivers autoinstall
  sudo reboot
EOF
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("nvidia-smi works, but this Python env cannot use CUDA.")
    print("Check that the installed PyTorch build matches the driver/CUDA runtime.")
    sys.exit(1)
print("CUDA OK:", torch.cuda.get_device_name(0))
PY

echo "GPU access is working in this host terminal."
