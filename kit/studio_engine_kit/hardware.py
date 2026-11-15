# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Hardware ids (studio-spec src/hardware.rs) and the launch-time device check.

The host detects the hardware and passes its id in ALLTERNIT_STUDIO_HARDWARE;
the engine never classifies the computer itself. What the engine still checks
is whether the device the host named can be used right now (the NVIDIA driver
answers): when it cannot, a job is refused with 503 `hardware_unavailable`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional

HARDWARE_ENV = "ALLTERNIT_STUDIO_HARDWARE"

# In the order of Hardware::ALL.
HARDWARE_IDS = ("apple-silicon", "cuda-linux", "cuda-windows", "rocm-linux", "xpu-linux", "xpu-windows", "cpu")

LABELS: Dict[str, str] = {
    "apple-silicon": "a Mac with Apple Silicon",
    "cuda-linux": "Linux with an NVIDIA GPU",
    "cuda-windows": "Windows with an NVIDIA GPU",
    "rocm-linux": "Linux with an AMD GPU",
    "xpu-linux": "Linux with an Intel GPU",
    "xpu-windows": "Windows with an Intel GPU",
    "cpu": "the CPU alone",
}

TORCH_BACKEND: Dict[str, str] = {
    "apple-silicon": "torch-mps",
    "cuda-linux": "torch-cuda", "cuda-windows": "torch-cuda",
    "rocm-linux": "torch-rocm",
    "xpu-linux": "torch-xpu", "xpu-windows": "torch-xpu",
    "cpu": "torch-cpu",
}

# The torch device string a trainer config names for each backend. ROCm builds
# of PyTorch expose the GPU as "cuda".
TORCH_DEVICE: Dict[str, str] = {
    "torch-mps": "mps", "torch-cuda": "cuda", "torch-rocm": "cuda", "torch-xpu": "xpu", "torch-cpu": "cpu",
}


def from_env(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The hardware id the host passed, or None when it is missing or unknown."""
    value = (env if env is not None else os.environ).get(HARDWARE_ENV, "").strip()
    return value if value in HARDWARE_IDS else None


def device_unavailable(hardware: Optional[str]) -> Optional[str]:
    """Why the device for `hardware` cannot be used right now, or None.

    Only NVIDIA has a cheap, dependable check (nvidia-smi ships with the
    driver on Linux and Windows). Other devices are left to the trainer, which
    fails with its own message if the device is gone.
    """
    if hardware is None:
        return (f"{HARDWARE_ENV} is not set to a hardware id ({', '.join(HARDWARE_IDS)}); "
                "the host passes the id it detected when it starts the engine.")
    if hardware not in ("cuda-linux", "cuda-windows"):
        return None
    smi = shutil.which("nvidia-smi")
    if not smi:
        if os.path.exists("/proc/driver/nvidia/version"):
            return None
        return "the host reported an NVIDIA GPU, but nvidia-smi is missing: the NVIDIA driver is not loaded."
    try:
        out = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the host reported an NVIDIA GPU, but nvidia-smi did not answer: {exc}"
    if out.returncode != 0 or not any(line.strip() for line in out.stdout.splitlines()):
        return "the host reported an NVIDIA GPU, but nvidia-smi lists none (driver not loaded or device lost)."
    return None
