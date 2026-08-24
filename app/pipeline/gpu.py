from __future__ import annotations

import os
import shutil
import subprocess
import sys

from app.paths import UNSLOTH_CACHE, ensure_dirs
from app.pipeline.llama_cpp_tools import prepend_windows_cmake


CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu130"
INSTALL_HINT = (
    "PyTorch is CPU-only, so Unsloth cannot see the GPU. "
    "In this venv run: .\\.venv\\Scripts\\pip install --upgrade torch torchvision "
    f"--index-url {CUDA_WHEEL_INDEX}"
)


def nvidia_gpu_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    name = (proc.stdout or "").strip().splitlines()
    return name[0].strip() if name else None


def probe_torch() -> dict[str, str | bool | None]:
    info: dict[str, str | bool | None] = {
        "torch": None,
        "cuda_built": None,
        "cuda_available": False,
        "device_name": None,
        "host_gpu": nvidia_gpu_name(),
        "hint": None,
    }
    try:
        import torch
    except ImportError:
        info["hint"] = (
            "PyTorch is not installed. Install a CUDA build before training: "
            f"pip install torch torchvision --index-url {CUDA_WHEEL_INDEX}"
        )
        return info

    info["torch"] = torch.__version__
    info["cuda_built"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        return info

    host = info["host_gpu"]
    if not info["cuda_built"] or "+cpu" in str(info["torch"]):
        extra = f" nvidia-smi sees {host}." if host else ""
        info["hint"] = INSTALL_HINT + extra
    elif host:
        info["hint"] = (
            f"CUDA PyTorch {info['torch']} is installed but cannot see {host}. "
            "Check NVIDIA drivers and that this process is not forced onto the CPU."
        )
    else:
        info["hint"] = (
            "No CUDA GPU is visible to PyTorch. Unsloth QLoRA needs an NVIDIA GPU."
        )
    return info


def require_cuda() -> dict[str, str | bool | None]:
    info = probe_torch()
    if not info["cuda_available"]:
        raise RuntimeError(str(info["hint"] or INSTALL_HINT))
    return info


def unsloth_worker_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Keep Unsloth kernel compiles out of the uvicorn --reload watch tree."""
    out = os.environ.copy() if env is None else env
    ensure_dirs()
    out.setdefault("PYTHONUNBUFFERED", "1")
    out["PYTHONIOENCODING"] = "utf-8"
    out["PYTHONUTF8"] = "1"
    out["UNSLOTH_COMPILE_LOCATION"] = str(UNSLOTH_CACHE)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(UNSLOTH_CACHE)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    if sys.platform == "win32":
        prepend_windows_cmake(out)
    return out


def spawn_python_worker(args: list[str], env: dict[str, str], **extra) -> subprocess.Popen:
    """Run a repo module as a subprocess; decode Unsloth emoji logs as UTF-8 on Windows."""
    env = unsloth_worker_env(env)
    flags = int(extra.pop("creationflags", 0) or 0)
    if sys.platform == "win32":
        flags |= subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        [sys.executable, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=flags,
        **extra,
    )
