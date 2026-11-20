"""Windows CUDA와 Mac MPS를 같은 규칙으로 고른다.

비어 있는 device는 CUDA만 자동 선택하고 Apple GPU(MPS)는 CPU로 떨어진다.
그래서 학습·추론 모두 여기서 장치를 정해 넘긴다.
"""

from __future__ import annotations

import os

_DEVICE: str | None = None


def configure_runtime() -> None:
    """torch를 불러오기 전에 Mac GPU 폴백과 학습 그래프 백엔드를 고정한다."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")


configure_runtime()


def yolo_device() -> str:
    """ultralytics device 값. CUDA면 '0', Apple GPU면 'mps', 없으면 'cpu'."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        _DEVICE = "0"
        return _DEVICE

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_built() and mps.is_available():
        _DEVICE = "mps"
        return _DEVICE

    _DEVICE = "cpu"
    return _DEVICE


def gpu_file_slug() -> str:
    """파일 이름에 넣을 그래픽카드 이름. 예: m1-pro, rtx-3060-ti."""
    device = yolo_device()
    if device == "cpu":
        raw = "cpu"
    elif device == "mps":
        raw = _apple_chip() or "mps"
    else:
        import torch

        raw = torch.cuda.get_device_name(int(device))
    return _file_slug(raw)


def _apple_chip() -> str:
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return ""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _file_slug(name: str) -> str:
    skip = {"nvidia", "geforce", "apple", "gpu"}
    parts: list[str] = []
    token: list[str] = []
    for char in name.lower():
        if char.isalnum():
            token.append(char)
            continue
        if token:
            parts.append("".join(token))
            token = []
    if token:
        parts.append("".join(token))
    kept = [part for part in parts if part not in skip]
    return "-".join(kept) or "gpu"


def device_label() -> str:
    device = yolo_device()
    if device == "mps":
        return "Apple GPU (MPS)"
    if device == "cpu":
        return "CPU"
    import torch

    index = int(device)
    return f"CUDA:{index} ({torch.cuda.get_device_name(index)})"


def use_amp(device: str | None = None) -> bool:
    """CUDA에서만 AMP를 켠다. MPS·CPU는 ultralytics도 AMP를 끄며, 켜면 학습이 깨진다."""
    device = yolo_device() if device is None else device
    return device not in {"cpu", "mps"}
