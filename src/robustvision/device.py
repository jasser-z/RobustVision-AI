from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_metadata(requested: str = "auto") -> dict:
    d = select_device(requested)
    return {
        "requested": requested,
        "actual": str(d),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(d) if d.type == "cuda" else None,
    }
