from __future__ import annotations

from pathlib import Path

import torch

from robustvision.models.registry import create_model


def save_checkpoint(path: str | Path, model: torch.nn.Module, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "state_dict": model.state_dict(), "metadata": metadata}
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load only project-generated state dict checkpoints; never load user-supplied uploads."""
    path = Path(path)
    if not path.is_file() or path.suffix not in {".pt", ".pth"}:
        raise FileNotFoundError("Checkpoint is not an approved local .pt/.pth file")
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload.get("metadata", {})
    name = metadata.get("model")
    if name not in {"baseline_cnn", "resnet18"}:
        raise ValueError("Checkpoint metadata has an unsupported model identifier")
    model = create_model(name, pretrained=False).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata
