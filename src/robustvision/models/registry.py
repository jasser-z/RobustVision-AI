from __future__ import annotations

import torch.nn as nn

from .cnn import BaselineCNN


def create_model(name: str, pretrained: bool = False) -> nn.Module:
    if name == "baseline_cnn":
        return BaselineCNN()
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    raise ValueError(f"Unknown model: {name}")


def model_metadata(name: str) -> dict:
    model = create_model(name)
    return {
        "id": name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "classes": 10,
        "checkpoint_required": True,
    }
