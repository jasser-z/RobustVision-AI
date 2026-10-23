from __future__ import annotations

import time

import torch
from PIL import Image
from torchvision.transforms import v2

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
TRANSFORM = v2.Compose(
    [
        v2.Resize((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)


def predict_image(model: torch.nn.Module, image: Image.Image, device: torch.device) -> dict:
    model.eval()
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        started = time.perf_counter()
        logits = model(tensor)
        elapsed = (time.perf_counter() - started) * 1000
        probabilities = torch.softmax(logits, 1)[0].cpu()
    idx = int(probabilities.argmax())
    return {
        "class_id": idx,
        "label": CLASSES[idx],
        "confidence": float(probabilities[idx]),
        "latency_ms": elapsed,
        "probabilities": [float(x) for x in probabilities],
    }
