from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from robustvision.models.inference import TRANSFORM


def _target_layer(model):
    if hasattr(model, "features"):
        return model.features[-3]
    if hasattr(model, "layer4"):
        return model.layer4[-1].conv2
    raise ValueError("Grad-CAM target layer unavailable")


def gradcam(
    model: torch.nn.Module, image: Image.Image, device: torch.device, class_id: int | None = None
) -> Image.Image:
    activations, gradients = [], []
    layer = _target_layer(model)
    hooks = [
        layer.register_forward_hook(lambda _, __, out: activations.append(out)),
        layer.register_full_backward_hook(lambda _, gin, gout: gradients.append(gout[0])),
    ]
    try:
        model.eval()
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
        logits = model(tensor)
        target = int(logits.argmax(1).item()) if class_id is None else class_id
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(1))[0]
        cam = (
            torch.nn.functional.interpolate(
                cam[None, None], size=image.size[::-1], mode="bilinear", align_corners=False
            )[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        base = np.asarray(image.convert("RGB"), dtype=np.float32)
        heat = np.zeros_like(base)
        heat[..., 0] = cam * 255
        heat[..., 1] = (1 - cam) * 70
        return Image.fromarray(np.uint8(0.55 * base + 0.45 * heat))
    finally:
        for hook in hooks:
            hook.remove()
