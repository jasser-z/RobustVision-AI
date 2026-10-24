from __future__ import annotations
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SEVERITY = {"gaussian_noise": [0.04, .07, .10, .15, .20], "gaussian_blur": [.4, .8, 1.2, 1.6, 2.0], "brightness": [.75, .60, .45, .30, .15], "contrast": [.80, .65, .50, .35, .20], "jpeg": [75, 55, 40, 25, 10], "salt_pepper": [.01, .03, .05, .08, .12]}

def _validate(name: str, severity: int) -> None:
    if name not in SEVERITY: raise ValueError(f"Unsupported corruption: {name}")
    if severity not in range(1, 6): raise ValueError("severity must be an integer from 1 to 5")

def apply_corruption(image: Image.Image, name: str, severity: int, seed: int | None = None) -> Image.Image:
    if not isinstance(image, Image.Image): raise TypeError("image must be a Pillow Image")
    _validate(name, severity); value = SEVERITY[name][severity - 1]; rgb = image.convert("RGB")
    if name == "gaussian_blur": return rgb.filter(ImageFilter.GaussianBlur(value))
    if name == "brightness": return ImageEnhance.Brightness(rgb).enhance(value)
    if name == "contrast": return ImageEnhance.Contrast(rgb).enhance(value)
    if name == "jpeg":
        b = BytesIO(); rgb.save(b, format="JPEG", quality=value); b.seek(0)
        with Image.open(b) as out: return out.convert("RGB").copy()
    arr = np.asarray(rgb, dtype=np.float32); rng = np.random.default_rng(seed)
    if name == "gaussian_noise": arr += rng.normal(0, value * 255, arr.shape)
    else:
        mask = rng.random(arr.shape[:2]); arr[mask < value / 2] = 0; arr[(mask >= value / 2) & (mask < value)] = 255
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
