from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_logits(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits.append(model(images.to(device)).cpu().numpy().astype(np.float32))
            labels.append(targets.numpy().astype(np.int64))
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    indices = np.asarray(
        getattr(loader.dataset, "indices", range(len(loader.dataset))), dtype=np.int64
    )
    if len(indices) != len(labels):
        raise ValueError("Stable example indices do not match collected labels")
    return logits, labels, indices


def software_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def artifact_metadata(
    checkpoint: str | Path,
    dataset_split: str,
    seed: int,
    preprocessing: dict,
) -> dict:
    checkpoint = Path(checkpoint)
    if dataset_split not in {"validation", "test"}:
        raise ValueError("dataset_split must be validation or test")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_split": dataset_split,
        "seed": seed,
        "preprocessing": preprocessing,
        "software": software_metadata(),
    }


def save_prediction_archive(
    path: str | Path,
    logits: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    **metadata,
) -> None:
    """Persist prediction arrays plus scalar provenance without pickle serialization."""
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    np.savez_compressed(
        path,
        logits=logits.astype(np.float32),
        probabilities=probabilities.astype(np.float32),
        labels=labels.astype(np.int64),
        predictions=probabilities.argmax(axis=1).astype(np.int64),
        indices=indices.astype(np.int64),
        **metadata,
    )
