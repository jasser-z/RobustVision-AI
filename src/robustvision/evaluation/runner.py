from __future__ import annotations

import time

import numpy as np
import torch

from .metrics import classification_metrics, expected_calibration_error


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    labels, predictions, probabilities = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, 1).cpu().numpy()
            probabilities.extend(probs.tolist())
            predictions.extend(probs.argmax(1).tolist())
            labels.extend(targets.tolist())
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(labels, predictions)
    metrics.update(
        {
            "ece": expected_calibration_error(probabilities, labels),
            "brier_score": float(
                np.mean(np.sum((np.asarray(probabilities) - np.eye(10)[labels]) ** 2, axis=1))
            ),
            "elapsed_seconds": elapsed,
            "throughput_images_per_second": len(labels) / elapsed if elapsed else 0.0,
            "probabilities": probabilities,
            "labels": labels,
            "predictions": predictions,
        }
    )
    return metrics
