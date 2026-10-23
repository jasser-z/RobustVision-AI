"""Clean-split calibration metrics and validation-only temperature scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from .metrics import expected_calibration_error


def _arrays(logits, labels) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must be a nonempty [examples, classes] array")
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("labels must have one value per logits row")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= logits.shape[1]:
        raise ValueError("labels are outside the logits class range")
    return logits, labels


def probabilities_from_logits(logits, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    shifted = logits / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibration_metrics(logits, labels, bins: int = 10, temperature: float = 1.0) -> dict:
    logits, labels = _arrays(logits, labels)
    probabilities = probabilities_from_logits(logits, temperature)
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1.0)).mean()
    one_hot = np.eye(probabilities.shape[1])[labels]
    return {
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "nll": float(nll),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(probabilities, labels, bins=bins),
        "samples": int(len(labels)),
        "bins": bins,
        "temperature": float(temperature),
    }


def reliability_bins(logits, labels, bins: int = 10, temperature: float = 1.0) -> list[dict]:
    logits, labels = _arrays(logits, labels)
    probabilities = probabilities_from_logits(logits, temperature)
    confidence, predicted = probabilities.max(axis=1), probabilities.argmax(axis=1)
    correct = predicted == labels
    edges = np.linspace(0, 1, bins + 1)
    result = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= low) & (
            (confidence < high) if index < bins - 1 else (confidence <= high)
        )
        result.append(
            {
                "bin_lower": float(low),
                "bin_upper": float(high),
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
                "confidence": float(confidence[mask].mean()) if mask.any() else None,
            }
        )
    return result


def fit_temperature(validation_logits, validation_labels, max_iterations: int = 100) -> float:
    """Fit one positive scalar strictly on validation logits and labels."""
    logits, labels = _arrays(validation_logits, validation_labels)
    logits_t = torch.tensor(logits, dtype=torch.float64)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=max_iterations, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = functional.cross_entropy(logits_t / log_temperature.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Temperature fitting produced an invalid result")
    return temperature
