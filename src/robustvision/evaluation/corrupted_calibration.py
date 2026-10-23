"""Provenance checks for applying frozen clean-validation temperatures."""

from __future__ import annotations

import json
from pathlib import Path


def load_clean_validation_temperature(path: str | Path, checkpoint_sha256: str) -> float:
    """Return a frozen temperature only when tied to the same clean validation artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validation = payload.get("artifacts", {}).get("validation", {})
    if validation.get("dataset_split") != "validation":
        raise ValueError("Calibration artifact does not establish a validation split")
    if validation.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Calibration artifact checkpoint hash does not match")
    temperature = float(payload.get("temperature", 0))
    if temperature <= 0:
        raise ValueError("Calibration artifact has an invalid frozen temperature")
    return temperature
