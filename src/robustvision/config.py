from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    device: str = "auto"
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    mlflow_uri: str = "file:./mlruns"
    max_upload_bytes: int = 5 * 1024 * 1024
    cors_origins: tuple[str, ...] = ("http://localhost:8501",)


def get_settings() -> Settings:
    origins = tuple(
        x.strip()
        for x in os.getenv("ROBUSTVISION_CORS_ORIGINS", "http://localhost:8501").split(",")
        if x.strip()
    )
    return Settings(
        device=os.getenv("ROBUSTVISION_DEVICE", "auto"),
        data_dir=Path(os.getenv("ROBUSTVISION_DATA_DIR", "data")),
        artifact_dir=Path(os.getenv("ROBUSTVISION_ARTIFACT_DIR", "artifacts")),
        mlflow_uri=os.getenv("ROBUSTVISION_MLFLOW_URI", "file:./mlruns"),
        max_upload_bytes=int(os.getenv("ROBUSTVISION_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        cors_origins=origins,
    )


def load_experiment_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Experiment configuration must be a mapping")
    return data
