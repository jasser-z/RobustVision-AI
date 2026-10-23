from __future__ import annotations

from pathlib import Path

from robustvision.training.checkpoints import load_checkpoint


def checkpoint_directory() -> Path:
    path = Path("artifacts/checkpoints")
    path.mkdir(parents=True, exist_ok=True)
    return path


def available_checkpoints() -> list[dict]:
    records = []
    for path in sorted(checkpoint_directory().glob("*.pt")):
        sidecar = path.with_suffix(".json")
        records.append(
            {
                "id": path.name,
                "path": str(path),
                "metadata_path": str(sidecar),
                "metadata": sidecar.read_text(encoding="utf-8") if sidecar.exists() else None,
            }
        )
    return records


def get_checkpoint(checkpoint_id: str, device):
    candidate = Path(checkpoint_id)
    if candidate.name != checkpoint_id or candidate.suffix not in {".pt", ".pth"}:
        raise ValueError("Checkpoint identifier is invalid")
    path = checkpoint_directory() / candidate.name
    return load_checkpoint(path, device)
