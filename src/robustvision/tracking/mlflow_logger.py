from __future__ import annotations

from pathlib import Path


def log_run(
    name: str,
    params: dict,
    metrics: dict,
    artifacts: list[str | Path] | None = None,
    uri: str = "file:./mlruns",
) -> str | None:
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            for artifact in artifacts or []:
                if Path(artifact).exists():
                    mlflow.log_artifact(str(artifact))
            return run.info.run_id
    except Exception:
        return None
