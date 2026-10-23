# Architecture

The Streamlit UI is an interactive client, FastAPI exposes validated programmatic operations, and `src/robustvision` holds the reusable ML/evaluation logic. This separation permits the same corruption and inference code to be exercised by tests, scripts, API, and UI. SQLite stores compact experiment metadata; reports are immutable Markdown artifacts. MLflow is configured as optional local file-backed tracking through `ROBUSTVISION_MLFLOW_URI`.
