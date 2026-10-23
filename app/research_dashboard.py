"""Checkpoint-first Streamlit interface. Every chart reads saved local artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from robustvision.data.corruptions import SEVERITY, apply_corruption
from robustvision.device import device_metadata, select_device
from robustvision.explainability.gradcam import gradcam
from robustvision.models.inference import predict_image
from robustvision.services import available_checkpoints, get_checkpoint
from robustvision.tracking.experiment_store import ExperimentStore

st.set_page_config(page_title="RobustVision AI", layout="wide")
st.title("RobustVision AI")
st.caption("Results are shown only when a saved local experiment/checkpoint exists.")
st.sidebar.json(device_metadata())
page = st.sidebar.radio(
    "Research workspace", ["Overview", "Image laboratory", "Robustness", "Experiments & reports"]
)
records = available_checkpoints()

if page == "Overview":
    st.subheader("Experiment status")
    st.metric("Saved trained checkpoints", len(records))
    st.write(
        "Use `scripts/train_model.py` to create a checkpoint, then `scripts/benchmark.py` to populate measured robustness results."
    )
    if records:
        st.dataframe(pd.DataFrame(records)[["id", "path"]], width="stretch")
        calibration = []
        for path in Path("artifacts/calibration").glob("*/calibration.json"):
            stored = json.loads(path.read_text(encoding="utf-8"))
            for state, metrics in stored["metrics"]["test"].items():
                calibration.append(
                    {
                        "checkpoint": path.parent.name,
                        "state": state,
                        "ECE (10 bins)": metrics["ece"],
                        "Brier score": metrics["brier_score"],
                        "NLL": metrics["nll"],
                        "temperature": metrics["temperature"],
                    }
                )
        if calibration:
            st.subheader("Saved clean-test calibration metrics")
            st.dataframe(pd.DataFrame(calibration), width="stretch")
            st.caption(
                "Temperature is fitted only on validation data. Corruption-aware results, when saved below, use that frozen clean-validation temperature."
            )
        corrupted_summary = Path(
            "artifacts/calibration_corrupted/corrupted_calibration_aggregate.csv"
        )
        if corrupted_summary.exists():
            st.subheader("Saved corruption-aware calibration")
            st.caption(
                "Metrics use frozen clean-validation temperatures across five corruptions at severities 1, 3, and 5."
            )
            corrupted_frame = pd.read_csv(corrupted_summary)
            st.dataframe(corrupted_frame, width="stretch")
            st.plotly_chart(
                px.line(
                    corrupted_frame,
                    x="severity",
                    y="ece",
                    color="model",
                    line_dash="state",
                    markers=True,
                    title="Mean corruption ECE by severity",
                ),
                width="stretch",
            )
            st.download_button(
                "Download corrupted calibration summary",
                corrupted_summary.read_bytes(),
                file_name="corrupted_calibration_aggregate.csv",
                mime="text/csv",
                key="download_corrupted_calibration_summary",
            )
        else:
            st.info(
                "No corruption-aware calibration summary is available. "
                "Run scripts/evaluate_corrupted_calibration.py after clean calibration."
            )

elif page == "Image laboratory":
    if not records:
        st.warning(
            "No trained checkpoint is available. Train one first; untrained inference is intentionally not offered here."
        )
    else:
        chosen = st.selectbox("Trained checkpoint", [x["id"] for x in records])
        uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        corruption = st.selectbox("Corruption", list(SEVERITY))
        severity = st.slider("Severity", 1, 5, 1)
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            altered = apply_corruption(image, corruption, severity, seed=42)
            a, b = st.columns(2)
            a.image(image, caption="Clean")
            b.image(altered, caption=f"{corruption}, severity {severity}")
            if st.button("Predict and explain"):
                device = select_device()
                model, metadata = get_checkpoint(chosen, device)
                before, after = (
                    predict_image(model, image, device),
                    predict_image(model, altered, device),
                )
                st.json(
                    {
                        "checkpoint": chosen,
                        "metadata": metadata,
                        "clean": before,
                        "corrupted": after,
                    }
                )
                st.image(
                    gradcam(model, image, device),
                    caption="Grad-CAM (prediction-specific; explanation, not causation)",
                )

elif page == "Robustness":
    path = Path("artifacts/analysis/validated_rows.csv")
    if not path.exists():
        st.info(
            "No validated multi-model analysis exists. Run scripts/analyze_benchmarks.py first."
        )
    else:
        frame = pd.read_csv(path)
        clean = frame[frame.corruption == "clean"]
        selected_models = st.multiselect(
            "Models", sorted(frame.model.unique()), default=sorted(frame.model.unique())
        )
        if not selected_models:
            st.info("Select at least one model to view saved benchmark results.")
            st.stop()
        frame = frame[frame.model.isin(selected_models)]
        selected_corruption = st.selectbox(
            "Corruption", sorted(frame.loc[frame.corruption != "clean", "corruption"].unique())
        )
        selected = frame[(frame.corruption == selected_corruption) | (frame.corruption == "clean")]
        if not clean.empty:
            metrics = clean.set_index("model").loc[selected_models, "accuracy"]
            st.caption(
                "Clean accuracy: "
                + ", ".join(f"{name}: {value:.2%}" for name, value in metrics.items())
            )
        st.dataframe(frame, width="stretch")
        st.plotly_chart(
            px.line(
                selected[selected.corruption != "clean"],
                x="severity",
                y="accuracy",
                color="model",
                markers=True,
                title=f"Measured accuracy: {selected_corruption.replace('_', ' ')}",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.line(
                frame[frame.corruption != "clean"],
                x="severity",
                y="accuracy_drop",
                color="model",
                facet_col="corruption",
                markers=True,
                title="Accuracy drop from clean baseline",
            ),
            width="stretch",
        )
        summary_path = Path("artifacts/analysis/model_summary.csv")
        if summary_path.exists():
            st.subheader("Clean vs mean corrupted accuracy")
            st.bar_chart(
                pd.read_csv(summary_path).set_index("model")[
                    ["clean_accuracy", "mean_corrupted_accuracy"]
                ]
            )
        st.download_button(
            "Download validated benchmark rows",
            path.read_bytes(),
            file_name="validated_rows.csv",
            mime="text/csv",
            key="download_validated_benchmark_rows",
        )

else:
    database = Path("artifacts/benchmarks/experiments.db")
    if database.exists():
        st.dataframe(pd.DataFrame(ExperimentStore(database).list()), width="stretch")
    else:
        st.info("No persisted benchmark experiments yet.")
    for report in Path("artifacts").rglob("*.md"):
        st.download_button(
            f"Download {report.name}",
            report.read_bytes(),
            file_name=report.name,
            mime="text/markdown",
            key=f"download_report_{report.relative_to('artifacts').as_posix()}",
        )
