from __future__ import annotations

import streamlit as st
from PIL import Image

from robustvision.data.corruptions import SEVERITY, apply_corruption
from robustvision.device import device_metadata, select_device
from robustvision.models.inference import predict_image
from robustvision.models.registry import create_model

st.set_page_config(page_title="RobustVision AI", layout="wide")
st.title("RobustVision AI")
st.caption(
    "Reproducible computer-vision reliability evaluation. Predictions below use untrained model weights unless you load a trained checkpoint."
)
st.sidebar.json(device_metadata())
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
model_name = st.selectbox("Model", ["baseline_cnn", "resnet18"])
corruption = st.selectbox("Corruption", list(SEVERITY))
severity = st.slider("Severity", 1, 5, 1)
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    altered = apply_corruption(image, corruption, severity, seed=42)
    left, right = st.columns(2)
    left.image(image, caption="Original")
    right.image(altered, caption=f"{corruption}, severity {severity}")
    if st.button("Run local inference"):
        try:
            device = select_device()
            model = create_model(model_name).to(device)
            before = predict_image(model, image, device)
            after = predict_image(model, altered, device)
            st.json(
                {
                    "before": before,
                    "after": after,
                    "warning": "No checkpoint was loaded; results are not meaningful accuracy claims.",
                }
            )
        except Exception as exc:
            st.error(str(exc))
else:
    st.info(
        "Upload an image to start the Image Laboratory. Run benchmarks from scripts/benchmark.py after training/loading a checkpoint."
    )
