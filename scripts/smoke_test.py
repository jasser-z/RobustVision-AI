from pathlib import Path

import numpy as np
from PIL import Image

from robustvision.data.corruptions import apply_corruption
from robustvision.device import select_device
from robustvision.evaluation.metrics import classification_metrics, expected_calibration_error
from robustvision.models.inference import predict_image
from robustvision.models.registry import create_model
from robustvision.reporting.report_generator import generate_report
from robustvision.tracking.experiment_store import ExperimentStore

image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
altered = apply_corruption(image, "gaussian_noise", 2, 42)
model = create_model("baseline_cnn")
result = predict_image(model, image, select_device("cpu"))
metrics = classification_metrics([0, 1], [0, 1])
ece = expected_calibration_error([[0.9, 0.1], [0.2, 0.8]], [0, 1])
root = Path("artifacts/smoke")
store = ExperimentStore(root / "experiments.db")
ident = store.add({"model": "baseline_cnn", "metrics": metrics})
report = generate_report(
    {"model": "baseline_cnn", "device": "cpu", "metrics": metrics}, root / "report.md"
)
print(
    {
        "prediction": result["label"],
        "ece": ece,
        "experiment_id": ident,
        "report": str(report),
        "corrupted_size": altered.size,
    }
)
