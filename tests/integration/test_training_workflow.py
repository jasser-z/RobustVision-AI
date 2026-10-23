from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from robustvision.device import select_device
from robustvision.evaluation.robustness import benchmark
from robustvision.evaluation.runner import evaluate
from robustvision.explainability.gradcam import gradcam
from robustvision.models.registry import create_model
from robustvision.training.checkpoints import load_checkpoint, save_checkpoint
from robustvision.training.trainer import train


def test_train_checkpoint_reload_evaluate_and_gradcam(tmp_path: Path):
    torch.manual_seed(5)
    features = torch.randn(12, 3, 32, 32)
    labels = torch.tensor([0, 1] * 6)
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)
    device = select_device("cpu")
    model = create_model("baseline_cnn").to(device)
    summary, _ = train(model, loader, loader, device, 1, 0.001, 0.0)
    path = save_checkpoint(tmp_path / "model.pt", model, {"model": "baseline_cnn", "quick": True})
    reloaded, metadata = load_checkpoint(path, device)
    metrics = evaluate(reloaded, loader, device)
    image = Image.new("RGB", (32, 32))
    explanation = gradcam(reloaded, image, device)
    assert metadata["model"] == "baseline_cnn"
    assert summary["history"] and metrics["samples"] == 12
    assert explanation.size == image.size


def test_robustness_benchmark_schema():
    class RawImages:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return Image.new("RGB", (32, 32), color=(index, 2, 3)), index % 2

    rows = benchmark(
        create_model("baseline_cnn"),
        RawImages(),
        select_device("cpu"),
        2,
        ["jpeg"],
        [1],
        42,
    )
    assert len(rows) == 2
    assert {"accuracy", "accuracy_drop", "samples"}.issubset(rows[1])
