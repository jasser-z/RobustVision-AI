import numpy as np
import pytest
import torch

from robustvision.evaluation.calibration import (
    calibration_metrics,
    fit_temperature,
    probabilities_from_logits,
    reliability_bins,
)
from robustvision.evaluation.prediction_artifacts import artifact_metadata, collect_logits


def test_temperature_scaling_is_positive_and_improves_overconfident_validation_nll():
    logits = np.array([[8.0, 0.0], [7.0, 0.0], [0.0, 8.0], [0.0, 7.0]])
    labels = np.array([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    before = calibration_metrics(logits, labels)
    after = calibration_metrics(logits, labels, temperature=temperature)
    assert temperature > 0
    assert after["nll"] < before["nll"]


def test_calibration_metrics_and_empty_reliability_bins_are_safe():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 1])
    metrics = calibration_metrics(logits, labels, bins=5)
    bins = reliability_bins(logits, labels, bins=5)
    assert metrics["accuracy"] == 1.0
    assert metrics["samples"] == 2
    assert len(bins) == 5
    assert any(row["count"] == 0 and row["accuracy"] is None for row in bins)


@pytest.mark.parametrize("temperature", [0, -1, float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        probabilities_from_logits(np.array([[1.0, 0.0]]), temperature)


def test_malformed_logits_and_labels_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        calibration_metrics(np.empty((0, 2)), np.array([], dtype=int))
    with pytest.raises(ValueError, match="one value"):
        calibration_metrics(np.array([[1.0, 0.0]]), np.array([0, 1]))


def test_prediction_artifact_metadata_and_stable_indices(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    metadata = artifact_metadata(checkpoint, "validation", 42, {"normalization": "fixture"})
    assert metadata["checkpoint_sha256"] and metadata["dataset_split"] == "validation"
    model = torch.nn.Identity()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(3, 2), torch.tensor([0, 1, 0])), batch_size=2
    )
    logits, labels, indices = collect_logits(model, loader, torch.device("cpu"))
    assert logits.shape == (3, 2) and labels.tolist() == [0, 1, 0] and indices.tolist() == [0, 1, 2]


def test_invalid_artifact_split_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test")
    with pytest.raises(ValueError, match="dataset_split"):
        artifact_metadata(checkpoint, "train", 42, {})
