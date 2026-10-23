import numpy as np
import pytest
from PIL import Image

from robustvision.data.corruptions import apply_corruption
from robustvision.device import select_device
from robustvision.evaluation.metrics import classification_metrics, expected_calibration_error
from robustvision.models.cnn import BaselineCNN


def test_cnn_forward():
    import torch

    assert BaselineCNN()(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_corruption_deterministic_and_size():
    im = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
    assert np.array_equal(
        np.asarray(apply_corruption(im, "gaussian_noise", 2, 7)),
        np.asarray(apply_corruption(im, "gaussian_noise", 2, 7)),
    )


def test_invalid_severity():
    with pytest.raises(ValueError):
        apply_corruption(Image.new("RGB", (2, 2)), "jpeg", 6)


def test_metrics():
    m = classification_metrics([0, 1], [0, 1])
    assert m["accuracy"] == 1 and m["samples"] == 2
    assert expected_calibration_error([[0.9, 0.1]], [0]) >= 0


def test_cpu_selection():
    assert str(select_device("cpu")) == "cpu"
