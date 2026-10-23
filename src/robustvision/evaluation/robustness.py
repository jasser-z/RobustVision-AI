from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from robustvision.data.corruptions import apply_corruption
from robustvision.data.datasets import MEAN, STD

from .runner import evaluate


class CorruptedDataset(Dataset):
    def __init__(self, dataset, corruption: str, severity: int, seed: int):
        self.dataset, self.corruption, self.severity, self.seed = (
            dataset,
            corruption,
            severity,
            seed,
        )
        self.transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if isinstance(image, torch.Tensor):
            raise TypeError("Robustness benchmark requires a raw-image dataset")
        image = apply_corruption(
            Image.fromarray(__import__("numpy").asarray(image)),
            self.corruption,
            self.severity,
            self.seed + index,
        )
        return self.transform(image), label


def benchmark(
    model,
    raw_dataset,
    device,
    batch_size: int,
    corruptions: list[str],
    severities: list[int],
    seed: int,
) -> list[dict]:
    DataLoader(raw_dataset, batch_size=batch_size)

    # Raw CIFAR needs tensor conversion; evaluation is run on a normalized wrapper for clean images.
    class CleanDataset(CorruptedDataset):
        def __getitem__(self, index):
            image, label = self.dataset[index]
            return self.transform(image), label

    clean_metrics = evaluate(
        model, DataLoader(CleanDataset(raw_dataset, "jpeg", 1, seed), batch_size=batch_size), device
    )
    rows = [
        {
            "corruption": "clean",
            "severity": 0,
            "accuracy": clean_metrics["accuracy"],
            "macro_f1": clean_metrics["macro_f1"],
            "samples": clean_metrics["samples"],
            "accuracy_drop": 0.0,
        }
    ]
    for corruption in corruptions:
        for severity in severities:
            metrics = evaluate(
                model,
                DataLoader(
                    CorruptedDataset(raw_dataset, corruption, severity, seed), batch_size=batch_size
                ),
                device,
            )
            rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "samples": metrics["samples"],
                    "accuracy_drop": clean_metrics["accuracy"] - metrics["accuracy"],
                }
            )
    return rows
