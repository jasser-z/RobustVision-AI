"""Create clean validation/test prediction artifacts and validation-fitted calibration results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robustvision.data.datasets import MEAN, STD, cifar_loaders
from robustvision.device import select_device
from robustvision.evaluation.calibration import (
    calibration_metrics,
    fit_temperature,
    probabilities_from_logits,
    reliability_bins,
)
from robustvision.evaluation.prediction_artifacts import artifact_metadata, collect_logits
from robustvision.training.checkpoints import load_checkpoint
from robustvision.training.reproducibility import seed_everything


def save_predictions(
    path: Path, logits: np.ndarray, labels: np.ndarray, indices: np.ndarray
) -> None:
    probabilities = probabilities_from_logits(logits).astype(np.float32)
    np.savez_compressed(
        path,
        logits=logits,
        probabilities=probabilities,
        labels=labels,
        predictions=probabilities.argmax(axis=1).astype(np.int64),
        indices=indices,
    )


def plot_reliability(before, after, output: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 6))
    for bins, label, color in [
        (before, "Uncalibrated", "#1f77b4"),
        (after, "Temperature-scaled", "#d62728"),
    ]:
        confidence = [row["confidence"] for row in bins if row["count"]]
        accuracy = [row["accuracy"] for row in bins if row["count"]]
        axis.plot(confidence, accuracy, marker="o", label=label, color=color)
    axis.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    axis.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy", title=title
    )
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=250)
    plt.close(fig)


parser = argparse.ArgumentParser(
    description="Evaluate clean calibration from local RobustVision checkpoints"
)
parser.add_argument(
    "--checkpoints",
    nargs="+",
    default=[
        "artifacts/checkpoints/baseline_cnn_20ep.pt",
        "artifacts/checkpoints/resnet18_finetuned.pt",
    ],
)
parser.add_argument("--data-dir", default="data")
parser.add_argument("--output", default="artifacts/calibration")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
parser.add_argument("--bins", type=int, default=10)
args = parser.parse_args()
if args.bins < 2:
    raise ValueError("--bins must be at least 2")
seed_everything(args.seed)
device = select_device(args.device)
_, validation_loader, test_loader, split = cifar_loaders(
    args.data_dir, args.batch_size, args.seed, download=False
)
for checkpoint_arg in args.checkpoints:
    checkpoint = Path(checkpoint_arg)
    model, checkpoint_metadata = load_checkpoint(checkpoint, device)
    model_id = checkpoint.stem
    output = Path(args.output) / model_id
    output.mkdir(parents=True, exist_ok=True)
    validation_logits, validation_labels, validation_indices = collect_logits(
        model, validation_loader, device
    )
    test_logits, test_labels, test_indices = collect_logits(model, test_loader, device)
    temperature = fit_temperature(validation_logits, validation_labels)
    save_predictions(
        output / "validation_predictions.npz",
        validation_logits,
        validation_labels,
        validation_indices,
    )
    save_predictions(output / "test_predictions.npz", test_logits, test_labels, test_indices)
    metrics = {
        "validation": {
            "uncalibrated": calibration_metrics(validation_logits, validation_labels, args.bins),
            "temperature_scaled": calibration_metrics(
                validation_logits, validation_labels, args.bins, temperature
            ),
        },
        "test": {
            "uncalibrated": calibration_metrics(test_logits, test_labels, args.bins),
            "temperature_scaled": calibration_metrics(
                test_logits, test_labels, args.bins, temperature
            ),
        },
    }
    reliability = {
        "test_uncalibrated": reliability_bins(test_logits, test_labels, args.bins),
        "test_temperature_scaled": reliability_bins(
            test_logits, test_labels, args.bins, temperature
        ),
    }
    plot_reliability(
        reliability["test_uncalibrated"],
        reliability["test_temperature_scaled"],
        output / "test_reliability.png",
        f"Clean test reliability: {model_id}",
    )
    metadata = {
        "artifacts": {
            split_name: artifact_metadata(
                checkpoint,
                split_name,
                args.seed,
                {"normalization_mean": MEAN, "normalization_std": STD},
            )
            for split_name in ["validation", "test"]
        },
        "model": checkpoint_metadata.get("model"),
        "checkpoint_metadata": checkpoint_metadata,
        "dataset": "CIFAR-10",
        "splits": split,
        "evaluation": {
            "validation": "temperature fit only",
            "test": "untouched final estimate",
            "seed": args.seed,
            "device": str(device),
            "bins": args.bins,
            "preprocessing": {"normalization_mean": MEAN, "normalization_std": STD},
        },
        "temperature": temperature,
        "metrics": metrics,
        "reliability_bins": reliability,
    }
    (output / "calibration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": model_id,
                "output": str(output),
                "temperature": temperature,
                "test": metrics["test"],
            },
            indent=2,
        )
    )
