"""Evaluate corrupted CIFAR-10 test calibration with frozen clean-validation temperatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from robustvision.device import select_device
from robustvision.evaluation.calibration import calibration_metrics, reliability_bins
from robustvision.evaluation.corrupted_calibration import load_clean_validation_temperature
from robustvision.evaluation.prediction_artifacts import (
    artifact_metadata,
    collect_logits,
    save_prediction_archive,
    sha256,
)
from robustvision.evaluation.robustness import CorruptedDataset
from robustvision.training.checkpoints import load_checkpoint
from robustvision.training.reproducibility import seed_everything


def reliability_plot(before, after, output: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 6))
    for bins, label, color in [
        (before, "Uncalibrated", "#1f77b4"),
        (after, "Clean-validation temperature", "#d62728"),
    ]:
        rows = [row for row in bins if row["count"]]
        axis.plot(
            [row["confidence"] for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            label=label,
            color=color,
        )
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
    description="Evaluate corrupted test calibration using clean-validation temperatures"
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
parser.add_argument("--clean-calibration-dir", default="artifacts/calibration")
parser.add_argument("--output", default="artifacts/calibration_corrupted")
parser.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
parser.add_argument("--bins", type=int, default=10)
args = parser.parse_args()
if not set(args.severities).issubset({1, 2, 3, 4, 5}):
    raise ValueError("--severities must be selected from 1..5")
seed_everything(args.seed)
device = select_device(args.device)
raw_test = CIFAR10(args.data_dir, train=False, download=False)
all_rows = []
for checkpoint_arg in args.checkpoints:
    checkpoint = Path(checkpoint_arg)
    checkpoint_hash = sha256(checkpoint)
    model, checkpoint_metadata = load_checkpoint(checkpoint, device)
    model_id = checkpoint.stem
    clean_json = Path(args.clean_calibration_dir) / model_id / "calibration.json"
    temperature = load_clean_validation_temperature(clean_json, checkpoint_hash)
    model_dir = Path(args.output) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": checkpoint_metadata["model"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "temperature_source": str(clean_json),
        "temperature": temperature,
        "dataset": "CIFAR-10 official test split",
        "seed": args.seed,
        "severities": args.severities,
        "conditions": [],
    }
    for corruption in ["gaussian_noise", "gaussian_blur", "brightness", "contrast", "jpeg"]:
        for severity in args.severities:
            filename = f"{corruption}_s{severity}.npz"
            archive_path = model_dir / filename
            try:
                archive = np.load(archive_path, allow_pickle=False)
                if (
                    str(archive["checkpoint_sha256"].item()) != checkpoint_hash
                    or str(archive["corruption"].item()) != corruption
                    or int(archive["severity"].item()) != severity
                ):
                    raise ValueError("stale prediction archive")
                logits, labels, indices = archive["logits"], archive["labels"], archive["indices"]
            except (FileNotFoundError, KeyError, ValueError, OSError):
                dataset = CorruptedDataset(raw_test, corruption, severity, args.seed)
                logits, labels, indices = collect_logits(
                    model, DataLoader(dataset, batch_size=args.batch_size), device
                )
                save_prediction_archive(
                    archive_path,
                    logits,
                    labels,
                    indices,
                    corruption=np.asarray(corruption),
                    severity=np.asarray(severity),
                    seed=np.asarray(args.seed),
                    checkpoint_sha256=np.asarray(checkpoint_hash),
                    temperature=np.asarray(temperature),
                )
            before, after = (
                calibration_metrics(logits, labels, args.bins),
                calibration_metrics(logits, labels, args.bins, temperature),
            )
            condition = {
                "corruption": corruption,
                "severity": severity,
                "artifact": artifact_metadata(
                    checkpoint, "test", args.seed, {"normalization": "CIFAR-10 mean/std"}
                ),
                "prediction_file": filename,
                "metrics": {"uncalibrated": before, "temperature_scaled": after},
            }
            manifest["conditions"].append(condition)
            all_rows.extend(
                [
                    {
                        "model": model_id,
                        "corruption": corruption,
                        "severity": severity,
                        "state": state,
                        **metrics,
                    }
                    for state, metrics in condition["metrics"].items()
                ]
            )
            if severity == max(args.severities):
                reliability_plot(
                    reliability_bins(logits, labels, args.bins),
                    reliability_bins(logits, labels, args.bins, temperature),
                    model_dir / f"reliability_{corruption}_s{severity}.png",
                    f"{model_id}: {corruption} severity {severity}",
                )
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
frame = pd.DataFrame(all_rows)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
frame.to_csv(output / "corrupted_calibration_summary.csv", index=False)
aggregate = frame.groupby(["model", "severity", "state"], as_index=False)[
    ["accuracy", "nll", "brier_score", "ece"]
].mean()
aggregate.to_csv(output / "corrupted_calibration_aggregate.csv", index=False)
for metric, label in [("ece", "ECE"), ("nll", "Negative log-likelihood")]:
    fig, axis = plt.subplots(figsize=(10, 6))
    for (model, state), subset in frame.groupby(["model", "state"]):
        grouped = subset.groupby("severity")[metric].mean()
        axis.plot(grouped.index, grouped.values, marker="o", label=f"{model} - {state}")
    axis.set(
        xlabel="Corruption severity",
        ylabel=label,
        title=f"Mean corrupted {label} across corruption types",
    )
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / f"mean_{metric}_by_severity.png", dpi=250)
    plt.close(fig)
print(
    json.dumps(
        {
            "conditions": len(frame) // 2,
            "rows": len(frame),
            "output": str(output),
            "device": str(device),
        },
        indent=2,
    )
)
