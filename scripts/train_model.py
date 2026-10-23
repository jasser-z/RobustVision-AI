from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustvision.data.datasets import cifar_loaders
from robustvision.device import device_metadata, select_device
from robustvision.evaluation.runner import evaluate
from robustvision.models.registry import create_model
from robustvision.tracking.mlflow_logger import log_run
from robustvision.training.checkpoints import save_checkpoint
from robustvision.training.reproducibility import seed_everything
from robustvision.training.trainer import train

parser = argparse.ArgumentParser(description="Train and evaluate a CIFAR-10 classifier")
parser.add_argument("--model", choices=["baseline_cnn", "resnet18"], default="baseline_cnn")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--feature-extraction", action="store_true")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--data-dir", default="data")
parser.add_argument("--output", default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
args = parser.parse_args()

seed_everything(args.seed)
device = select_device(args.device)
train_loader, validation_loader, test_loader, split = cifar_loaders(
    args.data_dir,
    args.batch_size,
    args.seed,
    quick_samples=512 if args.quick else None,
    download=True,
)
model = create_model(args.model, pretrained=args.pretrained).to(device)
if args.feature_extraction and args.model == "resnet18":
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
summary, _ = train(
    model,
    train_loader,
    validation_loader,
    device,
    args.epochs,
    args.learning_rate,
    args.weight_decay,
)
test = evaluate(model, test_loader, device)
output = Path(
    args.output or f"artifacts/checkpoints/{args.model}_{'quick' if args.quick else 'full'}.pt"
)
metadata = {
    "model": args.model,
    "pretrained_requested": args.pretrained,
    "feature_extraction": args.feature_extraction,
    "seed": args.seed,
    "device": device_metadata(args.device),
    "split": split,
    "quick": args.quick,
    "training": summary,
    "test_metrics": {
        k: v
        for k, v in test.items()
        if k not in {"probabilities", "labels", "predictions", "report", "confusion_matrix"}
    },
}
save_checkpoint(output, model, metadata)
metadata_path = output.with_suffix(".json")
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
run_id = log_run(
    f"train-{args.model}",
    {"model": args.model, "quick": args.quick, "seed": args.seed},
    test,
    [output, metadata_path],
)
print(
    json.dumps(
        {
            "checkpoint": str(output),
            "mlflow_run_id": run_id,
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "device": str(device),
            "elapsed_seconds": summary["elapsed_seconds"],
            "samples": test["samples"],
        },
        indent=2,
    )
)
