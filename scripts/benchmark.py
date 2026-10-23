from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torchvision.datasets import CIFAR10

from robustvision.device import select_device
from robustvision.evaluation.robustness import benchmark
from robustvision.tracking.experiment_store import ExperimentStore
from robustvision.training.checkpoints import load_checkpoint

parser = argparse.ArgumentParser(description="Run reproducible CIFAR-10 corruption benchmark")
parser.add_argument("checkpoint")
parser.add_argument("--data-dir", default="data")
parser.add_argument("--output", default="artifacts/benchmarks")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="auto")
args = parser.parse_args()
device = select_device(args.device)
model, metadata = load_checkpoint(args.checkpoint, device)
raw = CIFAR10(args.data_dir, train=False, download=False)
if args.quick:
    raw.data = raw.data[:100]
    raw.targets = raw.targets[:100]
rows = benchmark(
    model,
    raw,
    device,
    args.batch_size,
    ["gaussian_noise", "gaussian_blur", "brightness", "contrast", "jpeg"],
    [1, 2] if args.quick else [1, 2, 3, 4, 5],
    args.seed,
)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "robustness.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
payload = {
    "model": metadata["model"],
    "checkpoint": str(args.checkpoint),
    "quick": args.quick,
    "rows": rows,
}
(out / "robustness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
identifier = ExperimentStore(out / "experiments.db").add(payload)
print(json.dumps({"csv": str(csv_path), "experiment_id": identifier, "rows": len(rows)}, indent=2))
