from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Generate a traceable report from benchmark analysis artifacts"
)
parser.add_argument("--analysis", default="artifacts/analysis")
parser.add_argument("--output", default="reports/research_validation.md")
args = parser.parse_args()
analysis, output = Path(args.analysis), Path(args.output)
summary = pd.read_csv(analysis / "model_summary.csv")
per_corruption = pd.read_csv(analysis / "per_corruption_summary.csv")
cnn = json.loads(Path("artifacts/checkpoints/baseline_cnn_20ep.json").read_text(encoding="utf-8"))
resnet = json.loads(
    Path("artifacts/checkpoints/resnet18_finetuned.json").read_text(encoding="utf-8")
)
calibration_rows = []
calibration_figures = []
for checkpoint in ["baseline_cnn_20ep", "resnet18_finetuned"]:
    path = Path("artifacts/calibration") / checkpoint / "calibration.json"
    if path.exists():
        calibration = json.loads(path.read_text(encoding="utf-8"))
        for split in ["validation", "test"]:
            for state in ["uncalibrated", "temperature_scaled"]:
                metrics = calibration["metrics"][split][state]
                calibration_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "split": split,
                        "state": state,
                        "accuracy": metrics["accuracy"],
                        "nll": metrics["nll"],
                        "brier_score": metrics["brier_score"],
                        "ece": metrics["ece"],
                        "temperature": metrics["temperature"],
                    }
                )
        calibration_figures.append(str(path.parent / "test_reliability.png"))
corrupted_calibration_path = Path(
    "artifacts/calibration_corrupted/corrupted_calibration_aggregate.csv"
)
corrupted_calibration = (
    pd.read_csv(corrupted_calibration_path) if corrupted_calibration_path.exists() else None
)
corrupted_summary_path = Path("artifacts/calibration_corrupted/corrupted_calibration_summary.csv")
corrupted_by_type = None
corrupted_reliability_figures = []
if corrupted_summary_path.exists():
    corrupted_summary = pd.read_csv(corrupted_summary_path)
    corrupted_by_type = corrupted_summary.groupby(["model", "corruption", "state"], as_index=False)[
        ["accuracy", "nll", "brier_score", "ece"]
    ].mean()
    corrupted_reliability_figures = sorted(
        str(path) for path in Path("artifacts/calibration_corrupted").glob("*/reliability_*_s5.png")
    )


def table(frame):
    header = list(frame.columns)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in frame.itertuples(index=False):
        lines.append(
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


output.parent.mkdir(parents=True, exist_ok=True)
text = f"""# RobustVision AI: Research Validation Report

## Abstract

This report analyzes saved, full CIFAR-10 robustness benchmark outputs for a baseline CNN and a fine-tuned ResNet18. Every table and figure is derived from the benchmark CSVs validated by `scripts/analyze_benchmarks.py`; no values are manually entered.

## Research questions

1. How do architecture and image corruption affect CIFAR-10 classification reliability?
2. Which corruption/severity pairs create the largest measured degradation?
3. What reliability limitations remain after clean-data evaluation?

## Dataset, split, and reproducibility

CIFAR-10 uses 45,000 training images, a deterministic 5,000-image validation split selected with seed 42, and the untouched official 10,000-image test split. Best checkpoints are selected by validation accuracy, not test accuracy. Each clean/corrupted benchmark condition evaluates 10,000 test examples.

## Training provenance

| Model | Pretrained requested | Epochs | Device | Training elapsed seconds | Best validation accuracy |
| --- | --- | --- | --- | --- | --- |
| Baseline CNN | {cnn["pretrained_requested"]} | {len(cnn["training"]["history"])} | {cnn["device"]["cuda_name"]} | {cnn["training"]["elapsed_seconds"]:.2f} | {cnn["training"]["best_validation_accuracy"]:.4f} |
| Fine-tuned ResNet18 | {resnet["pretrained_requested"]} | {len(resnet["training"]["history"])} | {resnet["device"]["cuda_name"]} | {resnet["training"]["elapsed_seconds"]:.2f} | {resnet["training"]["best_validation_accuracy"]:.4f} |

## Benchmark aggregation

All 25 corrupted conditions per model contain the same sample count, so the unweighted mean over conditions equals a sample-weighted mean. This is a mean over corruption-severity conditions, not a claim about a natural image distribution.

{table(summary)}

## Per-corruption analysis

{table(per_corruption)}

## Calibration

ECE uses ten confidence bins and sums each bin's population-weighted absolute accuracy-confidence gap. Brier score is the mean sum of squared differences between predicted class probabilities and one-hot labels. NLL is mean negative log probability of the true class. A single positive temperature is fitted on validation logits/labels only, then frozen for the untouched test split; temperature scaling does not change predicted classes or accuracy.

{table(pd.DataFrame(calibration_rows)) if calibration_rows else "Calibration artifacts were not available when this report was generated."}

Clean reliability figures: {", ".join(f"`{path}`" for path in calibration_figures) if calibration_figures else "unavailable"}. Clean metrics are based on saved per-example prediction artifacts, not the aggregate benchmark CSVs.

## Corruption-aware calibration

The corruption-aware evaluation uses the temperature fitted on the clean validation split above. It does not fit on corrupted test labels. Five corruption types were evaluated at severities 1, 3, and 5 on the official test split; each aggregate below is the equally weighted mean over five corruption types at a given severity.

{table(corrupted_calibration) if corrupted_calibration is not None else "Corrupted calibration artifacts were not available when this report was generated."}

Per-model/per-corruption summary below equally weights severities 1, 3, and 5; every condition has 10,000 test samples.

{table(corrupted_by_type) if corrupted_by_type is not None else "Per-corruption summary unavailable."}

Generated figures: `artifacts/calibration_corrupted/mean_ece_by_severity.png`, `artifacts/calibration_corrupted/mean_nll_by_severity.png`, and the following saved severity-5 reliability diagrams: {", ".join(f"`{path}`" for path in corrupted_reliability_figures) if corrupted_reliability_figures else "unavailable"}. Clean-validation temperature scaling is not guaranteed to improve every shifted condition; results must be interpreted condition by condition.

## Figures

- `artifacts/analysis/figures/accuracy_heatmap.png`
- `artifacts/analysis/figures/clean_vs_corrupted.png`
- `artifacts/analysis/figures/accuracy_drop_by_corruption.png`
- `artifacts/analysis/figures/accuracy_by_severity_*.png`

## Limitations and threats to validity

- Benchmarks use the same held-out test set across measured conditions; results should not be used for further model selection without a new holdout protocol.
- Deterministic corruption seeds support repeatability but do not represent every real-world corruption distribution.
- The ResNet experiment starts from pretrained weights but uses CIFAR normalization rather than the ImageNet transforms packaged with the original weights. A controlled rerun should test preprocessing choices.
- Corruption-level calibration is available for the tested synthetic corruptions. Per-class calibration summaries have not been generated from the saved prediction artifacts.
- The CNN checkpoint sidecar records 0.7088 clean test accuracy while the benchmark CSV clean row is 0.7087. A later read-only evaluation of the current checkpoint reproduced 0.7087, matching the benchmark CSV. The sidecar therefore appears inconsistent; its exact historical cause cannot be established because these artifacts do not store checkpoint digests or complete evaluation provenance. This report uses each benchmark CSV's clean row for its corresponding corruption drops and does not merge records.

## Reproduction

```powershell
python scripts/analyze_benchmarks.py
python scripts/generate_research_report.py
streamlit run app/research_dashboard.py
```
"""
output.write_text(text, encoding="utf-8")
print(f"Wrote {output}")
