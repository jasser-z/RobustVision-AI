from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robustvision.evaluation.benchmark_analysis import load_benchmark, summarize


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


parser = argparse.ArgumentParser(description="Validate and analyze RobustVision benchmark CSVs")
parser.add_argument("--baseline", default="artifacts/benchmarks/baseline_cnn_20ep/robustness.csv")
parser.add_argument("--resnet", default="artifacts/benchmarks/resnet18_finetuned/robustness.csv")
parser.add_argument("--output", default="artifacts/analysis")
args = parser.parse_args()
out = Path(args.output)
figures = out / "figures"
figures.mkdir(parents=True, exist_ok=True)
data = pd.concat(
    [
        load_benchmark(args.baseline, "Baseline CNN"),
        load_benchmark(args.resnet, "Fine-tuned ResNet18"),
    ],
    ignore_index=True,
)
summary, per_corruption = summarize(data)
data.to_csv(out / "validated_rows.csv", index=False)
summary.to_csv(out / "model_summary.csv", index=False)
per_corruption.to_csv(out / "per_corruption_summary.csv", index=False)
sns.set_theme(style="whitegrid", context="talk")
corrupted = data[data.corruption != "clean"]
for corruption, subset in corrupted.groupby("corruption"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=subset, x="severity", y="accuracy", hue="model", marker="o", ax=ax)
    ax.set(
        title=f"Accuracy degradation: {corruption.replace('_', ' ').title()}",
        ylabel="Accuracy",
        xlabel="Severity",
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(figures / f"accuracy_by_severity_{corruption}.png", dpi=250)
    fig.savefig(figures / f"accuracy_by_severity_{corruption}.pdf")
    plt.close(fig)
heat = corrupted.pivot_table(index=["model", "corruption"], columns="severity", values="accuracy")
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(heat, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis", ax=ax)
ax.set_title("Accuracy by model, corruption, and severity")
fig.tight_layout()
fig.savefig(figures / "accuracy_heatmap.png", dpi=250)
fig.savefig(figures / "accuracy_heatmap.pdf")
plt.close(fig)
comparison = summary.melt(
    id_vars="model",
    value_vars=["clean_accuracy", "mean_corrupted_accuracy"],
    var_name="condition",
    value_name="accuracy",
)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=comparison, x="model", y="accuracy", hue="condition", ax=ax)
ax.set(title="Clean versus mean corrupted accuracy", ylabel="Accuracy", xlabel="")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(figures / "clean_vs_corrupted.png", dpi=250)
plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=per_corruption, x="corruption", y="mean_accuracy_drop", hue="model", ax=ax)
ax.set(
    title="Mean accuracy drop by corruption", ylabel="Accuracy drop (percentage points)", xlabel=""
)
ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}")
fig.tight_layout()
fig.savefig(figures / "accuracy_drop_by_corruption.png", dpi=250)
plt.close(fig)
report = [
    "# RobustVision AI Benchmark Analysis",
    "",
    "## Aggregation",
    "",
    "Each condition contains 10,000 test images, so the unweighted condition mean equals the sample-weighted condition mean. `mean_corrupted_accuracy` averages the 25 corruption-severity conditions; it is not an image-level pooled accuracy across a different population.",
    "",
    "## Model summary",
    "",
    markdown_table(summary),
    "",
    "## Per-corruption summary",
    "",
    markdown_table(per_corruption),
    "",
    "## Limitations",
    "",
    "These summaries are derived from saved benchmark CSVs. The CSVs do not contain logits or probabilities, so calibration curves cannot be recomputed from them. The ResNet experiment used CIFAR normalization rather than ImageNet weight transforms; it is a fine-tuned CIFAR result, but preprocessing comparability should be revisited in a future controlled rerun.",
]
(out / "benchmark_report.md").write_text("\n".join(report), encoding="utf-8")
print(f"Validated {len(data)} rows; wrote analysis to {out}")
