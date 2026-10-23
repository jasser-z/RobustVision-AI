"""Validated, sample-aware analysis of saved robustness benchmark CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"corruption", "severity", "accuracy", "macro_f1", "samples", "accuracy_drop"}
EXPECTED_CORRUPTIONS = {"gaussian_noise", "gaussian_blur", "brightness", "contrast", "jpeg"}


def load_benchmark(path: str | Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if len(frame) != 26 or (frame.corruption == "clean").sum() != 1:
        raise ValueError(f"{path}: expected one clean row and 25 corruption-severity rows")
    clean = frame[frame.corruption == "clean"].iloc[0]
    corrupted = frame[frame.corruption != "clean"].copy()
    if set(corrupted.corruption) != EXPECTED_CORRUPTIONS or set(corrupted.severity) != {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError(f"{path}: corruption types or severity levels are incomplete")
    if (frame.samples <= 0).any() or frame.samples.nunique() != 1:
        raise ValueError(f"{path}: conditions must have one positive, common sample count")
    if int(clean.severity) != 0 or abs(float(clean.accuracy_drop)) > 1e-10:
        raise ValueError(f"{path}: malformed clean baseline")
    expected = float(clean.accuracy) - corrupted.accuracy
    if not ((expected - corrupted.accuracy_drop).abs() < 1e-8).all():
        raise ValueError(f"{path}: accuracy_drop is inconsistent with clean accuracy")
    frame["model"] = model
    return frame


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_corruption = [], []
    for model, group in frame.groupby("model", sort=False):
        clean = group.loc[group.corruption == "clean"].iloc[0]
        corrupt = group[group.corruption != "clean"]
        # Conditions have an identical sample count. Therefore condition mean equals a
        # sample-weighted mean; this is validated by load_benchmark.
        worst = corrupt.loc[corrupt.accuracy.idxmin()]
        rows.append(
            {
                "model": model,
                "clean_accuracy": clean.accuracy,
                "mean_corrupted_accuracy": corrupt.accuracy.mean(),
                "mean_accuracy_drop": corrupt.accuracy_drop.mean(),
                "worst_corruption": worst.corruption,
                "worst_severity": int(worst.severity),
                "worst_accuracy": worst.accuracy,
                "samples_per_condition": int(clean.samples),
                "condition_count": len(corrupt),
            }
        )
        for corruption, subset in corrupt.groupby("corruption", sort=False):
            one, five = (
                subset.loc[subset.severity == 1].iloc[0],
                subset.loc[subset.severity == 5].iloc[0],
            )
            per_corruption.append(
                {
                    "model": model,
                    "corruption": corruption,
                    "mean_accuracy": subset.accuracy.mean(),
                    "mean_accuracy_drop": subset.accuracy_drop.mean(),
                    "severity_1_accuracy": one.accuracy,
                    "severity_5_accuracy": five.accuracy,
                    "severity_1_to_5_drop": one.accuracy - five.accuracy,
                    "samples_per_condition": int(one.samples),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_corruption)
