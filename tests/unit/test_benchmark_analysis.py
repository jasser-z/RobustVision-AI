from pathlib import Path

import pandas as pd
import pytest

from robustvision.evaluation.benchmark_analysis import load_benchmark, summarize


def valid_frame() -> pd.DataFrame:
    rows = [
        {
            "corruption": "clean",
            "severity": 0,
            "accuracy": 0.8,
            "macro_f1": 0.8,
            "samples": 10,
            "accuracy_drop": 0.0,
        }
    ]
    for corruption in ["gaussian_noise", "gaussian_blur", "brightness", "contrast", "jpeg"]:
        for severity in range(1, 6):
            accuracy = 0.8 - severity * 0.01
            rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "accuracy": accuracy,
                    "macro_f1": accuracy,
                    "samples": 10,
                    "accuracy_drop": 0.8 - accuracy,
                }
            )
    return pd.DataFrame(rows)


def test_validated_summary_uses_condition_average(tmp_path: Path):
    path = tmp_path / "benchmark.csv"
    valid_frame().to_csv(path, index=False)
    summary, per_corruption = summarize(load_benchmark(path, "fixture"))
    assert summary.iloc[0].clean_accuracy == 0.8
    assert summary.iloc[0].condition_count == 25
    assert len(per_corruption) == 5


def test_inconsistent_accuracy_drop_is_rejected(tmp_path: Path):
    frame = valid_frame()
    frame.loc[1, "accuracy_drop"] = 0.5
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="accuracy_drop"):
        load_benchmark(path, "fixture")


def test_unequal_sample_counts_are_rejected(tmp_path: Path):
    frame = valid_frame()
    frame.loc[1, "samples"] = 9
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="common sample count"):
        load_benchmark(path, "fixture")
