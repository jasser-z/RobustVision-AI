import json

import pytest

from robustvision.evaluation.corrupted_calibration import load_clean_validation_temperature


def write_payload(path, checkpoint_hash="abc", split="validation", temperature=1.5):
    path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "validation": {"dataset_split": split, "checkpoint_sha256": checkpoint_hash}
                },
                "temperature": temperature,
            }
        ),
        encoding="utf-8",
    )


def test_frozen_temperature_requires_matching_clean_validation_artifact(tmp_path):
    path = tmp_path / "calibration.json"
    write_payload(path)
    assert load_clean_validation_temperature(path, "abc") == 1.5
    with pytest.raises(ValueError, match="hash"):
        load_clean_validation_temperature(path, "other")


@pytest.mark.parametrize("split,temperature", [("test", 1.5), ("validation", 0)])
def test_test_labels_cannot_supply_a_temperature_artifact(tmp_path, split, temperature):
    path = tmp_path / "invalid.json"
    write_payload(path, split=split, temperature=temperature)
    with pytest.raises(ValueError):
        load_clean_validation_temperature(path, "abc")
