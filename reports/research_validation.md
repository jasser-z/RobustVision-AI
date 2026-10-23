# RobustVision AI: Research Validation Report

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
| Baseline CNN | False | 20 | NVIDIA GeForce RTX 4060 Laptop GPU | 146.44 | 0.7124 |
| Fine-tuned ResNet18 | True | 10 | NVIDIA GeForce RTX 4060 Laptop GPU | 215.67 | 0.8122 |

## Benchmark aggregation

All 25 corrupted conditions per model contain the same sample count, so the unweighted mean over conditions equals a sample-weighted mean. This is a mean over corruption-severity conditions, not a claim about a natural image distribution.

| model | clean_accuracy | mean_corrupted_accuracy | mean_accuracy_drop | worst_corruption | worst_severity | worst_accuracy | samples_per_condition | condition_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline CNN | 0.7087 | 0.4667 | 0.2420 | gaussian_noise | 5 | 0.1497 | 10000 | 25 |
| Fine-tuned ResNet18 | 0.8082 | 0.6143 | 0.1939 | gaussian_blur | 5 | 0.2783 | 10000 | 25 |

## Per-corruption analysis

| model | corruption | mean_accuracy | mean_accuracy_drop | severity_1_accuracy | severity_5_accuracy | severity_1_to_5_drop | samples_per_condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline CNN | gaussian_noise | 0.3361 | 0.3726 | 0.5920 | 0.1497 | 0.4423 | 10000 |
| Baseline CNN | gaussian_blur | 0.4062 | 0.3025 | 0.6941 | 0.2346 | 0.4595 | 10000 |
| Baseline CNN | brightness | 0.4773 | 0.2314 | 0.6896 | 0.2129 | 0.4767 | 10000 |
| Baseline CNN | contrast | 0.4922 | 0.2165 | 0.7104 | 0.2126 | 0.4978 | 10000 |
| Baseline CNN | jpeg | 0.6216 | 0.0871 | 0.6875 | 0.5001 | 0.1874 | 10000 |
| Fine-tuned ResNet18 | gaussian_noise | 0.5690 | 0.2392 | 0.7711 | 0.3482 | 0.4229 | 10000 |
| Fine-tuned ResNet18 | gaussian_blur | 0.5028 | 0.3054 | 0.8010 | 0.2783 | 0.5227 | 10000 |
| Fine-tuned ResNet18 | brightness | 0.6203 | 0.1879 | 0.7948 | 0.2969 | 0.4979 | 10000 |
| Fine-tuned ResNet18 | contrast | 0.6559 | 0.1523 | 0.8013 | 0.3891 | 0.4122 | 10000 |
| Fine-tuned ResNet18 | jpeg | 0.7237 | 0.0845 | 0.7866 | 0.5839 | 0.2027 | 10000 |

## Calibration

ECE uses ten confidence bins and sums each bin's population-weighted absolute accuracy-confidence gap. Brier score is the mean sum of squared differences between predicted class probabilities and one-hot labels. NLL is mean negative log probability of the true class. A single positive temperature is fitted on validation logits/labels only, then frozen for the untouched test split; temperature scaling does not change predicted classes or accuracy.

| checkpoint | split | state | accuracy | nll | brier_score | ece | temperature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_cnn_20ep | validation | uncalibrated | 0.7124 | 0.8397 | 0.3994 | 0.0204 | 1.0000 |
| baseline_cnn_20ep | validation | temperature_scaled | 0.7124 | 0.8396 | 0.3993 | 0.0177 | 0.9877 |
| baseline_cnn_20ep | test | uncalibrated | 0.7088 | 0.8293 | 0.3985 | 0.0159 | 1.0000 |
| baseline_cnn_20ep | test | temperature_scaled | 0.7088 | 0.8290 | 0.3983 | 0.0127 | 0.9877 |
| resnet18_finetuned | validation | uncalibrated | 0.8120 | 0.7169 | 0.2931 | 0.1072 | 1.0000 |
| resnet18_finetuned | validation | temperature_scaled | 0.8120 | 0.5618 | 0.2666 | 0.0140 | 1.8668 |
| resnet18_finetuned | test | uncalibrated | 0.8082 | 0.7621 | 0.3021 | 0.1100 | 1.0000 |
| resnet18_finetuned | test | temperature_scaled | 0.8082 | 0.5835 | 0.2746 | 0.0176 | 1.8668 |

Clean reliability figures: `artifacts\calibration\baseline_cnn_20ep\test_reliability.png`, `artifacts\calibration\resnet18_finetuned\test_reliability.png`. Clean metrics are based on saved per-example prediction artifacts, not the aggregate benchmark CSVs.

## Corruption-aware calibration

The corruption-aware evaluation uses the temperature fitted on the clean validation split above. It does not fit on corrupted test labels. Five corruption types were evaluated at severities 1, 3, and 5 on the official test split; each aggregate below is the equally weighted mean over five corruption types at a given severity.

| model | severity | state | accuracy | nll | brier_score | ece |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_cnn_20ep | 1 | temperature_scaled | 0.6747 | 0.9309 | 0.4429 | 0.0382 |
| baseline_cnn_20ep | 1 | uncalibrated | 0.6747 | 0.9309 | 0.4430 | 0.0399 |
| baseline_cnn_20ep | 3 | temperature_scaled | 0.4689 | 1.6684 | 0.6951 | 0.1165 |
| baseline_cnn_20ep | 3 | uncalibrated | 0.4689 | 1.6624 | 0.6941 | 0.1169 |
| baseline_cnn_20ep | 5 | temperature_scaled | 0.2620 | 2.8681 | 0.9529 | 0.2327 |
| baseline_cnn_20ep | 5 | uncalibrated | 0.2620 | 2.8493 | 0.9506 | 0.2294 |
| resnet18_finetuned | 1 | temperature_scaled | 0.7909 | 0.6356 | 0.2975 | 0.0175 |
| resnet18_finetuned | 1 | uncalibrated | 0.7909 | 0.8249 | 0.3270 | 0.1186 |
| resnet18_finetuned | 3 | temperature_scaled | 0.6398 | 1.1320 | 0.5002 | 0.0803 |
| resnet18_finetuned | 3 | uncalibrated | 0.6398 | 1.5764 | 0.5654 | 0.2170 |
| resnet18_finetuned | 5 | temperature_scaled | 0.3793 | 2.0640 | 0.8182 | 0.2016 |
| resnet18_finetuned | 5 | uncalibrated | 0.3793 | 3.0244 | 0.9513 | 0.3785 |

Per-model/per-corruption summary below equally weights severities 1, 3, and 5; every condition has 10,000 test samples.

| model | corruption | state | accuracy | nll | brier_score | ece |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_cnn_20ep | brightness | temperature_scaled | 0.4739 | 1.6488 | 0.6612 | 0.0477 |
| baseline_cnn_20ep | brightness | uncalibrated | 0.4739 | 1.6457 | 0.6612 | 0.0491 |
| baseline_cnn_20ep | contrast | temperature_scaled | 0.4869 | 1.6346 | 0.6497 | 0.0792 |
| baseline_cnn_20ep | contrast | uncalibrated | 0.4869 | 1.6302 | 0.6494 | 0.0806 |
| baseline_cnn_20ep | gaussian_blur | temperature_scaled | 0.4194 | 2.0535 | 0.7662 | 0.1928 |
| baseline_cnn_20ep | gaussian_blur | uncalibrated | 0.4194 | 2.0418 | 0.7644 | 0.1914 |
| baseline_cnn_20ep | gaussian_noise | temperature_scaled | 0.3512 | 2.6629 | 0.8941 | 0.2924 |
| baseline_cnn_20ep | gaussian_noise | uncalibrated | 0.3512 | 2.6424 | 0.8914 | 0.2886 |
| baseline_cnn_20ep | jpeg | temperature_scaled | 0.6114 | 1.1125 | 0.5134 | 0.0337 |
| baseline_cnn_20ep | jpeg | uncalibrated | 0.6114 | 1.1111 | 0.5132 | 0.0341 |
| resnet18_finetuned | brightness | temperature_scaled | 0.5984 | 1.2845 | 0.5327 | 0.0649 |
| resnet18_finetuned | brightness | uncalibrated | 0.5984 | 1.7255 | 0.5990 | 0.2125 |
| resnet18_finetuned | contrast | temperature_scaled | 0.6365 | 1.1752 | 0.4913 | 0.0614 |
| resnet18_finetuned | contrast | uncalibrated | 0.6365 | 1.6048 | 0.5543 | 0.2025 |
| resnet18_finetuned | gaussian_blur | temperature_scaled | 0.5036 | 1.6887 | 0.6797 | 0.1869 |
| resnet18_finetuned | gaussian_blur | uncalibrated | 0.5036 | 2.5656 | 0.7887 | 0.3294 |
| resnet18_finetuned | gaussian_noise | temperature_scaled | 0.5697 | 1.3431 | 0.5797 | 0.1249 |
| resnet18_finetuned | gaussian_noise | uncalibrated | 0.5697 | 1.9133 | 0.6714 | 0.2693 |
| resnet18_finetuned | jpeg | temperature_scaled | 0.7086 | 0.8945 | 0.4097 | 0.0609 |
| resnet18_finetuned | jpeg | uncalibrated | 0.7086 | 1.2337 | 0.4595 | 0.1765 |

Generated figures: `artifacts/calibration_corrupted/mean_ece_by_severity.png`, `artifacts/calibration_corrupted/mean_nll_by_severity.png`, and the following saved severity-5 reliability diagrams: `artifacts\calibration_corrupted\baseline_cnn_20ep\reliability_brightness_s5.png`, `artifacts\calibration_corrupted\baseline_cnn_20ep\reliability_contrast_s5.png`, `artifacts\calibration_corrupted\baseline_cnn_20ep\reliability_gaussian_blur_s5.png`, `artifacts\calibration_corrupted\baseline_cnn_20ep\reliability_gaussian_noise_s5.png`, `artifacts\calibration_corrupted\baseline_cnn_20ep\reliability_jpeg_s5.png`, `artifacts\calibration_corrupted\resnet18_finetuned\reliability_brightness_s5.png`, `artifacts\calibration_corrupted\resnet18_finetuned\reliability_contrast_s5.png`, `artifacts\calibration_corrupted\resnet18_finetuned\reliability_gaussian_blur_s5.png`, `artifacts\calibration_corrupted\resnet18_finetuned\reliability_gaussian_noise_s5.png`, `artifacts\calibration_corrupted\resnet18_finetuned\reliability_jpeg_s5.png`. Clean-validation temperature scaling is not guaranteed to improve every shifted condition; results must be interpreted condition by condition.

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
