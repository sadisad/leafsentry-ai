# Reproduced evaluation evidence

## Held-out model-level evaluation

[Full report](test-split-report.json), including all 128 sample predictions, confusion matrix, and threshold sweep. Dataset and archive identities are pinned and checksum-verified. Runtime: Python 3.12.3, torch 2.9.1+cpu, torchvision 0.24.1+cpu, Transformers 5.10.1; CPU.

| Metric | Observed result |
|---|---:|
| Samples | 128 |
| Top-1 accuracy | 96.875% |
| Macro-F1 | 0.969121 |
| Negative log likelihood | 0.131531 |
| Multiclass Brier | 0.050559 |
| Top-label ECE (10 equal-width bins) | 0.054865 |
| Accepted at confidence 0.75 / margin 0.20 | 127 / 128 |
| Score-policy coverage | 99.219% |
| Accepted-set accuracy | 97.638% |
| Accepted-set risk | 2.362% |

This run measures the pretrained model and score policy on the upstream Beans test split. It bypasses service image-quality gates, uses temperature 1.0 (not fitted calibration), and does not measure end-to-end service coverage. Thresholds were author-selected before this report; the sweep is descriptive only. No thresholds were fitted on this test set.

The upstream model card reports 94.53125% test accuracy. Reproducing a different value does not prove a model improvement: this execution uses a distinct runtime and explicitly pinned preprocessing. The project did not train new weights.

## Failure analysis

The model made 4 top-1 errors. The default policy abstained on one of them and still accepted three incorrect outputs. Abstention is not a safety guarantee.

| Dataset filename | Target | Top-1 output | Score-policy decision |
|---|---|---|---|
| `angular_leaf_spot_test.14.jpg` | angular_leaf_spot | bean_rust | accepted |
| `angular_leaf_spot_test.36.jpg` | angular_leaf_spot | bean_rust | accepted |
| `angular_leaf_spot_test.42.jpg` | angular_leaf_spot | bean_rust | accepted |
| `healthy_test.35.jpg` | healthy | bean_rust | abstained |

## Coverage versus risk

At confidence 0.80 (margin 0.20), coverage is 98.438% and selective risk 1.587%. At confidence 0.90, coverage is 92.188% and risk 1.695%. At confidence 0.95, coverage drops to 28.906% with zero observed errors in that small accepted subset. This is not a recommended deployment policy; zero errors in a small subset is not evidence of zero population risk. The non-monotone risk illustrates why confidence alone is insufficient.

## Three-sample wiring smoke

[Smoke JSON](model-smoke.json) records real CPU inference on the three upstream examples. All top-1 labels matched after sample checksum verification. These are illustrative model-card examples, not a random or independent test set. Latency includes model loading for the first sample and is not an HTTP throughput benchmark.

## Synthetic metric fixture

[Example JSON](example-report.json) comes from `examples/predictions.example.jsonl`, a synthetic fixture for checking metric semantics. Its perfect accuracy is not a model-performance result.

## Engineering verification

[Local verification JSON](local-verification.json) records the test matrix, coverage, real-model test, wheel/CLI, dependency audit, secret scan, non-root offline container, and [real-model browser QA](browser-qa.json). These are local observations; live CI state is available from the repository Actions badge. Security scans are time-scoped, not safety guarantees.

## Reproduce

Run `make model-smoke` or follow the pinned archive download and `make benchmark` in the root README. The evaluator rejects archive drift and unexpected sample counts; the smoke rejects model identity drift and mismatched sample checksums. GitHub's manual model evaluation workflow saves the same report format as downloadable artifacts.

No field, external-domain, expert, geography/device, fitted-calibration, or OOD claims are supported by these results.
