# Engineering case study

## Problem and scope

A classifier can always emit a label, even when an uploaded image is corrupted, blurred, unsupported, or ambiguous. LeafSentry focuses on the decision boundary around a model: what to accept, when to abstain, how to explain the output, and what evidence supports the system.

The scope is deliberately three bean-leaf classes. A broader farm advisor would require region-specific data, crop recognition, expert validation, and treatment safety that this project does not have.

## Implemented decisions

A stateless modular monolith separates image safety, model access, selection policy, and HTTP delivery. The predictor protocol permits deterministic CI without weights; the pinned real adapter is exercised separately. Image-quality failure skips inference, while confidence and top-two margin provide distinct abstention reasons after inference.

Raw logits preserve an explicit temperature layer. Temperature 1.0 does not improve calibration by itself. ECE, NLL, Brier score, and accepted-set risk make the limits measurable rather than hiding them behind a confidence threshold.

A CPU-specific lock avoids CUDA dependencies. A runtime audit found issues in an older Transformers pin; the dependency was updated and real inference re-exercised. A second check found that pip-audit silently skipped `+cpu` wheel versions. The audit wrapper now queries the corresponding upstream torch/torchvision release versions and fails on skips or incomplete evidence. The paired runtime was upgraded and the model benchmark reproduced. The API moves synchronous inference to a serialized worker thread, preserving liveness while preventing concurrent lazy loading. A restrictive CSP revealed a broken CDN-backed docs page; a local script-free reference removes that dependency.

## Evidence and interpretation

The three-sample smoke proves model loading, preprocessing, label mapping, and score policy wiring. The 128-image held-out report assesses the model and score policy on the upstream dataset; it bypasses the quality guard. The [evidence page](evaluation/README.md) records the results and limitations.

The default policy is not validation-selected. The held-out threshold sweep is explanatory: increased confidence can reduce coverage sharply, and selective risk need not decrease monotonically on a small sample. A production policy would require validation-selected thresholds, explicit failure costs, and an untouched external/field test.

## What remains unvalidated

OOD detection, crop identification, fitted calibration, expert-backed diagnosis, field/device/geography shifts, fairness, and throughput/SLO targets remain open. There is no treatment advice or claim that the system is safe for agricultural decisions.

## Interview walkthrough

Start with an accepted prediction, show a dark image abstaining before inference, then malformed input returning a safe error. Compare liveness to readiness before weights load. Open the confusion matrix and coverage-risk sweep, explain which three tasks the smoke verifies, and describe why high confidence is not OOD protection. Finish with the locked runtime, tests, and clean wheel/container execution.
