# Contributing

Use Python 3.11 or 3.12 and uv. Create a scoped branch and keep model weights, dataset images, secrets, and caches out of Git.

```bash
uv sync --frozen --extra dev --no-install-project
make verify
```

Write a failing regression test before changing behavior. Standard tests must not download model weights or need a GPU. New public response fields must preserve schema invariants and explain compatibility. Keep filenames, images, and unbounded IDs out of telemetry labels.

For ML changes, pin all artifacts, run `make model-smoke`, and reproduce the test evaluation. Document preprocessing changes and distinguish model-only evaluation from the guarded service. Do not select thresholds on the test set or describe temperature 1.0 as fitted calibration.

Dependency changes must update `uv.lock`, audit the runtime including ML extras, and exercise a real inference. Upgrade torch and torchvision together against their official CPU-wheel compatibility; automatic version-only bumps are ignored for this pair, while security updates remain monitored. Update evidence artifacts only from executions of the final code. CI and action permissions must stay least privilege; use full commit SHAs for actions.

Review [security](SECURITY.md), [model/system card](MODEL_CARD.md), and [operations](docs/OPERATIONS.md) before proposing public deployment changes.
