# Operations runbook

## Start and health semantics

`uv sync --frozen --extra ml && uv run --no-sync leafsentry-api` serves on port 8000. Compose publishes only `127.0.0.1:8000` by default. The first quality-passing prediction lazy-loads the pinned model; before it succeeds, `/health/ready` is 503. `/health/live` reports process liveness without triggering a download. Docker's health check uses liveness, not readiness, so the container can boot before weights are available.

Inference is serialized per process and runs outside the ASGI event loop. Adding workers duplicates weights and cache/memory demand. This repository does not implement autoscaling or a throughput SLO.

## Model cache

`HF_HOME` is `/home/leafsentry/.cache/huggingface` in the container. Compose mounts a named volume there. Cache writes require UID 10001 ownership. If permissions fail, fix only that volume; do not run the API as root. The runtime includes CPU torch/torchvision/transformers, not model weights. Allow first-load network access to the pinned upstream registry, then use the populated cache for offline smoke testing.

## Observation

`GET /metrics` exposes `leafsentry_predictions_total{outcome="accepted|abstained|error"}` and `leafsentry_prediction_duration_seconds` histograms. Latency includes request processing and model work; no filename, image bytes, or request ID is a metric label. Successful predictions log request ID, outcome, and latency. Unexpected errors log an internal trace while returning a safe public error.

Example PromQL:

```promql
sum(rate(leafsentry_predictions_total{outcome="error"}[5m]))
/ clamp_min(sum(rate(leafsentry_predictions_total[5m])), 0.001)
```

```promql
histogram_quantile(0.95, sum by (le) (rate(leafsentry_prediction_duration_seconds_bucket[5m])))
```

Scrape `/metrics` on a private network. An increase in abstentions requires inspection of input and policy behavior; it is not automatically evidence of drift or improved safety.

## Before public exposure

Provide TLS, access control if needed, body-size/rate/concurrency limits, and header/upload timeouts at the reverse proxy. The application file bound does not protect the earlier multipart parser against arbitrary total request bodies. See [security](../SECURITY.md). No public deployment or uptime claim is made by this repository.

The reference public demo is `https://leafsentry.syd.my.id`. Its checked-in deployment files bind the origin to `127.0.0.1:8010`, cap HTTP request bodies at 6 MiB, rate-limit predictions, restrict `/metrics` to localhost, and keep model loading offline from a pinned cache. The demo is still unauthenticated and carries no availability commitment.

## Rollback and shutdown

Tag images by immutable Git SHA for a deployment. Keep the previous image and cache volume; rollback by selecting the earlier image, then verify liveness, readiness after inference, and one prediction. `docker compose down` stops the app but retains the model cache. Do not use `down --volumes` unless cache deletion is intended.
