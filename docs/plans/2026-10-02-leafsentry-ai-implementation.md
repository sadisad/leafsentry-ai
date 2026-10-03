# LeafSentry AI Implementation Plan

This document records the original implementation sequence. Current behavior and measured evidence are described in the root README and model/system card.

**Goal:** Build and publish a verified, production-oriented computer-vision triage service that demonstrates safe inference, selective prediction, evaluation, API engineering, observability, and MLOps practices.

**Architecture:** A FastAPI app owns the untrusted upload boundary and delegates to an HTTP-independent service. The service runs a deterministic image guard, a protocol-based predictor, temperature scaling, and an abstention policy; an optional lazy Hugging Face adapter loads one pinned model while CI uses real in-process fakes.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, Pillow, NumPy, prometheus-client, optional PyTorch/Transformers, pytest, Ruff, mypy, uv, Docker, GitHub Actions.

---

### Task 1: Establish the package and public contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/leafsentry/__init__.py`
- Create: `src/leafsentry/config.py`
- Create: `src/leafsentry/schemas.py`
- Test: `tests/test_config.py`
- Test: `tests/test_schemas.py`

**Steps:**
1. Write one failing settings/defaults test and run it.
2. Implement the minimum immutable settings object and rerun the test.
3. Write one failing schema serialization test and run it.
4. Implement the minimum versioned response models and rerun it.
5. Run the focused tests, format, and commit.

### Task 2: Guard untrusted images

**Files:**
- Create: `src/leafsentry/errors.py`
- Create: `src/leafsentry/image_guard.py`
- Test: `tests/test_image_guard.py`

**Steps:**
1. Write and fail a valid JPEG decode test; implement decode + EXIF normalization.
2. Write and fail empty, truncated, unsupported-format, media-type mismatch, dimensions, and decompression-limit tests one behavior at a time; implement each minimal guard.
3. Write and fail dark, overexposed, low-contrast, and blurry quality tests one behavior at a time; implement deterministic measurements and findings.
4. Run all image-guard tests and commit.

### Task 3: Implement calibrated selective prediction

**Files:**
- Create: `src/leafsentry/prediction.py`
- Test: `tests/test_prediction.py`

**Steps:**
1. Write and fail stable softmax tests; implement temperature-scaled softmax.
2. Write and fail probability validation and entropy tests; implement them.
3. Write and fail accepted, low-confidence, and ambiguous-margin decisions; implement the policy.
4. Verify probabilities sum to one and all focused tests pass; commit.

### Task 4: Orchestrate inference and expose the API

**Files:**
- Create: `src/leafsentry/service.py`
- Create: `src/leafsentry/metrics.py`
- Create: `src/leafsentry/api.py`
- Create: `src/leafsentry/__main__.py`
- Test: `tests/fakes.py`
- Test: `tests/test_service.py`
- Test: `tests/test_api.py`

**Steps:**
1. Build the service vertical slice through failing accepted/abstained tests.
2. Add model-unavailable behavior through a failing test.
3. Add an app factory and fail/pass liveness, readiness, prediction, upload-error, and metrics contract tests sequentially.
4. Ensure responses do not echo filenames or bytes; verify metrics avoid unbounded labels.
5. Run service/API tests and commit.

### Task 5: Add reproducible evaluation

**Files:**
- Create: `src/leafsentry/evaluation.py`
- Create: `src/leafsentry/cli.py`
- Create: `examples/predictions.example.jsonl`
- Test: `tests/test_evaluation.py`
- Test: `tests/test_cli.py`

**Steps:**
1. Add accuracy and macro-F1 via failing tests.
2. Add NLL, multiclass Brier, and ECE via failing tests with hand-calculated fixtures.
3. Add coverage/selective-risk and threshold sweep via failing tests.
4. Add a JSONL CLI and validate output determinism in a failing-then-passing CLI test.
5. Run focused tests and commit.

### Task 6: Integrate the pinned open model

**Files:**
- Create: `src/leafsentry/backends/__init__.py`
- Create: `src/leafsentry/backends/huggingface.py`
- Create: `scripts/model_smoke.py`
- Test: `tests/test_huggingface_backend.py`
- Test: `tests/model/test_real_model.py`

**Steps:**
1. Write a failing adapter test using injected processor/model doubles; no PyTorch download.
2. Implement lazy loading, exact revision pinning, `trust_remote_code=False`, eval/no-grad behavior, and label extraction.
3. Download the three licensed upstream sample images at test runtime with checksums recorded in a manifest.
4. Install optional ML dependencies in an isolated environment and run the marked real-model smoke test.
5. Record only observed outputs and timings; commit.

### Task 7: Build the operator surface and observability

**Files:**
- Create: `src/leafsentry/static/index.html`
- Create: `src/leafsentry/static/app.css`
- Create: `src/leafsentry/static/app.js`
- Test: `tests/test_static.py`

**Steps:**
1. Write failing static-route and required-accessibility-marker tests.
2. Implement the **Operate** surface: upload/drop, preview, model state, result/abstention, uncertainty, quality findings, and limitations.
3. Add loading, error, keyboard focus, reduced-motion, responsive, and no-JavaScript states.
4. Exercise the UI against the local API, capture screenshots at desktop/mobile, run the slop diagnostic, and fix flagged issues.
5. Commit.

### Task 8: Package, secure, and document

**Files:**
- Create: `README.md`
- Create: `MODEL_CARD.md`
- Create: `SECURITY.md`
- Create: `CONTRIBUTING.md`
- Create: `LICENSE`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `.dockerignore`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `Makefile`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/model-smoke.yml`
- Create: `.github/dependabot.yml`

**Steps:**
1. Document the exact scope, architecture, API examples, reproducibility, upstream versus reproduced metrics, and limitations.
2. Add a non-root container with a health check and pinned lock file.
3. Add CI for tests, coverage, Ruff, mypy, build, and container build; keep weight download out of standard CI.
4. Add a manual model-smoke workflow pinned to action SHAs where practical.
5. Validate YAML, build the wheel and container, and commit.

### Task 9: Verify and publish

**Files:**
- Modify: GitHub profile repository `sadisad/sadisad/README.md`

**Steps:**
1. Run the complete local verification gate: format check, lint, type check, full non-model test suite with coverage, package build/install smoke, dependency audit, secret scan, Docker build/run/health/API smoke.
2. Review the entire diff and perform an independent code/security review; fix material findings and rerun all affected gates.
3. Create public repository `sadisad/leafsentry-ai`, push the verified history, set description/topics, and enable issues while disabling unused wiki/projects.
4. Verify GitHub Actions from the pushed SHA; fix failures rather than weakening checks.
5. Update the profile README Featured Projects section through an isolated clone/commit/push and verify the rendered source contains the link.
6. Report the exact repository URL, commit SHA, observed verification evidence, and any honest remaining limitations.
