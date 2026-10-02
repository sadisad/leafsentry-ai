from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from leafsentry.api import create_app
from leafsentry.config import Settings
from tests.fakes import FakePredictor, textured_png_bytes


def api_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "min_brightness": 5.0,
        "max_brightness": 250.0,
        "min_contrast": 5.0,
        "min_sharpness": 1.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def build_app(
    *, predictor: FakePredictor | None = None, settings: Settings | None = None
) -> FastAPI:
    return create_app(
        settings=settings or api_settings(),
        predictor=predictor or FakePredictor(),
        registry=CollectorRegistry(),
    )


def request(app: FastAPI, method: str, path: str, **kwargs: object) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_liveness_does_not_depend_on_model_readiness() -> None:
    response = request(build_app(predictor=FakePredictor(ready=False)), "GET", "/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_unavailable_model() -> None:
    response = request(build_app(predictor=FakePredictor(ready=False)), "GET", "/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_prediction_endpoint_returns_versioned_contract_without_filename() -> None:
    response = request(
        build_app(),
        "POST",
        "/v1/predictions",
        files={"file": ("private-farm-name.png", textured_png_bytes(), "image/png")},
        headers={"X-Request-ID": "portfolio-test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["request_id"] == "portfolio-test"
    assert body["decision"] == "accepted"
    assert body["prediction"] == "healthy"
    assert "private-farm-name" not in response.text


def test_prediction_endpoint_returns_safe_invalid_image_error() -> None:
    response = request(
        build_app(),
        "POST",
        "/v1/predictions",
        files={"file": ("bad.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"
    assert "Traceback" not in response.text


def test_prediction_endpoint_rejects_oversized_stream() -> None:
    response = request(
        build_app(settings=api_settings(max_upload_bytes=64)),
        "POST",
        "/v1/predictions",
        files={"file": ("large.png", b"x" * 65, "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_prediction_endpoint_returns_503_when_model_is_not_ready() -> None:
    response = request(
        build_app(predictor=FakePredictor(ready=False)),
        "POST",
        "/v1/predictions",
        files={"file": ("leaf.png", textured_png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_metrics_endpoint_exposes_bounded_outcome_labels() -> None:
    app = build_app()
    prediction = request(
        app,
        "POST",
        "/v1/predictions",
        files={"file": ("leaf.png", textured_png_bytes(), "image/png")},
    )
    assert prediction.status_code == 200

    response = request(app, "GET", "/metrics")

    assert response.status_code == 200
    assert "leafsentry_predictions_total" in response.text
    assert 'outcome="accepted"' in response.text
    assert "request_id" not in response.text
    assert "filename" not in response.text
