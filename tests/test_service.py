import pytest

from leafsentry.config import Settings
from leafsentry.errors import ModelUnavailableError
from leafsentry.schemas import Decision
from leafsentry.service import LeafSentryService
from tests.fakes import FakePredictor, solid_png_bytes, textured_png_bytes


def service_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "min_brightness": 5.0,
        "max_brightness": 250.0,
        "min_contrast": 5.0,
        "min_sharpness": 1.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_service_returns_an_accepted_prediction() -> None:
    predictor = FakePredictor()
    service = LeafSentryService(settings=service_settings(), predictor=predictor)

    response = service.predict(
        textured_png_bytes(), media_type="image/png", request_id="req-accepted"
    )

    assert response.decision is Decision.ACCEPTED
    assert response.prediction == "healthy"
    assert response.model.id == "nateraw/vit-base-beans"
    assert response.model.revision == service_settings().model_revision
    assert response.policy.min_confidence == service_settings().min_confidence
    assert response.quality.passed is True
    assert response.latency_ms >= 0.0
    assert predictor.calls == 1


def test_service_abstains_before_inference_for_poor_quality() -> None:
    predictor = FakePredictor()
    service = LeafSentryService(settings=service_settings(), predictor=predictor)

    response = service.predict(solid_png_bytes(), media_type="image/png", request_id="req-quality")

    assert response.decision is Decision.ABSTAINED
    assert response.prediction is None
    assert response.abstention_reasons == ["poor_image_quality"]
    assert predictor.calls == 0


def test_service_rejects_prediction_when_model_is_not_ready() -> None:
    service = LeafSentryService(settings=service_settings(), predictor=FakePredictor(ready=False))

    with pytest.raises(ModelUnavailableError):
        service.predict(textured_png_bytes(), media_type="image/png", request_id="req-down")
