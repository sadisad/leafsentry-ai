from leafsentry.schemas import (
    Decision,
    ModelIdentity,
    PredictionResponse,
    PredictionScore,
    QualityMetrics,
    QualityReport,
)


def test_prediction_response_serializes_a_versioned_accepted_decision() -> None:
    response = PredictionResponse(
        request_id="req-test",
        decision=Decision.ACCEPTED,
        prediction="healthy",
        confidence=0.91,
        margin=0.73,
        normalized_entropy=0.31,
        scores=[PredictionScore(label="healthy", probability=0.91)],
        quality=QualityReport(
            passed=True,
            findings=[],
            metrics=QualityMetrics(
                width=224,
                height=224,
                format="JPEG",
                mean_brightness=121.2,
                contrast=42.0,
                sharpness=91.3,
            ),
        ),
        model=ModelIdentity(id="model", revision="a" * 40),
        latency_ms=12.5,
    )

    payload = response.model_dump(mode="json")

    assert payload["schema_version"] == "1.0"
    assert payload["decision"] == "accepted"
    assert payload["prediction"] == "healthy"
    assert payload["abstention_reasons"] == []
    assert "not a field-validated diagnosis" in payload["disclaimer"]


def test_abstained_response_cannot_expose_a_prediction() -> None:
    quality = QualityReport(
        passed=False,
        findings=["too_dark"],
        metrics=QualityMetrics(
            width=224,
            height=224,
            format="PNG",
            mean_brightness=1.0,
            contrast=0.0,
            sharpness=0.0,
        ),
    )

    response = PredictionResponse(
        request_id="req-test",
        decision=Decision.ABSTAINED,
        prediction=None,
        confidence=None,
        margin=None,
        normalized_entropy=None,
        scores=[],
        quality=quality,
        abstention_reasons=["poor_image_quality"],
        model=ModelIdentity(id="model", revision="b" * 40),
        latency_ms=1.2,
    )

    assert response.prediction is None
    assert response.abstention_reasons == ["poor_image_quality"]
