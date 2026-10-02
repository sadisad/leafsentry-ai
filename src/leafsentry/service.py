"""HTTP-independent orchestration for image triage."""

from __future__ import annotations

import time

from leafsentry.config import Settings
from leafsentry.image_guard import inspect_image
from leafsentry.prediction import Predictor, SelectionPolicy, select_prediction
from leafsentry.schemas import (
    Decision,
    ModelIdentity,
    PolicySnapshot,
    PredictionResponse,
)


class LeafSentryService:
    def __init__(self, *, settings: Settings, predictor: Predictor) -> None:
        self.settings = settings
        self.predictor = predictor
        self.model_identity = ModelIdentity(
            id=settings.model_id,
            revision=settings.model_revision,
        )
        self.policy = SelectionPolicy(
            min_confidence=settings.min_confidence,
            min_margin=settings.min_margin,
            temperature=settings.temperature,
        )

    @property
    def is_ready(self) -> bool:
        return self.predictor.is_ready

    def predict(self, data: bytes, *, media_type: str, request_id: str) -> PredictionResponse:
        started = time.perf_counter()
        inspected = inspect_image(data, media_type=media_type, settings=self.settings)
        policy = PolicySnapshot(
            min_confidence=self.policy.min_confidence,
            min_margin=self.policy.min_margin,
            temperature=self.policy.temperature,
        )

        if not inspected.report.passed:
            return PredictionResponse(
                request_id=request_id,
                decision=Decision.ABSTAINED,
                prediction=None,
                confidence=None,
                margin=None,
                normalized_entropy=None,
                scores=[],
                quality=inspected.report,
                abstention_reasons=["poor_image_quality"],
                model=self.model_identity,
                policy=policy,
                latency_ms=(time.perf_counter() - started) * 1_000.0,
            )

        selection = select_prediction(
            self.predictor.predict(inspected.image),
            policy=self.policy,
        )
        return PredictionResponse(
            request_id=request_id,
            decision=selection.decision,
            prediction=selection.prediction,
            confidence=selection.confidence,
            margin=selection.margin,
            normalized_entropy=selection.normalized_entropy,
            scores=list(selection.scores),
            quality=inspected.report,
            abstention_reasons=list(selection.abstention_reasons),
            model=self.model_identity,
            policy=policy,
            latency_ms=(time.perf_counter() - started) * 1_000.0,
        )
