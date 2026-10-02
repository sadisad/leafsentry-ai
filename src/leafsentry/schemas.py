"""Versioned public and internal domain schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Decision(StrEnum):
    ACCEPTED = "accepted"
    ABSTAINED = "abstained"


class PredictionScore(StrictModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class QualityMetrics(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str
    mean_brightness: float = Field(ge=0.0, le=255.0)
    contrast: float = Field(ge=0.0)
    sharpness: float = Field(ge=0.0)


class QualityReport(StrictModel):
    passed: bool
    findings: list[str]
    metrics: QualityMetrics


class ModelIdentity(StrictModel):
    id: str
    revision: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")


class PolicySnapshot(StrictModel):
    min_confidence: float = Field(default=0.75, gt=0.0, lt=1.0)
    min_margin: float = Field(default=0.20, gt=0.0, lt=1.0)
    temperature: float = Field(default=1.0, gt=0.0)


class PredictionResponse(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    decision: Decision
    prediction: str | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    margin: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    scores: list[PredictionScore]
    quality: QualityReport
    abstention_reasons: list[str] = Field(default_factory=list)
    model: ModelIdentity
    policy: PolicySnapshot = Field(default_factory=PolicySnapshot)
    latency_ms: float = Field(ge=0.0)
    disclaimer: str = "Educational decision-support demo; this is not a field-validated diagnosis."

    @model_validator(mode="after")
    def decision_fields_are_consistent(self) -> PredictionResponse:
        if self.decision is Decision.ACCEPTED:
            if self.prediction is None or self.confidence is None:
                raise ValueError("accepted decisions require a prediction and confidence")
            if self.abstention_reasons:
                raise ValueError("accepted decisions cannot have abstention reasons")
        elif self.prediction is not None:
            raise ValueError("abstained decisions cannot expose a prediction")
        elif not self.abstention_reasons:
            raise ValueError("abstained decisions require at least one reason")
        return self


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorDetail


class HealthResponse(StrictModel):
    status: Literal["ok", "ready", "not_ready"]
    model: ModelIdentity
