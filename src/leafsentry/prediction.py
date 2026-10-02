"""Calibration, uncertainty, and selective-prediction policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image

from leafsentry.errors import ModelOutputError
from leafsentry.schemas import Decision, PredictionScore

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RawPrediction:
    labels: tuple[str, ...]
    logits: FloatArray


@runtime_checkable
class Predictor(Protocol):
    @property
    def is_ready(self) -> bool: ...

    def predict(self, image: Image.Image) -> RawPrediction: ...


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    min_confidence: float = 0.75
    min_margin: float = 0.20
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_confidence) or not 0.0 < self.min_confidence < 1.0:
            raise ValueError("min_confidence must be finite and between 0 and 1")
        if not math.isfinite(self.min_margin) or not 0.0 < self.min_margin < 1.0:
            raise ValueError("min_margin must be finite and between 0 and 1")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")


@dataclass(frozen=True, slots=True)
class Selection:
    decision: Decision
    prediction: str | None
    confidence: float
    margin: float
    normalized_entropy: float
    scores: tuple[PredictionScore, ...]
    abstention_reasons: tuple[str, ...]


def temperature_softmax(logits: npt.ArrayLike, *, temperature: float) -> FloatArray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ModelOutputError()
    scaled = values / temperature
    shifted = scaled - np.max(scaled)
    exponentials = np.exp(shifted)
    total = float(np.sum(exponentials))
    if not math.isfinite(total) or total <= 0.0:
        raise ModelOutputError()
    return np.asarray(exponentials / total, dtype=np.float64)


def normalized_entropy(probabilities: npt.ArrayLike) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not math.isclose(float(np.sum(values)), 1.0, rel_tol=1e-7, abs_tol=1e-7)
    ):
        raise ModelOutputError("Probabilities must be finite and sum to one.")
    nonzero = values[values > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return min(1.0, max(0.0, entropy / math.log(values.size)))


def _validate_raw(raw: RawPrediction) -> None:
    if raw.logits.ndim != 1 or raw.logits.size < 2:
        raise ModelOutputError()
    if len(raw.labels) != raw.logits.size:
        raise ModelOutputError()
    if len(set(raw.labels)) != len(raw.labels) or any(not label for label in raw.labels):
        raise ModelOutputError()
    if not np.all(np.isfinite(raw.logits)):
        raise ModelOutputError()


def select_prediction(raw: RawPrediction, *, policy: SelectionPolicy) -> Selection:
    _validate_raw(raw)
    probabilities = temperature_softmax(raw.logits, temperature=policy.temperature)
    ranked_indices = np.argsort(-probabilities, kind="stable")
    top_index = int(ranked_indices[0])
    runner_up_index = int(ranked_indices[1])
    confidence = float(probabilities[top_index])
    margin = confidence - float(probabilities[runner_up_index])

    reasons: list[str] = []
    if confidence < policy.min_confidence:
        reasons.append("low_confidence")
    if margin < policy.min_margin:
        reasons.append("ambiguous_top_classes")

    scores = tuple(
        PredictionScore(label=raw.labels[int(index)], probability=float(probabilities[index]))
        for index in ranked_indices
    )
    accepted = not reasons
    return Selection(
        decision=Decision.ACCEPTED if accepted else Decision.ABSTAINED,
        prediction=raw.labels[top_index] if accepted else None,
        confidence=confidence,
        margin=margin,
        normalized_entropy=normalized_entropy(probabilities),
        scores=scores,
        abstention_reasons=tuple(reasons),
    )
