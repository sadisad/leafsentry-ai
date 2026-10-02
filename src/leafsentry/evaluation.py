"""Reproducible classification, calibration, and selective-risk metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    target: str
    probabilities: dict[str, float]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    samples: int
    accepted: int
    accuracy: float
    macro_f1: float
    negative_log_likelihood: float
    multiclass_brier: float
    expected_calibration_error: float
    coverage: float
    selective_accuracy: float | None
    selective_risk: float | None


@dataclass(frozen=True, slots=True)
class SweepPoint:
    min_confidence: float
    min_margin: float
    coverage: float
    selective_accuracy: float | None
    selective_risk: float | None


def _validate(rows: list[EvaluationRow]) -> tuple[str, ...]:
    if not rows:
        raise ValueError("at least one evaluation row is required")
    labels = tuple(rows[0].probabilities)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("each row must contain at least two unique labels")
    expected = set(labels)
    for row in rows:
        values = row.probabilities
        if set(values) != expected or row.target not in expected:
            raise ValueError("all rows must share labels and include the target")
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("probabilities must be finite and non-negative")
        if not math.isclose(sum(values.values()), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("probabilities must sum to one")
    return labels


def _rank(row: EvaluationRow) -> list[tuple[str, float]]:
    return sorted(row.probabilities.items(), key=lambda item: (-item[1], item[0]))


def _macro_f1(rows: list[EvaluationRow], labels: tuple[str, ...]) -> float:
    predictions = [_rank(row)[0][0] for row in rows]
    values: list[float] = []
    for label in labels:
        tp = sum(
            prediction == label and row.target == label
            for row, prediction in zip(rows, predictions, strict=True)
        )
        fp = sum(
            prediction == label and row.target != label
            for row, prediction in zip(rows, predictions, strict=True)
        )
        fn = sum(
            prediction != label and row.target == label
            for row, prediction in zip(rows, predictions, strict=True)
        )
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(values) / len(values)


def _ece(rows: list[EvaluationRow], bins: int) -> float:
    if bins <= 0:
        raise ValueError("ece_bins must be positive")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for row in rows:
        label, confidence = _rank(row)[0]
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, label == row.target))
    result = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        result += len(bucket) / len(rows) * abs(accuracy - confidence)
    return result


def _selection(
    rows: list[EvaluationRow], min_confidence: float, min_margin: float
) -> tuple[int, float | None]:
    accepted = 0
    correct = 0
    for row in rows:
        ranked = _rank(row)
        confidence = ranked[0][1]
        margin = confidence - ranked[1][1]
        if confidence >= min_confidence and margin >= min_margin:
            accepted += 1
            correct += ranked[0][0] == row.target
    return accepted, None if accepted == 0 else correct / accepted


def evaluate(
    rows: list[EvaluationRow],
    *,
    min_confidence: float,
    min_margin: float,
    ece_bins: int = 10,
) -> EvaluationReport:
    labels = _validate(rows)
    predictions = [_rank(row)[0][0] for row in rows]
    accuracy = sum(
        prediction == row.target for row, prediction in zip(rows, predictions, strict=True)
    ) / len(rows)
    epsilon = 1e-15
    nll = -sum(math.log(max(row.probabilities[row.target], epsilon)) for row in rows) / len(rows)
    brier = sum(
        sum((row.probabilities[label] - float(label == row.target)) ** 2 for label in labels)
        for row in rows
    ) / len(rows)
    accepted, selective_accuracy = _selection(rows, min_confidence, min_margin)
    return EvaluationReport(
        samples=len(rows),
        accepted=accepted,
        accuracy=accuracy,
        macro_f1=_macro_f1(rows, labels),
        negative_log_likelihood=nll,
        multiclass_brier=brier,
        expected_calibration_error=_ece(rows, ece_bins),
        coverage=accepted / len(rows),
        selective_accuracy=selective_accuracy,
        selective_risk=None if selective_accuracy is None else 1.0 - selective_accuracy,
    )


def threshold_sweep(
    rows: list[EvaluationRow],
    *,
    confidence_thresholds: list[float],
    margin_thresholds: list[float],
) -> list[SweepPoint]:
    _validate(rows)
    points: list[SweepPoint] = []
    for confidence in sorted(confidence_thresholds):
        for margin in sorted(margin_thresholds):
            accepted, selective_accuracy = _selection(rows, confidence, margin)
            points.append(
                SweepPoint(
                    min_confidence=confidence,
                    min_margin=margin,
                    coverage=accepted / len(rows),
                    selective_accuracy=selective_accuracy,
                    selective_risk=None if selective_accuracy is None else 1.0 - selective_accuracy,
                )
            )
    return points
