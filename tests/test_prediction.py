import math

import numpy as np
import pytest

from leafsentry.errors import ModelOutputError
from leafsentry.prediction import (
    RawPrediction,
    SelectionPolicy,
    normalized_entropy,
    select_prediction,
    temperature_softmax,
)
from leafsentry.schemas import Decision


def test_temperature_softmax_is_stable_and_normalized() -> None:
    probabilities = temperature_softmax(np.array([1000.0, 999.0, 998.0]), temperature=2.0)

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert float(np.sum(probabilities)) == pytest.approx(1.0)
    assert probabilities.tolist() == pytest.approx([0.50648, 0.30720, 0.18632], abs=1e-5)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_temperature_softmax_requires_positive_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        temperature_softmax(np.array([1.0, 2.0]), temperature=temperature)


def test_normalized_entropy_has_known_bounds() -> None:
    assert normalized_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert normalized_entropy(np.array([1 / 3, 1 / 3, 1 / 3])) == pytest.approx(1.0)


def test_select_prediction_accepts_a_confident_separated_class() -> None:
    selection = select_prediction(
        RawPrediction(labels=("spot", "rust", "healthy"), logits=np.array([0.0, 0.0, 4.0])),
        policy=SelectionPolicy(min_confidence=0.75, min_margin=0.20, temperature=1.0),
    )

    assert selection.decision is Decision.ACCEPTED
    assert selection.prediction == "healthy"
    assert selection.confidence > 0.95
    assert selection.margin > 0.90
    assert selection.abstention_reasons == ()
    assert [score.label for score in selection.scores] == ["healthy", "spot", "rust"]


def test_select_prediction_abstains_on_low_confidence() -> None:
    selection = select_prediction(
        RawPrediction(labels=("spot", "rust", "healthy"), logits=np.array([0.0, 0.0, 0.0])),
        policy=SelectionPolicy(min_confidence=0.75, min_margin=0.10, temperature=1.0),
    )

    assert selection.decision is Decision.ABSTAINED
    assert selection.prediction is None
    assert selection.abstention_reasons == ("low_confidence", "ambiguous_top_classes")


def test_select_prediction_abstains_on_small_top_two_margin() -> None:
    selection = select_prediction(
        RawPrediction(labels=("spot", "rust", "healthy"), logits=np.array([3.0, 2.9, -4.0])),
        policy=SelectionPolicy(min_confidence=0.40, min_margin=0.20, temperature=1.0),
    )

    assert selection.confidence > 0.40
    assert selection.decision is Decision.ABSTAINED
    assert selection.abstention_reasons == ("ambiguous_top_classes",)


@pytest.mark.parametrize(
    "raw",
    [
        RawPrediction(labels=("a",), logits=np.array([1.0, 2.0])),
        RawPrediction(labels=("a", "b"), logits=np.array([[1.0, 2.0]])),
        RawPrediction(labels=("a", "b"), logits=np.array([1.0, math.nan])),
        RawPrediction(labels=("a", "a"), logits=np.array([1.0, 2.0])),
    ],
)
def test_select_prediction_rejects_malformed_model_output(raw: RawPrediction) -> None:
    with pytest.raises(ModelOutputError):
        select_prediction(raw, policy=SelectionPolicy())
