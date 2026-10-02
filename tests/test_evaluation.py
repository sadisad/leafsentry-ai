import pytest

from leafsentry.evaluation import EvaluationRow, evaluate, threshold_sweep


def sample_rows() -> list[EvaluationRow]:
    return [
        EvaluationRow(
            target="a",
            probabilities={"a": 0.8, "b": 0.1, "c": 0.1},
        ),
        EvaluationRow(
            target="b",
            probabilities={"a": 0.6, "b": 0.4, "c": 0.0},
        ),
        EvaluationRow(
            target="c",
            probabilities={"a": 0.05, "b": 0.05, "c": 0.9},
        ),
    ]


def test_evaluate_computes_classification_and_calibration_metrics() -> None:
    report = evaluate(sample_rows(), min_confidence=0.0, min_margin=0.0, ece_bins=1)

    assert report.samples == 3
    assert report.accuracy == pytest.approx(2 / 3)
    assert report.macro_f1 == pytest.approx((0.6666667 + 0.0 + 1.0) / 3)
    assert report.negative_log_likelihood == pytest.approx(0.414931599615)
    assert report.multiclass_brier == pytest.approx(0.265)
    assert report.expected_calibration_error == pytest.approx(0.1)


def test_evaluate_reports_coverage_and_selective_risk() -> None:
    report = evaluate(sample_rows(), min_confidence=0.75, min_margin=0.20, ece_bins=5)

    assert report.accepted == 2
    assert report.coverage == pytest.approx(2 / 3)
    assert report.selective_accuracy == pytest.approx(1.0)
    assert report.selective_risk == pytest.approx(0.0)


def test_threshold_sweep_is_deterministic() -> None:
    points = threshold_sweep(
        sample_rows(),
        confidence_thresholds=[0.0, 0.75],
        margin_thresholds=[0.0, 0.2],
    )

    assert [(point.min_confidence, point.min_margin) for point in points] == [
        (0.0, 0.0),
        (0.0, 0.2),
        (0.75, 0.0),
        (0.75, 0.2),
    ]
    assert points[-1].coverage == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "row",
    [
        EvaluationRow(target="missing", probabilities={"a": 1.0}),
        EvaluationRow(target="a", probabilities={"a": 0.4, "b": 0.4}),
        EvaluationRow(target="a", probabilities={"a": -0.1, "b": 1.1}),
    ],
)
def test_evaluate_rejects_invalid_probability_rows(row: EvaluationRow) -> None:
    with pytest.raises(ValueError):
        evaluate([row], min_confidence=0.0, min_margin=0.0)


def test_evaluate_handles_zero_coverage_without_division_error() -> None:
    report = evaluate(sample_rows(), min_confidence=0.99, min_margin=0.99)

    assert report.coverage == 0.0
    assert report.selective_accuracy is None
    assert report.selective_risk is None
