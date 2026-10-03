import pytest
from prometheus_client import CollectorRegistry

from leafsentry.config import Settings
from leafsentry.metrics import Metrics


@pytest.mark.parametrize(
    "values",
    [
        {"min_confidence": 1.0},
        {"min_margin": float("nan")},
        {"temperature": 0.0},
        {"max_upload_bytes": 0},
        {"min_image_dimension": 800, "max_image_dimension": 100},
    ],
)
def test_settings_reject_invalid_operational_values(values):
    with pytest.raises(ValueError):
        Settings(**values)


def test_metrics_reject_unbounded_labels():
    with pytest.raises(ValueError, match="outcome"):
        Metrics(CollectorRegistry()).observe(outcome="private-user-name", duration_seconds=1)
