from __future__ import annotations

from PIL import Image

from leafsentry.benchmark import BenchmarkSample, run_benchmark
from leafsentry.config import MODEL_ID, MODEL_REVISION, Settings
from tests.fakes import FakePredictor


def test_run_benchmark_reports_model_policy_and_selective_metrics() -> None:
    samples = [
        BenchmarkSample("healthy", "one.jpg", Image.new("RGB", (32, 32), "green")),
        BenchmarkSample("healthy", "two.jpg", Image.new("RGB", (32, 32), "green")),
    ]

    report = run_benchmark(
        samples,
        predictor=FakePredictor(),
        settings=Settings(),
        dataset={"id": "example/test", "revision": "abc", "sha256": "0" * 64},
    )

    assert report["model"] == {"id": MODEL_ID, "revision": MODEL_REVISION}
    assert report["dataset"]["samples"] == 2
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["coverage"] == 1.0
    assert report["confusion_matrix"]["healthy"]["healthy"] == 2
    assert len(report["threshold_sweep"]) > 1
    assert report["runtime"]["device"] == "cpu"
    assert len(report["predictions"]) == 2
    assert report["predictions"][0]["correct"] is True
