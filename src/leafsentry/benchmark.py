"""Pinned held-out dataset evaluation for the production inference backend."""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError

from leafsentry.config import Settings
from leafsentry.evaluation import EvaluationRow, evaluate, threshold_sweep
from leafsentry.prediction import Predictor, SelectionPolicy, select_prediction

_ALLOWED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_MAX_ARCHIVE_ENTRY_BYTES = 10 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


def _runtime_versions() -> dict[str, str]:
    runtime = {"python": platform.python_version(), "device": "cpu"}
    for package in ("torch", "torchvision", "transformers"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "not_installed"
    return runtime


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    label: str
    filename: str
    image: Image.Image


def load_image_archive(
    path: Path,
    *,
    expected_sha256: str,
    expected_samples: int,
    allowed_labels: frozenset[str],
) -> list[BenchmarkSample]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("dataset archive checksum does not match the pinned manifest")
    if expected_samples <= 0 or not allowed_labels:
        raise ValueError("dataset contract must define samples and labels")

    samples: list[BenchmarkSample] = []
    total_bytes = 0
    try:
        with ZipFile(path) as archive:
            files = sorted(
                (item for item in archive.infolist() if not item.is_dir()), key=lambda x: x.filename
            )
            for item in files:
                parts = PurePosixPath(item.filename).parts
                if (
                    len(parts) != 3
                    or parts[0] != "test"
                    or parts[1] not in allowed_labels
                    or PurePosixPath(item.filename).suffix.lower() not in _ALLOWED_SUFFIXES
                    or item.file_size <= 0
                    or item.file_size > _MAX_ARCHIVE_ENTRY_BYTES
                ):
                    raise ValueError(f"unsafe or unexpected archive entry: {item.filename}")
                total_bytes += item.file_size
                if total_bytes > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise ValueError("dataset archive exceeds the uncompressed size limit")
                payload = archive.read(item)
                with Image.open(BytesIO(payload)) as source:
                    source.verify()
                with Image.open(BytesIO(payload)) as source:
                    image = source.convert("RGB")
                    image.load()
                samples.append(BenchmarkSample(label=parts[1], filename=parts[2], image=image))
    except (BadZipFile, OSError, UnidentifiedImageError) as exc:
        raise ValueError("dataset archive contains an invalid image or ZIP structure") from exc

    if len(samples) != expected_samples:
        raise ValueError(f"dataset sample count is {len(samples)}, expected {expected_samples}")
    return samples


def run_benchmark(
    samples: list[BenchmarkSample],
    *,
    predictor: Predictor,
    settings: Settings,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    rows: list[EvaluationRow] = []
    predictions: list[tuple[str, str]] = []
    latencies: list[float] = []
    records: list[dict[str, Any]] = []
    policy = SelectionPolicy(
        min_confidence=settings.min_confidence,
        min_margin=settings.min_margin,
        temperature=settings.temperature,
    )

    for sample in samples:
        started = time.perf_counter()
        selected = select_prediction(predictor.predict(sample.image), policy=policy)
        latencies.append((time.perf_counter() - started) * 1_000.0)
        probabilities = {score.label: score.probability for score in selected.scores}
        rows.append(EvaluationRow(target=sample.label, probabilities=probabilities))
        predictions.append((sample.label, selected.scores[0].label))
        records.append(
            {
                "filename": sample.filename,
                "target": sample.label,
                "top_label": selected.scores[0].label,
                "correct": sample.label == selected.scores[0].label,
                "decision": selected.decision.value,
                "probabilities": probabilities,
            }
        )

    report = evaluate(
        rows,
        min_confidence=settings.min_confidence,
        min_margin=settings.min_margin,
    )
    labels = sorted(rows[0].probabilities)
    confusion = {
        target: {
            predicted: sum(actual == target and guess == predicted for actual, guess in predictions)
            for predicted in labels
        }
        for target in labels
    }
    sweep = threshold_sweep(
        rows,
        confidence_thresholds=[0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
        margin_thresholds=[0.0, 0.10, 0.20, 0.30],
    )
    ordered_latencies = sorted(latencies)
    p95_index = min(len(ordered_latencies) - 1, int(len(ordered_latencies) * 0.95))

    return {
        "schema_version": "1.0",
        "runtime": _runtime_versions(),
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {"id": settings.model_id, "revision": settings.model_revision},
        "dataset": {**dataset, "samples": len(samples), "split": "test"},
        "policy": {
            "min_confidence": settings.min_confidence,
            "min_margin": settings.min_margin,
            "temperature": settings.temperature,
        },
        "metrics": asdict(report),
        "confusion_matrix": confusion,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p95": ordered_latencies[p95_index],
            "scope": (
                "model loading (first sample), preprocessing, forward pass, and score policy "
                "on local CPU; not an HTTP/load benchmark"
            ),
        },
        "predictions": records,
        "threshold_sweep": [asdict(point) for point in sweep],
    }
