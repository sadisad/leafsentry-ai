"""Pinned real-model smoke test with checksum-verified upstream samples."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from leafsentry.backends.huggingface import HuggingFacePredictor
from leafsentry.config import MODEL_ID, MODEL_REVISION, Settings
from leafsentry.prediction import SelectionPolicy, select_prediction

_MAX_SAMPLE_BYTES = 10 * 1024 * 1024
_KNOWN_LABELS = frozenset({"angular_leaf_spot", "bean_rust", "healthy"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SampleSpec:
    filename: str
    expected_label: str
    sha256: str
    url: str


@dataclass(frozen=True, slots=True)
class SmokeManifest:
    model_id: str
    model_revision: str
    samples: tuple[SampleSpec, ...]


def load_manifest(path: Path) -> SmokeManifest:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    samples = tuple(SampleSpec(**sample) for sample in payload["samples"])
    manifest = SmokeManifest(
        model_id=str(payload["model_id"]),
        model_revision=str(payload["model_revision"]),
        samples=samples,
    )
    if manifest.model_id != MODEL_ID or manifest.model_revision != MODEL_REVISION:
        raise ValueError("smoke manifest model identity does not match the source pin")
    if not samples:
        raise ValueError("smoke manifest must contain at least one sample")
    for sample in samples:
        parsed = urlparse(sample.url)
        expected_path = f"/{MODEL_ID}/resolve/{MODEL_REVISION}/{sample.filename}"
        if Path(sample.filename).name != sample.filename:
            raise ValueError("sample filename must be a basename")
        if sample.expected_label not in _KNOWN_LABELS:
            raise ValueError("sample label is not part of the pinned model contract")
        if _SHA256.fullmatch(sample.sha256) is None:
            raise ValueError("sample checksum must be a lowercase SHA-256 digest")
        if (
            parsed.scheme != "https"
            or parsed.hostname != "huggingface.co"
            or parsed.path != expected_path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("sample URL must match the pinned Hugging Face model revision")
    return manifest


def _download_verified(spec: SampleSpec, target: Path) -> None:
    request = urllib.request.Request(  # noqa: S310 -- validated HTTPS allowlist
        spec.url, headers={"User-Agent": "LeafSentry/0.1"}
    )
    # URL scheme, host, path, and revision are allowlisted by load_manifest.
    with urllib.request.urlopen(  # noqa: S310
        request, timeout=60
    ) as response:
        data = response.read(_MAX_SAMPLE_BYTES + 1)
    if len(data) > _MAX_SAMPLE_BYTES:
        raise ValueError(f"sample {spec.filename} exceeds the download limit")
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.sha256:
        raise ValueError(f"checksum mismatch for {spec.filename}: {digest}")
    target.write_bytes(data)


def ensure_samples(manifest: SmokeManifest, directory: Path) -> list[tuple[SampleSpec, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
    resolved: list[tuple[SampleSpec, Path]] = []
    for spec in manifest.samples:
        target = directory / spec.filename
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != spec.sha256:
            _download_verified(spec, target)
        resolved.append((spec, target))
    return resolved


def run_smoke(manifest_path: Path, sample_directory: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    predictor = HuggingFacePredictor(
        model_id=manifest.model_id,
        revision=manifest.model_revision,
    )
    settings = Settings()
    policy = SelectionPolicy(
        min_confidence=settings.min_confidence,
        min_margin=settings.min_margin,
        temperature=settings.temperature,
    )
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for spec, path in ensure_samples(manifest, sample_directory):
        sample_started = time.perf_counter()
        with Image.open(path) as source:
            raw = predictor.predict(source.convert("RGB"))
        selection = select_prediction(raw, policy=policy)
        top_label = selection.scores[0].label
        results.append(
            {
                "filename": spec.filename,
                "expected_label": spec.expected_label,
                "top_label": top_label,
                "matched": top_label == spec.expected_label,
                "decision": selection.decision.value,
                "confidence": selection.confidence,
                "margin": selection.margin,
                "normalized_entropy": selection.normalized_entropy,
                "latency_ms": (time.perf_counter() - sample_started) * 1_000.0,
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {"id": manifest.model_id, "revision": manifest.model_revision},
        "runtime": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "device": "cpu",
        },
        "all_top1_matched": all(result["matched"] for result in results),
        "total_latency_ms": (time.perf_counter() - started) * 1_000.0,
        "samples": results,
        "manifest": asdict(manifest),
    }
