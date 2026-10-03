#!/usr/bin/env python3
"""Evaluate the pinned production model on the pinned Beans test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leafsentry.backends.huggingface import HuggingFacePredictor
from leafsentry.benchmark import load_image_archive, run_benchmark
from leafsentry.config import Settings

DATASET_ID = "AI-Lab-Makerere/beans"
DATASET_REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
TEST_ARCHIVE_SHA256 = "ca67b15d960d1e2fd9d23fd8498ce86818ead90755c630a43baf19ec4af09312"
TEST_SAMPLES = 128
LABELS = frozenset({"angular_leaf_spot", "bean_rust", "healthy"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = Settings()
    samples = load_image_archive(
        args.archive,
        expected_sha256=TEST_ARCHIVE_SHA256,
        expected_samples=TEST_SAMPLES,
        allowed_labels=LABELS,
    )
    report = run_benchmark(
        samples,
        predictor=HuggingFacePredictor(
            model_id=settings.model_id,
            revision=settings.model_revision,
        ),
        settings=settings,
        dataset={
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "archive_sha256": TEST_ARCHIVE_SHA256,
        },
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
