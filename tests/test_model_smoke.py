import json
from pathlib import Path

import pytest

from leafsentry.config import MODEL_ID, MODEL_REVISION
from leafsentry.model_smoke import load_manifest


def write_manifest(tmp_path: Path, **sample_overrides: str) -> Path:
    sample = {
        "filename": "healthy.jpeg",
        "expected_label": "healthy",
        "sha256": "a" * 64,
        "url": (f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/healthy.jpeg"),
    }
    sample.update(sample_overrides)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "samples": [sample],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_manifest_accepts_the_pinned_huggingface_sample(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path))

    assert manifest.samples[0].filename == "healthy.jpeg"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filename", "../healthy.jpeg"),
        ("expected_label", "unknown"),
        ("sha256", "not-a-checksum"),
        ("url", "http://huggingface.co/unsafe"),
        ("url", "https://example.com/healthy.jpeg"),
    ],
)
def test_load_manifest_rejects_unsafe_sample_fields(tmp_path: Path, field: str, value: str) -> None:
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path, **{field: value}))
