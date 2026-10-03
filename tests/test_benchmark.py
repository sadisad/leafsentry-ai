from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from leafsentry.benchmark import load_image_archive


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (32, 32), "green").save(stream, format="JPEG")
    return stream.getvalue()


def write_archive(path: Path, entries: dict[str, bytes]) -> str:
    with ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_image_archive_returns_sorted_labelled_images(tmp_path: Path) -> None:
    path = tmp_path / "test.zip"
    digest = write_archive(
        path,
        {
            "test/healthy/z.jpg": image_bytes(),
            "test/angular_leaf_spot/a.jpg": image_bytes(),
        },
    )

    samples = load_image_archive(
        path,
        expected_sha256=digest,
        expected_samples=2,
        allowed_labels=frozenset({"angular_leaf_spot", "healthy"}),
    )

    assert [(sample.label, sample.filename) for sample in samples] == [
        ("angular_leaf_spot", "a.jpg"),
        ("healthy", "z.jpg"),
    ]
    assert all(sample.image.mode == "RGB" for sample in samples)


def test_load_image_archive_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "test.zip"
    write_archive(path, {"test/healthy/a.jpg": image_bytes()})

    with pytest.raises(ValueError, match="checksum"):
        load_image_archive(
            path,
            expected_sha256="0" * 64,
            expected_samples=1,
            allowed_labels=frozenset({"healthy"}),
        )


@pytest.mark.parametrize(
    "name",
    [
        "../healthy/a.jpg",
        "test/unknown/a.jpg",
        "test/healthy/a.txt",
        "test/healthy/nested/a.jpg",
    ],
)
def test_load_image_archive_rejects_unsafe_or_unexpected_entries(tmp_path: Path, name: str) -> None:
    path = tmp_path / "test.zip"
    digest = write_archive(path, {name: image_bytes()})

    with pytest.raises(ValueError):
        load_image_archive(
            path,
            expected_sha256=digest,
            expected_samples=1,
            allowed_labels=frozenset({"healthy"}),
        )


def test_load_image_archive_rejects_wrong_sample_count(tmp_path: Path) -> None:
    path = tmp_path / "test.zip"
    digest = write_archive(path, {"test/healthy/a.jpg": image_bytes()})

    with pytest.raises(ValueError, match="sample count"):
        load_image_archive(
            path,
            expected_sha256=digest,
            expected_samples=2,
            allowed_labels=frozenset({"healthy"}),
        )
