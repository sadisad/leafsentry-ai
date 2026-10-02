from io import BytesIO

import pytest
from PIL import Image

from leafsentry.config import Settings
from leafsentry.errors import ImageInputError
from leafsentry.image_guard import inspect_image


def image_bytes(
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (128, 128),
    color: int | tuple[int, int, int] = (90, 130, 60),
    image_format: str = "JPEG",
) -> bytes:
    buffer = BytesIO()
    Image.new(mode, size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def test_inspect_image_decodes_a_valid_jpeg() -> None:
    inspected = inspect_image(
        image_bytes(),
        media_type="image/jpeg",
        settings=Settings(min_contrast=0.1, min_sharpness=0.1),
    )

    assert inspected.image.mode == "RGB"
    assert inspected.image.size == (128, 128)
    assert inspected.report.metrics.format == "JPEG"


@pytest.mark.parametrize("payload", [b"", b"not-an-image", image_bytes()[:20]])
def test_inspect_image_rejects_invalid_or_incomplete_bytes(payload: bytes) -> None:
    with pytest.raises(ImageInputError) as raised:
        inspect_image(payload, media_type="image/jpeg", settings=Settings())

    assert raised.value.code in {"empty_file", "invalid_image"}


def test_inspect_image_rejects_payload_over_byte_limit() -> None:
    payload = image_bytes()

    with pytest.raises(ImageInputError) as raised:
        inspect_image(
            payload,
            media_type="image/jpeg",
            settings=Settings(max_upload_bytes=len(payload) - 1),
        )

    assert raised.value.code == "upload_too_large"
    assert raised.value.status_code == 413


def test_inspect_image_rejects_unsupported_declared_media_type() -> None:
    with pytest.raises(ImageInputError) as raised:
        inspect_image(image_bytes(), media_type="application/octet-stream", settings=Settings())

    assert raised.value.code == "unsupported_media_type"
    assert raised.value.status_code == 415


def test_inspect_image_rejects_media_type_mismatch() -> None:
    with pytest.raises(ImageInputError) as raised:
        inspect_image(image_bytes(), media_type="image/png", settings=Settings())

    assert raised.value.code == "media_type_mismatch"


def test_inspect_image_rejects_dimensions_outside_limits() -> None:
    with pytest.raises(ImageInputError) as raised:
        inspect_image(
            image_bytes(size=(128, 128)),
            media_type="image/jpeg",
            settings=Settings(max_image_dimension=100, min_image_dimension=32),
        )

    assert raised.value.code == "invalid_image_dimensions"


def test_inspect_image_turns_decompression_warning_into_input_error(monkeypatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)

    with pytest.raises(ImageInputError) as raised:
        inspect_image(
            image_bytes(size=(128, 128)),
            media_type="image/jpeg",
            settings=Settings(max_image_pixels=1_000_000),
        )

    assert raised.value.code == "decompression_bomb"


@pytest.mark.parametrize(
    ("color", "finding"),
    [
        ((0, 0, 0), "too_dark"),
        ((255, 255, 255), "too_bright"),
        ((120, 120, 120), "low_contrast"),
        ((120, 120, 120), "blurry"),
    ],
)
def test_inspect_image_reports_quality_failures(color: tuple[int, int, int], finding: str) -> None:
    inspected = inspect_image(
        image_bytes(color=color, image_format="PNG"),
        media_type="image/png",
        settings=Settings(),
    )

    assert inspected.report.passed is False
    assert finding in inspected.report.findings
