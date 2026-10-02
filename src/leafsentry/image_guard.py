"""Safe image decoding and deterministic quality checks."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from leafsentry.config import Settings
from leafsentry.errors import ImageInputError
from leafsentry.schemas import QualityMetrics, QualityReport

_ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


@dataclass(frozen=True, slots=True)
class InspectedImage:
    image: Image.Image
    report: QualityReport


def _decode_image(data: bytes, media_type: str, settings: Settings) -> tuple[Image.Image, str]:
    if not data:
        raise ImageInputError("empty_file", "The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ImageInputError(
            "upload_too_large",
            f"The image exceeds the {settings.max_upload_bytes}-byte upload limit.",
            status_code=413,
        )
    if media_type not in set(_ALLOWED_FORMATS.values()):
        raise ImageInputError(
            "unsupported_media_type",
            "Supported media types are image/jpeg, image/png, and image/webp.",
            status_code=415,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                image_format = probe.format
                width, height = probe.size
                if image_format not in _ALLOWED_FORMATS:
                    raise ImageInputError(
                        "unsupported_image_format",
                        "Supported image formats are JPEG, PNG, and WebP.",
                        status_code=415,
                    )
                if _ALLOWED_FORMATS[image_format] != media_type:
                    raise ImageInputError(
                        "media_type_mismatch",
                        "The declared media type does not match the decoded image format.",
                    )
                if (
                    width < settings.min_image_dimension
                    or height < settings.min_image_dimension
                    or width > settings.max_image_dimension
                    or height > settings.max_image_dimension
                    or width * height > settings.max_image_pixels
                ):
                    raise ImageInputError(
                        "invalid_image_dimensions",
                        "Image dimensions are outside the configured safety limits.",
                    )
                probe.verify()

            with Image.open(BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
    except ImageInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageInputError(
            "decompression_bomb", "The decoded image exceeds safe pixel limits."
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageInputError(
            "invalid_image", "The uploaded bytes are not a valid complete image."
        ) from exc

    return image, image_format


def _sharpness(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = -4.0 * center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(laplacian))


def inspect_image(data: bytes, *, media_type: str, settings: Settings) -> InspectedImage:
    """Decode untrusted image bytes and return bounded quality measurements."""

    image, image_format = _decode_image(data, media_type, settings)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = _sharpness(gray)

    findings: list[str] = []
    if mean_brightness < settings.min_brightness:
        findings.append("too_dark")
    if mean_brightness > settings.max_brightness:
        findings.append("too_bright")
    if contrast < settings.min_contrast:
        findings.append("low_contrast")
    if sharpness < settings.min_sharpness:
        findings.append("blurry")

    metrics = QualityMetrics(
        width=image.width,
        height=image.height,
        format=image_format,
        mean_brightness=mean_brightness,
        contrast=contrast,
        sharpness=sharpness,
    )
    return InspectedImage(
        image=image,
        report=QualityReport(passed=not findings, findings=findings, metrics=metrics),
    )
