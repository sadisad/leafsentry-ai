"""Application configuration with reproducible model defaults."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

MODEL_ID = "nateraw/vit-base-beans"
MODEL_REVISION = "41f85ace09a4613c2c65495b3b8465c4ceee1d00"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings.

    Only operational thresholds may be overridden by environment variables. The
    production model identity stays pinned in source so a deployment cannot
    silently drift to a different set of weights.
    """

    app_name: str = "LeafSentry AI"
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    min_confidence: float = 0.75
    min_margin: float = 0.20
    temperature: float = 1.0
    max_upload_bytes: int = 5 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    min_image_dimension: int = 96
    max_image_dimension: int = 8_192
    min_brightness: float = 20.0
    max_brightness: float = 235.0
    min_contrast: float = 10.0
    min_sharpness: float = 8.0

    def __post_init__(self) -> None:
        for name in ("min_confidence", "min_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        for name in (
            "max_upload_bytes",
            "max_image_pixels",
            "min_image_dimension",
            "max_image_dimension",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_image_dimension > self.max_image_dimension:
            raise ValueError("min_image_dimension cannot exceed max_image_dimension")

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the documented environment variables."""

        def env_float(name: str, default: float) -> float:
            return float(os.getenv(f"LEAFSENTRY_{name}", str(default)))

        def env_int(name: str, default: int) -> int:
            return int(os.getenv(f"LEAFSENTRY_{name}", str(default)))

        defaults = cls()
        return cls(
            min_confidence=env_float("MIN_CONFIDENCE", defaults.min_confidence),
            min_margin=env_float("MIN_MARGIN", defaults.min_margin),
            temperature=env_float("TEMPERATURE", defaults.temperature),
            max_upload_bytes=env_int("MAX_UPLOAD_BYTES", defaults.max_upload_bytes),
            max_image_pixels=env_int("MAX_IMAGE_PIXELS", defaults.max_image_pixels),
            min_image_dimension=env_int("MIN_IMAGE_DIMENSION", defaults.min_image_dimension),
            max_image_dimension=env_int("MAX_IMAGE_DIMENSION", defaults.max_image_dimension),
            min_brightness=env_float("MIN_BRIGHTNESS", defaults.min_brightness),
            max_brightness=env_float("MAX_BRIGHTNESS", defaults.max_brightness),
            min_contrast=env_float("MIN_CONTRAST", defaults.min_contrast),
            min_sharpness=env_float("MIN_SHARPNESS", defaults.min_sharpness),
        )
