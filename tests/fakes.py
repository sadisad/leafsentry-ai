from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from leafsentry.errors import ModelUnavailableError
from leafsentry.prediction import RawPrediction


class FakePredictor:
    def __init__(
        self,
        *,
        logits: tuple[float, ...] = (0.0, 0.0, 4.0),
        ready: bool = True,
    ) -> None:
        self._logits = np.asarray(logits, dtype=np.float64)
        self._ready = ready
        self.calls = 0

    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict(self, image: Image.Image) -> RawPrediction:
        if not self._ready:
            raise ModelUnavailableError()
        self.calls += 1
        assert image.mode == "RGB"
        return RawPrediction(
            labels=("angular_leaf_spot", "bean_rust", "healthy"),
            logits=self._logits,
        )


def textured_png_bytes(size: tuple[int, int] = (128, 128)) -> bytes:
    image = Image.new("RGB", size, (56, 92, 46))
    draw = ImageDraw.Draw(image)
    for offset in range(0, min(size), 8):
        color = (210, 224, 160) if (offset // 8) % 2 == 0 else (22, 48, 18)
        draw.line((0, offset, size[0], size[1] - offset), fill=color, width=3)
        draw.line((offset, 0, size[0] - offset, size[1]), fill=color, width=2)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def solid_png_bytes(color: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (128, 128), color).save(buffer, format="PNG")
    return buffer.getvalue()
