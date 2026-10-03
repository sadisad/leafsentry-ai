from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from PIL import Image

from leafsentry.backends.huggingface import HuggingFacePredictor
from leafsentry.config import MODEL_ID, MODEL_REVISION


class FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[Image.Image, str]] = []

    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, object]:
        self.calls.append((images, return_tensors))
        return {"pixel_values": object()}


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            id2label={0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}
        )
        self.eval_calls = 0
        self.inputs: list[dict[str, object]] = []

    def eval(self) -> FakeModel:
        self.eval_calls += 1
        return self

    def __call__(self, **inputs: object) -> SimpleNamespace:
        self.inputs.append(inputs)
        return SimpleNamespace(logits=FakeTensor([[0.5, -1.0, 3.0]]))


class FakeTorch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


def test_injected_huggingface_backend_returns_labels_and_raw_logits() -> None:
    processor = FakeProcessor()
    model = FakeModel()
    predictor = HuggingFacePredictor(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        processor=processor,
        model=model,
        torch_module=FakeTorch(),
    )

    output = predictor.predict(Image.new("RGB", (224, 224), "green"))

    assert predictor.is_ready is True
    assert output.labels == ("angular_leaf_spot", "bean_rust", "healthy")
    assert output.logits.tolist() == [0.5, -1.0, 3.0]
    assert processor.calls[0][1] == "pt"
    assert model.inputs[0].keys() == {"pixel_values"}


def test_loaders_receive_pinned_revision_and_disable_remote_code() -> None:
    calls: list[tuple[str, str, bool, bool | None]] = []
    processor = FakeProcessor()
    model = FakeModel()

    def processor_loader(
        model_id: str,
        *,
        revision: str,
        trust_remote_code: bool,
        use_fast: bool,
    ) -> FakeProcessor:
        calls.append((model_id, revision, trust_remote_code, use_fast))
        return processor

    def model_loader(model_id: str, *, revision: str, trust_remote_code: bool) -> FakeModel:
        calls.append((model_id, revision, trust_remote_code, None))
        return model

    predictor = HuggingFacePredictor(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        processor_loader=processor_loader,
        model_loader=model_loader,
        torch_module=FakeTorch(),
    )
    assert predictor.is_ready is False

    predictor.predict(Image.new("RGB", (224, 224), "green"))

    assert calls == [
        (MODEL_ID, MODEL_REVISION, False, False),
        (MODEL_ID, MODEL_REVISION, False, None),
    ]
    assert model.eval_calls == 1
    assert predictor.is_ready is True
