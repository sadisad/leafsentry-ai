"""Lazy, revision-pinned Hugging Face image-classification backend."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

from leafsentry.errors import ModelOutputError, ModelUnavailableError
from leafsentry.prediction import RawPrediction


class Processor(Protocol):
    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, object]: ...


class ModelConfig(Protocol):
    id2label: dict[int | str, str]


class TensorLike(Protocol):
    def detach(self) -> TensorLike: ...

    def cpu(self) -> TensorLike: ...

    def numpy(self) -> object: ...


class ModelOutput(Protocol):
    logits: TensorLike


class Model(Protocol):
    config: ModelConfig

    def eval(self) -> Model: ...

    def __call__(self, **inputs: object) -> ModelOutput: ...


class TorchModule(Protocol):
    def inference_mode(self) -> Any: ...


ProcessorLoader = Callable[..., Processor]
ModelLoader = Callable[..., Model]


class HuggingFacePredictor:
    """Load a known model lazily and return raw logits for local calibration."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        processor: Processor | None = None,
        model: Model | None = None,
        torch_module: TorchModule | None = None,
        processor_loader: ProcessorLoader | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be supplied together")
        self.model_id = model_id
        self.revision = revision
        self._processor = processor
        self._model = model
        self._torch = torch_module
        self._processor_loader = processor_loader
        self._model_loader = model_loader
        self._ready = processor is not None and model is not None
        if self._ready and self._model is not None:
            self._model.eval()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _load(self) -> None:
        if self._ready:
            return
        try:
            if self._torch is None:
                self._torch = cast(TorchModule, importlib.import_module("torch"))
            if self._processor_loader is None or self._model_loader is None:
                transformers = importlib.import_module("transformers")
                self._processor_loader = cast(
                    ProcessorLoader, transformers.AutoImageProcessor.from_pretrained
                )
                self._model_loader = cast(
                    ModelLoader, transformers.AutoModelForImageClassification.from_pretrained
                )
            self._processor = self._processor_loader(
                self.model_id,
                revision=self.revision,
                trust_remote_code=False,
            )
            self._model = self._model_loader(
                self.model_id,
                revision=self.revision,
                trust_remote_code=False,
            ).eval()
        except Exception as exc:
            self._ready = False
            raise ModelUnavailableError("The pinned inference model could not be loaded.") from exc
        self._ready = True

    def predict(self, image: Image.Image) -> RawPrediction:
        self._load()
        if self._processor is None or self._model is None or self._torch is None:
            raise ModelUnavailableError()
        try:
            inputs = self._processor(images=image, return_tensors="pt")
            with self._torch.inference_mode():
                output = self._model(**inputs)
            logits = np.asarray(output.logits.detach().cpu().numpy(), dtype=np.float64)
            if logits.shape[0] != 1:
                raise ModelOutputError()
            labels = tuple(
                self._model.config.id2label.get(index)
                or self._model.config.id2label.get(str(index))
                or ""
                for index in range(logits.shape[1])
            )
            return RawPrediction(labels=labels, logits=logits[0])
        except (ModelOutputError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "The pinned inference model failed during inference."
            ) from exc
