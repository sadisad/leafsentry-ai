"""Domain errors that can be translated safely at the HTTP boundary."""

from __future__ import annotations


class LeafSentryError(Exception):
    """Base class for expected, user-safe failures."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ImageInputError(LeafSentryError):
    """The uploaded bytes are not a safe, supported image."""

    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(code, message, status_code=status_code)


class ModelUnavailableError(LeafSentryError):
    """The configured model cannot currently serve predictions."""

    def __init__(self, message: str = "The inference model is not ready.") -> None:
        super().__init__("model_unavailable", message, status_code=503)


class ModelOutputError(LeafSentryError):
    """A predictor returned malformed output."""

    def __init__(self, message: str = "The model returned an invalid output.") -> None:
        super().__init__("invalid_model_output", message, status_code=500)
