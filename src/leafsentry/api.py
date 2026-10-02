"""FastAPI boundary for LeafSentry AI."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from leafsentry.config import Settings
from leafsentry.errors import LeafSentryError
from leafsentry.metrics import Metrics
from leafsentry.prediction import Predictor
from leafsentry.schemas import ErrorDetail, ErrorResponse, HealthResponse, PredictionResponse
from leafsentry.service import LeafSentryService

_LOGGER = logging.getLogger("leafsentry.api")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_READ_CHUNK_BYTES = 64 * 1024


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if _REQUEST_ID.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex


async def _read_bounded(upload: UploadFile, limit: int) -> bytes:
    payload = bytearray()
    while len(payload) <= limit:
        chunk = await upload.read(min(_READ_CHUNK_BYTES, limit + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) > limit:
        from leafsentry.errors import ImageInputError

        raise ImageInputError(
            "upload_too_large",
            f"The image exceeds the {limit}-byte upload limit.",
            status_code=413,
        )
    return bytes(payload)


def _default_predictor(settings: Settings) -> Predictor:
    from leafsentry.backends.huggingface import HuggingFacePredictor

    return HuggingFacePredictor(model_id=settings.model_id, revision=settings.model_revision)


def create_app(
    *,
    settings: Settings | None = None,
    predictor: Predictor | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    runtime_predictor = predictor or _default_predictor(runtime_settings)
    service = LeafSentryService(settings=runtime_settings, predictor=runtime_predictor)
    metrics = Metrics(registry or CollectorRegistry())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(
        title=runtime_settings.app_name,
        version="0.1.0",
        description="Uncertainty-aware bean leaf image triage.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.metrics = metrics

    @app.exception_handler(LeafSentryError)
    async def leafsentry_error_handler(request: Request, exc: LeafSentryError) -> JSONResponse:
        request_id = _request_id(request)
        payload = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message, request_id=request_id)
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, _: RequestValidationError) -> JSONResponse:
        request_id = _request_id(request)
        payload = ErrorResponse(
            error=ErrorDetail(
                code="invalid_request",
                message="A multipart image field named 'file' is required.",
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        _LOGGER.exception("unexpected_error request_id=%s", request_id, exc_info=exc)
        payload = ErrorResponse(
            error=ErrorDetail(
                code="internal_error",
                message="The request could not be completed.",
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))

    @app.get("/health/live", response_model=HealthResponse, tags=["operations"])
    async def live() -> HealthResponse:
        return HealthResponse(status="ok", model=service.model_identity)

    @app.get("/health/ready", response_model=HealthResponse, tags=["operations"])
    async def ready() -> Response:
        status = "ready" if service.is_ready else "not_ready"
        body = HealthResponse(status=status, model=service.model_identity)
        return JSONResponse(
            status_code=200 if service.is_ready else 503,
            content=body.model_dump(mode="json"),
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(content=generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/predictions",
        response_model=PredictionResponse,
        tags=["inference"],
    )
    async def predict(request: Request, file: Annotated[UploadFile, File()]) -> PredictionResponse:
        started = time.perf_counter()
        request_id = _request_id(request)
        try:
            payload = await _read_bounded(file, runtime_settings.max_upload_bytes)
            response = service.predict(
                payload,
                media_type=file.content_type or "application/octet-stream",
                request_id=request_id,
            )
        except LeafSentryError:
            metrics.observe(outcome="error", duration_seconds=time.perf_counter() - started)
            raise
        finally:
            await file.close()

        metrics.observe(
            outcome=response.decision.value,
            duration_seconds=time.perf_counter() - started,
        )
        return response

    return app
