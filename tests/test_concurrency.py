import asyncio
import threading

import httpx

from leafsentry.api import create_app
from tests.fakes import FakePredictor, textured_png_bytes


def test_health_remains_responsive_during_serialized_model_inference() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingPredictor(FakePredictor):
        def predict(self, image):
            started.set()
            if not release.wait(timeout=0.2):
                raise RuntimeError("inference blocked the event loop")
            return super().predict(image)

    async def scenario():
        app = create_app(predictor=BlockingPredictor())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            task = asyncio.create_task(
                client.post(
                    "/v1/predictions",
                    files={"file": ("leaf.png", textured_png_bytes(), "image/png")},
                )
            )
            await asyncio.to_thread(started.wait, 1)
            try:
                health = await client.get("/health/live")
                assert health.status_code == 200
            finally:
                release.set()
            assert (await task).status_code == 200

    asyncio.run(scenario())


def test_two_concurrent_predictions_never_enter_backend_together() -> None:
    import time

    lock = threading.Lock()
    active = 0
    peak = 0

    class ObservedPredictor(FakePredictor):
        def predict(self, image):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                return super().predict(image)
            finally:
                with lock:
                    active -= 1

    async def scenario():
        app = create_app(predictor=ObservedPredictor())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/v1/predictions",
                        files={"file": ("leaf.png", textured_png_bytes(), "image/png")},
                    )
                    for _ in range(2)
                ]
            )
        assert [response.status_code for response in responses] == [200, 200]
        assert peak == 1

    asyncio.run(scenario())
