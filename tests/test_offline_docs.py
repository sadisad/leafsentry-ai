from leafsentry.api import create_app
from tests.fakes import FakePredictor
from tests.test_api import request


def test_api_docs_work_without_csp_blocked_external_assets():
    response = request(create_app(predictor=FakePredictor()), "GET", "/docs")
    assert response.status_code == 200
    assert "/v1/predictions" in response.text
    assert "/openapi.json" in response.text
    assert "cdn.jsdelivr.net" not in response.text
    assert "<script" not in response.text
