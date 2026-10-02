from importlib import resources

from leafsentry.api import create_app
from tests.fakes import FakePredictor
from tests.test_api import request


def test_root_serves_operator_console() -> None:
    response = request(create_app(predictor=FakePredictor()), "GET", "/")

    assert response.status_code == 200
    assert "LeafSentry" in response.text
    assert 'id="upload-form"' in response.text
    assert 'aria-live="polite"' in response.text
    assert "not a field-validated diagnosis" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_favicon_is_served_without_a_console_404() -> None:
    response = request(create_app(predictor=FakePredictor()), "GET", "/static/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")


def test_static_assets_are_packaged_and_use_accessible_states() -> None:
    static = resources.files("leafsentry").joinpath("static")
    css = static.joinpath("app.css").read_text(encoding="utf-8")
    javascript = static.joinpath("app.js").read_text(encoding="utf-8")

    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert "@media (max-width:" in css
    assert "aria-busy" in javascript
    assert "fetch(" in javascript
    assert "innerHTML" not in javascript
