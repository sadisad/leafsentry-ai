import hashlib
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from leafsentry import model_smoke
from leafsentry.config import MODEL_ID, MODEL_REVISION
from tests.fakes import FakePredictor


def png_bytes():
    stream = BytesIO()
    Image.new("RGB", (128, 128), "green").save(stream, format="PNG")
    return stream.getvalue()


def manifest_for(payload):
    return model_smoke.SmokeManifest(
        MODEL_ID,
        MODEL_REVISION,
        (
            model_smoke.SampleSpec(
                "healthy.jpeg",
                "healthy",
                hashlib.sha256(payload).hexdigest(),
                f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/healthy.jpeg",
            ),
        ),
    )


def fake_network(monkeypatch, payload):
    calls = []

    @contextmanager
    def open_url(request, timeout):
        calls.append((request.full_url, timeout))
        yield BytesIO(payload)

    monkeypatch.setattr(model_smoke.urllib.request, "urlopen", open_url)
    return calls


def test_verified_sample_is_downloaded_then_cache_reused(tmp_path, monkeypatch):
    payload = png_bytes()
    calls = fake_network(monkeypatch, payload)
    manifest = manifest_for(payload)
    result = model_smoke.ensure_samples(manifest, tmp_path)
    assert result[0][1].read_bytes() == payload
    model_smoke.ensure_samples(manifest, tmp_path)
    assert len(calls) == 1


def test_corrupt_cache_is_replaced_only_with_checksum_verified_bytes(tmp_path, monkeypatch):
    payload = png_bytes()
    (tmp_path / "healthy.jpeg").write_bytes(b"corrupt")
    calls = fake_network(monkeypatch, payload)
    result = model_smoke.ensure_samples(manifest_for(payload), tmp_path)
    assert result[0][1].read_bytes() == payload
    assert len(calls) == 1


def test_untrusted_download_checksum_never_creates_sample(tmp_path, monkeypatch):
    fake_network(monkeypatch, b"wrong bytes")
    with pytest.raises(ValueError, match="checksum"):
        model_smoke.ensure_samples(manifest_for(png_bytes()), tmp_path)
    assert not (tmp_path / "healthy.jpeg").exists()


def test_download_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(model_smoke, "_MAX_SAMPLE_BYTES", 5)
    fake_network(monkeypatch, b"123456")
    with pytest.raises(ValueError, match="limit"):
        model_smoke.ensure_samples(manifest_for(b"123456"), tmp_path)
    assert not (tmp_path / "healthy.jpeg").exists()


def test_smoke_orchestration_uses_injected_predictor_without_weights(tmp_path, monkeypatch):
    payload = png_bytes()
    manifest = manifest_for(payload)
    monkeypatch.setattr(model_smoke, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(model_smoke, "HuggingFacePredictor", lambda **_: FakePredictor())
    monkeypatch.setattr(model_smoke, "version", lambda _: "test-version")
    fake_network(monkeypatch, payload)
    report = model_smoke.run_smoke(Path("unused"), tmp_path)
    assert report["all_top1_matched"] is True
    assert report["samples"][0]["top_label"] == "healthy"
    assert report["runtime"]["torch"] == "test-version"
    assert report["model"]["revision"] == MODEL_REVISION
