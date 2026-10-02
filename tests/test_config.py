from leafsentry.config import MODEL_ID, MODEL_REVISION, Settings


def test_settings_have_safe_reproducible_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "LeafSentry AI"
    assert settings.model_id == MODEL_ID == "nateraw/vit-base-beans"
    assert settings.model_revision == MODEL_REVISION
    assert len(settings.model_revision) == 40
    assert 0.0 < settings.min_confidence < 1.0
    assert 0.0 < settings.min_margin < 1.0
    assert settings.temperature > 0.0
    assert settings.max_upload_bytes == 5 * 1024 * 1024


def test_settings_from_env_parses_supported_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LEAFSENTRY_MIN_CONFIDENCE", "0.81")
    monkeypatch.setenv("LEAFSENTRY_MAX_UPLOAD_BYTES", "12345")

    settings = Settings.from_env()

    assert settings.min_confidence == 0.81
    assert settings.max_upload_bytes == 12345
