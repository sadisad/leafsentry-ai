"""Quality thresholds must remain safe through direct and environment settings."""

import math

import pytest

from leafsentry.config import Settings

QUALITY_FIELDS = (
    "min_brightness",
    "max_brightness",
    "min_contrast",
    "min_sharpness",
)
INVALID_FINITE_THRESHOLDS = (
    ("min_brightness", -0.01),
    ("min_brightness", 255.01),
    ("max_brightness", -0.01),
    ("max_brightness", 255.01),
    ("min_contrast", -0.01),
    ("min_sharpness", -0.01),
)


@pytest.fixture(autouse=True)
def clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in (
        "min_confidence",
        "min_margin",
        "temperature",
        "max_upload_bytes",
        "max_image_pixels",
        "min_image_dimension",
        "max_image_dimension",
        *QUALITY_FIELDS,
    ):
        monkeypatch.delenv(f"LEAFSENTRY_{field.upper()}", raising=False)


@pytest.mark.parametrize("field", QUALITY_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_quality_thresholds_reject_nonfinite(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


@pytest.mark.parametrize("field", QUALITY_FIELDS)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_env_quality_thresholds_reject_nonfinite(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    monkeypatch.setenv(f"LEAFSENTRY_{field.upper()}", value)

    with pytest.raises(ValueError, match=field):
        Settings.from_env()


@pytest.mark.parametrize(("field", "value"), INVALID_FINITE_THRESHOLDS)
def test_quality_thresholds_reject_invalid_finite_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


@pytest.mark.parametrize(("field", "value"), INVALID_FINITE_THRESHOLDS)
def test_env_quality_thresholds_reject_invalid_finite_values(
    monkeypatch: pytest.MonkeyPatch, field: str, value: float
) -> None:
    monkeypatch.setenv(f"LEAFSENTRY_{field.upper()}", str(value))

    with pytest.raises(ValueError, match=field):
        Settings.from_env()


@pytest.mark.parametrize("from_env", [False, True])
def test_brightness_thresholds_reject_inverted_range(
    monkeypatch: pytest.MonkeyPatch, from_env: bool
) -> None:
    with pytest.raises(ValueError, match=r"min_brightness.*max_brightness"):
        if from_env:
            monkeypatch.setenv("LEAFSENTRY_MIN_BRIGHTNESS", "200")
            monkeypatch.setenv("LEAFSENTRY_MAX_BRIGHTNESS", "100")
            Settings.from_env()
        else:
            Settings(min_brightness=200.0, max_brightness=100.0)


@pytest.mark.parametrize("from_env", [False, True])
@pytest.mark.parametrize(
    ("minimum", "maximum", "contrast", "sharpness"),
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 255.0, 0.0, 0.0),
        (255.0, 255.0, 0.0, 0.0),
        (0.5, 254.5, 0.5, 0.5),
        (20.0, 235.0, 1_000.0, 100_000.0),
    ],
)
def test_quality_thresholds_accept_valid_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    from_env: bool,
    minimum: float,
    maximum: float,
    contrast: float,
    sharpness: float,
) -> None:
    if from_env:
        for field, value in zip(
            QUALITY_FIELDS, (minimum, maximum, contrast, sharpness), strict=True
        ):
            monkeypatch.setenv(f"LEAFSENTRY_{field.upper()}", str(value))
        settings = Settings.from_env()
    else:
        settings = Settings(
            min_brightness=minimum,
            max_brightness=maximum,
            min_contrast=contrast,
            min_sharpness=sharpness,
        )

    assert (
        settings.min_brightness,
        settings.max_brightness,
        settings.min_contrast,
        settings.min_sharpness,
    ) == (minimum, maximum, contrast, sharpness)


@pytest.mark.parametrize("from_env", [False, True])
def test_quality_defaults_remain_unchanged(from_env: bool) -> None:
    settings = Settings.from_env() if from_env else Settings()

    assert (
        settings.min_brightness,
        settings.max_brightness,
        settings.min_contrast,
        settings.min_sharpness,
    ) == (20.0, 235.0, 10.0, 8.0)
