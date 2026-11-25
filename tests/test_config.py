"""Tests for MS3 environment-backed settings."""

import pytest
from pydantic import ValidationError

from src.config.config import Settings


UPLOAD_LEASE_ENV = "UPLOAD_LEASE_SECONDS"
UPLOAD_LEASE_RENEW_INTERVAL_ENV = "UPLOAD_LEASE_RENEW_INTERVAL_SECONDS"


def clear_upload_lease_environment(monkeypatch):
    """Aísla las pruebas de lease de cualquier configuración heredada."""
    monkeypatch.delenv(UPLOAD_LEASE_ENV, raising=False)
    monkeypatch.delenv(UPLOAD_LEASE_RENEW_INTERVAL_ENV, raising=False)


def assert_validation_error_for_field(error, *field_names):
    """Comprueba el campo señalado sin depender del texto completo de Pydantic."""
    locations = {
        str(part).lower()
        for item in error.value.errors()
        for part in item["loc"]
    }
    assert any(name.lower() in locations for name in field_names)


def test_local_mongodb_default(monkeypatch):
    monkeypatch.delenv("MONGO_URL", raising=False)

    settings = Settings()

    assert settings.mongo_url == "mongodb://localhost:27017"


def test_mongodb_url_can_be_configured_from_environment(monkeypatch):
    uri = "mongodb://user:password@db:27017/?authSource=admin"
    monkeypatch.setenv("MONGO_URL", uri)

    settings = Settings()

    assert settings.mongo_url == uri


def test_upload_lease_settings_have_defaults(monkeypatch):
    clear_upload_lease_environment(monkeypatch)

    settings = Settings()

    assert settings.upload_lease_seconds == 120
    assert settings.upload_lease_renew_interval_seconds == 30


def test_upload_lease_settings_can_be_configured_from_environment(monkeypatch):
    clear_upload_lease_environment(monkeypatch)
    monkeypatch.setenv(UPLOAD_LEASE_ENV, "240")
    monkeypatch.setenv(UPLOAD_LEASE_RENEW_INTERVAL_ENV, "45")

    settings = Settings()

    assert settings.upload_lease_seconds == 240
    assert settings.upload_lease_renew_interval_seconds == 45


@pytest.mark.parametrize("value", ["0", "-1"])
def test_upload_lease_duration_must_be_positive(monkeypatch, value):
    clear_upload_lease_environment(monkeypatch)
    monkeypatch.setenv(UPLOAD_LEASE_ENV, value)

    with pytest.raises(ValidationError) as error:
        Settings()

    assert_validation_error_for_field(
        error,
        "upload_lease_seconds",
        UPLOAD_LEASE_ENV,
    )


@pytest.mark.parametrize("value", ["0", "-1"])
def test_upload_lease_renew_interval_must_be_positive(monkeypatch, value):
    clear_upload_lease_environment(monkeypatch)
    monkeypatch.setenv(UPLOAD_LEASE_RENEW_INTERVAL_ENV, value)

    with pytest.raises(ValidationError) as error:
        Settings()

    assert_validation_error_for_field(
        error,
        "upload_lease_renew_interval_seconds",
        UPLOAD_LEASE_RENEW_INTERVAL_ENV,
    )


@pytest.mark.parametrize(
    ("lease_seconds", "renew_interval_seconds"),
    [
        ("60", "60"),
        ("60", "61"),
    ],
)
def test_upload_lease_renew_interval_must_be_less_than_lease(
    monkeypatch,
    lease_seconds,
    renew_interval_seconds,
):
    clear_upload_lease_environment(monkeypatch)
    monkeypatch.setenv(UPLOAD_LEASE_ENV, lease_seconds)
    monkeypatch.setenv(UPLOAD_LEASE_RENEW_INTERVAL_ENV, renew_interval_seconds)

    with pytest.raises(ValidationError) as error:
        Settings()

    messages = " ".join(item["msg"].lower() for item in error.value.errors())
    assert "lease" in messages
    assert "interval" in messages or "renew" in messages


@pytest.mark.parametrize(
    ("environment_name", "invalid_value", "field_name"),
    [
        (UPLOAD_LEASE_ENV, "not-a-number", "upload_lease_seconds"),
        (UPLOAD_LEASE_ENV, "120.5", "upload_lease_seconds"),
        (
            UPLOAD_LEASE_RENEW_INTERVAL_ENV,
            "not-a-number",
            "upload_lease_renew_interval_seconds",
        ),
        (
            UPLOAD_LEASE_RENEW_INTERVAL_ENV,
            "30.5",
            "upload_lease_renew_interval_seconds",
        ),
    ],
)
def test_upload_lease_settings_reject_non_integer_values(
    monkeypatch,
    environment_name,
    invalid_value,
    field_name,
):
    clear_upload_lease_environment(monkeypatch)
    monkeypatch.setenv(environment_name, invalid_value)

    with pytest.raises(ValidationError) as error:
        Settings()

    assert_validation_error_for_field(error, field_name, environment_name)
