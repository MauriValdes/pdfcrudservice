"""RED tests for the environment-backed configuration of the worker."""

import importlib

import pytest
from pydantic import ValidationError


WORKER_ENV = (
    "PDF_WORKER_ENABLED",
    "PDF_WORKER_MAX_CONCURRENCY",
    "PDF_WORKER_POLL_INITIAL_SECONDS",
    "PDF_WORKER_POLL_MAX_SECONDS",
    "PDF_WORKER_MS2_TIMEOUT_SECONDS",
    "PDF_WORKER_SHUTDOWN_GRACE_SECONDS",
)


def _worker_config_api():
    """Import lazily so RED is caused by the not-yet-created config module."""
    return importlib.import_module("src.worker.worker_config")


def _clear_worker_environment(monkeypatch):
    for name in WORKER_ENV:
        monkeypatch.delenv(name, raising=False)
    from src.config.config import settings

    monkeypatch.setattr(settings, "extract_service_url", "http://ms2.test")


def test_worker_config_defaults_are_isolated_from_inherited_environment(
    monkeypatch,
):
    _clear_worker_environment(monkeypatch)
    monkeypatch.setenv("EXTRACT_SERVICE_TIMEOUT", "30")
    config = _worker_config_api().WorkerSettings()

    assert config.enabled is False
    assert config.max_concurrency == 2
    assert config.poll_initial_seconds == 1.0
    assert config.poll_max_seconds == 5.0
    assert config.ms2_timeout_seconds == 40.0
    assert config.shutdown_grace_seconds == 60.0
    assert config.ms2_base_url == "http://ms2.test"


def test_worker_config_reads_custom_values_from_prefixed_environment(
    monkeypatch,
):
    _clear_worker_environment(monkeypatch)
    monkeypatch.setenv("PDF_WORKER_ENABLED", "true")
    monkeypatch.setenv("PDF_WORKER_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("PDF_WORKER_POLL_INITIAL_SECONDS", "0.5")
    monkeypatch.setenv("PDF_WORKER_POLL_MAX_SECONDS", "4")
    monkeypatch.setenv("PDF_WORKER_MS2_TIMEOUT_SECONDS", "42.5")
    monkeypatch.setenv("PDF_WORKER_SHUTDOWN_GRACE_SECONDS", "75")

    config = _worker_config_api().WorkerSettings()

    assert config.enabled is True
    assert config.max_concurrency == 3
    assert config.poll_initial_seconds == 0.5
    assert config.poll_max_seconds == 4.0
    assert config.ms2_timeout_seconds == 42.5
    assert config.shutdown_grace_seconds == 75.0


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true", "nan", "inf"])
def test_worker_config_rejects_invalid_concurrency(monkeypatch, value):
    _clear_worker_environment(monkeypatch)
    monkeypatch.setenv("PDF_WORKER_MAX_CONCURRENCY", value)

    with pytest.raises(ValidationError):
        _worker_config_api().WorkerSettings()


@pytest.mark.parametrize(
    "field",
    [
        "max_concurrency",
        "poll_initial_seconds",
        "poll_max_seconds",
        "ms2_timeout_seconds",
        "shutdown_grace_seconds",
    ],
)
def test_worker_config_rejects_python_booleans_for_numeric_fields(
    monkeypatch, field
):
    _clear_worker_environment(monkeypatch)

    with pytest.raises(ValidationError):
        _worker_config_api().WorkerSettings(**{field: True})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PDF_WORKER_POLL_INITIAL_SECONDS", "0"),
        ("PDF_WORKER_POLL_INITIAL_SECONDS", "-1"),
        ("PDF_WORKER_POLL_INITIAL_SECONDS", "true"),
        ("PDF_WORKER_POLL_INITIAL_SECONDS", "nan"),
        ("PDF_WORKER_POLL_INITIAL_SECONDS", "inf"),
        ("PDF_WORKER_POLL_MAX_SECONDS", "0"),
        ("PDF_WORKER_POLL_MAX_SECONDS", "false"),
        ("PDF_WORKER_POLL_MAX_SECONDS", "nan"),
        ("PDF_WORKER_MS2_TIMEOUT_SECONDS", "-1"),
        ("PDF_WORKER_MS2_TIMEOUT_SECONDS", "true"),
        ("PDF_WORKER_MS2_TIMEOUT_SECONDS", "inf"),
        ("PDF_WORKER_SHUTDOWN_GRACE_SECONDS", "0"),
        ("PDF_WORKER_SHUTDOWN_GRACE_SECONDS", "false"),
        ("PDF_WORKER_SHUTDOWN_GRACE_SECONDS", "nan"),
    ],
)
def test_worker_config_rejects_invalid_positive_numeric_values(
    monkeypatch, name, value
):
    _clear_worker_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        _worker_config_api().WorkerSettings()


@pytest.mark.parametrize(
    ("initial", "maximum"),
    [("2", "1"), ("1.5", "1.4")],
)
def test_worker_config_requires_poll_max_at_least_poll_initial(
    monkeypatch, initial, maximum
):
    _clear_worker_environment(monkeypatch)
    monkeypatch.setenv("PDF_WORKER_POLL_INITIAL_SECONDS", initial)
    monkeypatch.setenv("PDF_WORKER_POLL_MAX_SECONDS", maximum)

    with pytest.raises(ValidationError):
        _worker_config_api().WorkerSettings()


@pytest.mark.parametrize("value", ["sometimes", "2", ""])
def test_worker_config_rejects_invalid_enabled_values(monkeypatch, value):
    _clear_worker_environment(monkeypatch)
    monkeypatch.setenv("PDF_WORKER_ENABLED", value)

    with pytest.raises(ValidationError):
        _worker_config_api().WorkerSettings()


@pytest.mark.parametrize(
    "url",
    [
        "relative/path",
        "http://",
        "ftp://ms2.test",
        "http://user:password@ms2.test",
        "http://@ms2.test",
        "http://ms2.test?",
        "http://ms2.test/#fragment",
        "http://ms2.test#",
    ],
)
def test_worker_config_rejects_ms2_urls_outside_client_contract(
    monkeypatch, url
):
    _clear_worker_environment(monkeypatch)
    from src.config.config import settings

    monkeypatch.setattr(settings, "extract_service_url", url)

    with pytest.raises(ValidationError):
        _worker_config_api().WorkerSettings()


def test_worker_config_accepts_at_sign_in_ms2_url_path(monkeypatch):
    _clear_worker_environment(monkeypatch)
    from src.config.config import settings

    url = "http://ms2.test/v1/@extract"
    monkeypatch.setattr(settings, "extract_service_url", url)

    config = _worker_config_api().WorkerSettings()

    assert config.ms2_base_url == url


def test_worker_timeout_is_independent_of_historical_ms3_timeout(monkeypatch):
    _clear_worker_environment(monkeypatch)
    monkeypatch.setenv("EXTRACT_SERVICE_TIMEOUT", "30")

    config = _worker_config_api().WorkerSettings()

    assert config.ms2_timeout_seconds == 40.0
