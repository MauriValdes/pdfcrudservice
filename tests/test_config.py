"""Tests for MS3 environment-backed settings."""

from src.config.config import Settings


def test_local_mongodb_default(monkeypatch):
    monkeypatch.delenv("MONGO_URL", raising=False)

    settings = Settings()

    assert settings.mongo_url == "mongodb://localhost:27017"


def test_mongodb_url_can_be_configured_from_environment(monkeypatch):
    uri = "mongodb://user:password@db:27017/?authSource=admin"
    monkeypatch.setenv("MONGO_URL", uri)

    settings = Settings()

    assert settings.mongo_url == uri
