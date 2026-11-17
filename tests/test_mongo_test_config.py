"""Fail-closed MongoDB testing configuration checks; no real server is used."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymongo.uri_parser import parse_uri

from src.config.config import MONGO_TEST_DATABASE, Settings, settings
from src.database.mongodb import MongoDB, db_connection


def _settings(monkeypatch, test_url=None, production_url="mongodb://prod:27017/pdf_database_dev"):
    monkeypatch.setenv("MONGO_URL", production_url)
    if test_url is None:
        monkeypatch.delenv("MONGO_TEST_URL", raising=False)
    else:
        monkeypatch.setenv("MONGO_TEST_URL", test_url)
    return Settings()


@pytest.mark.parametrize(
    "test_url",
    [
        "mongodb://tester:do-not-print-this@[invalid/pdf_database_test",
        "mongodb://tester:do-not-print-this@mongo.example:70000/pdf_database_test",
        "mongodb://tester:do-not-print-this@invalid host/pdf_database_test",
        "mongodb://tester:do-not-print-this@mongo.example/pdf_database_test?maxPoolSize=invalid",
        "mongodb://tester:do-not-print-this@mongo.example/pdf_database_test#fragment",
        "mongodb+srv://tester:do-not-print-this@cluster.example/pdf_database_test",
        "https://tester:do-not-print-this@mongo.example/pdf_database_test",
        "mongodb://tester:do-not-print-this@mongo.example/",
        "mongodb://tester:do-not-print-this@mongo.example/another_database",
        "mongodb://tester:do-not-print-this@mongo.example/pdf_database_dev",
    ],
)
def test_test_uri_rejects_invalid_values_without_echoing_credentials(
    monkeypatch, test_url
):
    config = _settings(monkeypatch, test_url=test_url)

    with pytest.raises(ValueError) as exc_info:
        config.get_mongo_test_url()

    assert "do-not-print-this" not in str(exc_info.value)
    assert "mongodb://" not in str(exc_info.value)


def test_empty_test_uri_is_rejected_without_echoing_value(monkeypatch):
    config = _settings(monkeypatch, test_url="  ")

    with pytest.raises(ValueError) as exc_info:
        config.get_mongo_test_url()

    assert "MONGO_TEST_URL" in str(exc_info.value)
    assert "mongodb://" not in str(exc_info.value)


def test_test_uri_is_required(monkeypatch):
    config = _settings(monkeypatch)
    with pytest.raises(ValueError, match="MONGO_TEST_URL"):
        config.get_mongo_test_url()


def test_test_uri_cannot_equal_production_uri(monkeypatch):
    uri = "mongodb://same.example/pdf_database_test"
    config = _settings(monkeypatch, test_url=uri, production_url=uri)

    with pytest.raises(ValueError) as exc_info:
        config.get_mongo_test_url()

    assert uri not in str(exc_info.value)


def test_test_uri_cannot_equal_production_with_normalized_host_and_default_port(
    monkeypatch,
):
    config = _settings(
        monkeypatch,
        test_url="mongodb://same.example:27017/pdf_database_test",
        production_url="mongodb://SAME.example/pdf_database_test",
    )

    with pytest.raises(ValueError):
        config.get_mongo_test_url()


@pytest.mark.parametrize(
    "production_url",
    [
        "mongodb://prod-user:prod-secret@[invalid/pdf_database_dev",
        "https://prod-user:prod-secret@prod.example/pdf_database_dev",
    ],
)
def test_invalid_production_uri_fails_closed_without_echoing_secrets(
    monkeypatch, production_url
):
    test_uri = "mongodb://test-user:test-secret@test.example/pdf_database_test"
    config = _settings(
        monkeypatch, test_url=test_uri, production_url=production_url
    )

    with pytest.raises(ValueError) as exc_info:
        config.get_mongo_test_url()

    message = str(exc_info.value)
    for secret in ("prod-user", "prod-secret", "test-user", "test-secret", test_uri):
        assert secret not in message


def test_production_srv_uri_is_rejected_before_uri_parser_or_dns(
    monkeypatch,
):
    test_uri = "mongodb://test-user:test-secret@test.example/pdf_database_test"
    production_uri = (
        "mongodb+srv://prod-user:prod-secret@cluster.example/pdf_database_dev"
    )
    config = _settings(
        monkeypatch, test_url=test_uri, production_url=production_uri
    )

    with patch(
        "src.config.config.parse_uri", wraps=parse_uri
    ) as parse_uri_mock:
        with pytest.raises(ValueError) as exc_info:
            config.get_mongo_test_url()

    parse_uri_mock.assert_called_once_with(test_uri)
    message = str(exc_info.value)
    for secret in (
        "prod-user",
        "prod-secret",
        "test-user",
        "test-secret",
        production_uri,
        test_uri,
    ):
        assert secret not in message


def test_test_uri_accepts_explicit_test_database_and_encoded_credentials(monkeypatch):
    uri = "mongodb://test-user:p%40ss@mongo.example:27017/pdf_database_test?authSource=pdf_database_test"
    config = _settings(monkeypatch, test_url=uri)

    assert config.get_mongo_test_url() == uri


def test_test_uri_parser_accepts_valid_multiple_hosts(monkeypatch):
    uri = (
        "mongodb://test-user@mongo-a.example:27017,mongo-b.example:27018/"
        "pdf_database_test?replicaSet=rs0"
    )
    config = _settings(monkeypatch, test_url=uri)

    assert config.get_mongo_test_url() == uri


@pytest.mark.asyncio
async def test_testing_manager_uses_independent_client_and_fixed_database(monkeypatch):
    test_uri = "mongodb://test-user:secret@mongo.example/pdf_database_test"
    monkeypatch.setattr(settings, "mongo_test_url", test_uri)
    monkeypatch.setattr(settings, "mongo_url", "mongodb://prod-user:secret@prod/pdf_database_dev")

    collection = MagicMock()
    collection.create_index = AsyncMock(return_value="checksum_unique")
    database = MagicMock()
    database.name = MONGO_TEST_DATABASE
    database.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = database
    client.admin.command = AsyncMock(return_value={"ok": 1})

    with patch("src.database.mongodb.AsyncIOMotorClient", return_value=client) as factory:
        manager = MongoDB.for_testing()
        await manager.connect()

    factory.assert_called_once_with(test_uri)
    client.__getitem__.assert_called_once_with(MONGO_TEST_DATABASE)
    assert manager.db.name == MONGO_TEST_DATABASE
    assert manager is not db_connection
    assert db_connection._testing is False
    assert db_connection._mongo_url is None


@pytest.mark.asyncio
async def test_production_manager_keeps_default_uri_and_database(monkeypatch):
    production_uri = "mongodb://prod-user:secret@prod.example/pdf_database_dev"
    monkeypatch.setattr(settings, "mongo_url", production_uri)
    monkeypatch.setattr(settings, "mongo_db_dev", "pdf_database_dev")

    collection = MagicMock()
    collection.create_index = AsyncMock(return_value="checksum_unique")
    database = MagicMock()
    database.name = "pdf_database_dev"
    database.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = database
    client.admin.command = AsyncMock(return_value={"ok": 1})

    with patch("src.database.mongodb.AsyncIOMotorClient", return_value=client) as factory:
        manager = MongoDB()
        await manager.connect()

    factory.assert_called_once_with(production_uri)
    client.__getitem__.assert_called_once_with("pdf_database_dev")
    assert manager._testing is False


def test_constructor_does_not_accept_testing_uri_overrides():
    arbitrary_uri = "mongodb://unvalidated:secret@host/pdf_database_test"

    with pytest.raises(TypeError):
        MongoDB(_testing=True, _mongo_url=arbitrary_uri)


@pytest.mark.asyncio
async def test_testing_manager_cannot_select_another_database(monkeypatch):
    monkeypatch.setattr(
        settings, "mongo_test_url", "mongodb://test-user@mongo/pdf_database_test"
    )
    manager = MongoDB.for_testing()

    with pytest.raises(ValueError):
        # connect() validates before constructing a client or contacting MongoDB.
        await manager.connect(database_name="pdf_database_dev")


@pytest.mark.asyncio
async def test_testing_ping_failure_does_not_log_or_raise_uri(monkeypatch, caplog):
    test_uri = "mongodb://test-user:super-secret@mongo.example/pdf_database_test"
    monkeypatch.setattr(settings, "mongo_test_url", test_uri)
    client = MagicMock()
    client.admin.command = AsyncMock(side_effect=RuntimeError(test_uri))

    with patch("src.database.mongodb.AsyncIOMotorClient", return_value=client):
        manager = MongoDB.for_testing()
        with pytest.raises(RuntimeError) as exc_info:
            await manager.connect()

    for secret in ("test-user", "super-secret", test_uri):
        assert secret not in str(exc_info.value)
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_testing_index_failure_does_not_log_or_raise_uri(monkeypatch, caplog):
    test_uri = "mongodb://test-user:super-secret@mongo.example/pdf_database_test"
    monkeypatch.setattr(settings, "mongo_test_url", test_uri)
    collection = MagicMock()
    collection.create_index = AsyncMock(side_effect=RuntimeError(test_uri))
    database = MagicMock()
    database.name = MONGO_TEST_DATABASE
    database.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = database
    client.admin.command = AsyncMock(return_value={"ok": 1})

    with patch("src.database.mongodb.AsyncIOMotorClient", return_value=client):
        manager = MongoDB.for_testing()
        with pytest.raises(RuntimeError) as exc_info:
            await manager.connect()

    for secret in ("test-user", "super-secret", test_uri):
        assert secret not in str(exc_info.value)
        assert secret not in caplog.text
