"""Pruebas de integración de índices contra MongoDB real de testing."""

import hashlib
import uuid

import pytest
import pytest_asyncio
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from src.config.config import MONGO_TEST_DATABASE, settings
from src.database.indexes import ensure_required_indexes
from src.database.mongodb import MongoDB


EXPECTED_INDEXES = {
    "checksum_unique": {
        "keys": [("checksum", 1)],
        "unique": True,
    },
    "pending_jobs_idx": {
        "keys": [
            ("status", 1),
            ("next_retry_at", 1),
            ("created_at", 1),
        ],
        "unique": False,
    },
    "expired_leases_idx": {
        "keys": [("status", 1), ("lease_until", 1)],
        "unique": False,
    },
}


@pytest_asyncio.fixture
async def testing_context():
    """Conecta solo a pdf_database_test y garantiza la desconexión."""
    manager = MongoDB.for_testing()
    validated_db = None
    try:
        assert manager._testing is True
        assert manager._database_name == MONGO_TEST_DATABASE == "pdf_database_test"

        await manager.connect()

        if (
            manager._testing is not True
            or manager.db is None
            or manager.db.name != MONGO_TEST_DATABASE
        ):
            raise RuntimeError("El fixture solo puede operar sobre pdf_database_test.")
        validated_db = manager.db

        auth_status = await manager.client.admin.command(
            {"connectionStatus": 1}
        )
        auth_info = auth_status["authInfo"]
        authenticated_users = {
            user["user"] for user in auth_info["authenticatedUsers"]
        }
        role_databases = {
            role["db"] for role in auth_info["authenticatedUserRoles"]
        }
        assert authenticated_users == {"pdf_ms3_test"}
        assert role_databases == {MONGO_TEST_DATABASE}

        server_info = await manager.client.server_info()
        version = server_info["version"]
        assert version.startswith("8.0."), (
            f"Se esperaba MongoDB 8.0.x, se obtuvo {version}"
        )

        yield manager, validated_db, version
    finally:
        if validated_db is not None and (
            manager._testing is not True
            or manager.db is None
            or manager.db.name != MONGO_TEST_DATABASE
        ):
            raise RuntimeError("Se bloqueó el uso fuera de pdf_database_test.")
        await manager.disconnect()


async def _indexes_by_name(collection):
    indexes = await collection.list_indexes().to_list(length=None)
    return {index["name"]: index for index in indexes}, indexes


def _assert_index_definition(index, name):
    expected = EXPECTED_INDEXES[name]
    assert list(index["key"].items()) == expected["keys"]
    assert index.get("unique", False) is expected["unique"]

    if name != "checksum_unique":
        assert "expireAfterSeconds" not in index
        assert index.get("sparse", False) is False
        assert "partialFilterExpression" not in index


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mongodb_real_crea_los_tres_indices_requeridos(testing_context):
    _, db, version = testing_context
    collection = db[settings.mongo_collection]
    indexes_by_name, indexes = await _indexes_by_name(collection)

    assert version.startswith("8.0.")
    assert set(EXPECTED_INDEXES).issubset(indexes_by_name)
    assert len([index for index in indexes if index["name"] == "checksum_unique"]) == 1
    for name in EXPECTED_INDEXES:
        _assert_index_definition(indexes_by_name[name], name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checksum_unique_tiene_definicion_real_y_unica(testing_context):
    _, db, _ = testing_context
    indexes_by_name, _ = await _indexes_by_name(db[settings.mongo_collection])

    assert "checksum_unique" in indexes_by_name
    _assert_index_definition(indexes_by_name["checksum_unique"], "checksum_unique")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pending_jobs_idx_tiene_definicion_real_no_unica_y_sin_opciones(
    testing_context,
):
    _, db, _ = testing_context
    indexes_by_name, _ = await _indexes_by_name(db[settings.mongo_collection])

    assert "pending_jobs_idx" in indexes_by_name
    _assert_index_definition(indexes_by_name["pending_jobs_idx"], "pending_jobs_idx")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expired_leases_idx_tiene_definicion_real_no_unica_y_sin_opciones(
    testing_context,
):
    _, db, _ = testing_context
    indexes_by_name, _ = await _indexes_by_name(db[settings.mongo_collection])

    assert "expired_leases_idx" in indexes_by_name
    _assert_index_definition(indexes_by_name["expired_leases_idx"], "expired_leases_idx")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_inicializacion_repetida_es_idempotente_en_mongodb(testing_context):
    _, db, _ = testing_context
    collection = db[settings.mongo_collection]
    before_by_name, before_indexes = await _indexes_by_name(collection)
    before_names = sorted(index["name"] for index in before_indexes)

    await ensure_required_indexes(collection)
    await ensure_required_indexes(collection)

    after_by_name, after_indexes = await _indexes_by_name(collection)
    after_names = sorted(index["name"] for index in after_indexes)
    assert after_names == before_names
    assert len(after_names) == len(set(after_names))
    for name in EXPECTED_INDEXES:
        _assert_index_definition(after_by_name[name], name)
        assert after_by_name[name] == before_by_name[name]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checksum_unique_rechaza_checksum_duplicado_y_limpia_solo_sus_docs(
    testing_context,
):
    manager, db, _ = testing_context
    collection = db[settings.mongo_collection]
    checksum = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    owned_documents = []
    cleanup_results = []

    try:
        first_id = ObjectId()
        owned_documents.append((first_id, checksum))
        await collection.insert_one({"_id": first_id, "checksum": checksum})

        duplicate_id = ObjectId()
        owned_documents.append((duplicate_id, checksum))
        try:
            await collection.insert_one(
                {"_id": duplicate_id, "checksum": checksum}
            )
        except DuplicateKeyError as error:
            assert error.details.get("keyPattern") == {"checksum": 1}
        else:
            pytest.fail("MongoDB permitió un checksum duplicado.")
    finally:
        if (
            manager._testing is not True
            or manager.db is not db
            or db.name != MONGO_TEST_DATABASE
        ):
            raise RuntimeError("Se bloqueó la limpieza fuera de pdf_database_test.")
        for document_id, owned_checksum in owned_documents:
            result = await collection.delete_one(
                {"_id": document_id, "checksum": owned_checksum}
            )
            cleanup_results.append((document_id, owned_checksum, result.deleted_count))

    assert [deleted_count for _, _, deleted_count in cleanup_results] == [1, 0]
    for document_id, owned_checksum, _ in cleanup_results:
        assert await collection.find_one(
            {"_id": document_id, "checksum": owned_checksum}
        ) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connect_de_testing_deja_ready_los_tres_indices(testing_context):
    manager, db, _ = testing_context
    indexes_by_name, _ = await _indexes_by_name(db[settings.mongo_collection])

    assert manager.connected is True
    assert manager.indexes_ready is True
    assert set(EXPECTED_INDEXES).issubset(indexes_by_name)
    for name in EXPECTED_INDEXES:
        _assert_index_definition(indexes_by_name[name], name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reconexion_de_testing_reverifica_los_tres_indices(testing_context):
    manager, _, _ = testing_context
    assert manager.connected is True
    assert manager.indexes_ready is True

    await manager.connect()

    indexes_by_name, _ = await _indexes_by_name(
        manager.db[settings.mongo_collection]
    )
    assert manager.connected is True
    assert manager.indexes_ready is True
    assert set(EXPECTED_INDEXES).issubset(indexes_by_name)
    for name in EXPECTED_INDEXES:
        _assert_index_definition(indexes_by_name[name], name)
