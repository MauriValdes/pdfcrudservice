"""Pruebas de integración real para finalización atómica de carga."""

import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from bson import ObjectId

from src.config.config import settings
from src.database.mongodb import MongoDB
from src.models.document import DocumentStatus
from src.repository.document_repository import DocumentRepository
from src.schemas.document import DocumentCreate


@pytest_asyncio.fixture
async def _isolated_testing_db():
    """Conecta a la base de testing y garantiza la desconexión."""
    testing_db = MongoDB.for_testing()
    try:
        await testing_db.connect()
        yield testing_db
    finally:
        await testing_db.disconnect()


def _unique_checksum() -> str:
    """Genera un checksum SHA-256 único para cada prueba."""
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


def _normalize_mongo_datetime(value: datetime) -> datetime:
    """Normaliza fechas MongoDB a UTC con precisión de milisegundos."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=(value.microsecond // 1000) * 1000)


async def _create_owned_reservation(repository, checksum, filename):
    """Crea una reserva y adquiere su propiedad de carga."""
    reservation, created = await repository.reserve_document(
        DocumentCreate(
            filename=filename,
            checksum=checksum,
            size_bytes=1024,
        )
    )
    assert created is True

    owner_id = str(uuid.uuid4())
    fencing_token = str(uuid.uuid4())
    lease_until = datetime.now(timezone.utc) + timedelta(minutes=15)
    owned_document, acquired = await repository.acquire_upload_ownership(
        checksum,
        owner_id,
        fencing_token,
        lease_until,
    )
    assert acquired is True
    assert owned_document is not None
    assert owned_document.id == reservation.id

    return reservation, owner_id, fencing_token, lease_until


@pytest.mark.parametrize("file_id_state", ["null", "absent"])
@pytest.mark.integration
@pytest.mark.asyncio
async def test_finalize_upload_associates_file_and_preserves_other_fields(
    _isolated_testing_db,
    file_id_state,
):
    """Asocia file_id y pasa a pending, sin alterar los demás campos."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository = DocumentRepository(testing_db.db)
        reservation, owner_id, fencing_token, _ = (
            await _create_owned_reservation(
                repository,
                checksum,
                f"finalize-upload-success-{file_id_state}.pdf",
            )
        )
        document_filter = {
            "_id": ObjectId(reservation.id),
            "checksum": checksum,
        }

        if file_id_state == "absent":
            preparation = await collection.update_one(
                document_filter,
                {"$unset": {"file_id": ""}},
            )
            assert preparation.matched_count == 1

        persisted_before = await collection.find_one(document_filter)
        assert persisted_before is not None
        if file_id_state == "null":
            assert persisted_before["file_id"] is None
        else:
            assert "file_id" not in persisted_before

        file_id = ObjectId()
        finalized = await repository.finalize_upload(
            reservation.id,
            owner_id,
            fencing_token,
            file_id,
        )

        assert finalized is True
        persisted_after = await collection.find_one(document_filter)
        assert persisted_after is not None
        assert persisted_after["status"] == DocumentStatus.PENDING.value
        assert isinstance(persisted_after["file_id"], ObjectId)
        assert persisted_after["file_id"] == file_id
        assert persisted_after["upload_owner_id"] == owner_id
        assert persisted_after["upload_fencing_token"] == fencing_token
        assert _normalize_mongo_datetime(
            persisted_after["upload_lease_until"]
        ) == _normalize_mongo_datetime(
            persisted_before["upload_lease_until"]
        )

        mutable_fields = {"file_id", "status", "updated_at"}
        assert {
            key: value
            for key, value in persisted_after.items()
            if key not in mutable_fields
        } == {
            key: value
            for key, value in persisted_before.items()
            if key not in mutable_fields
        }
    finally:
        await collection.delete_many({"checksum": checksum})


@pytest.mark.parametrize(
    "scenario",
    [
        "wrong_owner",
        "wrong_fencing_token",
        "expired_lease",
        "file_already_associated",
        "pending_status",
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_rejected_upload_finalization_leaves_document_unchanged(
    _isolated_testing_db,
    scenario,
):
    """Rechaza finalizaciones no elegibles sin modificar el documento."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository = DocumentRepository(testing_db.db)
        reservation, owner_id, fencing_token, _ = (
            await _create_owned_reservation(
                repository,
                checksum,
                f"finalize-upload-rejected-{scenario}.pdf",
            )
        )
        document_filter = {
            "_id": ObjectId(reservation.id),
            "checksum": checksum,
        }
        requested_owner = owner_id
        requested_token = fencing_token
        file_id = ObjectId()

        if scenario == "wrong_owner":
            requested_owner = str(uuid.uuid4())
        elif scenario == "wrong_fencing_token":
            requested_token = str(uuid.uuid4())
        elif scenario == "expired_lease":
            expired_lease = datetime.now(timezone.utc) - timedelta(minutes=1)
            preparation = await collection.update_one(
                document_filter,
                {"$set": {"upload_lease_until": expired_lease}},
            )
            assert preparation.matched_count == 1
        elif scenario == "file_already_associated":
            preparation = await collection.update_one(
                document_filter,
                {"$set": {"file_id": ObjectId()}},
            )
            assert preparation.matched_count == 1
        elif scenario == "pending_status":
            preparation = await collection.update_one(
                document_filter,
                {"$set": {"status": DocumentStatus.PENDING.value}},
            )
            assert preparation.matched_count == 1

        persisted_before = await collection.find_one(document_filter)
        assert persisted_before is not None

        finalized = await repository.finalize_upload(
            reservation.id,
            requested_owner,
            requested_token,
            file_id,
        )

        assert finalized is False
        persisted_after = await collection.find_one(document_filter)
        assert persisted_after == persisted_before
    finally:
        await collection.delete_many({"checksum": checksum})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_finalize_upload_returns_false_for_missing_document(
    _isolated_testing_db,
):
    """No crea un documento al finalizar un ObjectId inexistente."""
    testing_db = _isolated_testing_db
    collection = testing_db.db[settings.mongo_collection]
    document_id = ObjectId()
    repository = DocumentRepository(testing_db.db)

    try:
        finalized = await repository.finalize_upload(
            str(document_id),
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            ObjectId(),
        )

        assert finalized is False
        assert await collection.find_one({"_id": document_id}) is None
    finally:
        await collection.delete_one({"_id": document_id})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_upload_finalization_associates_only_one_file(
    _isolated_testing_db,
):
    """Dos finalizaciones concurrentes solo asocian el archivo ganador."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository_one = DocumentRepository(testing_db.db)
        repository_two = DocumentRepository(testing_db.db)
        reservation, owner_id, fencing_token, _ = (
            await _create_owned_reservation(
                repository_one,
                checksum,
                "finalize-upload-concurrent.pdf",
            )
        )
        document_filter = {
            "_id": ObjectId(reservation.id),
            "checksum": checksum,
        }
        persisted_before = await collection.find_one(document_filter)
        assert persisted_before is not None

        file_ids = (ObjectId(), ObjectId())
        results = await asyncio.gather(
            repository_one.finalize_upload(
                reservation.id,
                owner_id,
                fencing_token,
                file_ids[0],
            ),
            repository_two.finalize_upload(
                reservation.id,
                owner_id,
                fencing_token,
                file_ids[1],
            ),
        )

        assert results.count(True) == 1
        assert results.count(False) == 1
        winning_index = results.index(True)
        persisted_after = await collection.find_one(document_filter)
        assert persisted_after is not None
        assert persisted_after["status"] == DocumentStatus.PENDING.value
        assert isinstance(persisted_after["file_id"], ObjectId)
        assert persisted_after["file_id"] == file_ids[winning_index]
        assert persisted_after["upload_owner_id"] == owner_id
        assert persisted_after["upload_fencing_token"] == fencing_token
        assert _normalize_mongo_datetime(
            persisted_after["upload_lease_until"]
        ) == _normalize_mongo_datetime(
            persisted_before["upload_lease_until"]
        )
    finally:
        await collection.delete_many({"checksum": checksum})
