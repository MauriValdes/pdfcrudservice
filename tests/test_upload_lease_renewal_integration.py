"""Pruebas de integración real para renovación del lease de carga."""

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_upload_lease_updates_only_lease_fields(_isolated_testing_db):
    """Renueva el lease conservando el resto del documento sin cambios."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository = DocumentRepository(testing_db.db)
        reservation, owner_id, fencing_token, original_lease = (
            await _create_owned_reservation(
                repository,
                checksum,
                "renew-upload-lease-success.pdf",
            )
        )
        document_filter = {
            "_id": ObjectId(reservation.id),
            "checksum": checksum,
        }
        persisted_before = await collection.find_one(document_filter)
        assert persisted_before is not None
        old_lease = persisted_before["upload_lease_until"]
        new_lease = original_lease + timedelta(minutes=15)

        renewed = await repository.renew_upload_lease(
            reservation.id,
            owner_id,
            fencing_token,
            new_lease,
        )

        assert renewed is True
        persisted_after = await collection.find_one(document_filter)
        assert persisted_after is not None

        assert _normalize_mongo_datetime(
            persisted_after["upload_lease_until"]
        ) == _normalize_mongo_datetime(new_lease)
        assert _normalize_mongo_datetime(
            persisted_after["upload_lease_until"]
        ) > _normalize_mongo_datetime(old_lease)
        assert persisted_after["upload_owner_id"] == owner_id
        assert persisted_after["upload_fencing_token"] == fencing_token
        assert persisted_after["status"] == DocumentStatus.UPLOADING.value
        assert persisted_after["file_id"] == persisted_before["file_id"]

        mutable_fields = {"upload_lease_until", "updated_at"}
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
        "earlier_expiration",
        "equal_expiration",
        "pending_status",
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_rejected_upload_lease_renewal_leaves_document_unchanged(
    _isolated_testing_db,
    scenario,
):
    """Rechaza renovaciones inválidas y preserva íntegramente el documento."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository = DocumentRepository(testing_db.db)
        reservation, owner_id, fencing_token, original_lease = (
            await _create_owned_reservation(
                repository,
                checksum,
                f"renew-upload-lease-{scenario}.pdf",
            )
        )
        document_filter = {
            "_id": ObjectId(reservation.id),
            "checksum": checksum,
        }
        requested_owner = owner_id
        requested_token = fencing_token
        requested_lease = original_lease + timedelta(minutes=15)

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
        elif scenario == "earlier_expiration":
            persisted = await collection.find_one(document_filter)
            assert persisted is not None
            requested_lease = (
                _normalize_mongo_datetime(persisted["upload_lease_until"])
                - timedelta(minutes=1)
            )
        elif scenario == "equal_expiration":
            persisted = await collection.find_one(document_filter)
            assert persisted is not None
            requested_lease = _normalize_mongo_datetime(
                persisted["upload_lease_until"]
            )
        elif scenario == "pending_status":
            preparation = await collection.update_one(
                document_filter,
                {"$set": {"status": DocumentStatus.PENDING.value}},
            )
            assert preparation.matched_count == 1

        persisted_before = await collection.find_one(document_filter)
        assert persisted_before is not None

        renewed = await repository.renew_upload_lease(
            reservation.id,
            requested_owner,
            requested_token,
            requested_lease,
        )

        assert renewed is False
        persisted_after = await collection.find_one(document_filter)
        assert persisted_after == persisted_before
    finally:
        await collection.delete_many({"checksum": checksum})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_upload_lease_returns_false_for_missing_document(
    _isolated_testing_db,
):
    """No crea un documento al intentar renovar un ObjectId inexistente."""
    testing_db = _isolated_testing_db
    collection = testing_db.db[settings.mongo_collection]
    document_id = ObjectId()
    repository = DocumentRepository(testing_db.db)

    renewed = await repository.renew_upload_lease(
        str(document_id),
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        datetime.now(timezone.utc) + timedelta(minutes=15),
    )

    assert renewed is False
    assert await collection.find_one({"_id": document_id}) is None
