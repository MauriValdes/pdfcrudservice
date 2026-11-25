"""Pruebas de integración real para adquisición de propiedad de carga."""

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_upload_ownership_and_active_lease_protection(
    _isolated_testing_db,
):
    """Una adquisición concurrente gana y el lease vigente no se reemplaza."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository_one = DocumentRepository(testing_db.db)
        repository_two = DocumentRepository(testing_db.db)
        document_data = DocumentCreate(
            filename="concurrent-upload-ownership.pdf",
            checksum=checksum,
            size_bytes=1024,
        )
        reservation, created = await repository_one.reserve_document(document_data)
        assert created is True

        owners = (str(uuid.uuid4()), str(uuid.uuid4()))
        tokens = (str(uuid.uuid4()), str(uuid.uuid4()))
        leases = (
            datetime.now(timezone.utc) + timedelta(minutes=15),
            datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        results = await asyncio.gather(
            repository_one.acquire_upload_ownership(
                checksum, owners[0], tokens[0], leases[0]
            ),
            repository_two.acquire_upload_ownership(
                checksum, owners[1], tokens[1], leases[1]
            ),
        )

        assert sum(acquired for _, acquired in results) == 1
        assert {acquired for _, acquired in results} == {False, True}
        assert all(document.id == reservation.id for document, _ in results)

        winner_index = next(
            index for index, (_, acquired) in enumerate(results) if acquired
        )
        winning_owner = owners[winner_index]
        winning_token = tokens[winner_index]
        winning_lease = leases[winner_index]

        persisted = await collection.find_one({"checksum": checksum})
        assert persisted is not None
        assert persisted["upload_owner_id"] == winning_owner
        assert persisted["upload_fencing_token"] == winning_token
        assert _normalize_mongo_datetime(persisted["upload_lease_until"]) == (
            _normalize_mongo_datetime(winning_lease)
        )

        third_owner = str(uuid.uuid4())
        third_token = str(uuid.uuid4())
        third_lease = datetime.now(timezone.utc) + timedelta(minutes=15)
        third_document, third_acquired = (
            await repository_one.acquire_upload_ownership(
                checksum,
                third_owner,
                third_token,
                third_lease,
            )
        )

        assert third_acquired is False
        assert third_document is not None
        assert third_document.id == reservation.id

        persisted_after_third_attempt = await collection.find_one(
            {"checksum": checksum}
        )
        assert persisted_after_third_attempt is not None
        assert (
            persisted_after_third_attempt["upload_owner_id"] == winning_owner
        )
        assert (
            persisted_after_third_attempt["upload_fencing_token"]
            == winning_token
        )
        assert _normalize_mongo_datetime(
            persisted_after_third_attempt["upload_lease_until"]
        ) == _normalize_mongo_datetime(winning_lease)
    finally:
        await collection.delete_many({"checksum": checksum})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expired_upload_lease_can_be_reacquired(_isolated_testing_db):
    """Una nueva solicitud adquiere la reserva cuyo lease ya venció."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository = DocumentRepository(testing_db.db)
        reservation, created = await repository.reserve_document(
            DocumentCreate(
                filename="expired-upload-lease.pdf",
                checksum=checksum,
                size_bytes=2048,
            )
        )
        assert created is True

        original_owner = str(uuid.uuid4())
        original_token = str(uuid.uuid4())
        original_lease = datetime.now(timezone.utc) + timedelta(minutes=15)
        initially_acquired_document, acquired = (
            await repository.acquire_upload_ownership(
                checksum,
                original_owner,
                original_token,
                original_lease,
            )
        )
        assert acquired is True
        assert initially_acquired_document.id == reservation.id

        expired_lease = datetime.now(timezone.utc) - timedelta(minutes=1)
        expiration_result = await collection.update_one(
            {"_id": ObjectId(reservation.id), "checksum": checksum},
            {"$set": {"upload_lease_until": expired_lease}},
        )
        assert expiration_result.matched_count == 1

        replacement_owner = str(uuid.uuid4())
        replacement_token = str(uuid.uuid4())
        replacement_lease = datetime.now(timezone.utc) + timedelta(minutes=15)
        reacquired_document, reacquired = (
            await repository.acquire_upload_ownership(
                checksum,
                replacement_owner,
                replacement_token,
                replacement_lease,
            )
        )

        assert reacquired is True
        assert reacquired_document.id == reservation.id

        persisted = await collection.find_one({"checksum": checksum})
        assert persisted is not None
        assert persisted["upload_owner_id"] == replacement_owner
        assert persisted["upload_fencing_token"] == replacement_token
        assert _normalize_mongo_datetime(persisted["upload_lease_until"]) == (
            _normalize_mongo_datetime(replacement_lease)
        )
        assert persisted["status"] == DocumentStatus.UPLOADING.value
        assert persisted["file_id"] is None
    finally:
        await collection.delete_many({"checksum": checksum})


@pytest.mark.parametrize(
    "ineligible_case",
    ["pending", "uploading_with_file_id"],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ineligible_document_does_not_acquire_upload_ownership(
    _isolated_testing_db,
    ineligible_case,
):
    """No adquiere reservas que no están uploading o que ya tienen file_id."""
    testing_db = _isolated_testing_db
    checksum = _unique_checksum()
    collection = testing_db.db[settings.mongo_collection]

    try:
        repository = DocumentRepository(testing_db.db)
        reservation, created = await repository.reserve_document(
            DocumentCreate(
                filename=f"{ineligible_case}.pdf",
                checksum=checksum,
                size_bytes=512,
            )
        )
        assert created is True

        if ineligible_case == "pending":
            preparation_update = {"status": DocumentStatus.PENDING.value}
        else:
            preparation_update = {"file_id": ObjectId()}

        preparation_result = await collection.update_one(
            {"_id": ObjectId(reservation.id), "checksum": checksum},
            {"$set": preparation_update},
        )
        assert preparation_result.matched_count == 1

        persisted_before = await collection.find_one({"checksum": checksum})
        assert persisted_before is not None

        result_document, acquired = await repository.acquire_upload_ownership(
            checksum,
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        assert acquired is False
        assert result_document is not None
        assert result_document.id == reservation.id

        persisted_after = await collection.find_one({"checksum": checksum})
        assert persisted_after == persisted_before
    finally:
        await collection.delete_many({"checksum": checksum})
