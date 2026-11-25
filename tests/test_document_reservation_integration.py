"""Pruebas de integración real para la reserva atómica de documentos."""

import asyncio
import hashlib

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reserve_document_is_atomic_for_concurrent_same_checksum(
    _isolated_testing_db,
):
    """Solo una de dos reservas concurrentes del mismo checksum debe ganar."""
    testing_db = _isolated_testing_db
    checksum = hashlib.sha256(ObjectId().binary).hexdigest()
    document_data = DocumentCreate(
        filename="concurrent-reservation.pdf",
        checksum=checksum,
        size_bytes=1024,
    )
    first_repository = DocumentRepository(testing_db.db)
    second_repository = DocumentRepository(testing_db.db)
    collection = testing_db.db[settings.mongo_collection]

    try:
        first_result, second_result = await asyncio.gather(
            first_repository.reserve_document(document_data),
            second_repository.reserve_document(document_data),
        )

        results = (first_result, second_result)
        assert sum(created for _, created in results) == 1
        assert {created for _, created in results} == {False, True}

        first_document, _ = first_result
        second_document, _ = second_result
        assert first_document.id == second_document.id

        matching_count = await collection.count_documents({"checksum": checksum})
        assert matching_count == 1

        persisted_document = await collection.find_one({"checksum": checksum})
        assert persisted_document is not None
        assert persisted_document["status"] == DocumentStatus.UPLOADING.value
        assert persisted_document["file_id"] is None
        assert persisted_document["upload_owner_id"] is None
        assert persisted_document["upload_lease_until"] is None
        assert persisted_document["upload_fencing_token"] is None
    finally:
        await collection.delete_many({"checksum": checksum})
