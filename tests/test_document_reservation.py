"""Tests unitarios para reserve_document() en DocumentRepository.

Fase RED: validan el contrato aprobado antes de implementar la operación.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.repository.document_repository import DocumentRepository
from src.schemas.document import DocumentCreate


@pytest.fixture
def repo_with_mocks():
    """Crea un repositorio con una colección Motor mockeada."""
    collection = MagicMock()
    collection.insert_one = AsyncMock()
    collection.find_one = AsyncMock()
    collection.update_one = AsyncMock()
    collection.delete_one = AsyncMock()

    repo = DocumentRepository.__new__(DocumentRepository)
    repo.collection = collection
    return repo


@pytest.mark.asyncio
async def test_reserve_document_success(repo_with_mocks):
    """Reserva exitosa: persiste los campos esperados y no consulta antes."""
    repo = repo_with_mocks
    checksum = "abc123def456"
    filename = "test.pdf"
    size_bytes = 1024
    inserted_id = ObjectId()
    document_data = DocumentCreate(
        filename=filename,
        checksum=checksum,
        size_bytes=size_bytes,
    )
    insert_result = Mock()
    insert_result.inserted_id = inserted_id
    repo.collection.insert_one.return_value = insert_result

    result_document, success = await repo.reserve_document(document_data)

    assert success is True
    assert isinstance(result_document, DocumentModel)
    assert result_document.id == str(inserted_id)
    assert result_document.checksum == checksum
    assert result_document.filename == filename
    assert result_document.size_bytes == size_bytes
    assert result_document.status == DocumentStatus.UPLOADING
    assert result_document.file_id is None
    assert result_document.upload_owner_id is None
    assert result_document.upload_lease_until is None
    assert result_document.upload_fencing_token is None

    for timestamp in (result_document.created_at, result_document.updated_at):
        assert timestamp.tzinfo is not None
        assert timestamp.utcoffset().total_seconds() == 0
        assert timestamp.tzinfo == timezone.utc

    repo.collection.insert_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()

    insert_call = repo.collection.insert_one.call_args
    if insert_call.args:
        inserted_document = insert_call.args[0]
    else:
        inserted_document = insert_call.kwargs["document"]

    assert inserted_document["filename"] == filename
    assert inserted_document["checksum"] == checksum
    assert inserted_document["size_bytes"] == size_bytes
    assert inserted_document["status"] == DocumentStatus.UPLOADING
    for field in ("created_at", "updated_at"):
        timestamp = inserted_document[field]
        assert timestamp.tzinfo is not None
        assert timestamp.utcoffset().total_seconds() == 0
        assert timestamp.tzinfo == timezone.utc
    assert inserted_document["file_id"] is None
    assert inserted_document["upload_owner_id"] is None
    assert inserted_document["upload_lease_until"] is None
    assert inserted_document["upload_fencing_token"] is None


@pytest.mark.asyncio
async def test_reserve_document_duplicate_checksum(repo_with_mocks):
    """Un duplicado devuelve el documento existente sin sobrescribirlo."""
    repo = repo_with_mocks
    checksum = "abc123def456"
    document_data = DocumentCreate(
        filename="requested.pdf",
        checksum=checksum,
        size_bytes=1024,
    )
    existing_id = ObjectId()
    existing_created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    existing_updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    existing_document = {
        "_id": existing_id,
        "filename": "original.pdf",
        "checksum": checksum,
        "size_bytes": 2048,
        "status": DocumentStatus.COMPLETED.value,
        "content": "contenido ya procesado",
        "page_count": 3,
        "error_message": None,
        "created_at": existing_created_at,
        "updated_at": existing_updated_at,
        "file_id": None,
        "upload_owner_id": None,
        "upload_lease_until": None,
        "upload_fencing_token": None,
    }
    repo.collection.insert_one.side_effect = DuplicateKeyError(
        "E11000 duplicate key error index"
    )
    repo.collection.find_one.return_value = existing_document

    result_document, success = await repo.reserve_document(document_data)

    assert success is False
    assert isinstance(result_document, DocumentModel)
    assert result_document.id == str(existing_id)
    assert result_document.filename == "original.pdf"
    assert result_document.checksum == checksum
    assert result_document.size_bytes == 2048
    assert result_document.status == DocumentStatus.COMPLETED
    assert result_document.content == "contenido ya procesado"
    assert result_document.page_count == 3
    assert result_document.created_at == existing_created_at
    assert result_document.updated_at == existing_updated_at

    repo.collection.insert_one.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once_with({"checksum": checksum})
    repo.collection.update_one.assert_not_called()
    repo.collection.delete_one.assert_not_called()


@pytest.mark.asyncio
async def test_reserve_document_duplicate_keyerror_no_recovery(repo_with_mocks):
    """Si no se recupera el duplicado, propaga la misma excepción original."""
    repo = repo_with_mocks
    document_data = DocumentCreate(
        filename="test.pdf",
        checksum="abc123def456",
        size_bytes=1024,
    )
    original_error = DuplicateKeyError("E11000 duplicate key error index")
    repo.collection.insert_one.side_effect = original_error
    repo.collection.find_one.return_value = None

    with pytest.raises(DuplicateKeyError) as raised:
        await repo.reserve_document(document_data)

    assert raised.value is original_error
    repo.collection.insert_one.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once_with(
        {"checksum": document_data.checksum}
    )


@pytest.mark.asyncio
async def test_reserve_document_connect_error_insert(repo_with_mocks):
    """Un error de conectividad al insertar se traduce a DatabaseUnavailableError."""
    repo = repo_with_mocks
    document_data = DocumentCreate(
        filename="test.pdf",
        checksum="abc123def456",
        size_bytes=1024,
    )
    repo.collection.insert_one.side_effect = ServerSelectionTimeoutError(
        "Connection timed out"
    )

    with pytest.raises(DatabaseUnavailableError):
        await repo.reserve_document(document_data)

    repo.collection.insert_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_reserve_document_connect_error_recovery(repo_with_mocks):
    """Un error de conectividad al recuperar el duplicado se traduce."""
    repo = repo_with_mocks
    document_data = DocumentCreate(
        filename="test.pdf",
        checksum="abc123def456",
        size_bytes=1024,
    )
    repo.collection.insert_one.side_effect = DuplicateKeyError(
        "E11000 duplicate key error index"
    )
    repo.collection.find_one.side_effect = ServerSelectionTimeoutError(
        "Connection timed out"
    )

    with pytest.raises(DatabaseUnavailableError):
        await repo.reserve_document(document_data)

    repo.collection.insert_one.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once_with(
        {"checksum": document_data.checksum}
    )
