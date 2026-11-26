"""Pruebas unitarias RED del lector de PDFs reclamados para el worker."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from gridfs.errors import NoFile

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.worker.exceptions import ProcessingLeaseLostError
from src.worker.pdf_reader import read_claimed_pdf


MISSING = object()


def _claimed_document(*, file_id=MISSING):
    document = {
        "_id": str(ObjectId()),
        "filename": "claimed.pdf",
        "checksum": "checksum",
        "size_bytes": 128,
        "status": DocumentStatus.PROCESSING.value,
        "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "worker_id": "worker-a",
        "fencing_token": "fencing-token-a",
    }
    if file_id is not None:
        document["file_id"] = ObjectId() if file_id is MISSING else file_id
    return DocumentModel.model_validate(document)


def _repositories(*, read_result=b"%PDF-1.4\noriginal bytes"):
    processing_repository = SimpleNamespace(
        renew_processing_lease=AsyncMock(return_value=True)
    )
    gridfs_repository = SimpleNamespace(
        read=AsyncMock(return_value=read_result),
        delete=AsyncMock(return_value=True),
    )
    return processing_repository, gridfs_repository


@pytest.mark.asyncio
async def test_read_claimed_pdf_returns_exact_original_bytes():
    file_id = ObjectId()
    original_bytes = b"\x00%PDF-original\xff"
    document = _claimed_document(file_id=file_id)
    processing_repository, gridfs_repository = _repositories(
        read_result=original_bytes
    )

    result = await read_claimed_pdf(
        document,
        processing_repository=processing_repository,
        gridfs_repository=gridfs_repository,
    )

    assert result == original_bytes
    assert type(result) is bytes
    assert processing_repository.renew_processing_lease.await_count == 2
    gridfs_repository.read.assert_awaited_once_with(file_id)


@pytest.mark.asyncio
async def test_read_claimed_pdf_uses_document_file_id_not_document_id():
    file_id = ObjectId()
    document = _claimed_document(file_id=file_id)
    processing_repository, gridfs_repository = _repositories()

    await read_claimed_pdf(
        document,
        processing_repository=processing_repository,
        gridfs_repository=gridfs_repository,
    )

    assert document.id != str(file_id)
    gridfs_repository.read.assert_awaited_once_with(document.file_id)


@pytest.mark.parametrize(
    "document",
    [
        _claimed_document(file_id=None),
        _claimed_document(file_id=ObjectId()),
    ],
    ids=["file-id-absent", "file-id-invalid"],
)
@pytest.mark.asyncio
async def test_read_claimed_pdf_rejects_absent_or_invalid_file_id(document):
    if document.id and document.file_id is not None:
        document = document.model_copy(update={"file_id": "not-an-object-id"})
    processing_repository, gridfs_repository = _repositories()

    with pytest.raises((TypeError, ValueError)):
        await read_claimed_pdf(
            document,
            processing_repository=processing_repository,
            gridfs_repository=gridfs_repository,
        )

    processing_repository.renew_processing_lease.assert_not_awaited()
    gridfs_repository.read.assert_not_awaited()
    gridfs_repository.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_claimed_pdf_propagates_gridfs_no_file():
    document = _claimed_document()
    processing_repository, gridfs_repository = _repositories()
    gridfs_repository.read.side_effect = NoFile()

    with pytest.raises(NoFile):
        await read_claimed_pdf(
            document,
            processing_repository=processing_repository,
            gridfs_repository=gridfs_repository,
        )

    gridfs_repository.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_claimed_pdf_propagates_gridfs_read_errors():
    document = _claimed_document()
    processing_repository, gridfs_repository = _repositories()
    gridfs_repository.read.side_effect = RuntimeError("gridfs read failed")

    with pytest.raises(RuntimeError, match="gridfs read failed"):
        await read_claimed_pdf(
            document,
            processing_repository=processing_repository,
            gridfs_repository=gridfs_repository,
        )

    gridfs_repository.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_claimed_pdf_propagates_database_unavailable_during_ownership_check():
    document = _claimed_document()
    processing_repository, gridfs_repository = _repositories()
    processing_repository.renew_processing_lease.side_effect = (
        DatabaseUnavailableError()
    )

    with pytest.raises(DatabaseUnavailableError):
        await read_claimed_pdf(
            document,
            processing_repository=processing_repository,
            gridfs_repository=gridfs_repository,
        )

    processing_repository.renew_processing_lease.assert_awaited_once()
    gridfs_repository.read.assert_not_awaited()
    gridfs_repository.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_claimed_pdf_does_not_delete_after_successful_read():
    document = _claimed_document()
    processing_repository, gridfs_repository = _repositories()

    await read_claimed_pdf(
        document,
        processing_repository=processing_repository,
        gridfs_repository=gridfs_repository,
    )

    gridfs_repository.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_claimed_pdf_stops_before_read_when_ownership_is_lost():
    document = _claimed_document()
    processing_repository, gridfs_repository = _repositories()
    processing_repository.renew_processing_lease.return_value = False

    with pytest.raises(ProcessingLeaseLostError, match="lease"):
        await read_claimed_pdf(
            document,
            processing_repository=processing_repository,
            gridfs_repository=gridfs_repository,
        )

    processing_repository.renew_processing_lease.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
    )
    gridfs_repository.read.assert_not_awaited()
    gridfs_repository.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_claimed_pdf_discards_bytes_if_ownership_is_lost_during_read():
    document = _claimed_document()
    processing_repository, gridfs_repository = _repositories()
    processing_repository.renew_processing_lease.side_effect = [True, False]
    gridfs_repository.read.return_value = b"read but must be discarded"

    with pytest.raises(ProcessingLeaseLostError, match="lease"):
        await read_claimed_pdf(
            document,
            processing_repository=processing_repository,
            gridfs_repository=gridfs_repository,
        )

    assert processing_repository.renew_processing_lease.await_count == 2
    gridfs_repository.read.assert_awaited_once_with(document.file_id)
    gridfs_repository.delete.assert_not_awaited()
