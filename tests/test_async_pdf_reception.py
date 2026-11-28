"""TDD tests for asynchronous PDF reception orchestration."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from bson import ObjectId

from src.config.config import settings
from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentCreate
from src.services.document_service import (
    DocumentService,
    UploadLeaseLostError,
)


@pytest.fixture
def reception():
    repository = Mock()
    service = DocumentService(repository)
    service.reserve_or_classify = AsyncMock()
    service.acquire_upload_ownership = AsyncMock()
    service.save_and_finalize_upload = AsyncMock(return_value=True)
    service.call_extraction_service = AsyncMock()
    service.process_pdf_extraction = AsyncMock()
    repository.find_by_id = AsyncMock()
    return service, repository


def document(status, *, document_id=None, owner=None, token=None):
    return DocumentModel(
        id=str(document_id or ObjectId()),
        filename="stored.pdf",
        checksum="old-checksum",
        size_bytes=17,
        status=status,
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        upload_owner_id=owner,
        upload_fencing_token=token,
    )


def reserve(service, doc, classification):
    service.reserve_or_classify.return_value = (doc, classification)


def acquire(service, doc, acquired=True):
    service.acquire_upload_ownership.return_value = (doc, acquired)


@pytest.mark.asyncio
async def test_new_upload_is_accepted_with_exact_checksum_and_upload_metadata(
    reception, monkeypatch
):
    service, _ = reception
    data = b"%PDF bytes exactly as received"
    import hashlib

    checksum = hashlib.sha256(data).hexdigest()
    original = document(DocumentStatus.UPLOADING)
    original.checksum = checksum
    owned = document(
        DocumentStatus.UPLOADING, document_id=original.id,
        owner="new-owner", token="new-fencing",
    )
    owned.checksum = checksum
    reserve(service, original, "created")
    acquire(service, owned)
    file_id = ObjectId()
    generated = []
    monkeypatch.setattr(
        "src.services.document_service.ObjectId",
        lambda: generated.append(file_id) or file_id,
    )
    before = original.model_dump()

    accepted, classification = await service.receive_pdf(
        file_bytes=data, filename="incoming.pdf"
    )

    service.reserve_or_classify.assert_awaited_once()
    reserved_data = service.reserve_or_classify.await_args.args[0]
    assert reserved_data == DocumentCreate(
        filename="incoming.pdf", checksum=checksum, size_bytes=len(data)
    )
    service.acquire_upload_ownership.assert_awaited_once_with(
        original, settings.upload_lease_seconds
    )
    assert generated == [file_id]
    service.save_and_finalize_upload.assert_awaited_once_with(
        owned,
        file_id,
        "incoming.pdf",
        data,
        {
            "document_id": owned.id,
            "checksum": checksum,
            "upload_owner_id": "new-owner",
            "upload_fencing_token": "new-fencing",
        },
        ownership_acquired=True,
    )
    assert classification == "accepted"
    assert accepted.id == owned.id
    assert accepted.status == DocumentStatus.PENDING
    assert accepted.file_id is file_id
    assert original.model_dump() == before
    assert service.call_extraction_service.await_count == 0
    assert service.process_pdf_extraction.await_count == 0


@pytest.mark.parametrize(
    "status",
    [
        DocumentStatus.PENDING,
        DocumentStatus.PROCESSING,
        DocumentStatus.COMPLETED,
        DocumentStatus.FAILED,
    ],
)
@pytest.mark.asyncio
async def test_existing_document_returns_unchanged_without_upload(reception, status):
    service, _ = reception
    existing = document(status)
    before = existing.model_dump()
    reserve(service, existing, "existing")

    returned, classification = await service.receive_pdf(
        file_bytes=b"same bytes", filename="new-name.pdf"
    )

    assert returned is existing
    assert classification == "existing"
    assert returned.id == existing.id
    assert returned.created_at == existing.created_at
    assert returned.updated_at == existing.updated_at
    assert existing.model_dump() == before
    service.reserve_or_classify.assert_awaited_once()
    service.acquire_upload_ownership.assert_not_awaited()
    service.save_and_finalize_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_uploading_reservation_with_rejected_acquisition_returns_updated_existing(
    reception,
):
    service, _ = reception
    reservation = document(DocumentStatus.UPLOADING)
    active_owner = document(
        DocumentStatus.UPLOADING,
        document_id=reservation.id,
        owner="other-owner",
        token="other-token",
    )
    reserve(service, reservation, "existing_uploading")
    acquire(service, active_owner, False)

    returned, classification = await service.receive_pdf(
        file_bytes=b"bytes", filename="incoming.pdf"
    )

    assert returned is active_owner
    assert classification == "existing"
    service.save_and_finalize_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_uploading_reservation_can_be_acquired_and_completed(reception):
    service, _ = reception
    reservation = document(DocumentStatus.UPLOADING)
    owned = document(
        DocumentStatus.UPLOADING,
        document_id=reservation.id,
        owner="recovery-owner",
        token="recovery-token",
    )
    reserve(service, reservation, "existing_uploading")
    acquire(service, owned)

    returned, classification = await service.receive_pdf(
        file_bytes=b"bytes", filename="recovered.pdf"
    )

    assert classification == "accepted"
    assert returned.status == DocumentStatus.PENDING
    service.acquire_upload_ownership.assert_awaited_once_with(
        reservation, settings.upload_lease_seconds
    )
    assert service.save_and_finalize_upload.await_args.args[0] is owned


@pytest.mark.asyncio
async def test_acquisition_without_recoverable_document_fails_without_using_stale_reservation(
    reception,
):
    service, _ = reception
    reservation = document(DocumentStatus.UPLOADING)
    reserve(service, reservation, "created")
    service.acquire_upload_ownership.return_value = (None, False)

    with pytest.raises(UploadLeaseLostError):
        await service.receive_pdf(file_bytes=b"bytes", filename="x.pdf")

    service.save_and_finalize_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_false_finalization_raises_specific_error_without_read_or_compensation(
    reception,
):
    service, repository = reception
    reservation = document(DocumentStatus.UPLOADING)
    owned = document(
        DocumentStatus.UPLOADING, document_id=reservation.id,
        owner="owner", token="token",
    )
    reserve(service, reservation, "created")
    acquire(service, owned)
    service.save_and_finalize_upload.return_value = False
    service.delete_original_pdf = AsyncMock()

    with pytest.raises(Exception) as raised:
        await service.receive_pdf(file_bytes=b"bytes", filename="x.pdf")
    assert type(raised.value).__name__ == "UploadFinalizationRejectedError"

    repository.find_by_id.assert_not_awaited()
    service.delete_original_pdf.assert_not_awaited()


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        ("reserve", DatabaseUnavailableError()),
        ("acquire", DatabaseUnavailableError()),
        ("save", DatabaseUnavailableError()),
        ("save", UploadLeaseLostError("lost")),
        ("save", asyncio.CancelledError()),
    ],
)
@pytest.mark.asyncio
async def test_reception_propagates_operation_errors_without_compensation(
    reception, operation, error
):
    service, _ = reception
    reservation = document(DocumentStatus.UPLOADING)
    owned = document(
        DocumentStatus.UPLOADING, document_id=reservation.id,
        owner="owner", token="token",
    )
    if operation == "reserve":
        service.reserve_or_classify.side_effect = error
    else:
        reserve(service, reservation, "created")
        if operation == "acquire":
            service.acquire_upload_ownership.side_effect = error
        else:
            acquire(service, owned)
            service.save_and_finalize_upload.side_effect = error
    service.delete_original_pdf = AsyncMock()

    with pytest.raises(type(error)) as raised:
        await service.receive_pdf(file_bytes=b"bytes", filename="x.pdf")

    assert raised.value is error
    service.delete_original_pdf.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquisition_rejection_does_not_use_returned_credentials_or_create_file(
    reception, monkeypatch
):
    service, _ = reception
    reservation = document(DocumentStatus.UPLOADING)
    rejected = document(
        DocumentStatus.UPLOADING, document_id=reservation.id,
        owner="winner", token="winner-token",
    )
    reserve(service, reservation, "created")
    acquire(service, rejected, False)
    object_id = Mock(side_effect=AssertionError("ObjectId must not be generated"))
    monkeypatch.setattr("src.services.document_service.ObjectId", object_id)

    returned, kind = await service.receive_pdf(
        file_bytes=b"bytes", filename="x.pdf"
    )

    assert returned is rejected
    assert kind == "existing"
    object_id.assert_not_called()
    service.save_and_finalize_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_receives_actual_acquisition_flag_and_unique_id_per_authorized_attempt(
    reception, monkeypatch
):
    service, _ = reception
    reservation = document(DocumentStatus.UPLOADING)
    owned = document(
        DocumentStatus.UPLOADING, document_id=reservation.id,
        owner="owner", token="token",
    )
    reserve(service, reservation, "created")
    acquire(service, owned, True)
    ids = [ObjectId(), ObjectId()]
    generated = iter(ids)
    monkeypatch.setattr(
        "src.services.document_service.ObjectId", lambda: next(generated)
    )

    await service.receive_pdf(file_bytes=b"one", filename="one.pdf")
    await service.receive_pdf(file_bytes=b"two", filename="two.pdf")

    assert service.save_and_finalize_upload.await_count == 2
    calls = service.save_and_finalize_upload.await_args_list
    assert calls[0].args[1] is ids[0]
    assert calls[1].args[1] is ids[1]
    assert ids[0] != ids[1]
    assert all(call.kwargs["ownership_acquired"] is True for call in calls)


@pytest.mark.asyncio
async def test_accepted_snapshot_does_not_mutate_acquired_or_reserved_document(reception):
    service, _ = reception
    reservation = document(DocumentStatus.UPLOADING)
    owned = document(
        DocumentStatus.UPLOADING, document_id=reservation.id,
        owner="owner", token="token",
    )
    reserve(service, reservation, "created")
    acquire(service, owned)
    reservation_before = reservation.model_dump()
    owned_before = owned.model_dump()

    returned, _ = await service.receive_pdf(file_bytes=b"bytes", filename="x.pdf")

    assert returned is not owned
    assert returned.status == DocumentStatus.PENDING
    assert returned.file_id is service.save_and_finalize_upload.await_args.args[1]
    assert reservation.model_dump() == reservation_before
    assert owned.model_dump() == owned_before
    service.repository.find_by_id.assert_not_awaited()
    service.call_extraction_service.assert_not_awaited()
    service.process_pdf_extraction.assert_not_awaited()
