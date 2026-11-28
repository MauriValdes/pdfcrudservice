"""Pruebas unitarias del flujo de guardado y finalización de carga."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from bson import ObjectId

from src.core.exceptions import DatabaseUnavailableError
from src.services.document_service import DocumentService, UploadLeaseLostError


@pytest.fixture
def setup_service():
    repository = Mock()
    service = DocumentService(repository)
    document = SimpleNamespace(
        id=str(ObjectId()),
        upload_owner_id="owner-1",
        upload_fencing_token="token-1",
        status="uploading",
        file_id=None,
        marker="unchanged",
    )
    file_id = ObjectId()
    metadata = {"checksum": "abc"}
    service.save_original_pdf_with_lease = AsyncMock(return_value=file_id)
    repository.finalize_upload = AsyncMock(return_value=True)
    return service, repository, document, file_id, metadata


@pytest.mark.asyncio
async def test_success_saves_then_finalizes_once_with_same_file_and_arguments(
    setup_service,
):
    service, repository, document, file_id, metadata = setup_service
    order = []
    original_document_values = vars(document).copy()

    async def save(*args, **kwargs):
        order.append("save")
        return file_id

    async def finalize(*args, **kwargs):
        order.append("finalize")
        return True

    service.save_original_pdf_with_lease.side_effect = save
    repository.finalize_upload.side_effect = finalize

    result = await service.save_and_finalize_upload(
        document, file_id, "sample.pdf", b"pdf", metadata,
        ownership_acquired=True,
    )

    assert result is True
    assert order == ["save", "finalize"]
    service.save_original_pdf_with_lease.assert_awaited_once_with(
        document, file_id, "sample.pdf", b"pdf", metadata,
        ownership_acquired=True,
    )
    repository.finalize_upload.assert_awaited_once_with(
        document.id, document.upload_owner_id,
        document.upload_fencing_token, file_id,
    )
    assert service.save_original_pdf_with_lease.await_args.args[1] is file_id
    assert repository.finalize_upload.await_args.args[3] is file_id
    assert vars(document) == original_document_values

@pytest.mark.asyncio
async def test_rejected_acquisition_does_not_finalize(setup_service):
    service, repository, document, file_id, metadata = setup_service
    error = UploadLeaseLostError("not acquired")
    service.save_original_pdf_with_lease.side_effect = error

    with pytest.raises(UploadLeaseLostError) as raised:
        await service.save_and_finalize_upload(
            document, file_id, "sample.pdf", b"pdf", metadata,
            ownership_acquired=False,
        )

    assert raised.value is error
    service.save_original_pdf_with_lease.assert_awaited_once_with(
        document, file_id, "sample.pdf", b"pdf", metadata,
        ownership_acquired=False,
    )
    repository.finalize_upload.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        OSError("storage failed"),
        UploadLeaseLostError("lease lost"),
        asyncio.CancelledError(),
    ],
)
@pytest.mark.asyncio
async def test_storage_error_is_preserved_and_skips_finalization(setup_service, error):
    service, repository, document, file_id, metadata = setup_service
    service.save_original_pdf_with_lease.side_effect = error

    with pytest.raises(type(error)) as raised:
        await service.save_and_finalize_upload(
            document, file_id, "sample.pdf", b"pdf", metadata,
            ownership_acquired=True,
        )

    assert raised.value is error
    repository.finalize_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalization_false_is_returned_without_cleanup(setup_service):
    service, repository, document, file_id, metadata = setup_service
    repository.finalize_upload.return_value = False
    service.delete_original_pdf = AsyncMock()
    repository.find_by_id = AsyncMock()

    result = await service.save_and_finalize_upload(
        document, file_id, "sample.pdf", b"pdf", metadata,
        ownership_acquired=True,
    )

    assert result is False
    repository.finalize_upload.assert_awaited_once()
    service.delete_original_pdf.assert_not_awaited()
    repository.find_by_id.assert_not_awaited()


@pytest.mark.parametrize("error", [
    DatabaseUnavailableError(),
    asyncio.CancelledError(),
])
@pytest.mark.asyncio
async def test_finalization_error_propagates_without_compensation(setup_service, error):
    service, repository, document, file_id, metadata = setup_service
    repository.finalize_upload.side_effect = error
    service.delete_original_pdf = AsyncMock()
    repository.find_by_id = AsyncMock()

    with pytest.raises(type(error)) as raised:
        await service.save_and_finalize_upload(
            document, file_id, "sample.pdf", b"pdf", metadata,
            ownership_acquired=True,
        )

    assert raised.value is error
    repository.finalize_upload.assert_awaited_once()
    service.delete_original_pdf.assert_not_awaited()
    repository.find_by_id.assert_not_awaited()
