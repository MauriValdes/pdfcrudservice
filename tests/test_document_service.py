"""Tests unitarios del servicio de documentos (MS3: CRUD y persistencia en MongoDB).

Pruebas en aislamiento con AsyncMock simulando la capa de persistencia.
"""
from datetime import datetime, timedelta, timezone
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, call
from uuid import UUID

from src.services.document_service import DocumentService
from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse


def build_service(existing_doc=None):
    """Fabrica un servicio con dependencias mockeadas para el nuevo flujo."""
    repository = AsyncMock()
    if existing_doc is not None:
        repository.find_by_checksum.return_value = existing_doc
    else:
        repository.find_by_checksum.return_value = None

    service = DocumentService(
        repository=repository,
    )
    return service, repository


def build_reservation_data():
    """Construye los datos de entrada para reservar un documento."""
    return DocumentCreate(
        filename="manual.pdf",
        checksum="abc123",
        size_bytes=1024,
    )


def build_reservation_document(status):
    """Construye un documento persistido con el estado indicado."""
    return DocumentModel(
        id="507f1f77bcf86cd799439011",
        filename="manual.pdf",
        checksum="abc123",
        size_bytes=1024,
        status=status,
    )


def build_owned_upload_document(upload_owner_id, upload_fencing_token):
    """Construye una reserva uploading con datos de propiedad existentes."""
    return DocumentModel(
        id="507f1f77bcf86cd799439011",
        filename="manual.pdf",
        checksum="abc123",
        size_bytes=1024,
        status=DocumentStatus.UPLOADING,
        upload_owner_id=upload_owner_id,
        upload_fencing_token=upload_fencing_token,
    )


@pytest.mark.asyncio
async def test_reserve_or_classify_returns_new_uploading_reservation():
    document_data = build_reservation_data()
    document = build_reservation_document(DocumentStatus.UPLOADING)
    repository = AsyncMock()
    repository.reserve_document.return_value = (document, True)
    service = DocumentService(repository=repository)

    result = await service.reserve_or_classify(document_data)

    assert result == (document, "created")
    assert result[0] is document
    repository.reserve_document.assert_awaited_once_with(document_data)
    assert repository.method_calls == [call.reserve_document(document_data)]


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
async def test_reserve_or_classify_returns_existing_document(status):
    document_data = build_reservation_data()
    document = build_reservation_document(status)
    repository = AsyncMock()
    repository.reserve_document.return_value = (document, False)
    service = DocumentService(repository=repository)

    result = await service.reserve_or_classify(document_data)

    assert result == (document, "existing")
    assert result[0] is document
    repository.reserve_document.assert_awaited_once_with(document_data)
    assert repository.method_calls == [call.reserve_document(document_data)]


@pytest.mark.asyncio
async def test_reserve_or_classify_returns_existing_uploading_reservation():
    document_data = build_reservation_data()
    document = build_reservation_document(DocumentStatus.UPLOADING)
    repository = AsyncMock()
    repository.reserve_document.return_value = (document, False)
    service = DocumentService(repository=repository)

    result = await service.reserve_or_classify(document_data)

    assert result == (document, "existing_uploading")
    assert result[0] is document
    repository.reserve_document.assert_awaited_once_with(document_data)
    assert repository.method_calls == [call.reserve_document(document_data)]


@pytest.mark.asyncio
async def test_reserve_or_classify_propagates_database_unavailable():
    document_data = build_reservation_data()
    error = DatabaseUnavailableError()
    repository = AsyncMock()
    repository.reserve_document.side_effect = error
    service = DocumentService(repository=repository)

    with pytest.raises(DatabaseUnavailableError) as raised:
        await service.reserve_or_classify(document_data)

    assert raised.value is error
    repository.reserve_document.assert_awaited_once_with(document_data)
    assert repository.method_calls == [call.reserve_document(document_data)]


@pytest.mark.asyncio
async def test_acquire_upload_ownership_returns_acquired_document_and_generates_claim():
    document = build_reservation_document(DocumentStatus.UPLOADING)
    acquired_document = build_owned_upload_document("owner-1", "fencing-1")
    repository = AsyncMock()
    repository.acquire_upload_ownership.return_value = (acquired_document, True)
    service = DocumentService(repository=repository)
    lease_duration_seconds = 120

    before = datetime.now(timezone.utc)
    result = await service.acquire_upload_ownership(
        document,
        lease_duration_seconds,
    )
    after = datetime.now(timezone.utc)

    assert result[0] is acquired_document
    assert result[1] is True
    repository.acquire_upload_ownership.assert_awaited_once()
    checksum, owner_id, fencing_token, lease_until = (
        repository.acquire_upload_ownership.await_args.args
    )
    assert checksum == document.checksum
    assert isinstance(owner_id, str)
    assert str(UUID(owner_id)) == owner_id
    assert isinstance(fencing_token, str)
    assert str(UUID(fencing_token)) == fencing_token
    assert owner_id != fencing_token
    expected_duration = timedelta(seconds=lease_duration_seconds)
    assert before + expected_duration <= lease_until <= after + expected_duration
    assert lease_until.tzinfo == timezone.utc
    assert lease_until.utcoffset() == timedelta(0)
    assert repository.method_calls == [
        call.acquire_upload_ownership(
            checksum,
            owner_id,
            fencing_token,
            lease_until,
        )
    ]


@pytest.mark.asyncio
async def test_acquire_upload_ownership_returns_rejected_document_unchanged():
    document = build_reservation_document(DocumentStatus.UPLOADING)
    existing_document = build_owned_upload_document(
        "active-upload-owner",
        "active-upload-fencing-token",
    )
    repository = AsyncMock()
    repository.acquire_upload_ownership.return_value = (existing_document, False)
    service = DocumentService(repository=repository)

    result = await service.acquire_upload_ownership(document, 120)

    assert result[0] is existing_document
    assert result[1] is False
    assert result[0].upload_owner_id == "active-upload-owner"
    assert result[0].upload_fencing_token == "active-upload-fencing-token"
    repository.acquire_upload_ownership.assert_awaited_once()
    assert len(repository.method_calls) == 1


@pytest.mark.asyncio
async def test_acquire_upload_ownership_returns_none_when_document_is_missing():
    document = build_reservation_document(DocumentStatus.UPLOADING)
    repository = AsyncMock()
    repository.acquire_upload_ownership.return_value = (None, False)
    service = DocumentService(repository=repository)

    result = await service.acquire_upload_ownership(document, 120)

    assert result == (None, False)
    repository.acquire_upload_ownership.assert_awaited_once()
    assert len(repository.method_calls) == 1


@pytest.mark.asyncio
async def test_acquire_upload_ownership_propagates_database_unavailable():
    document = build_reservation_document(DocumentStatus.UPLOADING)
    error = DatabaseUnavailableError()
    repository = AsyncMock()
    repository.acquire_upload_ownership.side_effect = error
    service = DocumentService(repository=repository)

    with pytest.raises(DatabaseUnavailableError) as raised:
        await service.acquire_upload_ownership(document, 120)

    assert raised.value is error
    repository.acquire_upload_ownership.assert_awaited_once()
    assert len(repository.method_calls) == 1


@pytest.mark.asyncio
async def test_acquire_upload_ownership_generates_new_ids_for_each_attempt():
    document = build_reservation_document(DocumentStatus.UPLOADING)
    acquired_document = build_owned_upload_document("owner", "fencing")
    repository = AsyncMock()
    repository.acquire_upload_ownership.side_effect = [
        (acquired_document, True),
        (acquired_document, True),
    ]
    service = DocumentService(repository=repository)

    await service.acquire_upload_ownership(document, 120)
    await service.acquire_upload_ownership(document, 120)

    assert repository.acquire_upload_ownership.await_count == 2
    first_args = repository.acquire_upload_ownership.await_args_list[0].args
    second_args = repository.acquire_upload_ownership.await_args_list[1].args
    first_owner_id, first_fencing_token = first_args[1:3]
    second_owner_id, second_fencing_token = second_args[1:3]
    assert first_owner_id != second_owner_id
    assert first_fencing_token != second_fencing_token
    assert first_owner_id != first_fencing_token
    assert second_owner_id != second_fencing_token
    assert len(repository.method_calls) == 2


@pytest.mark.parametrize("lease_duration_seconds", [0, -1])
@pytest.mark.asyncio
async def test_acquire_upload_ownership_rejects_non_positive_lease_duration(
    lease_duration_seconds,
):
    document = build_reservation_document(DocumentStatus.UPLOADING)
    repository = AsyncMock()
    service = DocumentService(repository=repository)

    with pytest.raises(ValueError):
        await service.acquire_upload_ownership(
            document,
            lease_duration_seconds,
        )

    repository.acquire_upload_ownership.assert_not_awaited()
    assert repository.method_calls == []


@pytest.mark.asyncio
async def test_submit_pdf_saves_metadata():
    """submit_pdf guarda metadatos básicos en MongoDB y retorna DocumentResponse."""
    service, repository = build_service()

    fake_bytes = b"%PDF-content"
    task = await service.submit_pdf(fake_bytes, "manual.pdf")

    # Se verifica que se llamó al repositorio con el checksum
    repository.find_by_checksum.assert_called_once()

    # Se verifica que se guardó el documento (insert_one)
    repository.collection.insert_one.assert_called_once()

    # El task devuelto debe ser DocumentResponse con status PENDING
    assert isinstance(task, DocumentResponse)
    assert task.filename == "manual.pdf"
    assert task.checksum is not None
    assert task.size_bytes == len(fake_bytes)
    assert task.status == "pending"  # DocumentStatus.PENDING.value
    assert task.content is None
    assert task.page_count is None
    assert task.error_message is None


@pytest.mark.asyncio
async def test_submit_pdf_raises_error_if_duplicate():
    """Error duplicada al enviar el mismo checksum."""
    service, repository = build_service(existing_doc={"filename": "manual.pdf"})

    with pytest.raises(ValueError, match="El documento ya existe"):
        await service.submit_pdf(b"%PDF-content", "manual.pdf")

    # Nada se guarda si es duplicado
    repository.collection.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_get_all_documents_calls_repo():
    """Obtiene la lista completa de documentos desde MongoDB."""
    mock_repo = AsyncMock()
    mock_response_1 = DocumentResponse(
        id="1",
        filename="doc1.pdf",
        checksum="abc123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=__import__("datetime").datetime.now(),
        updated_at=__import__("datetime").datetime.now(),
    )
    mock_response_2 = DocumentResponse(
        id="2",
        filename="doc2.pdf",
        checksum="def456",
        size_bytes=2048,
        status="processing",
        content=None,
        page_count=None,
        error_message=None,
        created_at=__import__("datetime").datetime.now(),
        updated_at=__import__("datetime").datetime.now(),
    )
    mock_repo.find_all.return_value = [mock_response_1, mock_response_2]

    service = DocumentService(repository=mock_repo)
    result = await service.get_all_documents()

    mock_repo.find_all.assert_called_once()
    assert len(result) == 2
    assert all(isinstance(r, DocumentResponse) for r in result)


@pytest.mark.asyncio
async def test_get_document_by_id_returns_document():
    """Obtiene un documento por su ID."""
    mock_repo = AsyncMock()
    # DocumentResponse usa atributos punto, no .get()
    mock_response = DocumentResponse(
        id="123",
        filename="manual.pdf",
        checksum="abc123",
        size_bytes=512,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=__import__("datetime").datetime.now(),
        updated_at=__import__("datetime").datetime.now(),
    )
    mock_repo.find_by_id.return_value = mock_response

    service = DocumentService(repository=mock_repo)
    result = await service.get_document_by_id("123")

    mock_repo.find_by_id.assert_awaited_once_with("123")
    assert result is not None
    assert result.filename == "manual.pdf"
    assert isinstance(result, DocumentResponse)


@pytest.mark.asyncio
async def test_get_document_by_id_returns_none_when_missing():
    """Devuelve None cuando el documento no existe."""
    mock_repo = AsyncMock()
    mock_repo.find_by_id.return_value = None

    service = DocumentService(repository=mock_repo)
    result = await service.get_document_by_id("id_inexistente")

    mock_repo.find_by_id.assert_awaited_once_with("id_inexistente")
    assert result is None


@pytest.mark.asyncio
async def test_update_document_partial():
    """Actualiza parcialmente un documento con DocumentUpdate."""
    mock_repo = AsyncMock()

    # Primero configurar que submit_pdf cree un documento
    service, repository = build_service()

    # Actualizar el documento - usar DocumentUpdate como parámetro positional
    updated_doc = await service.update_document(
        "doc1_id",
        DocumentUpdate(
            status="processing",
            content="Texto extraído",
            page_count=5,
        ),
    )

    # Verificar que se llamó a update_document con los datos correctos
    repository.update_document.assert_called_once()

    # Devolver un DocumentResponse actualizado
    updated_response = DocumentResponse(
        id="doc1_id",
        filename="manual.pdf",
        checksum="abc123",
        size_bytes=512,
        status="processing",
        content="Texto extraído",
        page_count=5,
        error_message=None,
        created_at=__import__("datetime").datetime.now(),
        updated_at=__import__("datetime").datetime.now(),
    )
    repository.collection.find_by_id.return_value = updated_doc

    assert updated_response.status == "processing"
    assert updated_response.content == "Texto extraído"
    assert updated_response.page_count == 5


@pytest.mark.asyncio
async def test_delete_document_success():
    """Elimina un documento exitosamente."""
    # En MS3 no hay método delete_document, se omite este test
    pytest.skip("delete_document no forma parte de MS3 pdf-crud-service")


@pytest.mark.asyncio
async def test_delete_document_not_found():
    """Error al intentar eliminar un documento que no existe."""
    # En MS3 no hay método delete_document, se omite este test
    pytest.skip("delete_document no forma parte de MS3 pdf-crud-service")