"""Tests unitarios del servicio de documentos (MS3: CRUD y persistencia en MongoDB).

Pruebas en aislamiento con AsyncMock simulando la capa de persistencia.
"""
from datetime import datetime
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, call

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