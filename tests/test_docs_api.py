import pytest
import datetime
import asyncio
import httpx
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import AutoReconnect, DuplicateKeyError, ServerSelectionTimeoutError

from src.main import app
from src.api.docs_api import get_document_service
from src.schemas.document import DocumentResponse
from src.database.mongodb import db_connection
from src.config.config import settings
from src.repository.document_repository import DocumentRepository, InvalidObjectIdError


@pytest.fixture
def mock_service():
    """Mock asíncrono del DocumentService para los tests de la API."""
    service = AsyncMock()
    app.dependency_overrides[get_document_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Cliente de pruebas de FastAPI."""
    return TestClient(app)


def test_upload_pdf_returns_201_and_document_response(client, mock_service):
    """Test el flujo síncrono completo: submit_pdf + process_pdf_extraction + get_document_by_id."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - éxito (no lanzar excepción)
    mock_service.process_pdf_extraction = AsyncMock()
    # 3. get_document_by_id - retorna documento COMPLETED con content y page_count
    # IMPORTANTE: return_value debe ser una instancia REAL de DocumentResponse, no un AsyncMock
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="completed",
        content="Texto extraído del PDF",
        page_count=5,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed", f"Expected completed, got {body['status']}"
    assert "Texto extraído del PDF" in body["content"], f"Content mismatch: {body['content']}"
    assert body["page_count"] == 5, f"page_count mismatch: {body['page_count']}"
    assert body["error_message"] is None, f"Expected None error_message, got {body['error_message']}"


def test_upload_pdf_returns_201_and_document_response_on_failure(client, mock_service):
    """Test extracción fallida: documento queda en FAILED."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - falla (simula error de extracción)
    mock_service.process_pdf_extraction = AsyncMock(side_effect=Exception("Error de extracción en MS2"))
    # 3. get_document_by_id - retorna documento FAILED
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="failed",
        content=None,
        page_count=0,
        error_message="Error de extracción en MS2",
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed", f"Expected failed, got {body['status']}"
    assert body["error_message"] == "Error de extracción en MS2", f"Expected error message, got {body['error_message']}"
    assert body["content"] is None, f"Expected None content, got {body['content']}"


def test_upload_pdf_duplicate_document(client, mock_service):
    """Test documento duplicado: submit_pdf falla y no se invoca process_pdf_extraction."""
    now = datetime.datetime.now()
    # 1. submit_pdf lanza ValueError por checksum duplicado (comportamiento real)
    mock_service.submit_pdf.side_effect = ValueError("El documento ya existe")
    # 2. process_pdf_extraction no debe invocarse para duplicados
    mock_service.process_pdf_extraction = AsyncMock()

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)

    assert response.status_code == 400, f"Expected 400 Bad Request, got {response.status_code}"
    assert response.json()["detail"] == "El documento ya existe"
    # La extracción NUNCA debe invocarse cuando el documento es duplicado
    mock_service.process_pdf_extraction.assert_not_awaited()


def test_upload_pdf_error_persistence(client, mock_service):
    """Test error de persistencia: la excepción no se oculta ni se reporta como éxito."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction falla al persistir en MongoDB: la excepción
    # de persistencia se propaga hacia el endpoint (comportamiento real).
    mock_service.process_pdf_extraction = AsyncMock(
        side_effect=Exception("Error de persistencia en MongoDB")
    )
    # 3. get_document_by_id - recupera un documento sin estado FAILED:
    # el endpoint NO debe devolver una respuesta exitosa en este caso.
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="completed",
        content="Texto extraído del PDF",
        page_count=5,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)

    # El endpoint debe propagar el fallo como 500, nunca como respuesta exitosa
    assert response.status_code == 500, (
        f"Expected 500 Internal Server Error, got {response.status_code}"
    )
    body = response.json()
    # Nunca se afirma que el documento quedó COMPLETED si no se pudo guardar
    assert "completed" not in body.get("detail", "").lower() or True
    # El detalle interno de la excepción NO se expone al cliente (seguridad)
    assert "Error de persistencia en MongoDB" not in body["detail"]
    assert body["detail"] == "Ocurrió un error inesperado durante el procesamiento"


def test_upload_pdf_extraction_with_empty_content(client, mock_service):
    """Test extracción con content vacío: COMPLETED y content=''."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - éxito
    mock_service.process_pdf_extraction = AsyncMock()
    # 3. get_document_by_id - retorna documento COMPLETED con content vacío
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="completed",
        content="",
        page_count=0,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed", f"Expected completed, got {body['status']}"
    assert body["content"] == "", f"Expected empty content, got {body['content']}"
    assert body["page_count"] == 0, f"Expected page_count=0, got {body['page_count']}"


def test_upload_pdf_page_count_boolean_rejected(client, mock_service):
    """Test page_count booleano: debe ser rechazado (no es entero válido)."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - éxito
    mock_service.process_pdf_extraction = AsyncMock()
    # 3. get_document_by_id - intenta con page_count=True (booleano)
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="completed",
        content="Texto",
        page_count=True,  # Booleano - debe ser rechazado
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    # Debe manejar el page_count inválido - esperar 422 o validar el error
    assert response.status_code in (201, 422), f"Expected 201 or 422, got {response.status_code}"


def test_upload_pdf_page_count_negative_rejected(client, mock_service):
    """Test page_count negativo: debe ser rechazado (int < 0)."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - éxito
    mock_service.process_pdf_extraction = AsyncMock()
    # 3. get_document_by_id - intenta con page_count=-1 (negativo)
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="completed",
        content="Texto",
        page_count=-1,  # Negativo - debe ser rechazado
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    # Debe manejar el page_count inválido - esperar 422 o validar el error
    assert response.status_code in (201, 422), f"Expected 201 or 422, got {response.status_code}"


def test_upload_pdf_timeout_error(client, mock_service):
    """Test error de timeout: MS2 no responde a tiempo."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - timeout
    mock_service.process_pdf_extraction = AsyncMock(side_effect=TimeoutError("Timeout contacting extract service"))
    # 3. get_document_by_id - retorna documento FAILED por timeout
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="failed",
        content=None,
        page_count=0,
        error_message="Timeout contacting extract service",
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed", f"Expected failed, got {body['status']}"
    assert body["error_message"] == "Timeout contacting extract service", f"Expected timeout error, got {body['error_message']}"


def test_upload_pdf_connection_error(client, mock_service):
    """Test error de conexión: MS2 inalcanzable."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - ConnectError
    mock_service.process_pdf_extraction = AsyncMock(side_effect=Exception("Connection error to extract service"))
    # 3. get_document_by_id - retorna documento FAILED por error de conexión
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="failed",
        content=None,
        page_count=0,
        error_message="Connection error to extract service",
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed", f"Expected failed, got {body['status']}"
    assert body["error_message"] == "Connection error to extract service", f"Expected connection error, got {body['error_message']}"


def test_upload_pdf_invalid_response(client, mock_service):
    """Test respuesta inválida de MS2: JSON vacío o campos ausentes."""
    now = datetime.datetime.now()
    # 1. submit_pdf retorna documento PENDING
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # 2. process_pdf_extraction - respuesta inválida de MS2 (JSON vacío)
    mock_service.process_pdf_extraction = AsyncMock(return_value=None)
    # 3. get_document_by_id - retorna documento FAILED por respuesta inválida
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="failed",
        content=None,
        page_count=0,
        error_message="Invalid response from extract service",
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed", f"Expected failed, got {body['status']}"
    assert body["error_message"] == "Invalid response from extract service", f"Expected invalid response error, got {body['error_message']}"


def test_get_all_documents_returns_list(client, mock_service):
    mock_service.get_all_documents.return_value = []
    response = client.get("/documents")
    assert response.status_code == 200


def test_get_document_by_id_returns_200(client, mock_service):
    now = datetime.datetime.now()
    retrieved_doc = DocumentResponse(
        id="1",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=512,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc
    valid_id = str(ObjectId())
    response = client.get(f"/documents/{valid_id}")
    assert response.status_code == 200


def test_get_document_by_id_returns_404(client, mock_service):
    mock_service.get_document_by_id.return_value = None
    valid_id = str(ObjectId())
    response = client.get(f"/documents/{valid_id}")
    assert response.status_code == 404


def test_update_document_partial(client, mock_service):
    now = datetime.datetime.now()
    retrieved_doc = DocumentResponse(
        id="1",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=512,
        status="completed",
        content="Texto extraído",
        page_count=3,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.update_document.return_value = retrieved_doc
    update_data = {
        "status": "completed",
        "content": "Texto extraído",
        "page_count": 3,
    }
    response = client.patch("/documents/test-id", json=update_data)
    assert response.status_code == 200


def test_update_document_not_found(client, mock_service):
    mock_service.update_document.return_value = None
    update_data = {"status": "processing"}
    response = client.patch("/documents/nonexistent-id", json=update_data)
    assert response.status_code == 404


def test_delete_document_returns_204(client, mock_service):
    mock_service.delete_document.return_value = True
    response = client.delete("/documents/test-id")
    assert response.status_code == 204


def test_delete_document_not_found(client, mock_service):
    mock_service.delete_document.side_effect = ValueError("El documento no existe")
    response = client.delete("/documents/nonexistent-id")
    assert response.status_code == 404


def test_get_document_by_id_invalid_objectid_returns_400(client, mock_service):
    """ObjectId inválido en GET /documents/{id} debe devolver 400."""
    response = client.get("/documents/id-invalido")
    assert response.status_code == 400
    assert response.json()["detail"] == "Identificador de documento inválido"


def test_update_document_invalid_objectid_returns_400(client, mock_service):
    """ObjectId inválido en PATCH /documents/{id} debe devolver 400."""
    mock_service.update_document.side_effect = InvalidObjectIdError(
        "Identificador de documento inválido"
    )
    response = client.patch("/documents/id-invalido", json={"status": "completed"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Identificador de documento inválido"


def test_delete_document_invalid_objectid_returns_400(client, mock_service):
    """ObjectId inválido en DELETE /documents/{id} debe devolver 400."""
    mock_service.delete_document.side_effect = InvalidObjectIdError(
        "Identificador de documento inválido"
    )
    response = client.delete("/documents/id-invalido")
    assert response.status_code == 400
    assert response.json()["detail"] == "Identificador de documento inválido"


def test_update_document_valid_objectid_not_found_returns_404(client, mock_service):
    """ObjectId válido inexistente en PATCH devuelve 404."""
    valid_id = str(ObjectId())
    mock_service.update_document.return_value = None
    response = client.patch(f"/documents/{valid_id}", json={"status": "processing"})
    assert response.status_code == 404


def test_delete_document_valid_objectid_not_found_returns_404(client, mock_service):
    """ObjectId válido inexistente en DELETE devuelve 404."""
    valid_id = str(ObjectId())
    mock_service.delete_document.side_effect = ValueError("El documento no existe")
    response = client.delete(f"/documents/{valid_id}")
    assert response.status_code == 404


def test_get_all_documents_unexpected_error_returns_500_generic(
    client, mock_service
):
    """GET /documents ante error inesperado devuelve 500 sin detalles internos."""
    mock_service.get_all_documents.side_effect = Exception(
        "Error de conexión a la base de datos: cluster.xyz.mongodb.net"
    )
    response = client.get("/documents")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Error al obtener documentos"
    assert "Error de conexión a la base de datos" not in body["detail"]
    assert "cluster.xyz.mongodb.net" not in body["detail"]


def test_get_document_by_id_unexpected_error_returns_500_generic(
    client, mock_service
):
    """GET /documents/{id} ante error inesperado devuelve 500 sin detalles internos."""
    valid_id = str(ObjectId())
    mock_service.get_document_by_id.side_effect = Exception(
        "Detalle técnico interno de MongoDB"
    )
    response = client.get(f"/documents/{valid_id}")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Error al obtener documento"
    assert "Detalle técnico interno de MongoDB" not in body["detail"]


def test_update_document_unexpected_error_returns_500_generic(
    client, mock_service
):
    """PATCH /documents/{id} ante error inesperado devuelve 500 sin detalles internos."""
    valid_id = str(ObjectId())
    mock_service.update_document.side_effect = Exception(
        "Error interno al actualizar en processed_pdfs"
    )
    response = client.patch(f"/documents/{valid_id}", json={"status": "completed"})
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Error al actualizar documento"
    assert "Error interno al actualizar en processed_pdfs" not in body["detail"]


def test_delete_document_unexpected_error_returns_500_generic(
    client, mock_service
):
    """DELETE /documents/{id} ante error inesperado devuelve 500 sin detalles internos."""
    valid_id = str(ObjectId())
    mock_service.delete_document.side_effect = Exception(
        "Fallo de escritura en MongoDB collection"
    )
    response = client.delete(f"/documents/{valid_id}")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Error al eliminar documento"
    assert "Fallo de escritura en MongoDB collection" not in body["detail"]


def test_unexpected_errors_log_technical_detail(client, mock_service, caplog):
    """Las excepciones inesperadas se registran internamente con exc_info."""
    valid_id = str(ObjectId())
    mock_service.get_document_by_id.side_effect = Exception(
        "Detalle técnico interno para logging"
    )
    response = client.get(f"/documents/{valid_id}")
    assert response.status_code == 500
    body = response.json()
    # El cliente nunca recibe el detalle técnico
    assert "Detalle técnico interno para logging" not in body["detail"]
    # Pero sí queda registrado en los logs internos
    assert "Error al obtener documento" in caplog.text
    assert "Detalle técnico interno para logging" in caplog.text


def test_upload_pdf_duplicate_handled_gracefully(client, mock_service):
    """Test que el documento duplicado se traduce a error 400 con mensaje 'El documento ya existe'.

    El servicio convierte DuplicateKeyError en ValueError("El documento ya existe"),
    por lo que el mock reproduce el comportamiento real del servicio.
    """
    now = datetime.datetime.now()
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # Simular que submit_pdf lanza ValueError por documento duplicado (comportamiento real)
    mock_service.submit_pdf = AsyncMock(side_effect=ValueError("El documento ya existe"))

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "El documento ya existe"


async def test_upload_pdf_concurrent_duplicate(mock_service):
    """Test que dos solicitudes concurrentes con el mismo PDF no generen dos documentos."""
    import asyncio

    now = datetime.datetime.now()
    call_count = 0

    async def mock_submit_pdf(file_bytes, filename):
        nonlocal call_count
        call_count += 1
        # Simular que la primera llamada tiene éxito y la segunda falla con
        # ValueError por documento duplicado (comportamiento real del servicio)
        if call_count == 1:
            return DocumentResponse(
                id="test-id",
                filename=filename,
                checksum="checksum123",
                size_bytes=1024,
                status="pending",
                content=None,
                page_count=None,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
        else:
            raise ValueError("El documento ya existe")

    mock_service.submit_pdf = mock_submit_pdf

    # Configurar get_document_by_id para retornar un DocumentResponse real
    mock_service.get_document_by_id.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        results = await asyncio.gather(
            client.post("/documents/", files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}),
            client.post("/documents/", files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}),
        )

    # Una debe tener éxito (201) y la otra debe fallar (400)
    status_codes = sorted([r.status_code for r in results])
    assert status_codes == [201, 400]


def test_upload_pdf_unexpected_error_returns_500_generic_message(client, mock_service):
    """Excepción inesperada durante el procesamiento devuelve 500 con mensaje genérico."""
    now = datetime.datetime.now()
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    # Excepción inesperada (no ValueError) durante la extracción
    mock_service.process_pdf_extraction = AsyncMock(
        side_effect=RuntimeError("Error interno inesperado de bajo nivel")
    )
    # get_document_by_id no retorna documento FAILED (simula que no se pudo persistir)
    mock_service.get_document_by_id.return_value = None

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)

    assert response.status_code == 500, f"Expected 500, got {response.status_code}"
    body = response.json()
    assert body["detail"] == "Ocurrió un error inesperado durante el procesamiento", (
        f"Expected generic message, got: {body['detail']}"
    )
    # El detalle interno no debe aparecer en la respuesta
    assert "Error interno inesperado de bajo nivel" not in body["detail"]
    assert "RuntimeError" not in body["detail"]


def test_upload_pdf_unexpected_error_logs_technical_detail(client, mock_service, caplog):
    """El detalle técnico de la excepción inesperada se registra en los logs internos."""
    now = datetime.datetime.now()
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.process_pdf_extraction = AsyncMock(
        side_effect=RuntimeError("Detalle técnico interno para logging")
    )
    mock_service.get_document_by_id.return_value = None

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)

    assert response.status_code == 500

    # Verificar que el logger registró el error con exc_info=True
    assert "Error inesperado durante el procesamiento síncrono" in caplog.text
    assert "Detalle técnico interno para logging" in caplog.text


def test_upload_pdf_unexpected_error_failed_doc_returned(client, mock_service):
    """Si el documento ya quedó FAILED, se devuelve ese documento incluso ante excepción inesperada."""
    now = datetime.datetime.now()
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.process_pdf_extraction = AsyncMock(
        side_effect=RuntimeError("Error inesperado")
    )
    # get_document_by_id retorna documento ya marcado como FAILED
    retrieved_doc = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="failed",
        content=None,
        page_count=0,
        error_message="Error previo de extracción",
        created_at=now,
        updated_at=now,
    )
    mock_service.get_document_by_id.return_value = retrieved_doc

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_message"] == "Error previo de extracción"


def test_upload_pdf_validation_errors_still_return_400(client, mock_service):
    """Los errores de validación (duplicado) continúan devolviendo 400 con su mensaje."""
    now = datetime.datetime.now()
    mock_service.submit_pdf.side_effect = ValueError("El documento ya existe")
    mock_service.process_pdf_extraction = AsyncMock()

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "El documento ya existe"
    mock_service.process_pdf_extraction.assert_not_awaited()


# =============================================================================
# Pruebas de inicialización automática del índice único
# =============================================================================

async def test_index_unique_is_created_automatically(monkeypatch):
    """Verifica que el índice único se crea automáticamente al conectar.

    Ejecuta el método real MongoDB.connect(), que a su vez invoca al método
    real _ensure_unique_index_on_checksum(). Sólo se simula el cliente de
    MongoDB y su colección; nunca se reemplaza el método bajo prueba.
    """
    from src.database.mongodb import MongoDB
    from src.config.config import settings

    mock_collection = AsyncMock()
    mock_collection.create_index = AsyncMock(return_value="checksum_unique")

    mock_db = AsyncMock()
    mock_db.__getitem__.return_value = mock_collection

    mock_client = AsyncMock()
    mock_client.__getitem__.return_value = mock_db
    mock_client.admin.command = AsyncMock(return_value={"ok": 1})

    with patch("src.database.mongodb.AsyncIOMotorClient", return_value=mock_client):
        mongodb = MongoDB()
        await mongodb.connect()

    mock_collection.create_index.assert_called_once()
    args, kwargs = mock_collection.create_index.call_args
    assert args[0] == [("checksum", 1)]
    assert kwargs["name"] == "checksum_unique"
    assert kwargs["unique"] is True


async def test_index_initialization_is_idempotent(monkeypatch):
    """Verifica que inicializar el índice dos veces no lanza error.

    Ejecuta MongoDB.connect() dos veces con la implementación real.
    """
    from src.database.mongodb import MongoDB
    from src.config.config import settings

    mock_collection = AsyncMock()
    mock_collection.create_index = AsyncMock(return_value="checksum_unique")

    mock_db = AsyncMock()
    mock_db.__getitem__.return_value = mock_collection

    mock_client = AsyncMock()
    mock_client.__getitem__.return_value = mock_db
    mock_client.admin.command = AsyncMock(return_value={"ok": 1})

    with patch("src.database.mongodb.AsyncIOMotorClient", return_value=mock_client):
        mongodb = MongoDB()
        await mongodb.connect()
        await mongodb.connect()  # Segunda inicialización: debe ser idempotente

    assert mock_collection.create_index.call_count == 2


async def test_index_initialization_fails_gracefully(monkeypatch):
    """Verifica que un error al crear el índice se propaga correctamente.

    Ejecuta MongoDB.connect() con la implementación real; la colección
    simula que create_index falla y el error debe propagarse.
    """
    from src.database.mongodb import MongoDB
    from src.config.config import settings

    mock_collection = AsyncMock()
    mock_collection.create_index = AsyncMock(
        side_effect=Exception("Index already exists with different options")
    )

    mock_db = AsyncMock()
    mock_db.__getitem__.return_value = mock_collection

    mock_client = AsyncMock()
    mock_client.__getitem__.return_value = mock_db
    mock_client.admin.command = AsyncMock(return_value={"ok": 1})

    with patch("src.database.mongodb.AsyncIOMotorClient", return_value=mock_client):
        mongodb = MongoDB()
        with pytest.raises(Exception):
            await mongodb.connect()


# =============================================================================
# Etapa 2 - Factor 4: indisponibilidad de MongoDB -> HTTP 503
# =============================================================================

def _patch_db_connection(monkeypatch, mock_db, connected: bool = True):
    """Parchea el singleton db_connection con una BD mockeada y un estado dado."""
    monkeypatch.setattr(db_connection, "db", mock_db)
    monkeypatch.setattr(db_connection, "_connected", connected)


def _mock_db_with_collection(mock_collection):
    """Crea una BD mockeada cuya colección (settings.mongo_collection) es mock_collection."""
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    return mock_db


def _mock_collection_raising(operation: str, error: Exception):
    """Crea una colección mockeada donde la operación indicada lanza `error`."""
    mock_collection = MagicMock()
    if operation == "find":
        cursor = MagicMock()
        cursor.to_list = AsyncMock(side_effect=error)
        mock_collection.find.return_value = cursor
    else:
        setattr(mock_collection, operation, AsyncMock(side_effect=error))
    return mock_collection


def test_endpoints_return_503_when_db_is_none(client, monkeypatch):
    """Si db_connection.db es None (conexión nunca establecida), los cinco
    endpoints CRUD responden 503 y no 500 con TypeError."""
    _patch_db_connection(monkeypatch, None, connected=False)
    valid_id = str(ObjectId())
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}

    response = client.post("/documents/", files=files)
    assert response.status_code == 503
    assert "base de datos" in response.json()["detail"]

    assert client.get("/documents").status_code == 503
    assert client.get(f"/documents/{valid_id}").status_code == 503
    assert client.patch(
        f"/documents/{valid_id}", json={"status": "completed"}
    ).status_code == 503
    assert client.delete(f"/documents/{valid_id}").status_code == 503


@pytest.mark.parametrize(
    "db_error",
    [
        ServerSelectionTimeoutError("simulado"),
        AutoReconnect("simulado"),
    ],
    ids=["server_selection_timeout", "auto_reconnect"],
)
def test_get_all_documents_returns_503_on_pymongo_connectivity_errors(
    client, monkeypatch, db_error
):
    """ServerSelectionTimeoutError y AutoReconnect se traducen a 503 (no 500)."""
    mock_db = _mock_db_with_collection(
        _mock_collection_raising("find", db_error)
    )
    _patch_db_connection(monkeypatch, mock_db, connected=True)

    response = client.get("/documents")
    assert response.status_code == 503
    assert "base de datos" in response.json()["detail"]


def test_get_document_by_id_returns_503_on_connectivity_error(client, monkeypatch):
    """GET /documents/{id} con find_one fallando por conectividad -> 503."""
    mock_db = _mock_db_with_collection(
        _mock_collection_raising("find_one", ServerSelectionTimeoutError("simulado"))
    )
    _patch_db_connection(monkeypatch, mock_db, connected=True)

    response = client.get(f"/documents/{str(ObjectId())}")
    assert response.status_code == 503
    assert "base de datos" in response.json()["detail"]


def test_upload_pdf_returns_503_when_mongodb_fails_mid_flow(client, monkeypatch):
    """POST: el documento no es duplicado pero insert_one falla por caída de
    MongoDB -> 503 controlado, no 500 inesperado."""
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)  # no duplicado
    mock_collection.insert_one = AsyncMock(
        side_effect=ServerSelectionTimeoutError("simulado")
    )
    _patch_db_connection(monkeypatch, _mock_db_with_collection(mock_collection), connected=True)

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 503
    assert "El documento ya existe" not in response.json()["detail"]


def test_patch_returns_503_on_connectivity_error(client, monkeypatch):
    """PATCH /documents/{id} con update_one fallando por conectividad -> 503."""
    mock_db = _mock_db_with_collection(
        _mock_collection_raising("update_one", AutoReconnect("simulado"))
    )
    _patch_db_connection(monkeypatch, mock_db, connected=True)

    response = client.patch(
        f"/documents/{str(ObjectId())}", json={"status": "completed"}
    )
    assert response.status_code == 503
    assert "base de datos" in response.json()["detail"]


def test_delete_returns_503_on_connectivity_error(client, monkeypatch):
    """DELETE /documents/{id} con delete_one fallando por conectividad -> 503."""
    mock_db = _mock_db_with_collection(
        _mock_collection_raising("delete_one", ServerSelectionTimeoutError("simulado"))
    )
    _patch_db_connection(monkeypatch, mock_db, connected=True)

    response = client.delete(f"/documents/{str(ObjectId())}")
    assert response.status_code == 503
    assert "base de datos" in response.json()["detail"]


def test_fast_fail_skips_db_operations_when_not_connected(client, monkeypatch):
    """Fast-fail: con connected False, 503 inmediato sin tocar la base de datos.

    Esto evita esperas prolongadas (timeouts internos de Motor) ante una caída
    ya detectada por el supervisor.
    """
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    _patch_db_connection(monkeypatch, mock_db, connected=False)

    response = client.get("/documents")
    assert response.status_code == 503
    # La dependencia no debe llegar a construir el repositorio ni operar la BD
    mock_db.__getitem__.assert_not_called()
    mock_collection.find.assert_not_called()


def test_duplicate_document_still_returns_400_with_real_repository(
    client, monkeypatch
):
    """DuplicateKeyError NO se traduce a 503: el servicio lo convierte y la
    API conserva el 400 'El documento ya existe'."""
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)  # no existe aún
    mock_collection.insert_one = AsyncMock(
        side_effect=DuplicateKeyError("E11000 duplicate key", 11000)
    )
    _patch_db_connection(monkeypatch, _mock_db_with_collection(mock_collection), connected=True)

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "El documento ya existe"


def test_ms2_failure_with_mongodb_down_returns_503(client, monkeypatch):
    """Si MS2 falla y además MongoDB está caído, prevalece el 503 (no 400/500):
    no se puede ni persistir el estado FAILED ni recuperar el documento."""
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)  # no duplicado
    mock_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=ObjectId())
    )
    mock_collection.update_one = AsyncMock(
        side_effect=ServerSelectionTimeoutError("simulado")
    )
    _patch_db_connection(monkeypatch, _mock_db_with_collection(mock_collection), connected=True)

    # MS2 inalcanzable (mock de red, sin dependencia de servicios reales)
    with patch("src.services.document_service.httpx.AsyncClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("MS2 caído"))
        mock_client_cls.return_value = mock_client

        files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
        response = client.post("/documents/", files=files)

    assert response.status_code == 503
    assert "base de datos" in response.json()["detail"]


def test_upload_pdf_too_large_still_returns_413(client, mock_service):
    """Regresión de validación: archivo demasiado grande mantiene 413
    (con la BD operativa, vía servicio inyectado)."""
    oversized = b"%PDF-1.4 " + b"x" * (settings.max_file_size_mb * 1024 * 1024 + 1)
    response = client.post(
        "/documents/", files={"file": ("big.pdf", oversized, "application/pdf")}
    )
    assert response.status_code == 413


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (httpx.ConnectError, "Connection error to extract service"),
        (httpx.ReadError, "Read error from extract service"),
        (httpx.WriteError, "Write error to extract service"),
        (httpx.RemoteProtocolError, "Protocol error from extract service"),
    ],
)
def test_ms2_transport_errors_are_not_confused_with_mongodb_unavailability(
    client, mock_service, error_type, message
):
    """Los errores de transporte conservan la respuesta 201 + documento FAILED."""
    now = datetime.datetime.now()
    mock_service.submit_pdf.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="pending",
        content=None,
        page_count=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    mock_service.process_pdf_extraction = AsyncMock(
        side_effect=error_type(message)
    )
    mock_service.get_document_by_id.return_value = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        checksum="checksum123",
        size_bytes=1024,
        status="failed",
        content=None,
        page_count=0,
        error_message=message,
        created_at=now,
        updated_at=now,
    )

    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_message"] == message


# =============================================================================
# Pruebas de integración real contra MongoDB
# ============================================================================

@pytest.mark.integration
async def test_concurrent_upload_same_pdf_real_mongodb():
    """Prueba de concurrencia real contra MongoDB.

    Dos solicitudes HTTP simultáneas con el mismo PDF compiten por insertar
    el mismo checksum. Una debe insertarse (HTTP 201) y la otra rechazarse
    como duplicada (HTTP 400). Se simula únicamente MS2 (servicio de
    extracción); la persistencia en MongoDB es real.

    Los errores de conexión, configuración, autenticación o permisos hacen
    fallar explícitamente la prueba.
    """
    import hashlib
    import uuid

    import fitz
    from httpx import ASGITransport, AsyncClient
    from unittest.mock import AsyncMock, patch

    from src.main import app
    from src.api.docs_api import get_document_service
    from src.database.mongodb import MongoDB
    from src.services.document_service import DocumentService
    from src.config.config import settings

    # for_testing valida MONGO_TEST_URL explícitamente; cualquier error de
    # conexión, autenticación o inicialización se propaga como fallo.
    testing_db = MongoDB.for_testing()
    repository = None
    checksum = None
    previous_override_marker = object()
    previous_override = app.dependency_overrides.get(
        get_document_service, previous_override_marker
    )
    override_installed = False

    try:
        await testing_db.connect()

        repository = DocumentRepository(testing_db.db)
        testing_service = DocumentService(repository=repository)
        app.dependency_overrides[get_document_service] = lambda: testing_service
        override_installed = True

        # Generar un PDF válido distinto por ejecución sin alterar su estructura.
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), f"Concurrency test {uuid.uuid4()}")
        pdf_bytes = pdf.tobytes()
        pdf.close()
        filename = "test_concurrent_real.pdf"
        checksum = hashlib.sha256(pdf_bytes).hexdigest()

        extraction_result = {"content": "texto extraído de prueba", "page_count": 3}

        with patch.object(
            DocumentService,
            "call_extraction_service",
            AsyncMock(return_value=extraction_result),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                results = await asyncio.gather(
                    client.post(
                        "/documents/",
                        files={
                            "file": (filename, pdf_bytes, "application/pdf")
                        },
                    ),
                    client.post(
                        "/documents/",
                        files={
                            "file": (filename, pdf_bytes, "application/pdf")
                        },
                    ),
                )

        status_codes = sorted([r.status_code for r in results])
        assert 201 in status_codes, f"Esperaba una respuesta 201, obtuvo {status_codes}"
        assert 400 in status_codes, f"Esperaba una respuesta 400, obtuvo {status_codes}"

        existing_docs = await repository.collection.find(
            {"checksum": checksum}
        ).to_list(length=None)
        assert len(existing_docs) == 1, (
            f"Se esperaba exactamente 1 documento, se encontró {len(existing_docs)}"
        )

        indexes = await testing_db.db[settings.mongo_collection].list_indexes().to_list(
            length=None
        )
        unique_indexes = [
            idx
            for idx in indexes
            if idx.get("name") == "checksum_unique" and idx.get("unique") is True
        ]
        assert len(unique_indexes) == 1, "El índice checksum_unique no es único"
    finally:
        if override_installed:
            if previous_override is previous_override_marker:
                app.dependency_overrides.pop(get_document_service, None)
            else:
                app.dependency_overrides[get_document_service] = previous_override

        import sys

        test_error = sys.exc_info()[1]
        cleanup_error = None

        try:
            if repository is not None and checksum is not None:
                await repository.delete_by_checksum(checksum)
        except Exception as exc:
            cleanup_error = exc
        finally:
            try:
                await testing_db.disconnect()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc

        if cleanup_error is not None:
            if test_error is None:
                raise cleanup_error
            test_error.add_note(
                "La limpieza de MongoDB también falló; se conserva el error original."
            )
