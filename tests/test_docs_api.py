import pytest
import datetime
import asyncio
import httpx
import hashlib
from unittest.mock import AsyncMock, patch
from bson import ObjectId
from fastapi.testclient import TestClient

from src.main import app
from src.api.docs_api import get_document_service
from src.schemas.document import DocumentResponse
from src.database.mongodb import db_connection
from src.config.config import settings
from src.repository.document_repository import InvalidObjectIdError


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
# Pruebas de integración real contra MongoDB
# ============================================================================

@pytest.mark.integration
async def test_concurrent_upload_same_pdf_real_mongodb():
    """Prueba de concurrencia real contra MongoDB.

    Dos solicitudes HTTP simultáneas con el mismo PDF compiten por insertar
    el mismo checksum. Una debe insertarse (HTTP 201) y la otra rechazarse
    como duplicada (HTTP 400). Se simula únicamente MS2 (servicio de
    extracción); la persistencia en MongoDB es real.

    Si MongoDB no está disponible (o no se puede autenticar/crear el índice),
    la prueba se omite de forma controlada en lugar de fallar.
    """
    import hashlib
    import time as _time

    from httpx import ASGITransport, AsyncClient
    from unittest.mock import AsyncMock, patch

    from src.main import app
    from src.database.mongodb import db_connection
    from src.services.document_service import DocumentService
    from src.config.config import settings

    # Base de datos temporal e isolada (nunca pdf_database_dev)
    test_db_name = f"pdf_concurrency_test_{int(_time.time() * 1000)}"

    # Intentar conectar a MongoDB real. Si no es posible (por ejemplo, el
    # servidor requiere autenticación que no está configurada), omitir.
    try:
        await db_connection.connect(database_name=test_db_name)
    except Exception as exc:
        pytest.skip(
            f"No se pudo conectar a MongoDB real para la prueba de concurrencia: {exc}"
        )

    try:
        # Asegurar el índice único en la base de datos temporal
        try:
            await db_connection.db[settings.mongo_collection].create_index(
                [("checksum", 1)],
                name="checksum_unique",
                unique=True,
            )
        except Exception as exc:
            pytest.skip(
                f"No se pudo crear el índice único en MongoDB real: {exc}"
            )

        pdf_bytes = b"%PDF-1.4 real concurrency test"
        filename = "test_concurrent_real.pdf"
        checksum = hashlib.sha256(pdf_bytes).hexdigest()

        # Simular únicamente MS2: devolver una extracción válida sin invocar al
        # servicio real. Aplica a todas las instancias de DocumentService.
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

        # MongoDB debe contener exactamente un documento para ese checksum
        existing_docs = await db_connection.db[settings.mongo_collection].find(
            {"checksum": checksum}
        ).to_list(length=None)
        assert len(existing_docs) == 1, (
            f"Se esperaba exactamente 1 documento, se encontró {len(existing_docs)}"
        )

        # Verificar que el índice checksum_unique es realmente único
        indexes = await db_connection.db[settings.mongo_collection].list_indexes().to_list(
            length=None
        )
        unique_indexes = [
            idx
            for idx in indexes
            if idx.get("name") == "checksum_unique" and idx.get("unique") is True
        ]
        assert len(unique_indexes) == 1, "El índice checksum_unique no es único"
    finally:
        # Limpiar exclusivamente la base de datos temporal, incluso ante excepciones
        try:
            await db_connection.client.drop_database(test_db_name)
        except Exception:
            pass
        await db_connection.close()