import pytest
import datetime
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from src.main import app
from src.api.docs_api import get_document_service
from src.schemas.document import DocumentResponse


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
    files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/documents/", files=files)
    assert response.status_code == 201


def test_get_all_documents_returns_list(client, mock_service):
    mock_service.get_all_documents.return_value = []
    response = client.get("/documents")
    assert response.status_code == 200


def test_get_document_by_id_returns_200(client, mock_service):
    now = datetime.datetime.now()
    mock_service.get_document_by_id.return_value = DocumentResponse(
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
    response = client.get("/documents/test-id")
    assert response.status_code == 200


def test_get_document_by_id_returns_404(client, mock_service):
    mock_service.get_document_by_id.return_value = None
    response = client.get("/documents/nonexistent-id")
    assert response.status_code == 404


def test_update_document_partial(client, mock_service):
    now = datetime.datetime.now()
    mock_service.update_document.return_value = DocumentResponse(
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