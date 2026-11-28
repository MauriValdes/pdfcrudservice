import pytest
import datetime
import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError
import pymupdf

from src.main import app
from src.api.docs_api import get_document_service
from src.api.pdf_validation import PdfValidationExecutor
from src.schemas.document import DocumentResponse
from src.database.mongodb import db_connection
from src.config.config import settings
from src.repository.document_repository import DocumentRepository, InvalidObjectIdError


def make_valid_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "PDF CRUD API test")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


PDF_BYTES = make_valid_pdf()


@pytest.fixture(autouse=True)
def pdf_validation_executor():
    """Provide the managed executor without starting the MongoDB lifespan."""
    executor = PdfValidationExecutor()
    app.state.pdf_validation_executor = executor
    yield executor
    executor.shutdown(wait=True, cancel_futures=True)
    app.state.pdf_validation_executor = None


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


def test_upload_pdf_returns_202_receipt_without_sync_extraction(client, mock_service):
    """La API devuelve el ticket de recepcion, no el resultado de MS2."""
    now = datetime.datetime.now(datetime.timezone.utc)
    mock_service.receive_pdf.return_value = (
        DocumentResponse(
            id="test-id", filename="test.pdf", checksum="internal-checksum",
            size_bytes=len(PDF_BYTES), status="pending", content="private",
            page_count=5, error_message=None, created_at=now, updated_at=now,
        ),
        "accepted",
    )
    response = client.post(
        "/documents/", files={"file": ("test.pdf", PDF_BYTES, "application/pdf")}
    )
    assert response.status_code == 202
    assert response.json() == {
        "document_id": "test-id", "status": "pending",
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "status_url": "/documents/test-id",
    }
    assert response.headers["location"] == response.json()["status_url"]
    assert set(response.json()) == {"document_id", "status", "created_at", "status_url"}
    mock_service.receive_pdf.assert_awaited_once_with(
        file_bytes=PDF_BYTES, filename="test.pdf"
    )
    mock_service.submit_pdf.assert_not_awaited()
    mock_service.process_pdf_extraction.assert_not_awaited()
    mock_service.call_extraction_service.assert_not_awaited()
    mock_service.get_document_by_id.assert_not_awaited()


def test_upload_pdf_existing_document_returns_200_original_ticket(client, mock_service):
    """Los duplicados conservan identidad y fecha original, sin re-procesar."""
    created_at = datetime.datetime(2021, 4, 5, 6, 7, 8, tzinfo=datetime.timezone.utc)
    mock_service.receive_pdf.return_value = (
        DocumentResponse(
            id="existing-id", filename="original-name.pdf",
            checksum="internal-checksum", size_bytes=len(PDF_BYTES),
            status="completed", content="private", page_count=3,
            error_message=None, created_at=created_at, updated_at=created_at,
        ),
        "existing",
    )
    response = client.post(
        "/documents/", files={"file": ("new-name.pdf", PDF_BYTES, "application/pdf")}
    )
    assert response.status_code == 200
    assert response.json() == {
        "document_id": "existing-id", "status": "completed",
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "status_url": "/documents/existing-id",
    }
    assert response.headers["location"] == "/documents/existing-id"
    mock_service.receive_pdf.assert_awaited_once()
    mock_service.submit_pdf.assert_not_awaited()
    mock_service.process_pdf_extraction.assert_not_awaited()


def test_unexpected_reception_error_is_generic_500_and_logged(client, mock_service, caplog):
    mock_service.receive_pdf.side_effect = RuntimeError("private storage detail")
    response = client.post(
        "/documents/", files={"file": ("test.pdf", PDF_BYTES, "application/pdf")}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Ocurri\u00f3 un error inesperado durante la recepci\u00f3n del documento"
    )
    assert "private storage detail" not in response.text
    assert "private storage detail" in caplog.text


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
    files = {"file": ("test.pdf", PDF_BYTES, "application/pdf")}

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

    files = {"file": ("test.pdf", PDF_BYTES, "application/pdf")}
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


def test_upload_pdf_too_large_still_returns_413(client, mock_service):
    """El tamano excedido mantiene 413 antes de invocar receive_pdf()."""
    oversized = b"%PDF-1.4 " + b"x" * (
        settings.max_file_size_mb * 1024 * 1024 + 1
    )
    response = client.post(
        "/documents/", files={"file": ("big.pdf", oversized, "application/pdf")}
    )
    assert response.status_code == 413
    mock_service.receive_pdf.assert_not_awaited()


# =============================================================================
# Pruebas de integracion real contra MongoDB
# =============================================================================

@pytest.mark.integration
async def test_concurrent_upload_same_pdf_real_mongodb():
    """Prueba de concurrencia real contra MongoDB.

    Dos solicitudes HTTP simultáneas con el mismo PDF compiten por insertar
    el mismo checksum. Una debe devolver 202 y la otra 200 para el mismo
    documento. La persistencia del PDF en GridFS es real y no se ejecuta MS2.

    Los errores de conexión, configuración, autenticación o permisos hacen
    fallar explícitamente la prueba.
    """
    import hashlib
    import uuid

    from httpx import ASGITransport, AsyncClient
    from unittest.mock import AsyncMock, patch

    from src.database.mongodb import MongoDB
    from src.services.document_service import DocumentService
    from src.config.config import settings
    from src.repository.gridfs_repository import GridFSRepository

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
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), f"Concurrency test {uuid.uuid4()}")
        pdf_bytes = pdf.tobytes()
        pdf.close()
        filename = "test_concurrent_real.pdf"
        checksum = hashlib.sha256(pdf_bytes).hexdigest()

        extraction_mock = AsyncMock(
            side_effect=AssertionError("MS2 must not be called")
        )
        with patch.object(
            DocumentService,
            "call_extraction_service",
            extraction_mock,
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

        responses_by_status = {result.status_code: result for result in results}
        assert set(responses_by_status) == {200, 202}, (
            f"Esperaba respuestas 200 y 202, obtuvo "
            f"{sorted(responses_by_status)}"
        )
        accepted_response = responses_by_status[202]
        duplicate_response = responses_by_status[200]
        accepted_ticket = accepted_response.json()
        duplicate_ticket = duplicate_response.json()

        assert accepted_ticket["status"] == "pending"
        assert duplicate_ticket["status"] in {"uploading", "pending"}
        assert accepted_ticket["document_id"] == duplicate_ticket["document_id"]
        assert accepted_ticket["created_at"] == duplicate_ticket["created_at"]
        assert all(
            result.headers["location"] == result.json()["status_url"]
            for result in results
        )
        assert all(
            set(ticket) == {"document_id", "status", "created_at", "status_url"}
            for ticket in (accepted_ticket, duplicate_ticket)
        )

        existing_docs = await repository.collection.find(
            {"checksum": checksum}
        ).to_list(length=None)
        assert len(existing_docs) == 1, (
            f"Se esperaba exactamente 1 documento, se encontró {len(existing_docs)}"
        )
        stored_document = existing_docs[0]
        assert stored_document["status"] == "pending"
        assert str(stored_document["_id"]) == accepted_ticket["document_id"]
        assert str(stored_document["_id"]) == duplicate_ticket["document_id"]
        assert stored_document["file_id"] is not None
        stored_pdf = await GridFSRepository(testing_db.db).read(
            stored_document["file_id"]
        )
        assert stored_pdf == pdf_bytes
        extraction_mock.assert_not_awaited()

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
                created_document = await repository.collection.find_one(
                    {"checksum": checksum}
                )
                if created_document and created_document.get("file_id") is not None:
                    await repository._gridfs_repository().delete(
                        created_document["file_id"]
                    )
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
