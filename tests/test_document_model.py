"""Tests unitarios del modelo de persistencia de documentos."""
from datetime import datetime, timezone

from bson import ObjectId

from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentResponse


def build_document(**overrides):
    """Construye un DocumentModel válido con datos mínimos y overrides opcionales."""
    data = {
        "filename": "manual.pdf",
        "checksum": "abc123",
        "size_bytes": 1024,
    }
    data.update(overrides)
    return DocumentModel(**data)


def test_document_status_admits_uploading():
    """El enum acepta el nuevo estado intermedio de carga."""
    assert DocumentStatus.UPLOADING == "uploading"
    assert DocumentStatus("uploading") is DocumentStatus.UPLOADING


def test_document_model_accepts_upload_fields():
    """El modelo persiste la reserva de carga y la referencia al archivo en GridFS."""
    file_id = ObjectId()
    lease_until = datetime.now(timezone.utc)

    document = build_document(
        status=DocumentStatus.UPLOADING,
        file_id=file_id,
        upload_owner_id="upload-owner-1",
        upload_lease_until=lease_until,
        upload_fencing_token="fencing-token-1",
    )

    assert document.status == DocumentStatus.UPLOADING
    assert document.file_id == file_id
    assert document.upload_owner_id == "upload-owner-1"
    assert document.upload_lease_until == lease_until
    assert document.upload_fencing_token == "fencing-token-1"


def test_upload_fields_default_to_none():
    """Los nuevos campos son opcionales y no alteran el estado inicial existente."""
    document = build_document()

    assert document.file_id is None
    assert document.upload_owner_id is None
    assert document.upload_lease_until is None
    assert document.upload_fencing_token is None
    assert document.status == DocumentStatus.PENDING


def test_upload_fields_survive_serialization_round_trip():
    """model_dump y model_validate conservan ObjectId, fechas y tokens internos."""
    file_id = ObjectId()
    lease_until = datetime.now(timezone.utc)

    original = build_document(
        status=DocumentStatus.UPLOADING,
        file_id=file_id,
        upload_owner_id="upload-owner-1",
        upload_lease_until=lease_until,
        upload_fencing_token="fencing-token-1",
    )

    dumped = original.model_dump()
    restored = DocumentModel.model_validate(dumped)

    assert dumped["file_id"] == file_id
    assert restored.status == DocumentStatus.UPLOADING
    assert restored.file_id == file_id
    assert restored.upload_owner_id == "upload-owner-1"
    assert restored.upload_lease_until == lease_until
    assert restored.upload_fencing_token == "fencing-token-1"


def test_existing_fields_and_defaults_are_preserved():
    """Los campos y valores por defecto previos siguen funcionando."""
    document = build_document()

    assert document.id is None
    assert document.filename == "manual.pdf"
    assert document.checksum == "abc123"
    assert document.size_bytes == 1024
    assert document.content is None
    assert document.page_count is None
    assert document.error_message is None
    assert isinstance(document.created_at, datetime)
    assert isinstance(document.updated_at, datetime)


def test_document_response_does_not_expose_upload_fields():
    """El DTO de la API no incorpora los campos internos de carga."""
    internal_fields = {
        "file_id",
        "upload_owner_id",
        "upload_lease_until",
        "upload_fencing_token",
    }

    assert internal_fields.isdisjoint(DocumentResponse.model_fields)
