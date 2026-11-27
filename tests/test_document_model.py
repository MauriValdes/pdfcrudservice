"""Tests unitarios del modelo de persistencia de documentos."""
from datetime import datetime, timezone
import uuid

from bson import BSON, ObjectId

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


def test_processing_fields_have_expected_initial_values():
    """Un documento nuevo inicia sin propietario ni actividad de procesamiento."""
    document = build_document()

    assert document.worker_id is None
    assert document.lease_until is None
    assert document.heartbeat_at is None
    assert document.fencing_token is None
    assert document.processing_cycle_id is None
    assert document.attempt_count == 0
    assert document.next_retry_at is None


def test_document_model_accepts_processing_field_values():
    """El modelo acepta UUID como strings y valores temporales de procesamiento."""
    lease_until = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    heartbeat_at = datetime(2026, 1, 2, 3, 3, tzinfo=timezone.utc)
    next_retry_at = datetime(2026, 1, 2, 3, 5, tzinfo=timezone.utc)
    worker_id = str(uuid.uuid4())
    fencing_token = str(uuid.uuid4())
    processing_cycle_id = str(uuid.uuid4())

    document = build_document(
        worker_id=worker_id,
        lease_until=lease_until,
        heartbeat_at=heartbeat_at,
        fencing_token=fencing_token,
        processing_cycle_id=processing_cycle_id,
        attempt_count=2,
        next_retry_at=next_retry_at,
    )

    assert document.worker_id == worker_id
    assert document.lease_until == lease_until
    assert document.heartbeat_at == heartbeat_at
    assert document.fencing_token == fencing_token
    assert document.processing_cycle_id == processing_cycle_id
    assert document.attempt_count == 2
    assert document.next_retry_at == next_retry_at


def test_upload_and_processing_ownership_fields_are_independent():
    """La propiedad de carga no se confunde con la propiedad del worker."""
    file_id = ObjectId()
    upload_lease_until = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    processing_lease_until = datetime(2026, 1, 2, 3, 5, tzinfo=timezone.utc)
    upload_owner_id = str(uuid.uuid4())
    worker_id = str(uuid.uuid4())
    upload_fencing_token = str(uuid.uuid4())
    fencing_token = str(uuid.uuid4())

    document = build_document(
        status=DocumentStatus.UPLOADING,
        file_id=file_id,
        upload_owner_id=upload_owner_id,
        upload_lease_until=upload_lease_until,
        upload_fencing_token=upload_fencing_token,
        worker_id=worker_id,
        lease_until=processing_lease_until,
        heartbeat_at=processing_lease_until,
        fencing_token=fencing_token,
        processing_cycle_id=str(uuid.uuid4()),
        attempt_count=1,
        next_retry_at=None,
    )

    assert document.file_id == file_id
    assert document.upload_owner_id == upload_owner_id
    assert document.upload_lease_until == upload_lease_until
    assert document.upload_fencing_token == upload_fencing_token
    assert document.worker_id == worker_id
    assert document.lease_until == processing_lease_until
    assert document.heartbeat_at == processing_lease_until
    assert document.fencing_token == fencing_token


def test_legacy_document_validation_uses_processing_defaults():
    """Documentos persistidos sin campos de procesamiento siguen validándose."""
    legacy_data = {
        "filename": "legacy.pdf",
        "checksum": "legacy-checksum",
        "size_bytes": 512,
        "status": DocumentStatus.PENDING.value,
    }

    document = DocumentModel.model_validate(legacy_data)

    assert document.filename == "legacy.pdf"
    assert document.checksum == "legacy-checksum"
    assert document.size_bytes == 512
    assert document.status == DocumentStatus.PENDING
    assert document.worker_id is None
    assert document.lease_until is None
    assert document.heartbeat_at is None
    assert document.fencing_token is None
    assert document.processing_cycle_id is None
    assert document.attempt_count == 0
    assert document.next_retry_at is None


def test_processing_fields_survive_serialization_round_trip():
    """La serialización conserva valores y tipos de los campos de procesamiento."""
    lease_until = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    heartbeat_at = datetime(2026, 1, 2, 3, 3, tzinfo=timezone.utc)
    next_retry_at = datetime(2026, 1, 2, 3, 5, tzinfo=timezone.utc)
    worker_id = str(uuid.uuid4())
    fencing_token = str(uuid.uuid4())
    processing_cycle_id = str(uuid.uuid4())
    original = build_document(
        worker_id=worker_id,
        lease_until=lease_until,
        heartbeat_at=heartbeat_at,
        fencing_token=fencing_token,
        processing_cycle_id=processing_cycle_id,
        attempt_count=3,
        next_retry_at=next_retry_at,
    )

    dumped = original.model_dump()
    restored = DocumentModel.model_validate(dumped)

    assert dumped["worker_id"] == worker_id
    assert dumped["lease_until"] == lease_until
    assert dumped["heartbeat_at"] == heartbeat_at
    assert dumped["fencing_token"] == fencing_token
    assert dumped["processing_cycle_id"] == processing_cycle_id
    assert dumped["attempt_count"] == 3
    assert dumped["next_retry_at"] == next_retry_at
    assert isinstance(dumped["worker_id"], str)
    assert isinstance(dumped["lease_until"], datetime)
    assert isinstance(dumped["heartbeat_at"], datetime)
    assert isinstance(dumped["fencing_token"], str)
    assert isinstance(dumped["processing_cycle_id"], str)
    assert isinstance(dumped["attempt_count"], int)
    assert isinstance(dumped["next_retry_at"], datetime)
    assert restored.worker_id == worker_id
    assert restored.lease_until == lease_until
    assert restored.heartbeat_at == heartbeat_at
    assert restored.fencing_token == fencing_token
    assert restored.processing_cycle_id == processing_cycle_id
    assert restored.attempt_count == 3
    assert restored.next_retry_at == next_retry_at


NEW_PROCESSING_FIELDS = (
    "processing_cycle_started_at",
    "last_attempt_at",
    "last_error",
    "completed_at",
    "failed_at",
)


def test_new_processing_fields_default_to_none_and_accept_explicit_none():
    """Los nuevos metadatos son opcionales y no cambian defaults existentes."""
    document = build_document()
    explicit_none = build_document(
        processing_cycle_started_at=None,
        last_attempt_at=None,
        last_error=None,
        completed_at=None,
        failed_at=None,
    )

    assert all(getattr(document, field) is None for field in NEW_PROCESSING_FIELDS)
    assert all(
        getattr(explicit_none, field) is None for field in NEW_PROCESSING_FIELDS
    )
    assert document.status == DocumentStatus.PENDING
    assert document.attempt_count == 0
    assert document.content is None
    assert document.error_message is None


def test_legacy_document_without_new_processing_fields_remains_valid():
    """Documentos anteriores reciben defaults para los campos nuevos."""
    legacy_data = {
        # Repositories convert MongoDB's ObjectId to the model's string id.
        "id": str(ObjectId()),
        "filename": "legacy-processing.pdf",
        "checksum": "legacy-processing-checksum",
        "size_bytes": 256,
        "status": DocumentStatus.PENDING.value,
        "error_message": "mensaje público existente",
        "attempt_count": 2,
    }

    document = DocumentModel.model_validate(legacy_data)

    assert all(getattr(document, field) is None for field in NEW_PROCESSING_FIELDS)
    assert document.id == legacy_data["id"]
    assert document.status == DocumentStatus.PENDING
    assert document.attempt_count == 2
    assert document.error_message == "mensaje público existente"


def test_new_processing_timestamps_accept_utc_and_survive_model_round_trip():
    """Las cuatro fechas UTC sobreviven model_dump/model_validate."""
    timestamps = {
        "processing_cycle_started_at": datetime(
            2026, 11, 26, 10, 0, 0, 123000, tzinfo=timezone.utc
        ),
        "last_attempt_at": datetime(
            2026, 11, 26, 10, 1, 2, 234000, tzinfo=timezone.utc
        ),
        "completed_at": datetime(
            2026, 11, 26, 10, 2, 3, 345000, tzinfo=timezone.utc
        ),
        "failed_at": datetime(
            2026, 11, 26, 10, 3, 4, 456000, tzinfo=timezone.utc
        ),
    }

    document = build_document(**timestamps)
    dumped = document.model_dump()
    restored = DocumentModel.model_validate(dumped)

    for field, value in timestamps.items():
        assert getattr(document, field) == value
        assert dumped[field] == value
        assert getattr(restored, field) == value
        assert getattr(restored, field).tzinfo is timezone.utc


def test_new_processing_timestamps_survive_bson_millisecond_round_trip():
    """BSON mantiene fechas con precisión de milisegundos."""
    timestamps = {
        "processing_cycle_started_at": datetime(
            2026, 11, 26, 10, 0, 0, 123456, tzinfo=timezone.utc
        ),
        "last_attempt_at": datetime(
            2026, 11, 26, 10, 1, 2, 234567, tzinfo=timezone.utc
        ),
        "completed_at": datetime(
            2026, 11, 26, 10, 2, 3, 345678, tzinfo=timezone.utc
        ),
        "failed_at": datetime(
            2026, 11, 26, 10, 3, 4, 456789, tzinfo=timezone.utc
        ),
    }
    last_error = "classified internal error"
    document = build_document(**timestamps, last_error=last_error)

    stored_document = BSON.encode(
        document.model_dump(by_alias=True, exclude={"id"})
    )
    decoded_document = BSON(stored_document).decode()
    restored = DocumentModel.model_validate(decoded_document)

    for field, value in timestamps.items():
        expected_milliseconds = value.replace(
            microsecond=(value.microsecond // 1000) * 1000
        )
        expected_naive_utc = expected_milliseconds.replace(tzinfo=None)
        assert decoded_document[field] == expected_naive_utc
        assert getattr(restored, field) == expected_naive_utc
    assert decoded_document["last_error"] == last_error
    assert restored.last_error == last_error


def test_last_error_is_independent_and_survives_model_round_trip():
    """last_error es metadata interna distinta de error_message."""
    document = build_document(
        error_message="Error seguro para el cliente",
        last_error="HTTP 503 upstream unavailable",
    )

    restored = DocumentModel.model_validate(document.model_dump())

    assert document.last_error == "HTTP 503 upstream unavailable"
    assert document.error_message == "Error seguro para el cliente"
    assert restored.last_error == document.last_error
    assert restored.error_message == document.error_message
    assert build_document(last_error=None).last_error is None


def test_new_processing_fields_are_excluded_from_public_response_serialization():
    """El DTO público ignora metadata añadida al modelo de persistencia."""
    document = build_document(
        id=str(ObjectId()),
        processing_cycle_started_at=datetime(
            2026, 11, 26, 10, 0, tzinfo=timezone.utc
        ),
        last_attempt_at=datetime(2026, 11, 26, 10, 1, tzinfo=timezone.utc),
        last_error="internal classified error",
        completed_at=datetime(2026, 11, 26, 10, 2, tzinfo=timezone.utc),
        failed_at=datetime(2026, 11, 26, 10, 3, tzinfo=timezone.utc),
    )

    response = DocumentResponse.model_validate(document.model_dump())

    assert set(NEW_PROCESSING_FIELDS).isdisjoint(DocumentResponse.model_fields)
    assert set(NEW_PROCESSING_FIELDS).isdisjoint(response.model_dump())


ORIGINAL_CLEANUP_FIELDS = (
    "original_cleanup_status",
    "original_deleted_at",
)


def test_original_cleanup_fields_default_to_none_independently_of_document_status():
    """El estado del documento no registra ni autoriza limpieza implícitamente."""
    document = build_document(status=DocumentStatus.COMPLETED)

    assert all(
        getattr(document, field) is None for field in ORIGINAL_CLEANUP_FIELDS
    )
    assert document.status == DocumentStatus.COMPLETED


def test_legacy_document_without_original_cleanup_fields_remains_valid():
    """Documentos legacy reciben defaults sin cambiar estado ni file_id."""
    file_id = ObjectId()
    legacy_data = {
        "filename": "legacy-cleanup.pdf",
        "checksum": "legacy-cleanup-checksum",
        "size_bytes": 512,
        "status": DocumentStatus.COMPLETED.value,
        "content": "texto ya extraído",
        "file_id": file_id,
    }

    document = DocumentModel.model_validate(legacy_data)

    assert all(
        getattr(document, field) is None for field in ORIGINAL_CLEANUP_FIELDS
    )
    assert document.status == DocumentStatus.COMPLETED
    assert document.content == "texto ya extraído"
    assert document.file_id == file_id


def test_pending_original_cleanup_fields_survive_model_round_trip_with_file_id():
    """El marcador pending se serializa sin perder resultado ni referencia GridFS."""
    file_id = ObjectId()
    original = build_document(
        status=DocumentStatus.COMPLETED,
        content="texto extraído",
        page_count=3,
        file_id=file_id,
        original_cleanup_status="pending",
        original_deleted_at=None,
    )

    dumped = original.model_dump()
    restored = DocumentModel.model_validate(dumped)

    assert dumped["original_cleanup_status"] == "pending"
    assert dumped["original_deleted_at"] is None
    assert restored.original_cleanup_status == "pending"
    assert restored.original_deleted_at is None
    assert restored.status == DocumentStatus.COMPLETED
    assert restored.content == "texto extraído"
    assert restored.page_count == 3
    assert restored.file_id == file_id


def test_completed_original_cleanup_survives_bson_millisecond_round_trip():
    """BSON conserva la confirmación de limpieza en UTC a precisión de milisegundos."""
    deleted_at = datetime(
        2026, 11, 27, 12, 34, 56, 789456, tzinfo=timezone.utc
    )
    file_id = ObjectId()
    document = build_document(
        status=DocumentStatus.COMPLETED,
        file_id=file_id,
        original_cleanup_status="completed",
        original_deleted_at=deleted_at,
    )

    stored_document = BSON.encode(
        document.model_dump(by_alias=True, exclude={"id"})
    )
    decoded_document = BSON(stored_document).decode()
    restored = DocumentModel.model_validate(decoded_document)

    expected_milliseconds = deleted_at.replace(
        microsecond=(deleted_at.microsecond // 1000) * 1000
    )
    expected_naive_utc = expected_milliseconds.replace(tzinfo=None)
    assert decoded_document["original_cleanup_status"] == "completed"
    assert decoded_document["original_deleted_at"] == expected_naive_utc
    assert restored.original_cleanup_status == "completed"
    assert restored.original_deleted_at == expected_naive_utc
    assert restored.file_id == file_id
    assert restored.status == DocumentStatus.COMPLETED


def test_original_cleanup_fields_are_excluded_from_public_response():
    """Los campos internos de limpieza no forman parte del DTO público."""
    assert set(ORIGINAL_CLEANUP_FIELDS).isdisjoint(DocumentResponse.model_fields)
