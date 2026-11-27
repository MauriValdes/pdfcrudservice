"""Pruebas de integración de reclamos contra MongoDB real de testing."""

import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio
from bson import Decimal128, Int64, ObjectId

from src.config.config import MONGO_TEST_DATABASE, settings
from src.database.mongodb import MongoDB
from src.models.document import DocumentModel, DocumentStatus
from src.repository import processing_repository
from src.repository.document_repository import DocumentRepository
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import ProcessingRepository
from src.schemas.document import DocumentCreate


MISSING = object()
FIXED_NOW = datetime(2026, 11, 25, 12, 0, 0, 123000, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def testing_context():
    """Conecta exclusivamente a pdf_database_test y limpia solo sus fixtures."""
    manager = MongoDB.for_testing()
    resources: list[tuple[ObjectId, str, ObjectId | None]] = []
    validated_test_db = None
    try:
        await manager.connect()
        if (
            manager._testing is not True
            or manager.db is None
            or manager.db.name != MONGO_TEST_DATABASE
        ):
            raise RuntimeError(
                "El fixture solo puede operar sobre pdf_database_test."
            )
        validated_test_db = manager.db
        yield validated_test_db, resources
    finally:
        try:
            if validated_test_db is not None:
                if (
                    manager._testing is not True
                    or manager.db is not validated_test_db
                    or manager.db.name != MONGO_TEST_DATABASE
                ):
                    raise RuntimeError(
                        "Se bloqueó la limpieza fuera de pdf_database_test."
                    )
                collection = validated_test_db[settings.mongo_collection]
                gridfs = GridFSRepository(validated_test_db)
                for document_id, checksum, file_id in reversed(resources):
                    try:
                        if file_id is not None:
                            await gridfs.delete(file_id)
                    finally:
                        await collection.delete_one(
                            {"_id": document_id, "checksum": checksum}
                        )
        finally:
            await manager.disconnect()


def _freeze_claim_clock(monkeypatch, now=FIXED_NOW):
    """Fija el reloj del repositorio sin alterar las fechas del driver Mongo."""
    class FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return now

    monkeypatch.setattr(processing_repository, "datetime", FrozenDateTime)
    return now


def _build_minimal_pdf(unique_content: str) -> bytes:
    """Construye un PDF mínimo con contenido único para almacenar en GridFS."""
    stream = f"BT /F1 12 Tf 72 720 Td ({unique_content}) Tj ET\n".encode(
        "ascii"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Length {len(stream)} >>\n".encode("ascii")
            + b"stream\n"
            + stream
            + b"endstream"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def _normalize_datetime(value: datetime) -> datetime:
    """Normaliza datetimes BSON a UTC y precisión de milisegundos."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=(value.microsecond // 1000) * 1000)


async def _create_pending_pdf(db, resources, *, created_at=None, fields=None):
    """Crea uploading, persiste un PDF GridFS y finaliza la reserva a pending."""
    token = uuid.uuid4().hex
    pdf_bytes = _build_minimal_pdf(f"processing-claim-{token}")
    checksum = hashlib.sha256(pdf_bytes).hexdigest()
    filename = f"processing-claim-{token}.pdf"
    repository = DocumentRepository(db)
    gridfs = GridFSRepository(db)
    collection = db[settings.mongo_collection]
    file_id = ObjectId()

    reservation, created = await repository.reserve_document(
        DocumentCreate(
            filename=filename,
            checksum=checksum,
            size_bytes=len(pdf_bytes),
        )
    )
    assert created is True
    document_id = ObjectId(reservation.id)
    resources.append((document_id, checksum, file_id))

    owner_id = str(uuid.uuid4())
    upload_token = str(uuid.uuid4())
    upload_lease_until = datetime.now(timezone.utc) + timedelta(minutes=15)
    _, acquired = await repository.acquire_upload_ownership(
        checksum,
        owner_id,
        upload_token,
        upload_lease_until,
    )
    assert acquired is True

    await gridfs.save(
        file_id,
        filename,
        pdf_bytes,
        metadata={"checksum": checksum},
    )
    finalized = await repository.finalize_upload(
        reservation.id,
        owner_id,
        upload_token,
        file_id,
    )
    assert finalized is True

    updates: dict[str, Any] = dict(fields or {})
    if created_at is not None:
        updates["created_at"] = created_at
    if updates:
        result = await collection.update_one(
            {"_id": document_id, "checksum": checksum},
            {"$set": updates},
        )
        assert result.matched_count == 1

    return {
        "document_id": document_id,
        "checksum": checksum,
        "file_id": file_id,
        "pdf_bytes": pdf_bytes,
        "upload_owner_id": owner_id,
        "upload_fencing_token": upload_token,
        "upload_lease_until": upload_lease_until,
    }


async def _claim_processing_document(
    db,
    resources,
    monkeypatch,
    *,
    worker_id="heartbeat-worker",
    attempt_count=4,
    now=FIXED_NOW,
):
    """Crea un fixture registrado y lo reclama para preparar el heartbeat."""
    now = _freeze_claim_clock(monkeypatch, now)
    cycle_id = str(uuid.uuid4())
    document = await _create_pending_pdf(
        db,
        resources,
        fields={
            "attempt_count": attempt_count,
            "processing_cycle_id": cycle_id,
            "processing_cycle_started_at": now,
        },
    )
    claimed = await ProcessingRepository(db).claim_pending(worker_id)
    assert claimed is not None
    assert claimed.id == str(document["document_id"])
    return document, claimed, now, cycle_id


async def _insert_raw_pending(db, resources, *, file_id=MISSING, fields=None):
    """Inserta un documento de testing para filtros con valores no válidos."""
    token = uuid.uuid4().hex
    checksum = hashlib.sha256(token.encode("ascii")).hexdigest()
    document_id = ObjectId()
    document = {
        "_id": document_id,
        "filename": f"raw-claim-{token}.pdf",
        "checksum": checksum,
        "size_bytes": 1,
        "status": DocumentStatus.PENDING.value,
        "created_at": FIXED_NOW,
        "updated_at": FIXED_NOW,
        "attempt_count": 0,
    }
    if file_id is not MISSING:
        document["file_id"] = file_id
    document.update(fields or {})
    await db[settings.mongo_collection].insert_one(document)
    resources.append((document_id, checksum, None))
    return document_id


async def _unset_raw_fields(db, document_id, *field_names):
    if not field_names:
        return
    result = await db[settings.mongo_collection].update_one(
        {"_id": document_id},
        {"$unset": {field: "" for field in field_names}},
    )
    assert result.matched_count == 1


async def _insert_processing_attempt_document(
    db,
    resources,
    *,
    attempt_count=0,
    fields=None,
    unset_fields=(),
):
    """Inserta un documento processing propio para probar la autorización."""
    worker_id = "attempt-worker"
    fencing_token = "attempt-fencing-token"
    processing_cycle_id = "attempt-cycle-id"
    document_fields = {
        "status": DocumentStatus.PROCESSING.value,
        "worker_id": worker_id,
        "fencing_token": fencing_token,
        "processing_cycle_id": processing_cycle_id,
        "processing_cycle_started_at": FIXED_NOW - timedelta(minutes=1),
        "lease_until": FIXED_NOW,
        "updated_at": FIXED_NOW - timedelta(seconds=2),
    }
    if attempt_count is not MISSING:
        document_fields["attempt_count"] = attempt_count
    document_fields.update(fields or {})
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields=document_fields,
    )
    fields_to_unset = list(unset_fields)
    if attempt_count is MISSING:
        fields_to_unset.append("attempt_count")
    await _unset_raw_fields(db, document_id, *fields_to_unset)
    return {
        "document_id": document_id,
        "worker_id": worker_id,
        "fencing_token": fencing_token,
        "processing_cycle_id": processing_cycle_id,
    }


async def _insert_processing_completion_document(
    db,
    resources,
    *,
    attempt_count=1,
    fields=None,
    unset_fields=(),
):
    """Inserta un fixture propio con el último intento ya autorizado."""
    completion_fields = {
        "last_attempt_at": FIXED_NOW,
        "completed_at": None,
        "failed_at": None,
    }
    completion_fields.update(fields or {})
    return await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=attempt_count,
        fields=completion_fields,
        unset_fields=unset_fields,
    )


async def _claim_and_record_processing_attempt(
    db,
    resources,
    monkeypatch,
    *,
    attempt_count=0,
    now=FIXED_NOW,
    worker_id="completion-worker",
):
    """Prepara un PDF real, lo reclama y persiste el intento antes de completar."""
    document, claimed, captured_now, cycle_id = await _claim_processing_document(
        db,
        resources,
        monkeypatch,
        worker_id=worker_id,
        attempt_count=attempt_count,
        now=now,
    )
    authorized_attempt = await ProcessingRepository(db).record_ms2_attempt(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
        claimed.processing_cycle_id,
    )
    assert authorized_attempt == attempt_count + 1
    return document, claimed, captured_now, cycle_id, authorized_attempt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_persists_successful_claim_and_returns_document_model(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    document = await _create_pending_pdf(
        db,
        resources,
        fields={
            "attempt_count": 6,
            "processing_cycle_id": str(uuid.uuid4()),
            "processing_cycle_started_at": now - timedelta(minutes=1),
        },
    )
    repository = ProcessingRepository(db)
    worker_id = "integration-worker-success"
    collection = db[settings.mongo_collection]

    claimed = await repository.claim_pending(worker_id)

    assert isinstance(claimed, DocumentModel)
    assert claimed.id == str(document["document_id"])
    assert claimed.status == DocumentStatus.PROCESSING
    assert claimed.worker_id == worker_id
    assert _normalize_datetime(claimed.lease_until) == now + timedelta(seconds=60)
    assert _normalize_datetime(claimed.heartbeat_at) == now
    assert _normalize_datetime(claimed.updated_at) == now
    uuid.UUID(claimed.fencing_token)
    uuid.UUID(claimed.processing_cycle_id)
    assert _normalize_datetime(claimed.processing_cycle_started_at) == (
        now - timedelta(minutes=1)
    )
    assert claimed.fencing_token != claimed.processing_cycle_id
    assert claimed.attempt_count == 6
    assert claimed.file_id == document["file_id"]

    persisted = await collection.find_one({"_id": document["document_id"]})
    assert persisted is not None
    assert persisted["status"] == DocumentStatus.PROCESSING.value
    assert persisted["worker_id"] == worker_id
    assert _normalize_datetime(persisted["lease_until"]) == now + timedelta(
        seconds=60
    )
    assert _normalize_datetime(persisted["heartbeat_at"]) == now
    assert _normalize_datetime(persisted["updated_at"]) == now
    assert persisted["fencing_token"] == claimed.fencing_token
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert _normalize_datetime(persisted["processing_cycle_started_at"]) == (
        now - timedelta(minutes=1)
    )
    assert persisted["attempt_count"] == 6
    assert persisted["file_id"] == document["file_id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_returns_none_when_there_are_no_eligible_candidates(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    await _insert_raw_pending(
        db,
        resources,
        file_id=None,
    )

    result = await ProcessingRepository(db).claim_pending("worker-no-candidate")

    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expired_lease"),
    [
        (DocumentStatus.UPLOADING.value, False),
        (DocumentStatus.PROCESSING.value, True),
        (DocumentStatus.COMPLETED.value, False),
        (DocumentStatus.FAILED.value, False),
    ],
    ids=["uploading", "processing-expired-lease", "completed", "failed"],
)
async def test_claim_excludes_documents_outside_pending_state(
    testing_context, monkeypatch, status, expired_lease
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    fields = {"status": status}
    if expired_lease:
        fields["lease_until"] = now - timedelta(seconds=1)
    document = await _create_pending_pdf(db, resources, fields=fields)
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document["document_id"]})

    result = await ProcessingRepository(db).claim_pending("worker-status")

    after = await collection.find_one({"_id": document["document_id"]})
    assert result is None
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "file_id"),
    [
        ("missing", MISSING),
        ("null", None),
        ("string", "not-an-object-id"),
        ("integer", 42),
        ("empty-array", []),
    ],
    ids=["missing", "null", "string", "integer", "empty-array"],
)
async def test_claim_excludes_invalid_file_id_values_without_mutation(
    testing_context, monkeypatch, case, file_id
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=file_id,
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document_id})

    result = await ProcessingRepository(db).claim_pending("worker-invalid-file")

    after = await collection.find_one({"_id": document_id})
    assert result is None, f"file_id case {case!r} was unexpectedly claimed"
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_excludes_array_containing_objectids_without_mutation(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=[ObjectId(), ObjectId()],
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document_id})

    result = await ProcessingRepository(db).claim_pending("worker-array-file")

    after = await collection.find_one({"_id": document_id})
    assert after == before, (
        "MongoDB reclamó o modificó un documento cuyo file_id es un array "
        "con elementos ObjectId"
    )
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "next_retry_at", "eligible"),
    [
        ("missing", MISSING, True),
        ("null", None, True),
        ("past", FIXED_NOW - timedelta(seconds=1), True),
        ("equal", FIXED_NOW, True),
        ("future", FIXED_NOW + timedelta(seconds=1), False),
    ],
    ids=["missing", "null", "past", "equal", "future"],
)
async def test_claim_respects_next_retry_at_eligibility(
    testing_context,
    monkeypatch,
    case,
    next_retry_at,
    eligible,
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    cycle_id = str(uuid.uuid4())
    cycle_started_at = now - timedelta(minutes=1)
    document = await _create_pending_pdf(
        db,
        resources,
        fields={
            "processing_cycle_id": cycle_id,
            "processing_cycle_started_at": cycle_started_at,
        },
    )
    collection = db[settings.mongo_collection]
    document_filter = {"_id": document["document_id"]}
    if next_retry_at is MISSING:
        await collection.update_one(
            document_filter,
            {"$unset": {"next_retry_at": ""}},
        )
    else:
        await collection.update_one(
            document_filter,
            {"$set": {"next_retry_at": next_retry_at}},
        )
    before = await collection.find_one(document_filter)
    assert before["processing_cycle_id"] == cycle_id
    assert _normalize_datetime(before["processing_cycle_started_at"]) == (
        cycle_started_at
    )

    result = await ProcessingRepository(db).claim_pending("worker-retry-time")

    after = await collection.find_one(document_filter)
    if eligible:
        assert isinstance(result, DocumentModel), f"{case} should be eligible"
        assert result.id == str(document["document_id"])
        assert _normalize_datetime(result.heartbeat_at) == now
        assert after["status"] == DocumentStatus.PROCESSING.value
    else:
        assert result is None
        assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_selects_oldest_created_at_then_smallest_id(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    tied_time = FIXED_NOW - timedelta(minutes=2)
    oldest = await _create_pending_pdf(
        db,
        resources,
        created_at=FIXED_NOW - timedelta(minutes=3),
    )
    tied_a = await _create_pending_pdf(db, resources, created_at=tied_time)
    tied_b = await _create_pending_pdf(db, resources, created_at=tied_time)
    tied_first = min(
        (tied_a, tied_b),
        key=lambda item: item["document_id"],
    )
    repository = ProcessingRepository(db)

    first = await repository.claim_pending("worker-order-1")
    second = await repository.claim_pending("worker-order-2")

    assert first is not None
    assert first.id == str(oldest["document_id"])
    assert second is not None
    assert second.id == str(tied_first["document_id"])


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cycle_case",
    ["both-missing", "id-missing", "started-at-missing", "both-null", "existing"],
)
async def test_claim_initializes_or_preserves_cycle_on_real_mongodb(
    testing_context, monkeypatch, cycle_case
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    existing_cycle = str(uuid.uuid4())
    existing_started_at = FIXED_NOW - timedelta(minutes=2)
    fields = {}
    if cycle_case == "both-null":
        fields.update(
            {
                "processing_cycle_id": None,
                "processing_cycle_started_at": None,
            }
        )
    elif cycle_case in {"id-missing", "started-at-missing"}:
        fields.update(
            {
                "processing_cycle_id": None,
                "processing_cycle_started_at": None,
            }
        )
    elif cycle_case == "existing":
        fields.update(
            {
                "processing_cycle_id": existing_cycle,
                "processing_cycle_started_at": existing_started_at,
            }
        )
    document = await _create_pending_pdf(db, resources, fields=fields)
    unset_fields = {}
    if cycle_case in {"both-missing", "id-missing"}:
        unset_fields["processing_cycle_id"] = ""
    if cycle_case in {"both-missing", "started-at-missing"}:
        unset_fields["processing_cycle_started_at"] = ""
    if unset_fields:
        await db[settings.mongo_collection].update_one(
            {"_id": document["document_id"]},
            {"$unset": unset_fields},
        )

    claimed = await ProcessingRepository(db).claim_pending("worker-cycle")

    assert claimed is not None
    uuid.UUID(claimed.fencing_token)
    if cycle_case == "existing":
        assert claimed.processing_cycle_id == existing_cycle
        expected_started_at = existing_started_at
    else:
        uuid.UUID(claimed.processing_cycle_id)
        expected_started_at = FIXED_NOW
        assert claimed.processing_cycle_started_at is not None
        assert _normalize_datetime(claimed.processing_cycle_started_at) == FIXED_NOW
    assert claimed.fencing_token != claimed.processing_cycle_id
    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert persisted is not None
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert _normalize_datetime(
        persisted["processing_cycle_started_at"]
    ) == expected_started_at
    assert persisted["attempt_count"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_initializes_new_cycle_when_attempt_count_is_absent(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields={
            "processing_cycle_id": None,
            "processing_cycle_started_at": None,
        },
    )
    await _unset_raw_fields(db, document_id, "attempt_count")

    claimed = await ProcessingRepository(db).claim_pending("worker-missing-count")

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document_id}
    )
    assert claimed is not None
    uuid.UUID(claimed.processing_cycle_id)
    assert _normalize_datetime(claimed.processing_cycle_started_at) == now
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert _normalize_datetime(persisted["processing_cycle_started_at"]) == now
    assert "attempt_count" not in persisted


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_initial_cycle_ignores_upload_metadata(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields={
            "processing_cycle_id": None,
            "processing_cycle_started_at": None,
            "upload_owner_id": "upload-owner-still-present",
            "upload_lease_until": now + timedelta(minutes=10),
            "upload_fencing_token": "upload-token-still-present",
        },
    )

    claimed = await ProcessingRepository(db).claim_pending("worker-upload-independent")

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document_id}
    )
    assert claimed is not None
    assert _normalize_datetime(claimed.processing_cycle_started_at) == now
    assert persisted["upload_owner_id"] == "upload-owner-still-present"
    assert _normalize_datetime(persisted["upload_lease_until"]) == (
        now + timedelta(minutes=10)
    )
    assert persisted["upload_fencing_token"] == "upload-token-still-present"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_attempt_at", FIXED_NOW - timedelta(seconds=1)),
        ("last_error", "previous classified failure"),
        ("completed_at", FIXED_NOW - timedelta(seconds=1)),
        ("failed_at", FIXED_NOW - timedelta(seconds=1)),
        ("next_retry_at", FIXED_NOW - timedelta(seconds=1)),
        ("worker_id", "previous-worker"),
        ("lease_until", FIXED_NOW - timedelta(seconds=1)),
        ("heartbeat_at", FIXED_NOW - timedelta(seconds=1)),
        ("fencing_token", "previous-fencing-token"),
    ],
)
async def test_claim_rejects_empty_cycle_with_processing_evidence_without_mutation(
    testing_context, monkeypatch, field, value
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields={
            "processing_cycle_id": None,
            "processing_cycle_started_at": None,
            field: value,
        },
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document_id})

    result = await ProcessingRepository(db).claim_pending("worker-evidence-check")

    after = await collection.find_one({"_id": document_id})
    assert result is None, f"{field} is evidence of prior processing"
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", "manually supplied before processing"),
        ("page_count", 0),
        ("error_message", "manual annotation before processing"),
    ],
)
async def test_claim_allows_and_preserves_public_fields_without_prior_processing(
    testing_context, monkeypatch, field, value
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields={
            "processing_cycle_id": None,
            "processing_cycle_started_at": None,
            field: value,
        },
    )

    claimed = await ProcessingRepository(db).claim_pending("worker-public-fields")

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document_id}
    )
    assert claimed is not None
    assert getattr(claimed, field) == value
    assert persisted[field] == value
    assert uuid.UUID(claimed.processing_cycle_id)
    assert _normalize_datetime(claimed.processing_cycle_started_at) == now
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert _normalize_datetime(persisted["processing_cycle_started_at"]) == now
    assert claimed.attempt_count == 0
    assert persisted["attempt_count"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_error", ""),
        ("worker_id", False),
        ("last_attempt_at", 0),
        ("lease_until", []),
        ("heartbeat_at", [None]),
    ],
    ids=[
        "empty-string",
        "false",
        "zero",
        "empty-array",
        "array-containing-null",
    ],
)
async def test_claim_rejects_falsy_non_null_internal_metadata_without_mutation(
    testing_context, monkeypatch, field, value
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields={
            "processing_cycle_id": None,
            "processing_cycle_started_at": None,
            field: value,
        },
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document_id})

    try:
        result = await ProcessingRepository(db).claim_pending("worker-falsy-metadata")
    except Exception as error:
        result = error

    after = await collection.find_one({"_id": document_id})
    assert (result is None, after == before) == (True, True), (
        f"{field}={value!r} is present, non-null internal metadata"
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "attempt_count"),
    [
        ("explicit-null", None),
        ("negative", -1),
        ("boolean", True),
        ("boolean-false", False),
        ("array", [0]),
        ("string", "0"),
        ("double", 0.0),
        ("decimal128", Decimal128("0")),
    ],
    ids=[
        "explicit-null",
        "negative",
        "boolean",
        "boolean-false",
        "array-containing-zero",
        "string",
        "double",
        "decimal128",
    ],
)
async def test_claim_rejects_invalid_initial_attempt_count_without_mutation(
    testing_context, monkeypatch, case, attempt_count
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields={
            "processing_cycle_id": None,
            "processing_cycle_started_at": None,
            "attempt_count": attempt_count,
        },
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document_id})

    try:
        result = await ProcessingRepository(db).claim_pending("worker-count-check")
    except Exception as error:
        result = error

    after = await collection.find_one({"_id": document_id})
    assert result is None, f"attempt_count case {case!r} must not be claimed"
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "attempt_count", "expected_type"),
    [
        ("bson-int32-zero", 0, int),
        ("bson-int64-zero", Int64(0), Int64),
    ],
)
async def test_claim_accepts_bson_int32_and_int64_zero_attempt_counts(
    testing_context, monkeypatch, case, attempt_count, expected_type
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db,
        resources,
        file_id=ObjectId(),
        fields={
            "processing_cycle_id": None,
            "processing_cycle_started_at": None,
            "attempt_count": attempt_count,
        },
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document_id})
    assert type(before["attempt_count"]) is expected_type

    claimed = await ProcessingRepository(db).claim_pending(f"worker-{case}")

    after = await collection.find_one({"_id": document_id})
    assert claimed is not None
    assert type(after["attempt_count"]) is expected_type
    assert after["attempt_count"] == 0
    assert claimed.attempt_count == 0
    assert _normalize_datetime(claimed.processing_cycle_started_at) == now


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "fields", "unset_fields"),
    [
        (
            "id-without-start",
            {"processing_cycle_id": "existing-cycle"},
            ("processing_cycle_started_at",),
        ),
        (
            "id-with-null-start",
            {"processing_cycle_id": "existing-cycle", "processing_cycle_started_at": None},
            (),
        ),
        (
            "start-without-id",
            {"processing_cycle_started_at": FIXED_NOW},
            ("processing_cycle_id",),
        ),
        (
            "start-with-null-id",
            {"processing_cycle_id": None, "processing_cycle_started_at": FIXED_NOW},
            (),
        ),
        (
            "empty-id",
            {"processing_cycle_id": "", "processing_cycle_started_at": FIXED_NOW},
            (),
        ),
        (
            "array-id-containing-string",
            {"processing_cycle_id": ["existing-cycle"], "processing_cycle_started_at": FIXED_NOW},
            (),
        ),
        (
            "integer-id",
            {"processing_cycle_id": 7, "processing_cycle_started_at": FIXED_NOW},
            (),
        ),
        (
            "string-date",
            {"processing_cycle_id": "existing-cycle", "processing_cycle_started_at": "2026-11-25"},
            (),
        ),
        (
            "array-date-containing-date",
            {"processing_cycle_id": "existing-cycle", "processing_cycle_started_at": [FIXED_NOW]},
            (),
        ),
        (
            "future-date",
            {
                "processing_cycle_id": "existing-cycle",
                "processing_cycle_started_at": FIXED_NOW + timedelta(seconds=1),
            },
            (),
        ),
        (
            "both-absent-positive-count",
            {"attempt_count": 1},
            ("processing_cycle_id", "processing_cycle_started_at"),
        ),
    ],
)
async def test_claim_rejects_inconsistent_cycle_without_mutation(
    testing_context, monkeypatch, case, fields, unset_fields
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document_id = await _insert_raw_pending(
        db, resources, file_id=ObjectId(), fields=fields
    )
    await _unset_raw_fields(db, document_id, *unset_fields)
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document_id})

    try:
        result = await ProcessingRepository(db).claim_pending("worker-cycle-check")
    except Exception as error:
        result = error

    after = await collection.find_one({"_id": document_id})
    assert result is None, f"inconsistent cycle {case!r} must not be claimed"
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_preserves_expired_existing_cycle_and_fully_consumed_attempts(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    cycle_id = str(uuid.uuid4())
    cycle_started_at = now - timedelta(minutes=11)
    document = await _create_pending_pdf(
        db,
        resources,
        fields={
            "processing_cycle_id": cycle_id,
            "processing_cycle_started_at": cycle_started_at,
            "attempt_count": 5,
            "last_attempt_at": now - timedelta(seconds=2),
            "next_retry_at": now - timedelta(seconds=1),
        },
    )

    claimed = await ProcessingRepository(db).claim_pending("worker-expired-cycle")

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert claimed is not None
    assert claimed.processing_cycle_id == cycle_id
    assert _normalize_datetime(claimed.processing_cycle_started_at) == cycle_started_at
    assert claimed.attempt_count == 5
    assert persisted["processing_cycle_id"] == cycle_id
    assert _normalize_datetime(persisted["processing_cycle_started_at"]) == cycle_started_at
    assert persisted["attempt_count"] == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_automatic_retry_claim_preserves_original_cycle_and_window(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    cycle_id = str(uuid.uuid4())
    cycle_started_at = now - timedelta(minutes=4)
    document = await _create_pending_pdf(
        db,
        resources,
        fields={
            "processing_cycle_id": cycle_id,
            "processing_cycle_started_at": cycle_started_at,
            "attempt_count": 2,
            "last_attempt_at": now - timedelta(seconds=5),
            "last_error": "temporary upstream failure",
            "next_retry_at": now - timedelta(seconds=1),
        },
    )

    claimed = await ProcessingRepository(db).claim_pending("worker-auto-retry")

    assert claimed is not None
    assert claimed.processing_cycle_id == cycle_id
    assert _normalize_datetime(claimed.processing_cycle_started_at) == cycle_started_at
    assert claimed.attempt_count == 2
    assert _normalize_datetime(claimed.last_attempt_at) == now - timedelta(seconds=5)
    assert claimed.last_error == "temporary upstream failure"
    assert _normalize_datetime(claimed.next_retry_at) == now - timedelta(seconds=1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_preserves_retry_count_and_cycle_but_renews_fencing_token(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    existing_cycle = str(uuid.uuid4())
    previous_token = str(uuid.uuid4())
    document = await _create_pending_pdf(
        db,
        resources,
        fields={
            "attempt_count": 7,
            "processing_cycle_id": existing_cycle,
            "processing_cycle_started_at": FIXED_NOW - timedelta(minutes=3),
            "fencing_token": previous_token,
        },
    )

    claimed = await ProcessingRepository(db).claim_pending("worker-retry")

    assert claimed is not None
    assert claimed.attempt_count == 7
    assert claimed.processing_cycle_id == existing_cycle
    assert _normalize_datetime(claimed.processing_cycle_started_at) == (
        FIXED_NOW - timedelta(minutes=3)
    )
    assert claimed.fencing_token != previous_token
    uuid.UUID(claimed.fencing_token)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_first_claim_initializes_one_cycle_pair(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    document = await _create_pending_pdf(
        db,
        resources,
        fields={"attempt_count": 0},
    )
    collection = db[settings.mongo_collection]
    await _unset_raw_fields(
        db,
        document["document_id"],
        "processing_cycle_id",
        "processing_cycle_started_at",
    )
    before = await collection.find_one({"_id": document["document_id"]})
    assert "processing_cycle_id" not in before
    assert "processing_cycle_started_at" not in before
    assert before["attempt_count"] == 0

    first_repository = ProcessingRepository(db)
    second_repository = ProcessingRepository(db)
    barrier = asyncio.Barrier(2)

    async def claim_after_barrier(repository, worker_id):
        await barrier.wait()
        return await repository.claim_pending(worker_id)

    first, second = await asyncio.gather(
        claim_after_barrier(first_repository, "worker-first-cycle-a"),
        claim_after_barrier(second_repository, "worker-first-cycle-b"),
    )

    winners = [result for result in (first, second) if result is not None]
    losers = [result for result in (first, second) if result is None]
    assert len(winners) == 1
    assert len(losers) == 1
    winner = winners[0]

    persisted = await collection.find_one({"_id": document["document_id"]})
    assert persisted is not None
    assert uuid.UUID(persisted["processing_cycle_id"])
    assert _normalize_datetime(persisted["processing_cycle_started_at"]) == now
    assert persisted["processing_cycle_id"] == winner.processing_cycle_id
    assert _normalize_datetime(winner.processing_cycle_started_at) == now
    assert persisted["fencing_token"] == winner.fencing_token
    assert winner.attempt_count == persisted["attempt_count"] == 0
    assert await collection.count_documents(
        {
            "_id": document["document_id"],
            "processing_cycle_id": winner.processing_cycle_id,
            "processing_cycle_started_at": now,
        }
    ) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_concurrent_workers_cannot_claim_the_same_document(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document = await _create_pending_pdf(
        db,
        resources,
        fields={
            "attempt_count": 5,
            "processing_cycle_id": str(uuid.uuid4()),
            "processing_cycle_started_at": FIXED_NOW - timedelta(minutes=1),
        },
    )
    first_repository = ProcessingRepository(db)
    second_repository = ProcessingRepository(db)
    barrier = asyncio.Barrier(2)

    async def claim_after_barrier(repository, worker_id):
        await barrier.wait()
        return await repository.claim_pending(worker_id)

    first, second = await asyncio.gather(
        claim_after_barrier(first_repository, "worker-concurrent-a"),
        claim_after_barrier(second_repository, "worker-concurrent-b"),
    )

    winners = [result for result in (first, second) if result is not None]
    losers = [result for result in (first, second) if result is None]
    assert len(winners) == 1
    assert len(losers) == 1
    winner = winners[0]

    collection = db[settings.mongo_collection]
    persisted = await collection.find_one({"_id": document["document_id"]})
    assert persisted is not None
    assert persisted["status"] == DocumentStatus.PROCESSING.value
    assert persisted["worker_id"] == winner.worker_id
    assert persisted["worker_id"] in {
        "worker-concurrent-a",
        "worker-concurrent-b",
    }
    assert persisted["fencing_token"] == winner.fencing_token
    assert persisted["attempt_count"] == 5
    assert persisted["processing_cycle_id"] == winner.processing_cycle_id
    assert _normalize_datetime(
        persisted["processing_cycle_started_at"]
    ) == FIXED_NOW - timedelta(minutes=1)
    assert await collection.count_documents(
        {"_id": document["document_id"]}
    ) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_active_upload_lease_does_not_block_processing_claim(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document = await _create_pending_pdf(db, resources)
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document["document_id"]})
    assert _normalize_datetime(before["upload_lease_until"]) > datetime.now(
        timezone.utc
    )

    claimed = await ProcessingRepository(db).claim_pending("worker-active-upload")

    assert claimed is not None
    assert claimed.id == str(document["document_id"])
    persisted = await collection.find_one({"_id": document["document_id"]})
    assert persisted["upload_owner_id"] == before["upload_owner_id"]
    assert persisted["upload_fencing_token"] == before["upload_fencing_token"]
    assert persisted["upload_lease_until"] == before["upload_lease_until"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dollar_prefixed_worker_id_is_persisted_literally(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    document = await _create_pending_pdf(db, resources)
    worker_id = "$integrationWorker"

    claimed = await ProcessingRepository(db).claim_pending(worker_id)

    assert claimed is not None
    assert claimed.worker_id == worker_id
    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert persisted["worker_id"] == worker_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_processing_collection_inherits_durable_write_concern(testing_context):
    db, _ = testing_context
    collection = ProcessingRepository(db).collection

    assert collection.write_concern.document == {"w": 1, "j": True}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_processing_lease_persists_all_three_timestamps(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, now, _ = await _claim_processing_document(
        db, resources, monkeypatch
    )
    collection = db[settings.mongo_collection]
    await collection.update_one(
        {"_id": document["document_id"], "checksum": document["checksum"]},
        {
            "$set": {
                "lease_until": now + timedelta(seconds=10),
                "heartbeat_at": now - timedelta(seconds=5),
                "updated_at": now - timedelta(seconds=5),
            }
        },
    )

    renewed = await ProcessingRepository(db).renew_processing_lease(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
    )

    persisted = await collection.find_one({"_id": document["document_id"]})
    assert renewed is True
    assert _normalize_datetime(persisted["heartbeat_at"]) == now
    assert _normalize_datetime(persisted["lease_until"]) == now + timedelta(
        seconds=60
    )
    assert _normalize_datetime(persisted["updated_at"]) == now


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_processing_lease_preserves_cycle_attempts_and_upload_fields(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, now, cycle_id = await _claim_processing_document(
        db,
        resources,
        monkeypatch,
        attempt_count=9,
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document["document_id"]})

    renewed = await ProcessingRepository(db).renew_processing_lease(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
    )

    after = await collection.find_one({"_id": document["document_id"]})
    assert renewed is True
    assert after["processing_cycle_id"] == cycle_id == before["processing_cycle_id"]
    assert after["processing_cycle_started_at"] == before[
        "processing_cycle_started_at"
    ]
    assert _normalize_datetime(after["processing_cycle_started_at"]) == now
    assert after["attempt_count"] == 9 == before["attempt_count"]
    for field in (
        "upload_owner_id",
        "upload_lease_until",
        "upload_fencing_token",
    ):
        assert after[field] == before[field]
    assert _normalize_datetime(after["lease_until"]) >= now + timedelta(seconds=60)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_processing_lease_rejects_expired_lease(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, now, _ = await _claim_processing_document(
        db, resources, monkeypatch
    )
    collection = db[settings.mongo_collection]
    await collection.update_one(
        {"_id": document["document_id"]},
        {"$set": {"lease_until": now - timedelta(milliseconds=1)}},
    )
    before = await collection.find_one({"_id": document["document_id"]})

    renewed = await ProcessingRepository(db).renew_processing_lease(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
    )

    after = await collection.find_one({"_id": document["document_id"]})
    assert renewed is False
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_processing_lease_accepts_exact_expiration_boundary(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, now, _ = await _claim_processing_document(
        db, resources, monkeypatch
    )
    collection = db[settings.mongo_collection]
    await collection.update_one(
        {"_id": document["document_id"]},
        {"$set": {"lease_until": now}},
    )

    renewed = await ProcessingRepository(db).renew_processing_lease(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
    )

    persisted = await collection.find_one({"_id": document["document_id"]})
    assert renewed is True
    assert _normalize_datetime(persisted["lease_until"]) == now + timedelta(
        seconds=60
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_processing_lease_rejects_different_worker_id(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, _, _ = await _claim_processing_document(
        db, resources, monkeypatch
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document["document_id"]})

    renewed = await ProcessingRepository(db).renew_processing_lease(
        claimed.id,
        "another-worker",
        claimed.fencing_token,
    )

    after = await collection.find_one({"_id": document["document_id"]})
    assert renewed is False
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_processing_lease_rejects_different_fencing_token(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, _, _ = await _claim_processing_document(
        db, resources, monkeypatch
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": document["document_id"]})

    renewed = await ProcessingRepository(db).renew_processing_lease(
        claimed.id,
        claimed.worker_id,
        str(uuid.uuid4()),
    )

    after = await collection.find_one({"_id": document["document_id"]})
    assert renewed is False
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_renew_processing_lease_returns_false_for_missing_document(
    testing_context, monkeypatch
):
    db, _ = testing_context
    now = _freeze_claim_clock(monkeypatch)

    renewed = await ProcessingRepository(db).renew_processing_lease(
        str(ObjectId()),
        "heartbeat-worker",
        str(uuid.uuid4()),
    )

    assert renewed is False
    assert now == FIXED_NOW


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_heartbeats_do_not_move_lease_or_timestamps_backwards(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, now, _ = await _claim_processing_document(
        db, resources, monkeypatch
    )
    later_now = now + timedelta(seconds=5)
    clock_values = [now, later_now]

    class ConcurrentHeartbeatClock:
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return clock_values.pop(0)

    monkeypatch.setattr(processing_repository, "datetime", ConcurrentHeartbeatClock)
    repository = ProcessingRepository(db)

    first, second = await asyncio.gather(
        repository.renew_processing_lease(
            claimed.id,
            claimed.worker_id,
            claimed.fencing_token,
        ),
        repository.renew_processing_lease(
            claimed.id,
            claimed.worker_id,
            claimed.fencing_token,
        ),
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert first is True
    assert second is True
    assert clock_values == []
    assert _normalize_datetime(persisted["heartbeat_at"]) == later_now
    assert _normalize_datetime(persisted["updated_at"]) == later_now
    assert _normalize_datetime(persisted["lease_until"]) == later_now + timedelta(
        seconds=60
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_old_processing_owner_cannot_renew_after_new_owner_acquires_document(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, old_claim, now, _ = await _claim_processing_document(
        db, resources, monkeypatch
    )
    collection = db[settings.mongo_collection]
    new_worker_id = "new-heartbeat-owner"
    new_fencing_token = str(uuid.uuid4())
    new_lease_until = now + timedelta(seconds=90)
    await collection.update_one(
        {"_id": document["document_id"]},
        {
            "$set": {
                "worker_id": new_worker_id,
                "fencing_token": new_fencing_token,
                "lease_until": new_lease_until,
            }
        },
    )
    before = await collection.find_one({"_id": document["document_id"]})

    renewed = await ProcessingRepository(db).renew_processing_lease(
        old_claim.id,
        old_claim.worker_id,
        old_claim.fencing_token,
    )

    after = await collection.find_one({"_id": document["document_id"]})
    assert renewed is False
    assert after == before
    assert after["worker_id"] == new_worker_id
    assert after["fencing_token"] == new_fencing_token
    assert _normalize_datetime(after["lease_until"]) == new_lease_until


async def _prepare_expired_processing_document(
    db,
    resources,
    monkeypatch,
    *,
    worker_id="expired-owner",
    status=DocumentStatus.PROCESSING.value,
    old_worker_id=MISSING,
):
    document, claimed, now, cycle_id = await _claim_processing_document(
        db,
        resources,
        monkeypatch,
        worker_id=worker_id,
        attempt_count=8,
    )
    fields = {
        "lease_until": now - timedelta(seconds=1),
        "heartbeat_at": now - timedelta(seconds=2),
        "updated_at": now - timedelta(seconds=3),
        "next_retry_at": now + timedelta(minutes=3),
        "status": status,
        "upload_owner_id": "upload-owner-preserved",
        "upload_lease_until": now + timedelta(minutes=10),
        "upload_fencing_token": "upload-token-preserved",
    }
    if old_worker_id is None:
        fields["worker_id"] = None
    elif old_worker_id is not MISSING:
        fields["worker_id"] = old_worker_id
    update = {"$set": fields}
    if old_worker_id is MISSING:
        update["$unset"] = {"worker_id": ""}
    result = await db[settings.mongo_collection].update_one(
        {"_id": document["document_id"], "checksum": document["checksum"]},
        update,
    )
    assert result.matched_count == 1
    before = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    return document, claimed, now, cycle_id, before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_expired_processing_persists_new_owner_and_preserves_metadata(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, old_claim, now, cycle_id, before = (
        await _prepare_expired_processing_document(
            db, resources, monkeypatch, old_worker_id="expired-owner"
        )
    )
    new_worker_id = "recovery-owner"

    recovered = await ProcessingRepository(db).recover_expired_processing(
        new_worker_id
    )

    after = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"], "checksum": document["checksum"]}
    )
    assert recovered is not None
    assert recovered.id == str(document["document_id"])
    assert recovered.worker_id == new_worker_id
    assert after["worker_id"] == new_worker_id
    assert recovered.fencing_token != old_claim.fencing_token
    assert after["fencing_token"] != before["fencing_token"]
    assert recovered.fencing_token == after["fencing_token"]
    assert len(uuid.UUID(after["fencing_token"]).hex) == 32
    assert _normalize_datetime(after["lease_until"]) == now + timedelta(seconds=60)
    assert _normalize_datetime(recovered.lease_until) == _normalize_datetime(
        after["lease_until"]
    )
    assert _normalize_datetime(after["heartbeat_at"]) == now
    assert _normalize_datetime(recovered.heartbeat_at) == _normalize_datetime(
        after["heartbeat_at"]
    )
    assert after["status"] == DocumentStatus.PROCESSING.value
    assert after["processing_cycle_id"] == before["processing_cycle_id"] == cycle_id
    assert recovered.processing_cycle_id == after["processing_cycle_id"]
    assert after["processing_cycle_started_at"] == before[
        "processing_cycle_started_at"
    ]
    assert _normalize_datetime(after["processing_cycle_started_at"]) == now
    assert _normalize_datetime(recovered.processing_cycle_started_at) == now
    assert after["attempt_count"] == before["attempt_count"] == 8
    assert recovered.attempt_count == after["attempt_count"]
    assert after["next_retry_at"] == before["next_retry_at"]
    assert _normalize_datetime(after["updated_at"]) == _normalize_datetime(
        before["updated_at"]
    )
    for field in (
        "upload_owner_id",
        "upload_lease_until",
        "upload_fencing_token",
    ):
        assert after[field] == before[field]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_expired_processing_accepts_coherent_exhausted_cycle(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, _, now, cycle_id, before = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="expired-owner"
    )
    started_at = now - timedelta(minutes=11)
    update = await db[settings.mongo_collection].update_one(
        {"_id": document["document_id"]},
        {
            "$set": {
                "attempt_count": 5,
                "processing_cycle_started_at": started_at,
                "last_attempt_at": now - timedelta(seconds=5),
                "next_retry_at": now - timedelta(seconds=1),
            }
        },
    )
    assert update.matched_count == 1

    recovered = await ProcessingRepository(db).recover_expired_processing(
        "recovery-after-budget"
    )

    after = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert recovered is not None
    assert recovered.processing_cycle_id == cycle_id
    assert _normalize_datetime(recovered.processing_cycle_started_at) == started_at
    assert recovered.attempt_count == 5
    assert after["processing_cycle_id"] == before["processing_cycle_id"] == cycle_id
    assert _normalize_datetime(after["processing_cycle_started_at"]) == started_at
    assert after["attempt_count"] == 5
    assert _normalize_datetime(after["last_attempt_at"]) == now - timedelta(seconds=5)
    assert _normalize_datetime(after["next_retry_at"]) == now - timedelta(seconds=1)


@pytest.mark.parametrize(
    ("case", "set_fields", "unset_fields"),
    [
        ("both-absent", {}, ("processing_cycle_id", "processing_cycle_started_at")),
        ("id-without-start", {"processing_cycle_id": "cycle-id"}, ("processing_cycle_started_at",)),
        ("id-with-null-start", {"processing_cycle_started_at": None}, ()),
        ("start-without-id", {"processing_cycle_started_at": FIXED_NOW}, ("processing_cycle_id",)),
        ("start-with-null-id", {"processing_cycle_id": None}, ()),
        ("empty-id", {"processing_cycle_id": ""}, ()),
        ("array-id-containing-string", {"processing_cycle_id": ["cycle-id"]}, ()),
        ("integer-id", {"processing_cycle_id": 7}, ()),
        ("string-date", {"processing_cycle_started_at": "2026-11-25"}, ()),
        ("array-date-containing-date", {"processing_cycle_started_at": [FIXED_NOW]}, ()),
        (
            "future-date",
            {"processing_cycle_started_at": FIXED_NOW + timedelta(seconds=1)},
            (),
        ),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_recovery_rejects_inconsistent_cycles_without_mutation(
    testing_context, monkeypatch, case, set_fields, unset_fields
):
    db, resources = testing_context
    document, _, _, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id=f"expired-owner-{case}"
    )
    collection = db[settings.mongo_collection]
    if set_fields:
        result = await collection.update_one(
            {"_id": document["document_id"]}, {"$set": set_fields}
        )
        assert result.matched_count == 1
    await _unset_raw_fields(db, document["document_id"], *unset_fields)
    before = await collection.find_one({"_id": document["document_id"]})

    try:
        recovered = await ProcessingRepository(db).recover_expired_processing(
            f"recovery-inconsistent-{case}"
        )
    except Exception as error:
        recovered = error

    after = await collection.find_one({"_id": document["document_id"]})
    assert (recovered is None, after == before) == (True, True), (
        f"inconsistent cycle {case!r} must not be recovered or mutated"
    )


@pytest.mark.parametrize(
    ("case", "status", "old_worker_id", "eligible"),
    [
        ("expired-other-worker", DocumentStatus.PROCESSING.value, "old", True),
        ("expired-worker-missing", DocumentStatus.PROCESSING.value, MISSING, True),
        ("expired-worker-null", DocumentStatus.PROCESSING.value, None, True),
        ("expired-same-worker", DocumentStatus.PROCESSING.value, "requester", False),
        ("non-processing", DocumentStatus.PENDING.value, "old", False),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_expired_processing_applies_worker_and_status_eligibility(
    testing_context,
    monkeypatch,
    case,
    status,
    old_worker_id,
    eligible,
):
    db, resources = testing_context
    _, _, now, _, _ = await _prepare_expired_processing_document(
        db,
        resources,
        monkeypatch,
        status=status,
        old_worker_id=old_worker_id,
    )
    requester = "requester" if case == "expired-same-worker" else "new-worker"

    recovered = await ProcessingRepository(db).recover_expired_processing(requester)

    assert (recovered is not None) is eligible
    assert now == FIXED_NOW


@pytest.mark.parametrize("lease_delta", [timedelta(seconds=10), timedelta(0)])
@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_expired_processing_rejects_live_and_exact_boundary_leases(
    testing_context, monkeypatch, lease_delta
):
    db, resources = testing_context
    document, _, now, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="expired-owner"
    )
    collection = db[settings.mongo_collection]
    lease_until = now + lease_delta
    updated = await collection.update_one(
        {"_id": document["document_id"], "checksum": document["checksum"]},
        {"$set": {"lease_until": lease_until}},
    )
    assert updated.matched_count == 1
    before = await collection.find_one({"_id": document["document_id"]})

    recovered = await ProcessingRepository(db).recover_expired_processing(
        "new-worker"
    )

    after = await collection.find_one({"_id": document["document_id"]})
    assert recovered is None
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_recovery_has_one_winner_and_fences_old_owner(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, old_claim, now, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="expired-owner"
    )
    first_repository = ProcessingRepository(db)
    second_repository = ProcessingRepository(db)
    barrier = asyncio.Barrier(2)

    async def recover_after_barrier(repository, worker_id):
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return await repository.recover_expired_processing(worker_id)

    first, second = await asyncio.gather(
        recover_after_barrier(first_repository, "recovery-worker-a"),
        recover_after_barrier(second_repository, "recovery-worker-b"),
    )

    winners = [result for result in (first, second) if result is not None]
    assert len(winners) == 1
    winner = winners[0]
    loser_result = second if first is not None else first
    assert loser_result is None
    assert winner.worker_id in {"recovery-worker-a", "recovery-worker-b"}
    assert winner.fencing_token != old_claim.fencing_token

    collection = db[settings.mongo_collection]
    persisted = await collection.find_one({"_id": document["document_id"]})
    assert persisted["worker_id"] == winner.worker_id
    assert persisted["fencing_token"] == winner.fencing_token
    assert winner.fencing_token == persisted["fencing_token"]
    assert _normalize_datetime(persisted["lease_until"]) == now + timedelta(
        seconds=60
    )
    assert await first_repository.renew_processing_lease(
        old_claim.id,
        old_claim.worker_id,
        old_claim.fencing_token,
    ) is False
    assert await first_repository.renew_processing_lease(
        winner.id,
        winner.worker_id,
        winner.fencing_token,
    ) is True
    assert await second_repository.recover_expired_processing(
        "recovery-worker-loser"
    ) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_expired_processing_selects_oldest_lease_before_creation_order(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)

    newer_document, _, _, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="expired-owner-newer"
    )
    older_document, _, _, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="expired-owner-older"
    )
    collection = db[settings.mongo_collection]
    older_lease = now - timedelta(seconds=20)
    newer_lease = now - timedelta(seconds=5)
    await collection.update_one(
        {"_id": newer_document["document_id"]},
        {"$set": {"lease_until": newer_lease}},
    )
    await collection.update_one(
        {"_id": older_document["document_id"]},
        {"$set": {"lease_until": older_lease}},
    )

    recovered = await ProcessingRepository(db).recover_expired_processing(
        "lease-order-worker"
    )

    assert recovered is not None
    assert recovered.id == str(older_document["document_id"])
    assert _normalize_datetime(recovered.lease_until) == _normalize_datetime(
        now + timedelta(seconds=60)
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_expired_processing_breaks_lease_ties_by_object_id(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)

    first_document, _, _, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="tie-owner-first"
    )
    second_document, _, _, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="tie-owner-second"
    )
    collection = db[settings.mongo_collection]
    tied_lease = now - timedelta(seconds=10)
    for document in (first_document, second_document):
        updated = await collection.update_one(
            {"_id": document["document_id"]},
            {"$set": {"lease_until": tied_lease}},
        )
        assert updated.matched_count == 1

    expected_document_id = min(
        first_document["document_id"],
        second_document["document_id"],
    )
    recovered = await ProcessingRepository(db).recover_expired_processing(
        "lease-tie-worker"
    )

    assert recovered is not None
    assert recovered.id == str(expected_document_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_expired_processing_persists_dollar_prefixed_worker_id(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, _, _, _, _ = await _prepare_expired_processing_document(
        db, resources, monkeypatch, old_worker_id="expired-owner"
    )
    worker_id = "$recovery-worker"

    recovered = await ProcessingRepository(db).recover_expired_processing(worker_id)

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"], "checksum": document["checksum"]}
    )
    assert recovered is not None
    assert recovered.worker_id == worker_id
    assert persisted["worker_id"] == worker_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_authorizes_five_attempts_then_rejects_sixth(
    testing_context, monkeypatch
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(db, resources)
    repository = ProcessingRepository(db)

    returned_counts = []
    for expected_count in range(1, 6):
        result = await repository.record_ms2_attempt(
            str(fixture["document_id"]),
            fixture["worker_id"],
            fixture["fencing_token"],
            fixture["processing_cycle_id"],
        )
        assert result == expected_count
        returned_counts.append(result)

    collection = db[settings.mongo_collection]
    before_sixth = await collection.find_one({"_id": fixture["document_id"]})
    rejected = await repository.record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )
    after_sixth = await collection.find_one({"_id": fixture["document_id"]})

    assert returned_counts == [1, 2, 3, 4, 5]
    assert rejected is None
    assert after_sixth == before_sixth
    assert after_sixth["attempt_count"] == 5
    assert _normalize_datetime(after_sixth["last_attempt_at"]) == now
    assert _normalize_datetime(after_sixth["updated_at"]) == now


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_treats_missing_counter_as_zero(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=MISSING,
    )

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result == 1
    assert persisted["attempt_count"] == 1


@pytest.mark.parametrize(
    ("case", "attempt_count", "expected_type"),
    [
        ("bson-int32", 0, int),
        ("bson-int64", Int64(0), Int64),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_accepts_bson_integer_counters(
    testing_context, monkeypatch, case, attempt_count, expected_type
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=attempt_count,
    )

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result == 1
    assert type(persisted["attempt_count"]) is expected_type
    assert persisted["attempt_count"] == 1


@pytest.mark.parametrize(
    ("case", "fields"),
    [
        ("wrong-status", {"status": DocumentStatus.PENDING.value}),
        ("array-status", {"status": [DocumentStatus.PROCESSING.value]}),
        ("wrong-worker", {"worker_id": "other-worker"}),
        ("array-worker-containing-owner", {"worker_id": ["attempt-worker"]}),
        ("null-worker", {"worker_id": None}),
        ("wrong-fencing-token", {"fencing_token": "stale-token"}),
        (
            "array-fencing-token-containing-current",
            {"fencing_token": ["attempt-fencing-token"]},
        ),
        ("null-fencing-token", {"fencing_token": None}),
        ("wrong-cycle-id", {"processing_cycle_id": "different-cycle"}),
        (
            "array-cycle-id-containing-current",
            {"processing_cycle_id": ["attempt-cycle-id"]},
        ),
        ("empty-cycle-id", {"processing_cycle_id": ""}),
        ("integer-cycle-id", {"processing_cycle_id": 7}),
        ("expired-lease", {"lease_until": FIXED_NOW - timedelta(milliseconds=1)}),
        ("string-lease", {"lease_until": "2026-11-25T12:00:00Z"}),
        ("array-lease-containing-date", {"lease_until": [FIXED_NOW]}),
        ("null-lease", {"lease_until": None}),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_rejects_invalid_ownership_without_mutation(
    testing_context, monkeypatch, case, fields
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=0,
        fields=fields,
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    assert result is None, f"ownership case {case!r} must not authorize an attempt"
    assert after == before


@pytest.mark.parametrize(
    ("case", "started_at", "eligible", "unset_fields"),
    [
        ("cycle-started-now", FIXED_NOW, True, ()),
        (
            "one-millisecond-before-window-end",
            FIXED_NOW - timedelta(minutes=10) + timedelta(milliseconds=1),
            True,
            (),
        ),
        (
            "exact-window-end",
            FIXED_NOW - timedelta(minutes=10),
            False,
            (),
        ),
        (
            "one-millisecond-after-window-end",
            FIXED_NOW - timedelta(minutes=10) - timedelta(milliseconds=1),
            False,
            (),
        ),
        (
            "future-cycle-start",
            FIXED_NOW + timedelta(milliseconds=1),
            False,
            (),
        ),
        ("null-cycle-start", None, False, ()),
        ("wrong-type-cycle-start", "2026-11-25T11:00:00Z", False, ()),
        (
            "array-cycle-start-containing-date",
            [FIXED_NOW - timedelta(minutes=1)],
            False,
            (),
        ),
        ("missing-cycle-start", MISSING, False, ("processing_cycle_started_at",)),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_enforces_cycle_window_boundaries(
    testing_context, monkeypatch, case, started_at, eligible, unset_fields
):
    db, resources = testing_context
    now = _freeze_claim_clock(monkeypatch)
    fields = {}
    if started_at is not MISSING:
        fields["processing_cycle_started_at"] = started_at
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=0,
        fields=fields,
        unset_fields=unset_fields,
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    if eligible:
        assert result == 1, case
        assert after["attempt_count"] == 1
        assert _normalize_datetime(after["last_attempt_at"]) == now
    else:
        assert result is None, case
        assert after == before


@pytest.mark.parametrize(
    ("case", "attempt_count"),
    [
        ("explicit-null", None),
        ("negative", -1),
        ("five", 5),
        ("greater-than-five", 6),
        ("boolean-true", True),
        ("boolean-false", False),
        ("numeric-string", "0"),
        ("double-zero", 0.0),
        ("decimal128-zero", Decimal128("0")),
        ("array-containing-zero", [0]),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_rejects_invalid_counters_without_mutation(
    testing_context, monkeypatch, case, attempt_count
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=attempt_count,
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    assert result is None, f"attempt_count case {case!r} must be rejected"
    assert after == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_attempt_at", "not-a-date"),
        ("last_attempt_at", False),
        ("last_attempt_at", 0),
        ("last_attempt_at", [FIXED_NOW]),
        ("updated_at", "not-a-date"),
        ("updated_at", False),
        ("updated_at", 0),
        ("updated_at", [FIXED_NOW]),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_rejects_invalid_timestamp_types_without_repair(
    testing_context, monkeypatch, field, value
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=0,
        fields={field: value},
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    assert result is None
    assert after == before


@pytest.mark.parametrize(
    ("case", "field", "value", "expected"),
    [
        ("missing-last-attempt", "last_attempt_at", MISSING, FIXED_NOW),
        ("null-last-attempt", "last_attempt_at", None, FIXED_NOW),
        (
            "past-last-attempt",
            "last_attempt_at",
            FIXED_NOW - timedelta(seconds=5),
            FIXED_NOW,
        ),
        ("equal-last-attempt", "last_attempt_at", FIXED_NOW, FIXED_NOW),
        (
            "future-last-attempt",
            "last_attempt_at",
            FIXED_NOW + timedelta(seconds=5),
            FIXED_NOW + timedelta(seconds=5),
        ),
        ("missing-updated-at", "updated_at", MISSING, FIXED_NOW),
        ("null-updated-at", "updated_at", None, FIXED_NOW),
        (
            "past-updated-at",
            "updated_at",
            FIXED_NOW - timedelta(seconds=5),
            FIXED_NOW,
        ),
        ("equal-updated-at", "updated_at", FIXED_NOW, FIXED_NOW),
        (
            "future-updated-at",
            "updated_at",
            FIXED_NOW + timedelta(seconds=5),
            FIXED_NOW + timedelta(seconds=5),
        ),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_preserves_monotonic_valid_timestamps(
    testing_context, monkeypatch, case, field, value, expected
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fields = {} if value is MISSING else {field: value}
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=0,
        fields=fields,
        unset_fields=(field,) if value is MISSING else (),
    )

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result == 1, case
    assert _normalize_datetime(persisted["last_attempt_at"]) == (
        expected if field == "last_attempt_at" else FIXED_NOW
    )
    assert _normalize_datetime(persisted["updated_at"]) == (
        expected if field == "updated_at" else FIXED_NOW
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_does_not_regress_after_older_clock_capture(
    testing_context, monkeypatch
):
    db, resources = testing_context
    later_now = FIXED_NOW + timedelta(seconds=3)
    captured_times = iter((later_now, FIXED_NOW))

    class SequencedDateTime:
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return next(captured_times)

    monkeypatch.setattr(processing_repository, "datetime", SequencedDateTime)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=0,
        fields={
            "lease_until": FIXED_NOW + timedelta(minutes=1),
            "last_attempt_at": FIXED_NOW - timedelta(seconds=2),
            "updated_at": FIXED_NOW - timedelta(seconds=1),
        },
    )
    repository = ProcessingRepository(db)

    first = await repository.record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )
    second = await repository.record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert (first, second) == (1, 2)
    assert _normalize_datetime(persisted["last_attempt_at"]) == later_now
    assert _normalize_datetime(persisted["updated_at"]) == later_now


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_preserves_every_non_authorized_field(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fields = {
        "file_id": ObjectId(),
        "heartbeat_at": FIXED_NOW - timedelta(seconds=8),
        "next_retry_at": FIXED_NOW + timedelta(minutes=2),
        "last_error": "previous error",
        "error_message": "public error",
        "content": "existing content",
        "page_count": 7,
        "completed_at": None,
        "failed_at": None,
        "upload_owner_id": "upload-owner",
        "upload_lease_until": FIXED_NOW + timedelta(minutes=10),
        "upload_fencing_token": "upload-fencing",
    }
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=0,
        fields=fields,
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    authorized_fields = {"attempt_count", "last_attempt_at", "updated_at"}
    assert result == 1
    assert {
        key: value for key, value in after.items() if key not in authorized_fields
    } == {
        key: value for key, value in before.items() if key not in authorized_fields
    }
    assert after["attempt_count"] == 1
    assert _normalize_datetime(after["last_attempt_at"]) == FIXED_NOW
    assert _normalize_datetime(after["updated_at"]) == FIXED_NOW


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_returns_none_for_missing_document(
    testing_context, monkeypatch
):
    db, _ = testing_context
    _freeze_claim_clock(monkeypatch)

    result = await ProcessingRepository(db).record_ms2_attempt(
        str(ObjectId()),
        "worker-a",
        "fencing-a",
        "cycle-a",
    )

    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_ms2_attempt_rejects_stale_owner_after_real_recovery(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, old_claim, now, cycle_id, _ = (
        await _prepare_expired_processing_document(
            db,
            resources,
            monkeypatch,
            old_worker_id="attempt-old-owner",
        )
    )
    collection = db[settings.mongo_collection]
    reset = await collection.update_one(
        {"_id": document["document_id"]},
        {"$set": {"attempt_count": 0}},
    )
    assert reset.matched_count == 1
    repository = ProcessingRepository(db)

    recovered = await repository.recover_expired_processing("attempt-new-owner")
    assert recovered is not None
    assert recovered.fencing_token != old_claim.fencing_token
    assert recovered.processing_cycle_id == cycle_id

    before_stale_attempt = await collection.find_one(
        {"_id": document["document_id"]}
    )
    stale_result = await repository.record_ms2_attempt(
        old_claim.id,
        old_claim.worker_id,
        old_claim.fencing_token,
        cycle_id,
    )
    after_stale_attempt = await collection.find_one(
        {"_id": document["document_id"]}
    )
    new_owner_result = await repository.record_ms2_attempt(
        recovered.id,
        recovered.worker_id,
        recovered.fencing_token,
        recovered.processing_cycle_id,
    )

    persisted = await collection.find_one({"_id": document["document_id"]})
    assert stale_result is None
    assert after_stale_attempt == before_stale_attempt
    assert new_owner_result == 1
    assert persisted["attempt_count"] == 1
    assert _normalize_datetime(persisted["lease_until"]) > now


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_record_ms2_attempts_with_one_slot_have_one_winner(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=4,
    )
    barrier = asyncio.Barrier(2)
    repository = ProcessingRepository(db)

    async def record_after_barrier():
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return await repository.record_ms2_attempt(
            str(fixture["document_id"]),
            fixture["worker_id"],
            fixture["fencing_token"],
            fixture["processing_cycle_id"],
        )

    first, second = await asyncio.gather(
        record_after_barrier(),
        record_after_barrier(),
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert sorted(result for result in (first, second) if result is not None) == [5]
    assert sum(result is None for result in (first, second)) == 1
    assert persisted["attempt_count"] == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_record_ms2_attempts_never_exceed_five(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_attempt_document(
        db,
        resources,
        attempt_count=0,
    )
    barrier = asyncio.Barrier(8)
    repository = ProcessingRepository(db)

    async def record_after_barrier():
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return await repository.record_ms2_attempt(
            str(fixture["document_id"]),
            fixture["worker_id"],
            fixture["fencing_token"],
            fixture["processing_cycle_id"],
        )

    results = await asyncio.gather(
        *(record_after_barrier() for _ in range(8))
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    successful = [result for result in results if result is not None]
    assert sorted(successful) == [1, 2, 3, 4, 5]
    assert sum(result is None for result in results) == 3
    assert persisted["attempt_count"] == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_persists_result_and_keeps_original_gridfs_pdf(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, now, cycle_id, attempt_number = (
        await _claim_and_record_processing_attempt(db, resources, monkeypatch)
    )

    completed = await ProcessingRepository(db).complete_processing(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
        cycle_id,
        attempt_number,
        "Texto extraído del PDF",
        1,
    )

    collection = db[settings.mongo_collection]
    persisted = await collection.find_one({"_id": document["document_id"]})
    gridfs_file = await GridFSRepository(db).stat(document["file_id"])
    assert completed is True
    assert persisted["status"] == DocumentStatus.COMPLETED.value
    assert persisted["content"] == "Texto extraído del PDF"
    assert persisted["page_count"] == 1
    assert _normalize_datetime(persisted["completed_at"]) == now
    assert _normalize_datetime(persisted["updated_at"]) == now
    assert persisted["attempt_count"] == attempt_number == 1
    assert persisted["processing_cycle_id"] == cycle_id
    assert _normalize_datetime(persisted["processing_cycle_started_at"]) == now
    assert _normalize_datetime(persisted["last_attempt_at"]) == now
    assert "lease_until" not in persisted
    assert "next_retry_at" not in persisted
    assert "error_message" not in persisted
    assert "last_error" not in persisted
    assert "failed_at" not in persisted
    assert persisted["worker_id"] == claimed.worker_id
    assert persisted["fencing_token"] == claimed.fencing_token
    assert _normalize_datetime(persisted["heartbeat_at"]) == now
    assert persisted["file_id"] == document["file_id"]
    assert gridfs_file is not None
    assert await GridFSRepository(db).read(document["file_id"]) == document["pdf_bytes"]


@pytest.mark.parametrize(
    ("content", "page_count"),
    [
        ("", 0),
        ("Texto Unicode: café, 東京, 😀", 0),
        ("línea uno\nlínea dos\nlínea tres", 2),
        ("$content debe persistirse literalmente", 1),
        ("$$ROOT tampoco es una expresión", 1),
        ("comillas: ' \" ; dólar: $ ; backslash: \\", 1),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_persists_exact_content_without_normalization(
    testing_context, monkeypatch, content, page_count
):
    db, resources = testing_context
    document, claimed, _, cycle_id, attempt_number = (
        await _claim_and_record_processing_attempt(db, resources, monkeypatch)
    )

    completed = await ProcessingRepository(db).complete_processing(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
        cycle_id,
        attempt_number,
        content,
        page_count,
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert completed is True
    assert persisted["content"] == content
    assert persisted["page_count"] == page_count


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_allows_the_fifth_authorized_attempt(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, now, cycle_id, attempt_number = (
        await _claim_and_record_processing_attempt(
            db,
            resources,
            monkeypatch,
            attempt_count=4,
        )
    )

    completed = await ProcessingRepository(db).complete_processing(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
        cycle_id,
        attempt_number,
        "quinto intento",
        1,
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert completed is True
    assert attempt_number == persisted["attempt_count"] == 5
    assert persisted["content"] == "quinto intento"
    assert _normalize_datetime(persisted["completed_at"]) == now


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_allows_completion_after_ten_minutes_with_live_lease(
    testing_context, monkeypatch
):
    db, resources = testing_context
    completion_now = FIXED_NOW + timedelta(minutes=11)
    document, claimed, _, cycle_id, attempt_number = (
        await _claim_and_record_processing_attempt(
            db,
            resources,
            monkeypatch,
            now=FIXED_NOW,
        )
    )
    lease_until = completion_now + timedelta(seconds=1)
    lease_result = await db[settings.mongo_collection].update_one(
        {"_id": document["document_id"]},
        {"$set": {"lease_until": lease_until}},
    )
    assert lease_result.matched_count == 1

    class CompletionDateTime:
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return completion_now

    monkeypatch.setattr(processing_repository, "datetime", CompletionDateTime)
    completed = await ProcessingRepository(db).complete_processing(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
        cycle_id,
        attempt_number,
        "persistido después de la ventana de autorización",
        1,
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert completed is True
    assert _normalize_datetime(persisted["completed_at"]) == completion_now
    assert persisted["attempt_count"] == attempt_number == 1


@pytest.mark.parametrize(
    ("case", "lease_until", "expected"),
    [
        ("lease-equal-now", FIXED_NOW, True),
        (
            "lease-one-millisecond-expired",
            FIXED_NOW - timedelta(milliseconds=1),
            False,
        ),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_enforces_lease_boundary(
    testing_context, monkeypatch, case, lease_until, expected
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        fields={"lease_until": lease_until},
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        1,
        "resultado",
        1,
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    assert result is expected, case
    if expected:
        assert after["status"] == DocumentStatus.COMPLETED.value
    else:
        assert after == before


@pytest.mark.parametrize(
    ("case", "updated_at", "expected_updated_at"),
    [
        ("missing", MISSING, FIXED_NOW),
        ("null", None, FIXED_NOW),
        (
            "future",
            FIXED_NOW + timedelta(minutes=2),
            FIXED_NOW + timedelta(minutes=2),
        ),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_preserves_monotonic_updated_at(
    testing_context, monkeypatch, case, updated_at, expected_updated_at
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fields = {} if updated_at is MISSING else {"updated_at": updated_at}
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        fields=fields,
        unset_fields=("updated_at",) if updated_at is MISSING else (),
    )

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        1,
        "resultado",
        0,
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is True, case
    assert _normalize_datetime(persisted["updated_at"]) == expected_updated_at
    assert _normalize_datetime(persisted["completed_at"]) == FIXED_NOW


@pytest.mark.parametrize(
    ("case", "fields"),
    [
        ("existing-completed-at", {"completed_at": FIXED_NOW - timedelta(seconds=1)}),
        ("existing-failed-at", {"failed_at": FIXED_NOW - timedelta(seconds=1)}),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_rejects_existing_terminal_timestamps(
    testing_context, monkeypatch, case, fields
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        fields=fields,
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        1,
        "resultado",
        1,
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    assert result is False, case
    assert after == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_accepts_missing_terminal_timestamps(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        attempt_count=1,
        unset_fields=("completed_at", "failed_at"),
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})
    assert "completed_at" not in before
    assert "failed_at" not in before

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        1,
        "resultado sin marcas terminales previas",
        1,
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    assert result is True
    assert after["status"] == DocumentStatus.COMPLETED.value
    assert after["content"] == "resultado sin marcas terminales previas"
    assert _normalize_datetime(after["completed_at"]) == FIXED_NOW
    assert "failed_at" not in after
    assert "lease_until" not in after
    assert after["processing_cycle_id"] == fixture["processing_cycle_id"]
    assert (
        _normalize_datetime(after["processing_cycle_started_at"])
        == _normalize_datetime(before["processing_cycle_started_at"])
    )
    assert after["attempt_count"] == before["attempt_count"] == 1


@pytest.mark.parametrize(
    ("case", "fields", "unset_fields"),
    [
        ("missing-document", {}, ()),
        ("wrong-status", {"status": DocumentStatus.PENDING.value}, ()),
        ("array-status", {"status": [DocumentStatus.PROCESSING.value]}, ()),
        ("wrong-worker", {"worker_id": "other-worker"}, ()),
        ("array-worker", {"worker_id": ["attempt-worker"]}, ()),
        ("null-worker", {"worker_id": None}, ()),
        ("wrong-fencing", {"fencing_token": "old-token"}, ()),
        ("array-fencing", {"fencing_token": ["attempt-fencing-token"]}, ()),
        ("wrong-cycle", {"processing_cycle_id": "old-cycle"}, ()),
        ("empty-cycle", {"processing_cycle_id": ""}, ()),
        ("integer-cycle", {"processing_cycle_id": 8}, ()),
        ("array-cycle", {"processing_cycle_id": ["attempt-cycle-id"]}, ()),
        ("missing-cycle", {}, ("processing_cycle_id",)),
        ("future-cycle-start", {"processing_cycle_started_at": FIXED_NOW + timedelta(milliseconds=1)}, ()),
        ("array-cycle-start", {"processing_cycle_started_at": [FIXED_NOW]}, ()),
        ("string-cycle-start", {"processing_cycle_started_at": "not-a-date"}, ()),
        ("missing-cycle-start", {}, ("processing_cycle_started_at",)),
        ("missing-lease", {}, ("lease_until",)),
        ("null-lease", {"lease_until": None}, ()),
        ("string-lease", {"lease_until": "not-a-date"}, ()),
        ("array-lease", {"lease_until": [FIXED_NOW]}, ()),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_rejects_invalid_ownership_without_mutation(
    testing_context, monkeypatch, case, fields, unset_fields
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    if case == "missing-document":
        document_id = ObjectId()
        fixture = {
            "document_id": document_id,
            "worker_id": "attempt-worker",
            "fencing_token": "attempt-fencing-token",
            "processing_cycle_id": "attempt-cycle-id",
        }
        before = None
    else:
        fixture = await _insert_processing_completion_document(
            db,
            resources,
            fields=fields,
            unset_fields=unset_fields,
        )
        before = await db[settings.mongo_collection].find_one(
            {"_id": fixture["document_id"]}
        )

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        1,
        "resultado",
        1,
    )

    after = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is False, f"case {case!r} must not complete"
    assert after == before


@pytest.mark.parametrize(
    ("case", "attempt_count", "attempt_number"),
    [
        ("older-attempt", 2, 1),
        ("zero-counter", 0, 1),
        ("counter-over-budget", 6, 1),
        ("missing-counter", MISSING, 1),
        ("null-counter", None, 1),
        ("boolean-counter", True, 1),
        ("double-counter", 1.0, 1),
        ("string-counter", "1", 1),
        ("array-counter", [1], 1),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_rejects_nonmatching_attempt_without_mutation(
    testing_context, monkeypatch, case, attempt_count, attempt_number
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        attempt_count=attempt_count,
    )
    before = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        attempt_number,
        "resultado",
        1,
    )

    after = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is False, f"attempt case {case!r} must not complete"
    assert after == before


@pytest.mark.parametrize(
    ("case", "fields", "unset_fields"),
    [
        ("missing-last-attempt", {}, ("last_attempt_at",)),
        ("null-last-attempt", {"last_attempt_at": None}, ()),
        ("string-last-attempt", {"last_attempt_at": "not-a-date"}, ()),
        ("bool-last-attempt", {"last_attempt_at": True}, ()),
        ("number-last-attempt", {"last_attempt_at": 1}, ()),
        ("array-last-attempt", {"last_attempt_at": [FIXED_NOW]}, ()),
        ("array-updated-at", {"updated_at": [FIXED_NOW]}, ()),
        ("string-updated-at", {"updated_at": "not-a-date"}, ()),
        ("bool-updated-at", {"updated_at": False}, ()),
        ("number-updated-at", {"updated_at": 1}, ()),
        ("array-completed-at", {"completed_at": [FIXED_NOW]}, ()),
        ("date-completed-at", {"completed_at": FIXED_NOW}, ()),
        ("array-failed-at", {"failed_at": [FIXED_NOW]}, ()),
        ("date-failed-at", {"failed_at": FIXED_NOW}, ()),
        ("array-next-retry", {"next_retry_at": [FIXED_NOW]}, ()),
        ("string-next-retry", {"next_retry_at": "not-a-date"}, ()),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_rejects_malformed_timestamps_without_mutation(
    testing_context, monkeypatch, case, fields, unset_fields
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        fields=fields,
        unset_fields=unset_fields,
    )
    before = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        1,
        "resultado",
        1,
    )

    after = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is False, f"timestamp case {case!r} must not complete"
    assert after == before


@pytest.mark.parametrize(
    ("case", "attempt_count", "expected_type"),
    [
        ("int32", 1, int),
        ("int64", Int64(1), Int64),
    ],
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_accepts_bson_int32_and_int64_attempt_counts(
    testing_context, monkeypatch, case, attempt_count, expected_type
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        attempt_count=attempt_count,
    )

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        1,
        "resultado",
        0,
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is True, case
    assert type(persisted["attempt_count"]) is expected_type
    assert persisted["attempt_count"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_unsets_exact_fields_and_preserves_the_rest(
    testing_context, monkeypatch
):
    db, resources = testing_context
    _freeze_claim_clock(monkeypatch)
    preserved = {
        "heartbeat_at": FIXED_NOW - timedelta(seconds=4),
        "processing_cycle_started_at": FIXED_NOW - timedelta(minutes=1),
        "last_attempt_at": FIXED_NOW + timedelta(seconds=5),
        "file_id": ObjectId(),
        "custom_metadata": {"nested": ["keep", 1]},
        "upload_owner_id": "upload-owner",
        "upload_lease_until": FIXED_NOW + timedelta(minutes=20),
        "upload_fencing_token": "upload-token",
    }
    removable = {
        "next_retry_at": FIXED_NOW - timedelta(seconds=1),
        "error_message": "old public error",
        "last_error": "old internal error",
    }
    fixture = await _insert_processing_completion_document(
        db,
        resources,
        attempt_count=3,
        fields={**preserved, **removable},
    )
    collection = db[settings.mongo_collection]
    before = await collection.find_one({"_id": fixture["document_id"]})

    result = await ProcessingRepository(db).complete_processing(
        str(fixture["document_id"]),
        fixture["worker_id"],
        fixture["fencing_token"],
        fixture["processing_cycle_id"],
        3,
        "$literal-sensitive result",
        0,
    )

    after = await collection.find_one({"_id": fixture["document_id"]})
    removed = {
        "lease_until",
        "next_retry_at",
        "error_message",
        "last_error",
        "failed_at",
    }
    changed = {
        "status",
        "content",
        "page_count",
        "completed_at",
        "updated_at",
    }
    assert result is True
    assert not (removed & after.keys())
    assert {
        key: value for key, value in after.items() if key not in removed | changed
    } == {
        key: value for key, value in before.items() if key not in removed | changed
    }
    assert after["worker_id"] == fixture["worker_id"]
    assert after["fencing_token"] == fixture["fencing_token"]
    assert _normalize_datetime(after["heartbeat_at"]) == _normalize_datetime(
        preserved["heartbeat_at"]
    )
    assert after["processing_cycle_id"] == fixture["processing_cycle_id"]
    assert after["attempt_count"] == 3
    assert _normalize_datetime(after["last_attempt_at"]) == _normalize_datetime(
        preserved["last_attempt_at"]
    )
    assert after["file_id"] == preserved["file_id"]
    assert after["custom_metadata"] == preserved["custom_metadata"]
    assert after["upload_owner_id"] == preserved["upload_owner_id"]
    assert _normalize_datetime(after["upload_lease_until"]) == _normalize_datetime(
        preserved["upload_lease_until"]
    )
    assert after["upload_fencing_token"] == preserved["upload_fencing_token"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_processing_repeat_does_not_overwrite_confirmed_result(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, _, cycle_id, attempt_number = (
        await _claim_and_record_processing_attempt(db, resources, monkeypatch)
    )
    repository = ProcessingRepository(db)

    first = await repository.complete_processing(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
        cycle_id,
        attempt_number,
        "resultado ganador",
        1,
    )
    before_repeat = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    second = await repository.complete_processing(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
        cycle_id,
        attempt_number,
        "resultado repetido diferente",
        99,
    )
    after_repeat = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )

    assert first is True
    assert second is False
    assert after_repeat == before_repeat


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_completions_have_one_winner_and_do_not_overwrite(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, _, cycle_id, attempt_number = (
        await _claim_and_record_processing_attempt(db, resources, monkeypatch)
    )
    repository = ProcessingRepository(db)
    barrier = asyncio.Barrier(2)

    async def complete(content, page_count):
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return await repository.complete_processing(
            claimed.id,
            claimed.worker_id,
            claimed.fencing_token,
            cycle_id,
            attempt_number,
            content,
            page_count,
        )

    results = await asyncio.gather(
        complete("resultado A", 1),
        complete("resultado B", 2),
    )

    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    assert sorted(results) == [False, True]
    expected_result = (
        ("resultado A", 1)
        if results[0] is True
        else ("resultado B", 2)
    )
    assert (
        persisted["content"],
        persisted["page_count"],
    ) == expected_result
    assert persisted["status"] == DocumentStatus.COMPLETED.value
    assert persisted["attempt_count"] == attempt_number


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recovery_fences_old_completion_and_allows_new_owner_after_attempt(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, old_claim, now, cycle_id = await _claim_processing_document(
        db,
        resources,
        monkeypatch,
        worker_id="completion-old-owner",
        attempt_count=0,
    )
    repository = ProcessingRepository(db)
    old_attempt = await repository.record_ms2_attempt(
        old_claim.id,
        old_claim.worker_id,
        old_claim.fencing_token,
        cycle_id,
    )
    assert old_attempt == 1
    expired = await db[settings.mongo_collection].update_one(
        {"_id": document["document_id"]},
        {"$set": {"lease_until": now - timedelta(milliseconds=1)}},
    )
    assert expired.matched_count == 1

    recovered = await repository.recover_expired_processing("completion-new-owner")
    assert recovered is not None
    assert recovered.fencing_token != old_claim.fencing_token
    before_stale_completion = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )

    old_result = await repository.complete_processing(
        old_claim.id,
        old_claim.worker_id,
        old_claim.fencing_token,
        cycle_id,
        old_attempt,
        "resultado antiguo",
        1,
    )
    after_stale_completion = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    new_attempt = await repository.record_ms2_attempt(
        recovered.id,
        recovered.worker_id,
        recovered.fencing_token,
        recovered.processing_cycle_id,
    )
    new_result = await repository.complete_processing(
        recovered.id,
        recovered.worker_id,
        recovered.fencing_token,
        recovered.processing_cycle_id,
        new_attempt,
        "resultado propietario nuevo",
        1,
    )
    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )

    assert old_result is False
    assert after_stale_completion == before_stale_completion
    assert new_attempt == 2
    assert new_result is True
    assert persisted["content"] == "resultado propietario nuevo"
    assert persisted["attempt_count"] == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_heartbeat_racing_completion_cannot_mutate_completed_document(
    testing_context, monkeypatch
):
    db, resources = testing_context
    document, claimed, _, cycle_id, attempt_number = (
        await _claim_and_record_processing_attempt(db, resources, monkeypatch)
    )
    repository = ProcessingRepository(db)
    barrier = asyncio.Barrier(2)

    async def renew():
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return await repository.renew_processing_lease(
            claimed.id,
            claimed.worker_id,
            claimed.fencing_token,
        )

    async def complete():
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return await repository.complete_processing(
            claimed.id,
            claimed.worker_id,
            claimed.fencing_token,
            cycle_id,
            attempt_number,
            "resultado concurrente",
            1,
        )

    heartbeat_result, completion_result = await asyncio.gather(renew(), complete())
    persisted_after_race = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )
    heartbeat_after_completion = await repository.renew_processing_lease(
        claimed.id,
        claimed.worker_id,
        claimed.fencing_token,
    )
    persisted_after_late_heartbeat = await db[settings.mongo_collection].find_one(
        {"_id": document["document_id"]}
    )

    assert heartbeat_result in (True, False)
    assert completion_result is True
    assert persisted_after_race["status"] == DocumentStatus.COMPLETED.value
    assert "lease_until" not in persisted_after_race
    assert heartbeat_after_completion is False
    assert persisted_after_late_heartbeat == persisted_after_race
