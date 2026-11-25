"""Integración del flujo conjunto de persistencia de una carga PDF."""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from bson import ObjectId

from src.config.config import settings
from src.database.mongodb import MongoDB
from src.models.document import DocumentStatus
from src.repository.document_repository import DocumentRepository
from src.repository.gridfs_repository import GridFSRepository
from src.schemas.document import DocumentCreate
from src.services.document_service import DocumentService


@pytest_asyncio.fixture
async def _isolated_testing_db():
    """Conecta a la base de testing y garantiza la desconexión."""
    testing_db = MongoDB.for_testing()
    try:
        await testing_db.connect()
        yield testing_db
    finally:
        await testing_db.disconnect()


def _build_minimal_pdf(unique_content: str) -> bytes:
    """Genera un PDF mínimo válido con texto único en su página."""
    stream = (
        f"BT /F1 12 Tf 72 720 Td ({unique_content}) Tj ET\n"
    ).encode("ascii")
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
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
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


def _normalize_mongo_datetime(value: datetime) -> datetime:
    """Normaliza fechas BSON a UTC y precisión de milisegundos."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=(value.microsecond // 1000) * 1000)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_persistence_flow_with_real_mongodb_and_gridfs(
    _isolated_testing_db,
):
    """Reserva, almacena, renueva, finaliza y recupera el PDF original."""
    testing_db = _isolated_testing_db
    checksum_token = str(uuid.uuid4())
    file_id = ObjectId()
    collection = testing_db.db[settings.mongo_collection]
    repository = DocumentRepository(testing_db.db)
    service = DocumentService(repository=repository)
    gridfs_repository = GridFSRepository(testing_db.db)

    pdf_bytes = _build_minimal_pdf(f"upload-flow-{checksum_token}")
    checksum = hashlib.sha256(pdf_bytes).hexdigest()
    filename = f"upload-flow-{checksum_token}.pdf"
    document_filter = {"checksum": checksum}

    try:
        reservation, created = await repository.reserve_document(
            DocumentCreate(
                filename=filename,
                checksum=checksum,
                size_bytes=len(pdf_bytes),
            )
        )
        assert created is True
        assert reservation.id is not None

        persisted = await collection.find_one(document_filter)
        assert persisted is not None
        assert persisted["_id"] == ObjectId(reservation.id)
        assert persisted["status"] == DocumentStatus.UPLOADING.value
        assert persisted["file_id"] is None

        upload_owner_id = str(uuid.uuid4())
        upload_fencing_token = str(uuid.uuid4())
        initial_lease_until = datetime.now(timezone.utc) + timedelta(
            minutes=15
        )
        owned_document, acquired = await repository.acquire_upload_ownership(
            checksum,
            upload_owner_id,
            upload_fencing_token,
            initial_lease_until,
        )
        assert acquired is True
        assert owned_document is not None
        assert owned_document.id == reservation.id

        stored_file_id = await service.save_original_pdf(
            file_id=file_id,
            filename=filename,
            data=pdf_bytes,
            metadata={"checksum": checksum},
        )
        assert stored_file_id == file_id

        after_gridfs_save = await collection.find_one(document_filter)
        assert after_gridfs_save is not None
        assert after_gridfs_save["_id"] == ObjectId(reservation.id)
        assert after_gridfs_save["status"] == DocumentStatus.UPLOADING.value
        assert after_gridfs_save["file_id"] is None

        renewed_lease_until = (
            _normalize_mongo_datetime(
                after_gridfs_save["upload_lease_until"]
            )
            + timedelta(minutes=15)
        )
        renewed = await repository.renew_upload_lease(
            reservation.id,
            upload_owner_id,
            upload_fencing_token,
            renewed_lease_until,
        )
        assert renewed is True

        finalized = await repository.finalize_upload(
            reservation.id,
            upload_owner_id,
            upload_fencing_token,
            stored_file_id,
        )
        assert finalized is True

        persisted_final = await collection.find_one(document_filter)
        assert persisted_final is not None
        assert persisted_final["_id"] == ObjectId(reservation.id)
        assert persisted_final["status"] == DocumentStatus.PENDING.value
        assert isinstance(persisted_final["file_id"], ObjectId)
        assert persisted_final["file_id"] == stored_file_id
        assert persisted_final["checksum"] == checksum
        assert persisted_final["size_bytes"] == len(pdf_bytes)
        assert persisted_final["upload_owner_id"] == upload_owner_id
        assert (
            persisted_final["upload_fencing_token"]
            == upload_fencing_token
        )
        assert _normalize_mongo_datetime(
            persisted_final["upload_lease_until"]
        ) == _normalize_mongo_datetime(renewed_lease_until)
        assert await collection.count_documents(document_filter) == 1

        recovered_pdf = await gridfs_repository.read(stored_file_id)
        assert recovered_pdf == pdf_bytes
        assert hashlib.sha256(recovered_pdf).hexdigest() == (
            persisted_final["checksum"]
        )
    finally:
        try:
            await gridfs_repository.delete(file_id)
        finally:
            await collection.delete_one({"checksum": checksum})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_stores_authorized_upload_durably_without_finalizing(
    _isolated_testing_db,
):
    """Reserva, adquiere y almacena con el servicio sin asociar ni finalizar."""
    testing_db = _isolated_testing_db
    db = testing_db.db
    collection = db[settings.mongo_collection]
    repository = DocumentRepository(db)
    service = DocumentService(repository=repository)
    gridfs_repository = GridFSRepository(db)

    unique_token = str(uuid.uuid4())
    pdf_bytes = _build_minimal_pdf(f"durable-upload-{unique_token}")
    checksum = hashlib.sha256(pdf_bytes).hexdigest()
    filename = f"durable-upload-{unique_token}.pdf"
    document_id = None
    file_id = None

    service.call_extraction_service = AsyncMock()
    service.process_pdf_extraction = AsyncMock()

    try:
        document, classification = await service.reserve_or_classify(
            DocumentCreate(
                filename=filename,
                checksum=checksum,
                size_bytes=len(pdf_bytes),
            )
        )
        if classification == "created":
            document_id = document.id

        assert classification == "created"
        assert document_id is not None
        assert document.status == DocumentStatus.UPLOADING
        assert document.file_id is None

        owned_document, acquired = await service.acquire_upload_ownership(
            document,
            settings.upload_lease_seconds,
        )

        assert acquired is True
        assert owned_document is not None
        assert owned_document.id == document_id
        assert owned_document.upload_owner_id
        assert owned_document.upload_fencing_token
        assert owned_document.upload_lease_until is not None
        lease_until = _normalize_mongo_datetime(
            owned_document.upload_lease_until
        )
        assert lease_until > datetime.now(timezone.utc)

        # El ID se crea para este intento, después de confirmar la adquisición.
        file_id = ObjectId()
        assert isinstance(file_id, ObjectId)
        metadata = {
            "document_id": owned_document.id,
            "checksum": owned_document.checksum,
            "upload_owner_id": owned_document.upload_owner_id,
            "upload_fencing_token": owned_document.upload_fencing_token,
        }
        assert metadata == {
            "document_id": document_id,
            "checksum": checksum,
            "upload_owner_id": owned_document.upload_owner_id,
            "upload_fencing_token": owned_document.upload_fencing_token,
        }

        stored_file_id = await service.save_original_pdf(
            file_id=file_id,
            filename=filename,
            data=pdf_bytes,
            metadata=metadata,
        )

        assert stored_file_id == file_id
        file_info = await gridfs_repository.stat(file_id)
        assert file_info is not None
        assert file_info.length == len(pdf_bytes)
        assert file_info.metadata == metadata

        recovered_pdf = await gridfs_repository.read(file_id)
        assert recovered_pdf == pdf_bytes
        assert hashlib.sha256(recovered_pdf).hexdigest() == checksum

        persisted_document = await collection.find_one(
            {"_id": ObjectId(document_id)}
        )
        assert persisted_document is not None
        assert persisted_document["status"] == DocumentStatus.UPLOADING.value
        assert persisted_document["file_id"] is None
        assert persisted_document["checksum"] == checksum

        service.call_extraction_service.assert_not_awaited()
        service.process_pdf_extraction.assert_not_awaited()
    finally:
        try:
            if file_id is not None:
                await db["fs.files"].delete_one({"_id": file_id})
        finally:
            try:
                if file_id is not None:
                    await db["fs.chunks"].delete_many({"files_id": file_id})
            finally:
                if document_id is not None:
                    await collection.delete_one(
                        {"_id": ObjectId(document_id)}
                    )
