"""Pruebas RED de lectura de PDFs del worker contra MongoDB/GridFS reales."""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from bson import ObjectId

from src.config.config import MONGO_TEST_DATABASE, settings
from src.database.mongodb import MongoDB
from src.models.document import DocumentStatus
from src.repository.document_repository import DocumentRepository
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import ProcessingRepository
from src.schemas.document import DocumentCreate
from src.worker.pdf_reader import read_claimed_pdf


@pytest_asyncio.fixture
async def testing_context():
    """Conecta a pdf_database_test y limpia solo los recursos del fixture."""
    manager = MongoDB.for_testing()
    resources: list[tuple[ObjectId, str, ObjectId]] = []
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
                        await gridfs.delete(file_id)
                    finally:
                        await collection.delete_one(
                            {"_id": document_id, "checksum": checksum}
                        )
        finally:
            await manager.disconnect()


def _build_minimal_pdf(unique_text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({unique_text}) Tj ET\n".encode("ascii")
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_reads_claimed_pdf_bytes_from_real_gridfs(testing_context):
    db, resources = testing_context
    token = uuid.uuid4().hex
    pdf_bytes = _build_minimal_pdf(f"worker-gridfs-{token}")
    checksum = hashlib.sha256(pdf_bytes).hexdigest()
    filename = f"worker-gridfs-{token}.pdf"
    file_id = ObjectId()

    documents = DocumentRepository(db)
    gridfs = GridFSRepository(db)
    processing = ProcessingRepository(db)
    reservation, created = await documents.reserve_document(
        DocumentCreate(
            filename=filename,
            checksum=checksum,
            size_bytes=len(pdf_bytes),
        )
    )
    assert created is True
    document_id = ObjectId(reservation.id)
    resources.append((document_id, checksum, file_id))

    upload_owner_id = f"worker-gridfs-owner-{token}"
    upload_fencing_token = str(uuid.uuid4())
    upload_lease_until = datetime.now(timezone.utc) + timedelta(minutes=15)
    _, acquired = await documents.acquire_upload_ownership(
        checksum,
        upload_owner_id,
        upload_fencing_token,
        upload_lease_until,
    )
    assert acquired is True

    stored_file_id = await gridfs.save(
        file_id,
        filename,
        pdf_bytes,
        metadata={"checksum": checksum},
    )
    assert stored_file_id == file_id
    finalized = await documents.finalize_upload(
        reservation.id,
        upload_owner_id,
        upload_fencing_token,
        stored_file_id,
    )
    assert finalized is True

    claimed = await processing.claim_pending("worker-gridfs-reader")
    assert claimed is not None
    assert claimed.status is DocumentStatus.PROCESSING
    assert claimed.file_id == file_id

    recovered_bytes = await read_claimed_pdf(
        claimed,
        processing_repository=processing,
        gridfs_repository=gridfs,
    )

    assert recovered_bytes == pdf_bytes
    assert hashlib.sha256(recovered_bytes).hexdigest() == checksum
    file_info = await gridfs.stat(file_id)
    assert file_info is not None
    assert file_info.length == len(pdf_bytes)
    persisted = await db[settings.mongo_collection].find_one(
        {"_id": document_id, "checksum": checksum}
    )
    assert persisted is not None
    assert persisted["status"] == DocumentStatus.PROCESSING.value
    assert persisted["file_id"] == file_id
