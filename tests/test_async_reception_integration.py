"""HTTP integration tests for durable asynchronous PDF reception."""

import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from httpx import ASGITransport, AsyncClient

from src.api.docs_api import get_document_service
from src.api.pdf_validation import PdfValidationExecutor
from src.config.config import settings
from src.core.exceptions import DatabaseUnavailableError
from src.database.mongodb import MongoDB
from src.main import app
from src.models.document import DocumentStatus
from src.repository.document_repository import DocumentRepository
from src.repository.gridfs_repository import GridFSRepository
from src.schemas.document import DocumentCreate
from src.services.document_service import DocumentService


def _make_pdf(token: str) -> bytes:
    """Build a small structurally valid one-page PDF without shared parser state."""
    stream = f"BT /F1 12 Tf 72 720 Td ({token}) Tj ET\n".encode("ascii")
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


def _bson_utc_datetime(value: datetime) -> datetime:
    """Normalize a MongoDB datetime to aware UTC at BSON millisecond precision."""
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=(value.microsecond // 1000) * 1000)


@asynccontextmanager
async def _isolated_http_environment():
    """Own a real testing DB, API executor and narrowly scoped cleanup."""
    testing_db = MongoDB.for_testing()
    checksums: set[str] = set()
    executor = PdfValidationExecutor()
    previous_executor = getattr(app.state, "pdf_validation_executor", None)
    previous_marker = object()
    previous_override = app.dependency_overrides.get(
        get_document_service, previous_marker
    )

    async def cleanup():
        collection = testing_db.db[settings.mongo_collection]
        for checksum in checksums:
            documents = await collection.find(
                {"checksum": checksum}
            ).to_list(length=None)
            file_ids = {
                doc["file_id"]
                for doc in documents
                if isinstance(doc.get("file_id"), ObjectId)
            }
            gridfs_files = await testing_db.db["fs.files"].find(
                {"metadata.checksum": checksum}, {"_id": 1}
            ).to_list(length=None)
            file_ids.update(file["_id"] for file in gridfs_files)
            for file_id in file_ids:
                # Delete only the files tied to this test's unique checksum.
                await testing_db.db["fs.files"].delete_one({"_id": file_id})
                await testing_db.db["fs.chunks"].delete_many(
                    {"files_id": file_id}
                )
            await collection.delete_many({"checksum": checksum})

    async def finish(original_error=None):
        errors = []

        async def attempt(operation):
            try:
                await operation()
            except BaseException as error:
                errors.append(error)

        if testing_db.connected and testing_db.db is not None:
            await attempt(cleanup)
        await attempt(testing_db.disconnect)

        def restore_override():
            if previous_override is previous_marker:
                app.dependency_overrides.pop(get_document_service, None)
            else:
                app.dependency_overrides[get_document_service] = previous_override

        def restore_executor_state():
            app.state.pdf_validation_executor = previous_executor

        async def shutdown_executor():
            executor.shutdown(wait=True, cancel_futures=True)

        for operation in (restore_override, restore_executor_state):
            try:
                operation()
            except BaseException as error:
                errors.append(error)
        await attempt(shutdown_executor)

        if original_error is not None:
            for error in errors:
                original_error.add_note(
                    f"La limpieza/restauración también falló: {error!r}"
                )
            return
        if errors:
            first_error, *other_errors = errors
            for error in other_errors:
                first_error.add_note(
                    f"Otra operación de limpieza/restauración falló: {error!r}"
                )
            raise first_error

    try:
        app.state.pdf_validation_executor = executor
        await testing_db.connect()
        repository = DocumentRepository(testing_db.db)
        service = DocumentService(repository=repository)
        app.dependency_overrides[get_document_service] = lambda: service
        yield testing_db, repository, service, checksums
    except BaseException as original_error:
        await finish(original_error)
        raise
    else:
        await finish()


async def _post(client, filename: str, payload: bytes):
    return await client.post(
        "/documents/",
        files={"file": (filename, payload, "application/pdf")},
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_new_reception_then_duplicate_is_durable_and_idempotent():
    async with _isolated_http_environment() as (db, repository, service, checksums):
        token = uuid.uuid4().hex
        pdf_bytes = _make_pdf(token)
        checksum = hashlib.sha256(pdf_bytes).hexdigest()
        checksums.add(checksum)
        gridfs = GridFSRepository(db.db)
        extraction = AsyncMock(
            side_effect=AssertionError("PDF reception must not call MS2")
        )
        service.call_extraction_service = extraction
        process_extraction = AsyncMock(
            side_effect=AssertionError("PDF reception must not start extraction")
        )
        service.process_pdf_extraction = process_extraction
        save_count = 0
        original_save = gridfs.save

        async def count_real_save(*args, **kwargs):
            nonlocal save_count
            save_count += 1
            return await original_save(*args, **kwargs)

        repository._gridfs_repository = lambda: gridfs
        gridfs.save = count_real_save

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            accepted = await _post(client, f"first-{token}.pdf", pdf_bytes)
            first_document = await repository.collection.find_one(
                {"checksum": checksum}
            )
            assert first_document is not None
            original_file_id = first_document["file_id"]
            original_created_at = first_document["created_at"]
            duplicate = await _post(client, f"second-{token}.pdf", pdf_bytes)

        assert accepted.status_code == 202
        assert duplicate.status_code == 200
        accepted_ticket = accepted.json()
        duplicate_ticket = duplicate.json()
        assert accepted_ticket["status"] == "pending"
        assert accepted_ticket["document_id"] == duplicate_ticket["document_id"]
        assert accepted_ticket["created_at"] == duplicate_ticket["created_at"]
        assert duplicate_ticket["status"] == "pending"
        assert accepted.headers["location"] == accepted_ticket["status_url"]
        assert duplicate.headers["location"] == duplicate_ticket["status_url"]
        assert accepted_ticket["status_url"] == (
            f"/documents/{accepted_ticket['document_id']}"
        )
        assert set(accepted_ticket) == {
            "document_id", "status", "created_at", "status_url"
        }

        docs = await repository.collection.find(
            {"checksum": checksum}
        ).to_list(length=None)
        assert len(docs) == 1
        document = docs[0]
        assert document["_id"] == ObjectId(accepted_ticket["document_id"])
        assert document["filename"] == f"first-{token}.pdf"
        assert document["created_at"] == original_created_at
        assert document["file_id"] == original_file_id
        assert document["status"] == DocumentStatus.PENDING.value
        assert document["checksum"] == checksum
        assert document["size_bytes"] == len(pdf_bytes)
        assert isinstance(document["file_id"], ObjectId)
        assert await db.db["fs.files"].count_documents(
            {
                "metadata.document_id": str(document["_id"]),
                "metadata.checksum": checksum,
            }
        ) == 1
        info = await gridfs.stat(document["file_id"])
        stored_bytes = await gridfs.read(document["file_id"])
        assert info is not None
        assert info.length == len(pdf_bytes)
        assert stored_bytes == pdf_bytes
        assert hashlib.sha256(stored_bytes).hexdigest() == checksum
        assert save_count == 1
        extraction.assert_not_awaited()
        process_extraction.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("mime", 400),
        ("empty", 400),
        ("oversized", 413),
        ("signature", 422),
        ("structure", 422),
    ],
)
async def test_http_invalid_uploads_create_no_persistent_artifacts(case, expected_status):
    async with _isolated_http_environment() as (db, repository, _service, checksums):
        token = uuid.uuid4().hex
        valid_pdf = _make_pdf(token)
        payload = {
            "mime": valid_pdf,
            "empty": b"",
            "oversized": b"%PDF-1.4\n" + b"x" * (
                settings.max_file_size_mb * 1024 * 1024 + 1
            ),
            "signature": f"not-a-pdf-{token}".encode(),
            "structure": b"%PDF-1.7\ninvalid structure " + token.encode(),
        }[case]
        checksum = hashlib.sha256(payload).hexdigest()
        checksums.add(checksum)
        content_type = "text/plain" if case == "mime" else "application/pdf"
        original_reserve = repository.reserve_document
        reserve_spy = AsyncMock(wraps=original_reserve)
        repository.reserve_document = reserve_spy
        gridfs = GridFSRepository(db.db)
        save_spy = AsyncMock(wraps=gridfs.save)
        gridfs.save = save_spy
        repository._gridfs_repository = lambda: gridfs

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/documents/",
                files={"file": (f"{case}-{token}.pdf", payload, content_type)},
            )

        assert response.status_code == expected_status
        reserve_spy.assert_not_awaited()
        save_spy.assert_not_awaited()
        assert await repository.collection.count_documents(
            {"checksum": checksum}
        ) == 0
        assert await db.db["fs.files"].count_documents(
            {"metadata.checksum": checksum}
        ) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_recovers_expired_upload_reservation_without_file():
    async with _isolated_http_environment() as (db, repository, service, checksums):
        token = uuid.uuid4().hex
        pdf_bytes = _make_pdf(token)
        checksum = hashlib.sha256(pdf_bytes).hexdigest()
        checksums.add(checksum)
        original_owner = f"abandoned-owner-{token}"
        original_fence = f"abandoned-fence-{token}"
        reservation, created = await repository.reserve_document(
            DocumentCreate(
                filename=f"original-{token}.pdf",
                checksum=checksum,
                size_bytes=len(pdf_bytes),
            )
        )
        assert created is True
        original_document = await repository.collection.find_one(
            {"_id": ObjectId(reservation.id)}
        )
        assert original_document is not None
        original_created_at = _bson_utc_datetime(
            original_document["created_at"]
        )
        await repository.collection.update_one(
            {"_id": ObjectId(reservation.id)},
            {
                "$set": {
                    "upload_owner_id": original_owner,
                    "upload_fencing_token": original_fence,
                    "upload_lease_until": datetime.now(timezone.utc)
                    - timedelta(minutes=1),
                    "file_id": None,
                }
            },
        )
        service.call_extraction_service = AsyncMock(
            side_effect=AssertionError("PDF reception must not call MS2")
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await _post(client, f"retry-{token}.pdf", pdf_bytes)

        assert response.status_code == 202
        ticket = response.json()
        status_url = f"/documents/{reservation.id}"
        expected_created_at = (
            original_created_at.isoformat().replace("+00:00", "Z")
        )
        assert ticket == {
            "document_id": reservation.id,
            "status": "pending",
            "created_at": expected_created_at,
            "status_url": status_url,
        }
        assert response.headers["location"] == status_url
        persisted = await repository.collection.find_one(
            {"checksum": checksum}
        )
        assert persisted["_id"] == ObjectId(reservation.id)
        assert _bson_utc_datetime(persisted["created_at"]) == original_created_at
        assert persisted["status"] == DocumentStatus.PENDING.value
        assert persisted["upload_owner_id"] != original_owner
        assert persisted["upload_fencing_token"] != original_fence
        assert isinstance(persisted["file_id"], ObjectId)
        gridfs = GridFSRepository(db.db)
        info = await gridfs.stat(persisted["file_id"])
        stored = await gridfs.read(persisted["file_id"])
        assert info is not None and info.length == len(pdf_bytes)
        assert stored == pdf_bytes
        assert hashlib.sha256(stored).hexdigest() == checksum
        assert await repository.collection.count_documents(
            {"checksum": checksum}
        ) == 1
        assert await db.db["fs.files"].count_documents(
            {"metadata.checksum": checksum}
        ) == 1
        service.call_extraction_service.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_database_unavailable_during_reception_returns_503():
    async with _isolated_http_environment() as (_db, _repository, service, _checksums):
        token = uuid.uuid4().hex
        pdf_bytes = _make_pdf(token)
        service.repository.reserve_document = AsyncMock(
            side_effect=DatabaseUnavailableError()
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await _post(client, f"{token}.pdf", pdf_bytes)
        assert response.status_code == 503
        service.repository.reserve_document.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_rejected_finalization_keeps_unassociated_gridfs_file():
    async with _isolated_http_environment() as (db, repository, _service, checksums):
        token = uuid.uuid4().hex
        pdf_bytes = _make_pdf(token)
        checksum = hashlib.sha256(pdf_bytes).hexdigest()
        checksums.add(checksum)
        gridfs = GridFSRepository(db.db)
        repository._gridfs_repository = lambda: gridfs
        finalize_calls = 0
        original_finalize = repository.finalize_upload

        async def reject_after_real_storage(*args, **kwargs):
            nonlocal finalize_calls
            finalize_calls += 1
            assert await gridfs.stat(args[3]) is not None
            return False

        repository.finalize_upload = reject_after_real_storage
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await _post(client, f"{token}.pdf", pdf_bytes)

        assert response.status_code == 503
        assert finalize_calls == 1
        doc = await repository.collection.find_one({"checksum": checksum})
        assert doc is not None
        assert doc["status"] == DocumentStatus.UPLOADING.value
        assert doc["file_id"] is None
        files = await db.db["fs.files"].find(
            {"metadata.checksum": checksum}
        ).to_list(length=None)
        assert len(files) == 1
        assert await gridfs.read(files[0]["_id"]) == pdf_bytes
        # Explicitly verify no compensating delete occurred; cleanup is test-owned.
        assert await db.db["fs.files"].count_documents(
            {"_id": files[0]["_id"]}
        ) == 1
        repository.finalize_upload = original_finalize


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_uncertain_finalization_keeps_real_success_without_retry():
    async with _isolated_http_environment() as (db, repository, _service, checksums):
        token = uuid.uuid4().hex
        pdf_bytes = _make_pdf(token)
        checksum = hashlib.sha256(pdf_bytes).hexdigest()
        checksums.add(checksum)
        gridfs = GridFSRepository(db.db)
        repository._gridfs_repository = lambda: gridfs
        original_finalize = repository.finalize_upload
        finalize_calls = 0

        async def apply_then_lose_confirmation(*args, **kwargs):
            nonlocal finalize_calls
            finalize_calls += 1
            applied = await original_finalize(*args, **kwargs)
            assert applied is True
            raise DatabaseUnavailableError()

        repository.finalize_upload = apply_then_lose_confirmation
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await _post(client, f"{token}.pdf", pdf_bytes)

        assert response.status_code == 503
        assert finalize_calls == 1
        doc = await repository.collection.find_one({"checksum": checksum})
        assert doc is not None
        assert doc["status"] == DocumentStatus.PENDING.value
        assert isinstance(doc["file_id"], ObjectId)
        assert await gridfs.read(doc["file_id"]) == pdf_bytes
        assert hashlib.sha256(await gridfs.read(doc["file_id"])).hexdigest() == checksum
        assert await db.db["fs.files"].count_documents(
            {"metadata.checksum": checksum}
        ) == 1
        # One invocation proves the endpoint did not retry the uncertain write.
        assert finalize_calls == 1
