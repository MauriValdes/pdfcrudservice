"""End-to-end worker pipeline checks against isolated test MongoDB."""

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import AutoReconnect

from src.config.config import MONGO_TEST_DATABASE, settings
from src.database.mongodb import MongoDB
from src.repository.document_repository import DocumentRepository
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import ProcessingRepository
from src.schemas.document import DocumentCreate
from src.worker.document_processor import (
    ProcessingResult,
    process_claimed_document,
)
from src.worker.circuit_breaker import CircuitBreaker, CircuitBreakerState
from src.worker.ms2_client import MS2Client, MS2HTTPError
from src.worker.worker_lifecycle import WorkerLifecycle
from src.worker.worker_runner import ProcessingWorker


_EXPECTED_TEST_USER = "pdf_ms3_test"
_MOCK_MS2_BASE_URL = "http://ms2.test"
_SUCCESS_CONTENT = "integral worker extraction"
_SUCCESS_PAGE_COUNT = 3
_FIXTURE_TIMEOUT_SECONDS = 8


def _bson_utc(value: datetime) -> datetime:
    """Normalize MongoDB's timezone-naive UTC dates for assertions."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _current_bson_utc() -> datetime:
    """Capture one UTC instant at MongoDB's persisted millisecond precision."""
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=(now.microsecond // 1000) * 1000)


def _freeze_worker_clock(monkeypatch, now: datetime):
    """Freeze repository and processor time while retaining the real retry policy."""
    from src.repository import processing_repository as repository_module
    from src.worker import document_processor as processor_module

    clock = SimpleNamespace(now=now)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return clock.now

    monkeypatch.setattr(repository_module, "datetime", FrozenDateTime)
    monkeypatch.setattr(processor_module, "_utc_now", lambda: clock.now)
    real_plan_retry = processor_module.plan_retry

    def deterministic_plan_retry(*args, **kwargs):
        kwargs.setdefault("uniform", lambda lower, _upper: lower)
        return real_plan_retry(*args, **kwargs)

    monkeypatch.setattr(
        processor_module,
        "plan_retry",
        deterministic_plan_retry,
    )
    return clock


class _ControlledMonotonicClock:
    """Mutable monotonic source injected through the public breaker API."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def _advance_worker_clocks(wall_clock, monotonic_clock, seconds: float) -> None:
    """Advance the independent UTC and monotonic test clocks coherently."""
    wall_clock.now += timedelta(seconds=seconds)
    monotonic_clock.value += seconds


def _build_valid_pdf(unique_text: str) -> bytes:
    """Build a minimal, valid PDF with unique content for this test fixture."""
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


def _assert_test_uri(uri: str) -> None:
    """Validate the dedicated testing identity without disclosing the URI."""
    parsed = urlsplit(uri)
    database_name = unquote(parsed.path.lstrip("/").split("/", 1)[0])
    username = unquote(parsed.username or "")
    auth_source = parse_qs(parsed.query).get("authSource", [None])[0]
    assert database_name == MONGO_TEST_DATABASE == "pdf_database_test"
    assert username == _EXPECTED_TEST_USER
    assert auth_source == MONGO_TEST_DATABASE


def _verify_test_identity_and_role(uri: str) -> None:
    """Read-only identity check before MongoDB.for_testing() can write indexes."""
    _assert_test_uri(uri)
    client = MongoClient(uri, w=1, journal=True, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        auth_info = client.admin.command(
            {"connectionStatus": 1, "showPrivileges": True}
        )["authInfo"]
        assert {
            "user": _EXPECTED_TEST_USER,
            "db": MONGO_TEST_DATABASE,
        } in auth_info["authenticatedUsers"]
        assert auth_info["authenticatedUserRoles"] == [
            {"role": "readWrite", "db": MONGO_TEST_DATABASE}
        ]
        assert client.write_concern.document == {"w": 1, "j": True}
    finally:
        client.close()


@pytest_asyncio.fixture
async def worker_integration_context():
    """Open only the test database and clean only IDs created by this fixture."""
    manager = MongoDB.for_testing()
    assert manager._testing is True
    assert manager._database_name == MONGO_TEST_DATABASE == "pdf_database_test"
    _verify_test_identity_and_role(manager._mongo_url)

    resources: list[tuple[ObjectId, str, ObjectId]] = []
    connected = False
    initial_gridfs_counts = None
    initial_document_count = None
    try:
        await manager.connect()
        connected = True
        assert manager._testing is True
        assert manager.db is not None
        assert manager.db.name == MONGO_TEST_DATABASE
        assert manager.client.write_concern.document == {"w": 1, "j": True}
        assert manager.db.write_concern.document == {"w": 1, "j": True}

        collection = manager.db[settings.mongo_collection]
        initial_document_count = await collection.count_documents({})
        now = datetime.now(timezone.utc)
        assert await collection.count_documents({"status": "pending"}) == 0
        assert (
            await collection.count_documents(
                {"status": "processing", "lease_until": {"$lt": now}}
            )
            == 0
        )
        required_indexes = {
            "checksum_unique",
            "pending_jobs_idx",
            "expired_leases_idx",
        }
        assert required_indexes.issubset(
            await collection.index_information()
        )

        initial_gridfs_counts = {
            name: await manager.db[name].count_documents({})
            for name in ("fs.files", "fs.chunks")
        }
        gridfs = GridFSRepository(manager.db)
        assert initial_gridfs_counts == {"fs.files": 0, "fs.chunks": 0}
        context = SimpleNamespace(
            manager=manager,
            db=manager.db,
            collection=collection,
            gridfs=gridfs,
            resources=resources,
            initial_gridfs_counts=initial_gridfs_counts,
            initial_document_count=initial_document_count,
        )
        yield context
    finally:
        cleanup_errors = []
        try:
            if connected:
                assert manager._testing is True
                assert manager.db is not None
                assert manager.db.name == MONGO_TEST_DATABASE
                collection = manager.db[settings.mongo_collection]
                gridfs = GridFSRepository(manager.db)
                for document_id, checksum, file_id in reversed(resources):
                    try:
                        await asyncio.wait_for(
                            gridfs.delete(file_id),
                            timeout=_FIXTURE_TIMEOUT_SECONDS,
                        )
                    except Exception as error:
                        cleanup_errors.append(error)
                    try:
                        await asyncio.wait_for(
                            collection.delete_one(
                                {"_id": document_id, "checksum": checksum}
                            ),
                            timeout=_FIXTURE_TIMEOUT_SECONDS,
                        )
                    except Exception as error:
                        cleanup_errors.append(error)
                    for verification in (
                        collection.find_one(
                            {"_id": document_id, "checksum": checksum}
                        ),
                        manager.db["fs.files"].find_one({"_id": file_id}),
                        manager.db["fs.chunks"].find_one(
                            {"files_id": file_id}
                        ),
                    ):
                        try:
                            assert await verification is None
                        except Exception as error:
                            cleanup_errors.append(error)

                final_gridfs_counts = {
                    name: await manager.db[name].count_documents({})
                    for name in ("fs.files", "fs.chunks")
                }
                final_document_count = await collection.count_documents({})
                if (
                    initial_document_count is not None
                    and final_document_count != initial_document_count
                ):
                    cleanup_errors.append(
                        AssertionError(
                            "Worker integration fixture document count changed"
                        )
                    )
                if (
                    initial_gridfs_counts is not None
                    and final_gridfs_counts != initial_gridfs_counts
                ):
                    cleanup_errors.append(
                        AssertionError(
                            "GridFS fixture counts changed after cleanup"
                        )
                    )
        except Exception as error:
            cleanup_errors.append(error)
        finally:
            try:
                await manager.disconnect()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise ExceptionGroup("MongoDB fixture cleanup failures", cleanup_errors)


async def _create_pending_pdf(context, label: str, *, fields=None) -> dict:
    """Persist one unique PDF by using the upload ownership/finalization APIs."""
    token = uuid4().hex
    pdf_bytes = _build_valid_pdf(f"worker-u86-{label}-{token}")
    checksum = hashlib.sha256(pdf_bytes).hexdigest()
    filename = f"worker-u86-{label}-{token}.pdf"
    file_id = ObjectId()

    documents = DocumentRepository(context.db)
    reservation, created = await documents.reserve_document(
        DocumentCreate(
            filename=filename,
            checksum=checksum,
            size_bytes=len(pdf_bytes),
        )
    )
    assert created is True
    document_id = ObjectId(reservation.id)
    context.resources.append((document_id, checksum, file_id))

    owner_id = f"worker-u86-upload-{token}"
    upload_fencing_token = str(uuid4())
    upload_lease_until = datetime.now(timezone.utc) + timedelta(minutes=15)
    _, acquired = await documents.acquire_upload_ownership(
        checksum,
        owner_id,
        upload_fencing_token,
        upload_lease_until,
    )
    assert acquired is True
    assert (
        await context.gridfs.save(
            file_id,
            filename,
            pdf_bytes,
            metadata={"checksum": checksum},
        )
        == file_id
    )
    assert await documents.finalize_upload(
        reservation.id,
        owner_id,
        upload_fencing_token,
        file_id,
    )

    if fields:
        result = await context.collection.update_one(
            {"_id": document_id, "checksum": checksum, "status": "pending"},
            {"$set": fields},
        )
        assert result.matched_count == 1

    return {
        "document_id": document_id,
        "checksum": checksum,
        "file_id": file_id,
        "filename": filename,
        "pdf_bytes": pdf_bytes,
    }


async def _process_with_breaker(
    context,
    repository,
    document,
    client,
    circuit_breaker,
    *,
    busy_retry_seconds=0.01,
):
    """Exercise the production processor with real repos, breaker and client."""
    return await process_claimed_document(
        document,
        processing_repository=repository,
        gridfs_repository=context.gridfs,
        ms2_client=MS2Client(
            client,
            base_url=_MOCK_MS2_BASE_URL,
            timeout_seconds=2.0,
        ),
        circuit_breaker=circuit_breaker,
        circuit_breaker_busy_retry_seconds=busy_retry_seconds,
    )


async def _open_breaker_with_integrated_failures(
    context,
    repository,
    client,
    circuit_breaker,
    requests,
    label,
):
    """Open through five real, eligible MS2 failures on distinct documents."""
    fixtures = []
    for index in range(5):
        fixture = await _create_pending_pdf(
            context,
            f"{label}-seed-{index}",
        )
        fixtures.append(fixture)
        claimed = await repository.claim_pending(
            f"worker-u452-{label}-seed-{index}"
        )
        assert claimed is not None
        assert claimed.id == str(fixture["document_id"])
        request_count_before = len(requests)

        outcome = await _process_with_breaker(
            context,
            repository,
            claimed,
            client,
            circuit_breaker,
        )

        assert outcome is ProcessingResult.RETRY_SCHEDULED
        assert len(requests) == request_count_before + 1
        persisted = await context.collection.find_one(
            {"_id": fixture["document_id"]}
        )
        assert persisted["status"] == "pending"
        assert persisted["attempt_count"] == 1
        assert persisted["last_error"] == "ms2_http_503"
        assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
        assert (
            _bson_utc(persisted["processing_cycle_started_at"])
            == _bson_utc(claimed.processing_cycle_started_at)
        )
        assert persisted.get("worker_id") is None
        assert persisted.get("fencing_token") is None
        assert persisted.get("lease_until") is None
        assert persisted.get("heartbeat_at") is None
        assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]

    assert circuit_breaker.state is CircuitBreakerState.OPEN
    return fixtures


async def _assert_no_foreign_eligible_work(context, owned_document_ids):
    """Assert all pending/expired work belongs to the current test fixture."""
    foreign = {"_id": {"$nin": list(owned_document_ids)}}
    now = datetime.now(timezone.utc)
    assert (
        await context.collection.count_documents(
            {"status": "pending", **foreign}
        )
        == 0
    )
    assert (
        await context.collection.count_documents(
            {
                "status": "processing",
                "lease_until": {"$lt": now},
                **foreign,
            }
        )
        == 0
    )


class _AmbiguousTransitionRepository:
    """Wrap a real repository and lose one selected write acknowledgement."""

    def __init__(
        self,
        repository,
        *,
        operation,
        apply_before_raise,
        before_write=None,
    ):
        self._repository = repository
        self._operation = operation
        self._apply_before_raise = apply_before_raise
        self._before_write = before_write
        self.write_attempts = 0
        self.real_write_results = []
        self.write_arguments = None
        self.observation_reads = 0
        self.exhaustion_writes = 0

    def __getattr__(self, name):
        return getattr(self._repository, name)

    async def _write_with_lost_acknowledgement(self, operation, *args):
        if operation != self._operation:
            return await getattr(self._repository, operation)(*args)

        self.write_attempts += 1
        self.write_arguments = args
        if self._before_write is not None:
            await self._before_write()
        if self._apply_before_raise:
            result = await getattr(self._repository, operation)(*args)
            self.real_write_results.append(result)
        raise AutoReconnect("controlled acknowledgement loss")

    async def complete_processing(self, *args):
        return await self._write_with_lost_acknowledgement(
            "complete_processing",
            *args,
        )

    async def schedule_transient_retry(self, *args):
        return await self._write_with_lost_acknowledgement(
            "schedule_transient_retry",
            *args,
        )

    async def observe_transition_state(self, *args, **kwargs):
        self.observation_reads += 1
        return await self._repository.observe_transition_state(
            *args,
            **kwargs,
        )

    async def fail_exhausted_processing(self, *args, **kwargs):
        self.exhaustion_writes += 1
        return await self._repository.fail_exhausted_processing(
            *args,
            **kwargs,
        )


def _assert_ambiguous_transition_diagnostic(
    caplog,
    *,
    resolution,
    operation,
    document_id,
):
    records = [
        record
        for record in caplog.records
        if hasattr(record, "transition_resolution")
    ]
    assert len(records) == 1
    record = records[0]
    assert record.transition_resolution == resolution
    assert record.transition_operation == operation
    assert record.document_id == str(document_id)
    assert record.exception_type == "AutoReconnect"
    assert "controlled acknowledgement loss" not in caplog.text


async def _wait_for_tasks(tasks, timeout=_FIXTURE_TIMEOUT_SECONDS):
    """Wait without implicit cancellation; explicitly cancel and collect on timeout."""
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        for task in pending:
            if not task.done() and not task.cancelling():
                task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if pending:
        raise TimeoutError(
            f"{len(pending)} task(s) exceeded the {timeout}s safety timeout"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ambiguous_completion_after_real_write_is_observed_unconfirmed(
    worker_integration_context,
    monkeypatch,
    caplog,
):
    context = worker_integration_context
    _freeze_worker_clock(monkeypatch, _current_bson_utc())
    fixture = await _create_pending_pdf(context, "ambiguous-complete-applied")
    repository = ProcessingRepository(context.db)
    claimed = await repository.claim_pending(
        f"worker-u461-complete-{uuid4()}"
    )
    assert claimed is not None
    assert claimed.id == str(fixture["document_id"])
    await _assert_no_foreign_eligible_work(
        context,
        {fixture["document_id"]},
    )

    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "content": _SUCCESS_CONTENT,
                "page_count": _SUCCESS_PAGE_COUNT,
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    controlled_repository = _AmbiguousTransitionRepository(
        repository,
        operation="complete_processing",
        apply_before_raise=True,
    )
    try:
        result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                controlled_repository,
                claimed,
                client,
                CircuitBreaker(),
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
    finally:
        await client.aclose()

    persisted = await context.collection.find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is ProcessingResult.RESULT_NOT_CONFIRMED
    assert persisted["status"] == "completed"
    assert persisted["content"] == _SUCCESS_CONTENT
    assert persisted["page_count"] == _SUCCESS_PAGE_COUNT
    assert persisted["attempt_count"] == 1
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert persisted["worker_id"] == claimed.worker_id
    assert persisted["fencing_token"] == claimed.fencing_token
    assert persisted["original_cleanup_status"] == "pending"
    assert controlled_repository.write_attempts == 1
    assert controlled_repository.real_write_results == [True]
    assert controlled_repository.observation_reads == 1
    assert controlled_repository.exhaustion_writes == 0
    assert len(requests) == 1
    _assert_ambiguous_transition_diagnostic(
        caplog,
        resolution="OBSERVED_UNCONFIRMED",
        operation="complete_processing",
        document_id=fixture["document_id"],
    )
    assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ambiguous_completion_not_applied_remains_recoverable(
    worker_integration_context,
    monkeypatch,
    caplog,
):
    context = worker_integration_context
    clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    fixture = await _create_pending_pdf(context, "ambiguous-complete-not-applied")
    repository = ProcessingRepository(context.db)
    claimed = await repository.claim_pending(
        f"worker-u461-complete-uncertain-{uuid4()}"
    )
    assert claimed is not None
    assert claimed.id == str(fixture["document_id"])

    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"content": _SUCCESS_CONTENT, "page_count": _SUCCESS_PAGE_COUNT},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    controlled_repository = _AmbiguousTransitionRepository(
        repository,
        operation="complete_processing",
        apply_before_raise=False,
    )
    try:
        result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                controlled_repository,
                claimed,
                client,
                CircuitBreaker(),
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
    finally:
        await client.aclose()

    persisted = await context.collection.find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is ProcessingResult.RESULT_NOT_CONFIRMED
    assert persisted["status"] == "processing"
    assert persisted["attempt_count"] == 1
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert persisted["worker_id"] == claimed.worker_id
    assert persisted["fencing_token"] == claimed.fencing_token
    assert persisted.get("content") is None
    assert persisted.get("page_count") is None
    assert controlled_repository.write_attempts == 1
    assert controlled_repository.real_write_results == []
    assert controlled_repository.observation_reads == 1
    assert controlled_repository.exhaustion_writes == 0
    assert len(requests) == 1
    _assert_ambiguous_transition_diagnostic(
        caplog,
        resolution="UNKNOWN",
        operation="complete_processing",
        document_id=fixture["document_id"],
    )
    assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]

    clock.now += timedelta(seconds=61)
    successor_worker_id = f"worker-u461-recovered-{uuid4()}"
    recovered = await repository.recover_expired_processing(
        successor_worker_id
    )
    assert recovered is not None
    assert recovered.id == claimed.id
    assert recovered.worker_id == successor_worker_id
    assert recovered.fencing_token != claimed.fencing_token
    assert recovered.processing_cycle_id == claimed.processing_cycle_id
    assert recovered.attempt_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ambiguous_retry_after_real_write_is_observed_unconfirmed(
    worker_integration_context,
    monkeypatch,
    caplog,
):
    context = worker_integration_context
    clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    fixture = await _create_pending_pdf(context, "ambiguous-retry-applied")
    repository = ProcessingRepository(context.db)
    claimed = await repository.claim_pending(
        f"worker-u461-retry-{uuid4()}"
    )
    assert claimed is not None
    assert claimed.id == str(fixture["document_id"])

    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            503,
            headers={"Retry-After": "0"},
            json={"detail": "private simulated upstream failure"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    controlled_repository = _AmbiguousTransitionRepository(
        repository,
        operation="schedule_transient_retry",
        apply_before_raise=True,
    )
    try:
        result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                controlled_repository,
                claimed,
                client,
                CircuitBreaker(),
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
    finally:
        await client.aclose()

    persisted = await context.collection.find_one(
        {"_id": fixture["document_id"]}
    )
    requested_retry_at = controlled_repository.write_arguments[-1]
    cycle_deadline = _bson_utc(
        claimed.processing_cycle_started_at
    ) + timedelta(minutes=10)
    bson_requested = _bson_utc(requested_retry_at)
    bson_requested = bson_requested.replace(
        microsecond=(bson_requested.microsecond // 1000) * 1000
    )
    cycle_deadline = cycle_deadline.replace(
        microsecond=(cycle_deadline.microsecond // 1000) * 1000
    )
    effective_retry_at = min(bson_requested, cycle_deadline)

    assert result is ProcessingResult.RESULT_NOT_CONFIRMED
    assert persisted["status"] == "pending"
    assert persisted["attempt_count"] == 1
    assert persisted["last_error"] == "ms2_http_503"
    assert _bson_utc(persisted["next_retry_at"]) == effective_retry_at
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert _bson_utc(persisted["processing_cycle_started_at"]) == _bson_utc(
        claimed.processing_cycle_started_at
    )
    assert persisted.get("worker_id") is None
    assert persisted.get("fencing_token") is None
    assert controlled_repository.write_attempts == 1
    assert controlled_repository.real_write_results == [True]
    assert controlled_repository.observation_reads == 1
    assert controlled_repository.exhaustion_writes == 0
    assert len(requests) == 1
    _assert_ambiguous_transition_diagnostic(
        caplog,
        resolution="OBSERVED_UNCONFIRMED",
        operation="schedule_transient_retry",
        document_id=fixture["document_id"],
    )
    assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]
    assert clock.now < _bson_utc(persisted["next_retry_at"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ambiguous_completion_observes_successor_after_real_recovery(
    worker_integration_context,
    monkeypatch,
    caplog,
):
    context = worker_integration_context
    clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    fixture = await _create_pending_pdf(context, "ambiguous-successor-owner")
    repository = ProcessingRepository(context.db)
    claimed = await repository.claim_pending(
        f"worker-u461-original-{uuid4()}"
    )
    assert claimed is not None
    assert claimed.id == str(fixture["document_id"])
    successor_worker_id = f"worker-u461-successor-{uuid4()}"
    recovered_claims = []

    async def recover_expired_claim():
        clock.now += timedelta(seconds=61)
        successor = await repository.recover_expired_processing(
            successor_worker_id
        )
        assert successor is not None
        recovered_claims.append(successor)

    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"content": _SUCCESS_CONTENT, "page_count": _SUCCESS_PAGE_COUNT},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    controlled_repository = _AmbiguousTransitionRepository(
        repository,
        operation="complete_processing",
        apply_before_raise=True,
        before_write=recover_expired_claim,
    )
    try:
        result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                controlled_repository,
                claimed,
                client,
                CircuitBreaker(),
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
    finally:
        await client.aclose()

    assert len(recovered_claims) == 1
    recovered = recovered_claims[0]
    assert recovered.id == claimed.id
    assert recovered.worker_id == successor_worker_id
    assert recovered.fencing_token != claimed.fencing_token
    assert recovered.processing_cycle_id == claimed.processing_cycle_id
    assert recovered.attempt_count == 1

    persisted = await context.collection.find_one(
        {"_id": fixture["document_id"]}
    )
    assert result is ProcessingResult.RESULT_NOT_CONFIRMED
    assert persisted["status"] == "processing"
    assert persisted["worker_id"] == successor_worker_id
    assert persisted["fencing_token"] == recovered.fencing_token
    assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
    assert persisted["attempt_count"] == 1
    assert persisted.get("content") is None
    assert persisted.get("page_count") is None
    assert _bson_utc(persisted["lease_until"]) > clock.now
    assert controlled_repository.write_attempts == 1
    assert controlled_repository.real_write_results == [False]
    assert controlled_repository.observation_reads == 1
    assert controlled_repository.exhaustion_writes == 0
    assert len(requests) == 1
    _assert_ambiguous_transition_diagnostic(
        caplog,
        resolution="UNKNOWN",
        operation="complete_processing",
        document_id=fixture["document_id"],
    )
    assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifecycle_completes_active_work_before_orderly_shutdown(
    worker_integration_context,
):
    context = worker_integration_context
    first = await _create_pending_pdf(context, "lifecycle-first")
    second = await _create_pending_pdf(context, "lifecycle-second")
    owned_document_ids = {first["document_id"], second["document_id"]}
    await _assert_no_foreign_eligible_work(context, owned_document_ids)
    request_entered = asyncio.Event()
    release_response = asyncio.Event()
    processing_finished = asyncio.Event()
    worker_stop_seen = asyncio.Event()
    close_order = []
    requests = []
    workers = []
    worker_managers = []
    clients = []

    async def transport_handler(request):
        requests.append((request, request.content))
        request_entered.set()
        await release_response.wait()
        return httpx.Response(
            200,
            json={
                "content": _SUCCESS_CONTENT,
                "page_count": _SUCCESS_PAGE_COUNT,
            },
        )

    def mongo_factory():
        worker_manager = MongoDB.for_testing()
        worker_managers.append(worker_manager)
        assert worker_manager._testing is True
        assert (
            worker_manager._database_name
            == MONGO_TEST_DATABASE
            == "pdf_database_test"
        )
        _verify_test_identity_and_role(worker_manager._mongo_url)
        original_close = worker_manager.close

        async def observe_close():
            close_order.append("mongo.close")
            await original_close()

        worker_manager.close = observe_close
        return worker_manager

    def http_client_factory():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(transport_handler)
        )
        clients.append(client)
        original_aclose = client.aclose

        async def observe_aclose():
            close_order.append("http.close")
            await original_aclose()

        client.aclose = observe_aclose
        return client

    async def observed_processor(document, **dependencies):
        result = await process_claimed_document(document, **dependencies)
        processing_finished.set()
        return result

    def worker_factory(*args, **kwargs):
        worker = ProcessingWorker(
            *args,
            processor=observed_processor,
            max_concurrency=1,
            poll_initial_seconds=0.01,
            poll_max_seconds=0.05,
            **kwargs,
        )
        original_request_stop = worker.request_stop

        def observe_stop():
            original_request_stop()
            worker_stop_seen.set()

        worker.request_stop = observe_stop
        workers.append(worker)
        return worker

    lifecycle = WorkerLifecycle(
        mongo_factory=mongo_factory,
        http_client_factory=http_client_factory,
        worker_factory=worker_factory,
        ms2_base_url=_MOCK_MS2_BASE_URL,
        ms2_timeout_seconds=2.0,
        shutdown_grace_seconds=2.0,
    )
    stop_event = asyncio.Event()
    lifecycle_task = asyncio.create_task(lifecycle.run(stop_event))

    try:
        await asyncio.wait_for(
            request_entered.wait(), timeout=_FIXTURE_TIMEOUT_SECONDS
        )
        request, body = requests[0]
        assert request.method == "POST"
        assert request.url.path == "/extract"
        assert request.headers["content-type"].startswith(
            "multipart/form-data;"
        )
        assert b'name="file"' in body
        assert f'filename="{first["filename"]}"'.encode() in body
        assert b"application/pdf" in body
        assert first["pdf_bytes"] in body
        assert len(requests) == 1

        stop_event.set()
        await asyncio.wait_for(
            worker_stop_seen.wait(), timeout=_FIXTURE_TIMEOUT_SECONDS
        )
        assert not lifecycle_task.done()
        assert close_order == []
        first_before_release = await context.collection.find_one(
            {"_id": first["document_id"]}
        )
        second_before_release = await context.collection.find_one(
            {"_id": second["document_id"]}
        )
        assert first_before_release["status"] == "processing"
        assert second_before_release["status"] == "pending"
        await _assert_no_foreign_eligible_work(
            context, owned_document_ids
        )

        release_response.set()
        await asyncio.wait_for(
            processing_finished.wait(), timeout=_FIXTURE_TIMEOUT_SECONDS
        )
        await _wait_for_tasks([lifecycle_task])
        lifecycle_task.result()

        persisted = await context.collection.find_one(
            {"_id": first["document_id"]}
        )
        waiting = await context.collection.find_one(
            {"_id": second["document_id"]}
        )
        assert len(requests) == 1
        assert persisted["status"] == "completed"
        assert persisted["content"] == _SUCCESS_CONTENT
        assert persisted["page_count"] == _SUCCESS_PAGE_COUNT
        assert persisted["attempt_count"] == 1
        assert isinstance(persisted["processing_cycle_id"], str)
        assert persisted["processing_cycle_id"]
        assert isinstance(persisted["processing_cycle_started_at"], datetime)
        assert persisted["worker_id"]
        assert persisted["fencing_token"]
        assert persisted["original_cleanup_status"] == "pending"
        assert waiting["status"] == "pending"
        assert waiting.get("worker_id") is None
        assert await context.gridfs.stat(first["file_id"]) is not None
        assert close_order == [
            "http.close",
            "mongo.close",
        ]
        assert clients[0].is_closed
        assert worker_managers[0].client is None
        assert worker_managers[0].db is None
    finally:
        release_response.set()
        stop_event.set()
        if workers:
            workers[0].request_stop()
        try:
            await _wait_for_tasks([lifecycle_task])
        finally:
            for client in clients:
                if not client.is_closed:
                    await client.aclose()
            for worker_manager in worker_managers:
                if worker_manager.client is not None:
                    await worker_manager.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_real_workers_observe_per_worker_capacity_and_unique_claims(
    worker_integration_context,
):
    context = worker_integration_context
    documents = [
        await _create_pending_pdf(context, f"parallel-{index}")
        for index in range(5)
    ]
    owned_document_ids = {item["document_id"] for item in documents}
    await _assert_no_foreign_eligible_work(context, owned_document_ids)
    first_four_started = asyncio.Event()
    release_responses = asyncio.Event()
    all_processed = asyncio.Event()
    in_flight_http = 0
    peak_http = 0
    request_count = 0
    active_by_worker = [0, 0]
    peak_by_worker = [0, 0]
    observed_worker_ids = [set(), set()]
    claimed_ids = [[], []]
    completed_ids = []
    workers = []
    tasks = []
    clients = []

    async def transport_handler(request):
        nonlocal in_flight_http, peak_http, request_count
        request_count += 1
        in_flight_http += 1
        peak_http = max(peak_http, in_flight_http)
        if request_count == 4:
            first_four_started.set()
        try:
            await release_responses.wait()
            return httpx.Response(
                200,
                json={
                    "content": "parallel result",
                    "page_count": 1,
                },
            )
        finally:
            in_flight_http -= 1

    processing_repository = ProcessingRepository(context.db)
    gridfs_repository = GridFSRepository(context.db)

    def make_observer(worker_index):
        async def observe(document, **dependencies):
            active_by_worker[worker_index] += 1
            peak_by_worker[worker_index] = max(
                peak_by_worker[worker_index],
                active_by_worker[worker_index],
            )
            observed_worker_ids[worker_index].add(document.worker_id)
            claimed_ids[worker_index].append(document.id)
            try:
                result = await process_claimed_document(
                    document,
                    **dependencies,
                )
                completed_ids.append(document.id)
                if len(completed_ids) == len(documents):
                    all_processed.set()
                return result
            finally:
                active_by_worker[worker_index] -= 1

        return observe

    for worker_index in range(2):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(transport_handler)
        )
        clients.append(client)
        worker = ProcessingWorker(
            processing_repository,
            gridfs_repository,
            MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=CircuitBreaker(),
            processor=make_observer(worker_index),
            max_concurrency=2,
            poll_initial_seconds=0.01,
            poll_max_seconds=0.05,
        )
        workers.append(worker)
        tasks.append(asyncio.create_task(worker.run()))

    try:
        await asyncio.wait_for(
            first_four_started.wait(), timeout=_FIXTURE_TIMEOUT_SECONDS
        )
        await _assert_no_foreign_eligible_work(
            context, owned_document_ids
        )
        assert request_count == 4
        assert in_flight_http == 4
        assert peak_http == 4
        assert peak_by_worker == [2, 2]
        worker_ids = set.union(*observed_worker_ids)
        assert len(worker_ids) == 2
        assert worker_ids.isdisjoint({""})
        assert set(claimed_ids[0]).isdisjoint(set(claimed_ids[1]))

        fixture_ids = [item["document_id"] for item in documents]
        states = await context.collection.find(
            {"_id": {"$in": fixture_ids}},
            {"_id": 1, "status": 1},
        ).to_list(length=len(fixture_ids))
        assert Counter(record["status"] for record in states) == {
            "processing": 4,
            "pending": 1,
        }

        release_responses.set()
        await asyncio.wait_for(
            all_processed.wait(), timeout=_FIXTURE_TIMEOUT_SECONDS
        )
        for worker in workers:
            worker.request_stop()
        await _wait_for_tasks(tasks)
        for task in tasks:
            task.result()

        assert request_count == 5
        assert peak_http == 4
        assert len(completed_ids) == 5
        assert Counter(claimed_ids[0] + claimed_ids[1]) == Counter(
            str(document_id) for document_id in fixture_ids
        )
        assert Counter(completed_ids) == Counter(
            str(document_id) for document_id in fixture_ids
        )
        for fixture in documents:
            persisted = await context.collection.find_one(
                {"_id": fixture["document_id"]}
            )
            assert persisted["status"] == "completed"
            assert persisted["attempt_count"] == 1
    finally:
        release_responses.set()
        for worker in workers:
            worker.request_stop()
        try:
            await _wait_for_tasks(tasks)
        finally:
            for client in clients:
                if not client.is_closed:
                    await client.aclose()


@pytest.mark.integration
@pytest.mark.parametrize(
    "outcome",
    ["permanent", "exhausted", "transient"],
)
@pytest.mark.asyncio
async def test_processor_persists_only_authorized_failure_outcomes(
    worker_integration_context,
    outcome,
    monkeypatch,
):
    context = worker_integration_context
    fields = None
    if outcome == "exhausted":
        now = datetime.now(timezone.utc)
        fields = {
            "processing_cycle_id": str(uuid4()),
            "processing_cycle_started_at": now - timedelta(minutes=11),
            "attempt_count": 1,
            "last_attempt_at": now - timedelta(minutes=10),
        }
    fixture = await _create_pending_pdf(
        context,
        f"failure-{outcome}",
        fields=fields,
    )
    await _assert_no_foreign_eligible_work(
        context, {fixture["document_id"]}
    )
    requests = []

    async def transport_handler(request):
        requests.append(request)
        if outcome == "permanent":
            return httpx.Response(422, json={"detail": "private upstream detail"})
        if outcome == "transient":
            return httpx.Response(503, json={"detail": "private upstream detail"})
        return httpx.Response(
            200,
            json={"content": "unexpected response", "page_count": 1},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    repository = ProcessingRepository(context.db)
    try:
        claimed = await repository.claim_pending(f"worker-u86-{outcome}")
        assert claimed is not None
        assert claimed.id == str(fixture["document_id"])
        await _assert_no_foreign_eligible_work(
            context, {fixture["document_id"]}
        )
        processor = process_claimed_document(
            claimed,
            processing_repository=repository,
            gridfs_repository=context.gridfs,
            ms2_client=MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=CircuitBreaker(),
            circuit_breaker_busy_retry_seconds=0.01,
        )

        if outcome == "transient":
            from src.worker import document_processor as processor_module

            planned_at = datetime.now(timezone.utc)
            real_plan_retry = processor_module.plan_retry

            def deterministic_plan_retry(*args, **kwargs):
                kwargs.setdefault("uniform", lambda lower, _upper: lower)
                return real_plan_retry(*args, **kwargs)

            monkeypatch.setattr(
                processor_module,
                "_utc_now",
                lambda: planned_at,
            )
            monkeypatch.setattr(
                processor_module,
                "plan_retry",
                deterministic_plan_retry,
            )
            result = await asyncio.wait_for(
                processor, timeout=_FIXTURE_TIMEOUT_SECONDS
            )
        else:
            result = await asyncio.wait_for(
                processor, timeout=_FIXTURE_TIMEOUT_SECONDS
            )

        persisted = await context.collection.find_one(
            {"_id": fixture["document_id"]}
        )
        assert await context.gridfs.stat(fixture["file_id"]) is not None
        if outcome == "permanent":
            assert result is ProcessingResult.FAILED_PERMANENT
            assert len(requests) == 1
            assert persisted["status"] == "failed"
            assert persisted["attempt_count"] == 1
            assert persisted["last_error"] == "pdf_unprocessable"
            assert persisted["error_message"] == (
                "No se pudo procesar el archivo PDF."
            )
            assert "private upstream detail" not in persisted["error_message"]
            assert "content" not in persisted
            assert "page_count" not in persisted
            assert persisted["original_cleanup_status"] is None
        elif outcome == "exhausted":
            assert result is ProcessingResult.FAILED_EXHAUSTED
            assert requests == []
            assert persisted["status"] == "failed"
            assert persisted["attempt_count"] == 1
            assert persisted["last_error"] == "cycle_window_exhausted"
            assert persisted["error_message"] == (
                "Se agotó el tiempo disponible para el procesamiento."
            )
            assert "content" not in persisted
            assert "page_count" not in persisted
        else:
            assert result is ProcessingResult.RETRY_SCHEDULED
            assert len(requests) == 1
            assert persisted["status"] == "pending"
            assert persisted["attempt_count"] == 1
            assert persisted["last_error"] == "ms2_http_503"
            next_retry_at = persisted["next_retry_at"]
            if next_retry_at.tzinfo is None:
                next_retry_at = next_retry_at.replace(tzinfo=timezone.utc)
            assert next_retry_at > planned_at
            cycle_started_at = claimed.processing_cycle_started_at
            if cycle_started_at.tzinfo is None:
                cycle_started_at = cycle_started_at.replace(tzinfo=timezone.utc)
            assert next_retry_at <= cycle_started_at + timedelta(minutes=10)
            assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
            assert (
                persisted["processing_cycle_started_at"]
                == claimed.processing_cycle_started_at
            )
            for field in (
                "worker_id",
                "fencing_token",
                "lease_until",
                "heartbeat_at",
            ):
                assert persisted.get(field) is None
            assert persisted["file_id"] == fixture["file_id"]
            assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]
            for field in (
                "content",
                "page_count",
                "completed_at",
                "failed_at",
                "error_message",
            ):
                assert persisted.get(field) is None
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("first_response", "expected_failure_code"),
    [
        ("http-503", "ms2_http_503"),
        ("invalid-success", "ms2_invalid_response"),
    ],
)
@pytest.mark.asyncio
async def test_automatic_retry_preserves_cycle_and_worker_processes_other_job(
    worker_integration_context,
    monkeypatch,
    first_response,
    expected_failure_code,
):
    context = worker_integration_context
    retry_fixture = await _create_pending_pdf(context, "automatic-retry")
    clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    repository = ProcessingRepository(context.db)
    requests = []
    retry_requests = []

    async def transport_handler(request):
        requests.append(request)
        if retry_fixture["filename"].encode() in request.content:
            retry_requests.append(request)
            if len(retry_requests) == 1:
                if first_response == "http-503":
                    return httpx.Response(
                        503,
                        headers={"Retry-After": "0"},
                        json={"detail": "private upstream response"},
                    )
                return httpx.Response(
                    200,
                    json={"content": "malformed result", "page_count": True},
                )
            return httpx.Response(
                200,
                json={"content": "retried extraction", "page_count": 2},
            )
        return httpx.Response(
            200,
            json={"content": "other job extraction", "page_count": 1},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    worker = None
    worker_task = None
    try:
        first_claim = await repository.claim_pending("worker-u86-first")
        assert first_claim is not None
        assert first_claim.id == str(retry_fixture["document_id"])
        first_cycle_id = first_claim.processing_cycle_id
        first_cycle_started_at = _bson_utc(
            first_claim.processing_cycle_started_at
        )
        first_fencing_token = first_claim.fencing_token
        circuit_breaker = CircuitBreaker()

        first_result = await process_claimed_document(
            first_claim,
            processing_repository=repository,
            gridfs_repository=context.gridfs,
            ms2_client=MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=circuit_breaker,
            circuit_breaker_busy_retry_seconds=0.01,
        )
        assert first_result is ProcessingResult.RETRY_SCHEDULED
        assert len(retry_requests) == 1
        retry_pending = await context.collection.find_one(
            {"_id": retry_fixture["document_id"]}
        )
        assert retry_pending["status"] == "pending"
        assert retry_pending["attempt_count"] == 1
        assert retry_pending["last_error"] == expected_failure_code
        for field in ("content", "page_count", "completed_at"):
            assert retry_pending.get(field) is None
        next_retry_at = _bson_utc(retry_pending["next_retry_at"])
        assert next_retry_at > clock.now
        assert next_retry_at <= first_cycle_started_at + timedelta(minutes=10)
        assert retry_pending["processing_cycle_id"] == first_cycle_id
        assert _bson_utc(
            retry_pending["processing_cycle_started_at"]
        ) == first_cycle_started_at
        for field in (
            "worker_id",
            "fencing_token",
            "lease_until",
            "heartbeat_at",
        ):
            assert retry_pending.get(field) is None
        assert retry_pending["file_id"] == retry_fixture["file_id"]
        assert (
            await context.gridfs.read(retry_fixture["file_id"])
            == retry_fixture["pdf_bytes"]
        )

        other_fixture = await _create_pending_pdf(
            context, "available-during-backoff"
        )
        owned_ids = {
            retry_fixture["document_id"],
            other_fixture["document_id"],
        }
        await _assert_no_foreign_eligible_work(context, owned_ids)
        other_completed = asyncio.Event()
        other_results = []

        async def observe_processing(document, **dependencies):
            result = await process_claimed_document(document, **dependencies)
            if document.id == str(other_fixture["document_id"]):
                other_results.append(result)
                other_completed.set()
            return result

        worker = ProcessingWorker(
            repository,
            context.gridfs,
            MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=CircuitBreaker(),
            processor=observe_processing,
            max_concurrency=1,
            poll_initial_seconds=0.01,
            poll_max_seconds=0.05,
        )
        worker_task = asyncio.create_task(worker.run())
        await asyncio.wait_for(
            other_completed.wait(),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        worker.request_stop()
        await _wait_for_tasks([worker_task])
        worker_task.result()

        assert other_results == [ProcessingResult.COMPLETED]
        assert len(requests) == 2
        other_persisted = await context.collection.find_one(
            {"_id": other_fixture["document_id"]}
        )
        assert other_persisted["status"] == "completed"
        assert retry_pending["status"] == "pending"
        assert len(retry_requests) == 1

        assert await repository.claim_pending("worker-u86-too-early") is None
        clock.now = next_retry_at
        second_claim = await repository.claim_pending("worker-u86-retry")
        assert second_claim is not None
        assert second_claim.id == first_claim.id
        assert second_claim.processing_cycle_id == first_cycle_id
        assert _bson_utc(
            second_claim.processing_cycle_started_at
        ) == first_cycle_started_at
        assert second_claim.fencing_token != first_fencing_token
        assert second_claim.attempt_count == 1

        second_result = await process_claimed_document(
            second_claim,
            processing_repository=repository,
            gridfs_repository=context.gridfs,
            ms2_client=MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=circuit_breaker,
            circuit_breaker_busy_retry_seconds=0.01,
        )
        assert second_result is ProcessingResult.COMPLETED
        assert len(requests) == 3
        assert len(retry_requests) == 2
        completed = await context.collection.find_one(
            {"_id": retry_fixture["document_id"]}
        )
        assert completed["status"] == "completed"
        assert completed["content"] == "retried extraction"
        assert completed["page_count"] == 2
        assert completed["attempt_count"] == 2
        assert completed["processing_cycle_id"] == first_cycle_id
        assert _bson_utc(
            completed["processing_cycle_started_at"]
        ) == first_cycle_started_at
        assert completed["original_cleanup_status"] == "pending"
        assert completed["file_id"] == retry_fixture["file_id"]
        assert "next_retry_at" not in completed
        assert (
            await context.gridfs.read(retry_fixture["file_id"])
            == retry_fixture["pdf_bytes"]
        )
    finally:
        if worker is not None:
            worker.request_stop()
        if worker_task is not None:
            await _wait_for_tasks([worker_task])
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transient_retries_exhaust_five_attempts_without_sixth_call(
    worker_integration_context,
    monkeypatch,
):
    context = worker_integration_context
    fixture = await _create_pending_pdf(context, "attempts-exhausted")
    clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    repository = ProcessingRepository(context.db)
    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            503,
            json={"detail": "private transient failure"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    circuit_breaker = CircuitBreaker()
    cycle_id = None
    cycle_started_at = None
    previous_fencing_token = None
    try:
        for attempt_number in range(1, 6):
            if attempt_number > 1:
                clock.now = next_retry_at
            claimed = await repository.claim_pending(
                f"worker-u86-attempt-{attempt_number}"
            )
            assert claimed is not None
            if attempt_number == 1:
                cycle_id = claimed.processing_cycle_id
                cycle_started_at = _bson_utc(
                    claimed.processing_cycle_started_at
                )
            else:
                assert claimed.processing_cycle_id == cycle_id
                assert _bson_utc(
                    claimed.processing_cycle_started_at
                ) == cycle_started_at
                assert claimed.fencing_token != previous_fencing_token
            assert claimed.attempt_count == attempt_number - 1
            previous_fencing_token = claimed.fencing_token

            result = await process_claimed_document(
                claimed,
                processing_repository=repository,
                gridfs_repository=context.gridfs,
                ms2_client=MS2Client(
                    client,
                    base_url=_MOCK_MS2_BASE_URL,
                    timeout_seconds=2.0,
                ),
                circuit_breaker=circuit_breaker,
                circuit_breaker_busy_retry_seconds=0.01,
            )
            persisted = await context.collection.find_one(
                {"_id": fixture["document_id"]}
            )
            assert persisted["attempt_count"] == attempt_number
            assert persisted["processing_cycle_id"] == cycle_id
            assert _bson_utc(
                persisted["processing_cycle_started_at"]
            ) == cycle_started_at

            if attempt_number < 5:
                assert result is ProcessingResult.RETRY_SCHEDULED
                assert persisted["status"] == "pending"
                assert persisted["last_error"] == "ms2_http_503"
                next_retry_at = _bson_utc(persisted["next_retry_at"])
                assert next_retry_at > clock.now
                assert next_retry_at <= cycle_started_at + timedelta(minutes=10)
            else:
                assert result is ProcessingResult.FAILED_EXHAUSTED
                assert persisted["status"] == "failed"
                assert persisted["last_error"] == "attempts_exhausted"
                assert persisted["error_message"] == (
                    "Se agotaron los intentos de procesamiento."
                )
                assert "next_retry_at" not in persisted

        assert len(requests) == 5
        assert await repository.claim_pending("worker-u86-no-sixth-attempt") is None
        assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_at_cycle_deadline_is_claimed_then_exhausted_without_http(
    worker_integration_context,
    monkeypatch,
):
    context = worker_integration_context
    fixture = await _create_pending_pdf(context, "cycle-window-exhausted")
    clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    repository = ProcessingRepository(context.db)
    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            503,
            headers={"Retry-After": "600"},
            json={"detail": "private retry-after response"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    circuit_breaker = CircuitBreaker()
    try:
        first_claim = await repository.claim_pending("worker-u86-window-first")
        assert first_claim is not None
        cycle_started_at = _bson_utc(
            first_claim.processing_cycle_started_at
        )
        deadline = cycle_started_at + timedelta(minutes=10)
        result = await process_claimed_document(
            first_claim,
            processing_repository=repository,
            gridfs_repository=context.gridfs,
            ms2_client=MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=circuit_breaker,
            circuit_breaker_busy_retry_seconds=0.01,
        )
        assert result is ProcessingResult.RETRY_SCHEDULED
        assert len(requests) == 1
        pending = await context.collection.find_one(
            {"_id": fixture["document_id"]}
        )
        assert pending["status"] == "pending"
        assert pending["attempt_count"] == 1
        assert _bson_utc(pending["next_retry_at"]) == deadline

        clock.now = deadline - timedelta(milliseconds=1)
        assert await repository.claim_pending("worker-u86-before-deadline") is None
        clock.now = deadline
        at_deadline = await repository.claim_pending("worker-u86-at-deadline")
        assert at_deadline is not None
        assert at_deadline.id == first_claim.id
        assert at_deadline.processing_cycle_id == first_claim.processing_cycle_id
        assert _bson_utc(
            at_deadline.processing_cycle_started_at
        ) == cycle_started_at

        exhausted = await process_claimed_document(
            at_deadline,
            processing_repository=repository,
            gridfs_repository=context.gridfs,
            ms2_client=MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=circuit_breaker,
            circuit_breaker_busy_retry_seconds=0.01,
        )
        assert exhausted is ProcessingResult.FAILED_EXHAUSTED
        assert len(requests) == 1
        failed = await context.collection.find_one(
            {"_id": fixture["document_id"]}
        )
        assert failed["status"] == "failed"
        assert failed["attempt_count"] == 1
        assert failed["last_error"] == "cycle_window_exhausted"
        assert failed["error_message"] == (
            "Se agotó el tiempo disponible para el procesamiento."
        )
        assert "next_retry_at" not in failed
        assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_technical_ms2_failure_is_persisted_and_alerted_safely(
    worker_integration_context,
    monkeypatch,
    caplog,
):
    context = worker_integration_context
    fixture = await _create_pending_pdf(context, "technical-failure")
    _freeze_worker_clock(monkeypatch, _current_bson_utc())
    repository = ProcessingRepository(context.db)
    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            401,
            json={"detail": "sensitive internal upstream detail"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    from src.worker import document_processor as processor_module

    try:
        claimed = await repository.claim_pending("worker-u86-technical")
        assert claimed is not None
        with caplog.at_level(logging.ERROR, logger=processor_module.__name__):
            result = await process_claimed_document(
                claimed,
                processing_repository=repository,
                gridfs_repository=context.gridfs,
                ms2_client=MS2Client(
                    client,
                    base_url=_MOCK_MS2_BASE_URL,
                    timeout_seconds=2.0,
                ),
                circuit_breaker=CircuitBreaker(),
                circuit_breaker_busy_retry_seconds=0.01,
            )

        assert result is ProcessingResult.FAILED_TECHNICAL
        assert len(requests) == 1
        failed = await context.collection.find_one(
            {"_id": fixture["document_id"]}
        )
        assert failed["status"] == "failed"
        assert failed["attempt_count"] == 1
        assert failed["last_error"] == "ms2_http_401"
        assert failed["error_message"] == (
            "No se pudo completar el procesamiento debido a un problema técnico "
            "del servicio."
        )
        assert "next_retry_at" not in failed
        assert failed["file_id"] == fixture["file_id"]
        assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]

        alert_records = [
            record
            for record in caplog.records
            if record.levelno == logging.ERROR
            and "event_id=worker.ms2.technical_failure" in record.getMessage()
        ]
        assert len(alert_records) == 1
        alert = alert_records[0].getMessage()
        assert f"document_id={fixture['document_id']}" in alert
        assert "error_code=ms2_http_401" in alert
        assert "http_status=401" in alert
        assert "sensitive internal upstream detail" not in alert
        assert fixture["filename"] not in alert
        assert "Authorization" not in alert
        assert "secret" not in alert
    finally:
        await client.aclose()


class _HeartbeatGateRepository:
    """Delegate real repository operations while holding periodic heartbeats."""

    def __init__(self, repository):
        self._repository = repository
        self._renewal_calls = 0
        self._allow_later_renewals = asyncio.Event()

    def __getattr__(self, name):
        return getattr(self._repository, name)

    async def renew_processing_lease(
        self,
        document_id,
        worker_id,
        fencing_token,
    ):
        self._renewal_calls += 1
        # run_with_heartbeat and pdf_reader perform three immediate ownership
        # checks; only the periodic heartbeat is gated for this race scenario.
        if self._renewal_calls > 3:
            await self._allow_later_renewals.wait()
        return await self._repository.renew_processing_lease(
            document_id,
            worker_id,
            fencing_token,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expired_owner_result_is_rejected_after_real_recovery(
    worker_integration_context,
):
    context = worker_integration_context
    fixture = await _create_pending_pdf(context, "stale-owner")
    owned_document_ids = {fixture["document_id"]}
    await _assert_no_foreign_eligible_work(context, owned_document_ids)
    repository = ProcessingRepository(context.db)
    old_claim = await repository.claim_pending("worker-u86-original")
    assert old_claim is not None
    assert old_claim.id == str(fixture["document_id"])

    request_entered = asyncio.Event()
    release_response = asyncio.Event()
    requests = []

    async def transport_handler(request):
        requests.append(request)
        request_entered.set()
        await release_response.wait()
        return httpx.Response(
            200,
            json={"content": "stale result must not persist", "page_count": 2},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    controlled_repository = _HeartbeatGateRepository(repository)
    circuit_breaker = CircuitBreaker()
    processing_task = asyncio.create_task(
        process_claimed_document(
            old_claim,
            processing_repository=controlled_repository,
            gridfs_repository=context.gridfs,
            ms2_client=MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=circuit_breaker,
            circuit_breaker_busy_retry_seconds=0.01,
        )
    )

    try:
        await asyncio.wait_for(
            request_entered.wait(), timeout=_FIXTURE_TIMEOUT_SECONDS
        )
        old_cycle_id = old_claim.processing_cycle_id
        in_flight_document = await context.collection.find_one(
            {"_id": fixture["document_id"]}
        )
        attempt_count_before_recovery = in_flight_document["attempt_count"]
        assert old_claim.attempt_count == 0
        assert attempt_count_before_recovery == 1

        expired_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        expired = await context.collection.update_one(
            {
                "_id": fixture["document_id"],
                "status": "processing",
                "worker_id": old_claim.worker_id,
                "fencing_token": old_claim.fencing_token,
            },
            {"$set": {"lease_until": expired_at}},
        )
        assert expired.matched_count == 1

        new_worker_id = f"worker-u86-recovery-{uuid4()}"
        recovered = await repository.recover_expired_processing(new_worker_id)
        assert recovered is not None
        assert recovered.id == old_claim.id
        assert recovered.worker_id == new_worker_id
        assert recovered.fencing_token != old_claim.fencing_token
        assert recovered.processing_cycle_id == old_cycle_id
        assert recovered.attempt_count == attempt_count_before_recovery
        await _assert_no_foreign_eligible_work(
            context, owned_document_ids
        )

        release_response.set()
        await _wait_for_tasks([processing_task])
        old_result = processing_task.result()
        assert old_result is ProcessingResult.RESULT_NOT_CONFIRMED
        assert len(requests) == 1

        persisted = await context.collection.find_one(
            {"_id": fixture["document_id"]}
        )
        assert persisted["status"] == "processing"
        assert persisted["worker_id"] == new_worker_id
        assert persisted["fencing_token"] == recovered.fencing_token
        assert persisted["processing_cycle_id"] == old_cycle_id
        assert persisted["attempt_count"] == 1
        assert persisted.get("content") is None
        assert persisted.get("page_count") is None
        assert persisted.get("completed_at") is None
        assert await context.gridfs.stat(fixture["file_id"]) is not None
    finally:
        release_response.set()
        if not processing_task.done() and not processing_task.cancelling():
            processing_task.cancel()
        try:
            await _wait_for_tasks([processing_task])
        finally:
            controlled_repository._allow_later_renewals.set()
            await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_shared_breaker_open_releases_claims_and_worker_continues_to_next_job(
    worker_integration_context,
    monkeypatch,
):
    context = worker_integration_context
    wall_clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    monotonic_clock = _ControlledMonotonicClock()
    circuit_breaker = CircuitBreaker(monotonic_clock=monotonic_clock)
    repository = ProcessingRepository(context.db)
    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            503,
            headers={"Retry-After": "600"},
            json={"detail": "private open-breaker seed failure"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    seed_fixtures = []
    worker = None
    worker_task = None
    try:
        seed_fixtures = await _open_breaker_with_integrated_failures(
            context,
            repository,
            client,
            circuit_breaker,
            requests,
            "open-shared",
        )
        assert len({item["document_id"] for item in seed_fixtures}) == 5
        assert len(requests) == 5

        deferred = [
            await _create_pending_pdf(context, f"open-rejected-{index}")
            for index in range(2)
        ]
        released = []
        active_processors = 0
        peak_processors = 0
        two_released = asyncio.Event()

        async def observe_processing(document, **dependencies):
            nonlocal active_processors, peak_processors
            active_processors += 1
            peak_processors = max(peak_processors, active_processors)
            try:
                outcome = await process_claimed_document(
                    document,
                    **dependencies,
                )
                released.append((document.id, outcome))
                if len(released) == len(deferred):
                    two_released.set()
                return outcome
            finally:
                active_processors -= 1

        worker = ProcessingWorker(
            repository,
            context.gridfs,
            MS2Client(
                client,
                base_url=_MOCK_MS2_BASE_URL,
                timeout_seconds=2.0,
            ),
            circuit_breaker=circuit_breaker,
            processor=observe_processing,
            max_concurrency=1,
            poll_initial_seconds=0.01,
            poll_max_seconds=0.05,
        )
        worker_task = asyncio.create_task(worker.run())
        await asyncio.wait_for(
            two_released.wait(),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        worker.request_stop()
        await _wait_for_tasks([worker_task])
        worker_task.result()

        assert circuit_breaker.state is CircuitBreakerState.OPEN
        assert len(requests) == 5
        assert peak_processors == 1
        assert len(released) == 2
        assert {document_id for document_id, _ in released} == {
            str(item["document_id"]) for item in deferred
        }
        assert all(
            outcome is ProcessingResult.CIRCUIT_BREAKER_RELEASED
            for _, outcome in released
        )

        for fixture in deferred:
            persisted = await context.collection.find_one(
                {"_id": fixture["document_id"]}
            )
            assert persisted["status"] == "pending"
            assert persisted["attempt_count"] == 0
            assert isinstance(persisted["processing_cycle_id"], str)
            assert persisted["processing_cycle_id"]
            assert isinstance(persisted["processing_cycle_started_at"], datetime)
            assert _bson_utc(persisted["next_retry_at"]) == (
                wall_clock.now + timedelta(seconds=30)
            )
            for field in (
                "worker_id",
                "fencing_token",
                "lease_until",
                "heartbeat_at",
            ):
                assert persisted.get(field) is None
            assert await context.gridfs.read(fixture["file_id"]) == fixture["pdf_bytes"]
    finally:
        if worker is not None:
            worker.request_stop()
        if worker_task is not None:
            await _wait_for_tasks([worker_task])
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_half_open_probe_is_exclusive_busy_claim_releases_and_success_closes(
    worker_integration_context,
    monkeypatch,
):
    context = worker_integration_context
    wall_clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    monotonic_clock = _ControlledMonotonicClock()
    circuit_breaker = CircuitBreaker(monotonic_clock=monotonic_clock)
    repository = ProcessingRepository(context.db)
    requests = []
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()

    async def transport_handler(request):
        requests.append(request)
        if len(requests) <= 5:
            return httpx.Response(
                503,
                headers={"Retry-After": "600"},
                json={"detail": "private breaker seed failure"},
            )
        if len(requests) == 6:
            probe_started.set()
            await release_probe.wait()
        return httpx.Response(
            200,
            json={"content": "half-open probe succeeded", "page_count": 2},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    probe_task = None
    try:
        await _open_breaker_with_integrated_failures(
            context,
            repository,
            client,
            circuit_breaker,
            requests,
            "half-open-exclusive",
        )
        _advance_worker_clocks(wall_clock, monotonic_clock, 30.0)
        assert monotonic_clock.value == 30.0

        probe = await _create_pending_pdf(context, "half-open-probe")
        busy = await _create_pending_pdf(context, "half-open-busy")
        normal = await _create_pending_pdf(context, "half-open-after-close")
        probe_claim = await repository.claim_pending("worker-u452-half-open-probe")
        assert probe_claim is not None
        assert probe_claim.id == str(probe["document_id"])
        probe_task = asyncio.create_task(
            _process_with_breaker(
                context,
                repository,
                probe_claim,
                client,
                circuit_breaker,
            )
        )
        await asyncio.wait_for(
            probe_started.wait(),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        probe_processing = await context.collection.find_one(
            {"_id": probe["document_id"]}
        )
        assert probe_processing["status"] == "processing"
        assert probe_processing["attempt_count"] == 1
        assert len(requests) == 6

        busy_claim = await repository.claim_pending("worker-u452-half-open-busy")
        assert busy_claim is not None
        assert busy_claim.id == str(busy["document_id"])
        busy_result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                repository,
                busy_claim,
                client,
                circuit_breaker,
                busy_retry_seconds=0.01,
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        assert busy_result is ProcessingResult.CIRCUIT_BREAKER_RELEASED
        busy_persisted = await context.collection.find_one(
            {"_id": busy["document_id"]}
        )
        assert busy_persisted["status"] == "pending"
        assert busy_persisted["attempt_count"] == 0
        assert _bson_utc(busy_persisted["next_retry_at"]) == (
            wall_clock.now + timedelta(seconds=0.01)
        )
        for field in (
            "worker_id",
            "fencing_token",
            "lease_until",
            "heartbeat_at",
        ):
            assert busy_persisted.get(field) is None
        assert len(requests) == 6
        assert circuit_breaker.state is CircuitBreakerState.HALF_OPEN

        release_probe.set()
        probe_result = await asyncio.wait_for(
            probe_task,
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        probe_task = None
        assert probe_result is ProcessingResult.COMPLETED
        assert circuit_breaker.state is CircuitBreakerState.CLOSED
        probe_completed = await context.collection.find_one(
            {"_id": probe["document_id"]}
        )
        assert probe_completed["status"] == "completed"
        assert probe_completed["attempt_count"] == 1
        assert probe_completed["content"] == "half-open probe succeeded"

        normal_claim = await repository.claim_pending("worker-u452-after-half-open")
        assert normal_claim is not None
        assert normal_claim.id == str(normal["document_id"])
        normal_result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                repository,
                normal_claim,
                client,
                circuit_breaker,
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        assert normal_result is ProcessingResult.COMPLETED
        assert circuit_breaker.state is CircuitBreakerState.CLOSED
        assert len(requests) == 7
        assert await context.gridfs.read(busy["file_id"]) == busy["pdf_bytes"]
    finally:
        release_probe.set()
        if probe_task is not None:
            await _wait_for_tasks([probe_task])
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_half_open_eligible_failure_reopens_and_open_rejects_until_probe_window(
    worker_integration_context,
    monkeypatch,
):
    context = worker_integration_context
    wall_clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    monotonic_clock = _ControlledMonotonicClock()
    circuit_breaker = CircuitBreaker(monotonic_clock=monotonic_clock)
    repository = ProcessingRepository(context.db)
    requests = []

    async def transport_handler(request):
        requests.append(request)
        retry_after = "600"
        status_code = 503
        response_json = {"detail": "private transient failure"}
        if len(requests) == 6:
            retry_after = "600"
        return httpx.Response(
            status_code,
            headers={"Retry-After": retry_after},
            json=response_json,
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    try:
        await _open_breaker_with_integrated_failures(
            context,
            repository,
            client,
            circuit_breaker,
            requests,
            "half-open-reopen",
        )
        _advance_worker_clocks(wall_clock, monotonic_clock, 30.0)
        probe = await _create_pending_pdf(context, "half-open-failure-probe")
        probe_claim = await repository.claim_pending("worker-u452-half-open-failure")
        assert probe_claim is not None
        assert probe_claim.id == str(probe["document_id"])

        failure_result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                repository,
                probe_claim,
                client,
                circuit_breaker,
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        assert failure_result is ProcessingResult.RETRY_SCHEDULED
        assert circuit_breaker.state is CircuitBreakerState.OPEN
        assert len(requests) == 6
        failed_probe = await context.collection.find_one(
            {"_id": probe["document_id"]}
        )
        assert failed_probe["status"] == "pending"
        assert failed_probe["attempt_count"] == 1
        assert failed_probe["last_error"] == "ms2_http_503"
        assert failed_probe["processing_cycle_id"] == probe_claim.processing_cycle_id
        assert _bson_utc(failed_probe["processing_cycle_started_at"]) == (
            _bson_utc(probe_claim.processing_cycle_started_at)
        )
        assert failed_probe.get("worker_id") is None
        assert failed_probe.get("fencing_token") is None
        assert failed_probe.get("lease_until") is None
        assert failed_probe.get("heartbeat_at") is None
        assert await context.gridfs.read(probe["file_id"]) == probe["pdf_bytes"]

        _advance_worker_clocks(wall_clock, monotonic_clock, 29.999)
        rejected = await _create_pending_pdf(context, "reopened-open-rejection")
        rejected_claim = await repository.claim_pending("worker-u452-reopened-open")
        assert rejected_claim is not None
        assert rejected_claim.id == str(rejected["document_id"])
        rejected_result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                repository,
                rejected_claim,
                client,
                circuit_breaker,
                busy_retry_seconds=0.01,
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        assert rejected_result is ProcessingResult.CIRCUIT_BREAKER_RELEASED
        assert circuit_breaker.state is CircuitBreakerState.OPEN
        assert len(requests) == 6
        rejected_persisted = await context.collection.find_one(
            {"_id": rejected["document_id"]}
        )
        assert rejected_persisted["status"] == "pending"
        assert rejected_persisted["attempt_count"] == 0
        assert _bson_utc(rejected_persisted["next_retry_at"]) == (
            wall_clock.now + timedelta(seconds=0.01)
        )
        assert await context.gridfs.read(rejected["file_id"]) == rejected["pdf_bytes"]
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cycle_window_exhausts_atomically_while_breaker_is_open(
    worker_integration_context,
    monkeypatch,
):
    context = worker_integration_context
    wall_clock = _freeze_worker_clock(monkeypatch, _current_bson_utc())
    monotonic_clock = _ControlledMonotonicClock()
    circuit_breaker = CircuitBreaker(monotonic_clock=monotonic_clock)
    repository = ProcessingRepository(context.db)
    requests = []

    async def transport_handler(request):
        requests.append(request)
        return httpx.Response(
            503,
            headers={"Retry-After": "600"},
            json={"detail": "private exhaustion seed failure"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport_handler)
    )
    target = await _create_pending_pdf(context, "open-cycle-window-exhaustion")
    try:
        target_claim = await repository.claim_pending(
            "worker-u452-open-cycle-target"
        )
        assert target_claim is not None
        assert target_claim.id == str(target["document_id"])
        cycle_started_at = _bson_utc(target_claim.processing_cycle_started_at)

        # Preserve the real lease through repository renewals while advancing
        # UTC and monotonic time together. Open the breaker only 29.999s before
        # this document's genuine ten-minute cycle deadline.
        for _ in range(19):
            assert await repository.renew_processing_lease(
                target_claim.id,
                target_claim.worker_id,
                target_claim.fencing_token,
            )
            _advance_worker_clocks(wall_clock, monotonic_clock, 30.0)
        assert await repository.renew_processing_lease(
            target_claim.id,
            target_claim.worker_id,
            target_claim.fencing_token,
        )
        _advance_worker_clocks(wall_clock, monotonic_clock, 0.001)
        assert wall_clock.now == cycle_started_at + timedelta(
            seconds=570.001
        )

        await _open_breaker_with_integrated_failures(
            context,
            repository,
            client,
            circuit_breaker,
            requests,
            "cycle-window-open",
        )
        assert circuit_breaker.state is CircuitBreakerState.OPEN
        assert len(requests) == 5

        _advance_worker_clocks(wall_clock, monotonic_clock, 29.999)
        assert wall_clock.now == cycle_started_at + timedelta(minutes=10)
        assert monotonic_clock.value == 600.0

        result = await asyncio.wait_for(
            _process_with_breaker(
                context,
                repository,
                target_claim,
                client,
                circuit_breaker,
                busy_retry_seconds=0.01,
            ),
            timeout=_FIXTURE_TIMEOUT_SECONDS,
        )
        assert result is ProcessingResult.FAILED_EXHAUSTED
        assert circuit_breaker.state is CircuitBreakerState.OPEN
        assert len(requests) == 5

        exhausted = await context.collection.find_one(
            {"_id": target["document_id"]}
        )
        assert exhausted["status"] == "failed"
        assert exhausted["attempt_count"] == 0
        assert exhausted["processing_cycle_id"] == target_claim.processing_cycle_id
        assert _bson_utc(exhausted["processing_cycle_started_at"]) == cycle_started_at
        assert exhausted["last_error"] == "cycle_window_exhausted"
        assert exhausted["error_message"] == (
            "Se agotó el tiempo disponible para el procesamiento."
        )
        assert "next_retry_at" not in exhausted
        assert exhausted["worker_id"] == target_claim.worker_id
        assert exhausted["fencing_token"] == target_claim.fencing_token
        assert exhausted.get("lease_until") is None
        assert _bson_utc(exhausted["heartbeat_at"]) == wall_clock.now
        assert await context.gridfs.read(target["file_id"]) == target["pdf_bytes"]
    finally:
        await client.aclose()
