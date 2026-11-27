"""End-to-end worker pipeline checks against isolated test MongoDB."""

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from bson import ObjectId
from pymongo import MongoClient

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
from src.worker.ms2_client import MS2Client, MS2HTTPError
from src.worker.worker_lifecycle import WorkerLifecycle
from src.worker.worker_runner import ProcessingWorker


_EXPECTED_TEST_USER = "pdf_ms3_test"
_MOCK_MS2_BASE_URL = "http://ms2.test"
_SUCCESS_CONTENT = "integral worker extraction"
_SUCCESS_PAGE_COUNT = 3
_FIXTURE_TIMEOUT_SECONDS = 8


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
    try:
        await manager.connect()
        connected = True
        assert manager._testing is True
        assert manager.db is not None
        assert manager.db.name == MONGO_TEST_DATABASE
        assert manager.client.write_concern.document == {"w": 1, "j": True}
        assert manager.db.write_concern.document == {"w": 1, "j": True}

        collection = manager.db[settings.mongo_collection]
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
        )

        if outcome == "transient":
            with pytest.raises(MS2HTTPError) as raised:
                await asyncio.wait_for(
                    processor, timeout=_FIXTURE_TIMEOUT_SECONDS
                )
            assert raised.value.status_code == 503
            assert raised.value.classification == "transient"
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
            assert len(requests) == 1
            assert persisted["status"] == "processing"
            assert persisted["attempt_count"] == 1
            assert persisted["worker_id"] == claimed.worker_id
            assert persisted["fencing_token"] == claimed.fencing_token
            assert persisted["processing_cycle_id"] == claimed.processing_cycle_id
            for field in (
                "content",
                "page_count",
                "completed_at",
                "failed_at",
                "error_message",
                "last_error",
                "next_retry_at",
            ):
                assert persisted.get(field) is None
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
