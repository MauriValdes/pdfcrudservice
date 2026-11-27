"""RED tests for basic processing of one already-claimed document."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from bson import ObjectId
from gridfs.errors import NoFile

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.worker.exceptions import ProcessingLeaseLostError
from src.worker.ms2_client import (
    MS2HTTPError,
    MS2InvalidResponseError,
    MS2TransportError,
)


def _document(**updates):
    values = {
        "_id": str(ObjectId()),
        "filename": "claimed.pdf",
        "checksum": "claimed-checksum",
        "size_bytes": 32,
        "status": DocumentStatus.PROCESSING.value,
        "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "file_id": ObjectId(),
        "worker_id": "worker-8-2",
        "fencing_token": str(uuid4()),
        "processing_cycle_id": str(uuid4()),
        "attempt_count": 0,
    }
    return DocumentModel.model_validate(values).model_copy(update=updates)


def _load_processor():
    # Import deferred so RED is caused by the not-yet-created processor module.
    from src.worker import document_processor

    return document_processor


def _dependencies(*, pdf_bytes=b"%PDF-test", extraction=None):
    repository = SimpleNamespace(
        renew_processing_lease=AsyncMock(return_value=True),
        record_ms2_attempt=AsyncMock(return_value=1),
        complete_processing=AsyncMock(return_value=True),
        fail_processing=AsyncMock(),
        fail_exhausted_processing=AsyncMock(),
        schedule_transient_retry=AsyncMock(),
        release_for_circuit_open=AsyncMock(),
    )
    gridfs = SimpleNamespace(read=AsyncMock(return_value=pdf_bytes))
    ms2 = SimpleNamespace(
        extract=AsyncMock(
            return_value=extraction
            or SimpleNamespace(content="extracted text", page_count=2)
        )
    )
    return repository, gridfs, ms2


def _run_inline(monkeypatch, processor_module):
    async def inline_heartbeat(
        document,
        *,
        processing_repository,
        operation,
    ):
        return await operation()

    monkeypatch.setattr(
        processor_module,
        "run_with_heartbeat",
        inline_heartbeat,
    )
    monkeypatch.setattr(
        processor_module,
        "read_claimed_pdf",
        AsyncMock(return_value=b"%PDF-test"),
    )


@pytest.mark.asyncio
async def test_real_heartbeat_reader_and_fakes_complete_empty_extraction_in_order():
    """Exercise the real heartbeat and reader together, without external services."""
    processor = _load_processor()
    document = _document()
    events = []
    extraction = SimpleNamespace(content="", page_count=0)

    class ProcessingRepository:
        async def renew_processing_lease(self, document_id, worker_id, token):
            events.append(("renew", document_id, worker_id, token))
            return True

        async def record_ms2_attempt(
            self, document_id, worker_id, token, cycle_id
        ):
            events.append(("authorize", document_id, worker_id, token, cycle_id))
            return 3

        async def complete_processing(
            self,
            document_id,
            worker_id,
            token,
            cycle_id,
            attempt_number,
            content,
            page_count,
        ):
            events.append(
                (
                    "complete",
                    document_id,
                    worker_id,
                    token,
                    cycle_id,
                    attempt_number,
                    content,
                    page_count,
                )
            )
            return True

    class GridFS:
        async def read(self, file_id):
            events.append(("gridfs_read", file_id))
            return b"%PDF-real-reader"

    class MS2:
        async def extract(self, pdf_bytes, filename):
            events.append(("extract", pdf_bytes, filename))
            return extraction

    repository = ProcessingRepository()
    outcome = await processor.process_claimed_document(
        document,
        processing_repository=repository,
        gridfs_repository=GridFS(),
        ms2_client=MS2(),
    )

    identity = (document.id, document.worker_id, document.fencing_token)
    assert outcome is processor.ProcessingResult.COMPLETED
    assert events == [
        ("renew", *identity),  # supervisor's initial ownership check
        ("renew", *identity),  # reader checks ownership before GridFS
        ("gridfs_read", document.file_id),
        ("renew", *identity),  # reader checks ownership after GridFS
        ("authorize", *identity, document.processing_cycle_id),
        ("extract", b"%PDF-real-reader", document.filename),
        (
            "complete",
            *identity,
            document.processing_cycle_id,
            3,
            "",
            0,
        ),
    ]


@pytest.mark.parametrize(
    ("confirmed", "expected"),
    [
        (True, "COMPLETED"),
        (False, "RESULT_NOT_CONFIRMED"),
    ],
)
@pytest.mark.asyncio
async def test_terminal_write_result_is_returned_without_repeating_operations(
    monkeypatch, confirmed, expected
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = 4
    repository.complete_processing.return_value = confirmed

    outcome = await processor.process_claimed_document(
        document,
        processing_repository=repository,
        gridfs_repository=gridfs,
        ms2_client=ms2,
    )

    assert outcome is getattr(processor.ProcessingResult, expected)
    processor.read_claimed_pdf.assert_awaited_once()
    repository.record_ms2_attempt.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
        document.processing_cycle_id,
    )
    ms2.extract.assert_awaited_once_with(b"%PDF-test", document.filename)
    repository.complete_processing.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
        document.processing_cycle_id,
        4,
        "extracted text",
        2,
    )
    repository.fail_processing.assert_not_awaited()
    repository.fail_exhausted_processing.assert_not_awaited()
    repository.schedule_transient_retry.assert_not_awaited()
    repository.release_for_circuit_open.assert_not_awaited()


@pytest.mark.asyncio
async def test_initial_ownership_rejection_prevents_read_authorization_and_http():
    processor = _load_processor()
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    repository.renew_processing_lease.return_value = False

    with pytest.raises(ProcessingLeaseLostError):
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    gridfs.read.assert_not_awaited()
    repository.record_ms2_attempt.assert_not_awaited()
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_lease_loss_during_real_safe_read_prevents_authorization():
    processor = _load_processor()
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    # Initial supervisor renew, reader pre-read renew, then reader post-read loss.
    repository.renew_processing_lease.side_effect = [True, True, False]

    with pytest.raises(ProcessingLeaseLostError):
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    gridfs.read.assert_awaited_once_with(document.file_id)
    repository.record_ms2_attempt.assert_not_awaited()
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_snapshot_survives_mutation_of_original_during_read(
    monkeypatch,
):
    processor = _load_processor()
    entered_read = asyncio.Event()
    release_read = asyncio.Event()
    expected = _document()
    document = expected.model_copy(deep=True)
    repository, gridfs, ms2 = _dependencies()
    captured_snapshots = []

    async def blocked_read(snapshot, **_kwargs):
        captured_snapshots.append(snapshot)
        entered_read.set()
        await release_read.wait()
        return b"%PDF-test"

    async def inline_heartbeat(
        snapshot,
        *,
        processing_repository,
        operation,
    ):
        captured_snapshots.append(snapshot)
        return await operation()

    monkeypatch.setattr(processor, "run_with_heartbeat", inline_heartbeat)
    monkeypatch.setattr(processor, "read_claimed_pdf", blocked_read)
    task = asyncio.create_task(
        processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )
    )
    try:
        await asyncio.wait_for(entered_read.wait(), timeout=1.0)
        document.id = str(ObjectId())
        document.worker_id = "mutated-worker"
        document.fencing_token = "mutated-token"
        document.processing_cycle_id = "mutated-cycle"
        document.file_id = ObjectId()
        document.filename = "mutated.pdf"
        release_read.set()
        assert await task is processor.ProcessingResult.COMPLETED
    finally:
        release_read.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    snapshot = captured_snapshots[0]
    assert snapshot is not document
    assert snapshot.id == expected.id
    assert snapshot.worker_id == expected.worker_id
    assert snapshot.fencing_token == expected.fencing_token
    assert snapshot.processing_cycle_id == expected.processing_cycle_id
    assert snapshot.file_id == expected.file_id
    assert snapshot.filename == expected.filename
    repository.record_ms2_attempt.assert_awaited_once_with(
        expected.id,
        expected.worker_id,
        expected.fencing_token,
        expected.processing_cycle_id,
    )
    ms2.extract.assert_awaited_once_with(b"%PDF-test", expected.filename)
    repository.complete_processing.assert_awaited_once_with(
        expected.id,
        expected.worker_id,
        expected.fencing_token,
        expected.processing_cycle_id,
        1,
        "extracted text",
        2,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"id": None},
        {"id": "not-an-object-id"},
        {"id": 42},
        {"worker_id": None},
        {"worker_id": ""},
        {"fencing_token": None},
        {"fencing_token": ""},
        {"processing_cycle_id": None},
        {"processing_cycle_id": ""},
        {"file_id": "not-an-object-id"},
        {"filename": None},
        {"filename": ""},
    ],
    ids=[
        "document-id-missing",
        "document-id-invalid",
        "document-id-wrong-type",
        "worker-id-missing",
        "worker-id-empty",
        "fencing-token-missing",
        "fencing-token-empty",
        "cycle-id-missing",
        "cycle-id-empty",
        "file-id-not-object-id",
        "filename-missing",
        "filename-empty",
    ],
)
@pytest.mark.asyncio
async def test_invalid_claim_arguments_are_rejected_before_dependencies(updates):
    processor = _load_processor()
    document = _document(**updates)
    repository, gridfs, ms2 = _dependencies()

    with pytest.raises((TypeError, ValueError)):
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    repository.renew_processing_lease.assert_not_awaited()
    repository.record_ms2_attempt.assert_not_awaited()
    gridfs.read.assert_not_awaited()
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_atomic_authorization_none_stops_before_http_and_persistence(
    monkeypatch,
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document(attempt_count=5)
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = None
    repository.fail_exhausted_processing.return_value = False

    outcome = await processor.process_claimed_document(
        document,
        processing_repository=repository,
        gridfs_repository=gridfs,
        ms2_client=ms2,
    )

    assert outcome is processor.ProcessingResult.ATTEMPT_NOT_AUTHORIZED
    processor.read_claimed_pdf.assert_awaited_once()
    repository.record_ms2_attempt.assert_awaited_once()
    repository.fail_exhausted_processing.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
        document.processing_cycle_id,
    )
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()
    repository.fail_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorization_exception_propagates_without_retry_or_http(monkeypatch):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    error = DatabaseUnavailableError("authorization outcome uncertain")
    repository.record_ms2_attempt.side_effect = error

    with pytest.raises(DatabaseUnavailableError) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    repository.record_ms2_attempt.assert_awaited_once()
    repository.fail_exhausted_processing.assert_not_awaited()
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_attempt_count_never_replaces_atomic_authorization(
    monkeypatch,
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document(attempt_count=5)
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = 2

    outcome = await processor.process_claimed_document(
        document,
        processing_repository=repository,
        gridfs_repository=gridfs,
        ms2_client=ms2,
    )

    assert outcome is processor.ProcessingResult.COMPLETED
    repository.record_ms2_attempt.assert_awaited_once()
    ms2.extract.assert_awaited_once()
    assert repository.complete_processing.await_args.args[4] == 2


@pytest.mark.parametrize(
    "error",
    [
        MS2TransportError("timeout"),
        MS2InvalidResponseError("invalid response"),
    ],
    ids=["transport-error", "invalid-success-response"],
)
@pytest.mark.asyncio
async def test_transport_and_invalid_response_errors_propagate_without_terminal_persistence(
    monkeypatch, error
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    ms2.extract.side_effect = error

    with pytest.raises(type(error)) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    repository.record_ms2_attempt.assert_awaited_once()
    ms2.extract.assert_awaited_once()
    repository.complete_processing.assert_not_awaited()
    repository.fail_processing.assert_not_awaited()
    repository.schedule_transient_retry.assert_not_awaited()


@pytest.mark.parametrize(
    ("status_code", "failure_code"),
    [
        (400, "ms2_invalid_input"),
        (413, "pdf_too_large"),
        (422, "pdf_unprocessable"),
    ],
)
@pytest.mark.parametrize(
    ("confirmed", "expected"),
    [
        (True, "FAILED_PERMANENT"),
        (False, "RESULT_NOT_CONFIRMED"),
    ],
)
@pytest.mark.asyncio
async def test_known_permanent_http_errors_persist_once_with_exact_failure_code(
    monkeypatch,
    status_code,
    failure_code,
    confirmed,
    expected,
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    attempt_number = 4
    repository.record_ms2_attempt.return_value = attempt_number
    repository.fail_processing.return_value = confirmed
    error = MS2HTTPError(status_code, "permanent_input", None)
    ms2.extract.side_effect = error

    outcome = await processor.process_claimed_document(
        document,
        processing_repository=repository,
        gridfs_repository=gridfs,
        ms2_client=ms2,
    )

    assert outcome is getattr(processor.ProcessingResult, expected)
    repository.record_ms2_attempt.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
        document.processing_cycle_id,
    )
    ms2.extract.assert_awaited_once_with(b"%PDF-test", document.filename)
    repository.fail_processing.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
        document.processing_cycle_id,
        attempt_number,
        failure_code,
    )
    repository.complete_processing.assert_not_awaited()
    repository.fail_exhausted_processing.assert_not_awaited()
    repository.schedule_transient_retry.assert_not_awaited()
    repository.release_for_circuit_open.assert_not_awaited()


@pytest.mark.asyncio
async def test_permanent_failure_write_exception_propagates_without_repeating(
    monkeypatch,
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = 2
    error = DatabaseUnavailableError("permanent failure acknowledgement uncertain")
    repository.fail_processing.side_effect = error
    ms2.extract.side_effect = MS2HTTPError(413, "permanent_input", None)

    with pytest.raises(DatabaseUnavailableError) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    ms2.extract.assert_awaited_once()
    repository.fail_processing.assert_awaited_once()
    repository.complete_processing.assert_not_awaited()
    repository.fail_exhausted_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_none_authorization_confirms_exhaustion_when_conditional_write_matches(
    monkeypatch,
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document(attempt_count=5)
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = None
    repository.fail_exhausted_processing.return_value = True

    outcome = await processor.process_claimed_document(
        document,
        processing_repository=repository,
        gridfs_repository=gridfs,
        ms2_client=ms2,
    )

    assert outcome is processor.ProcessingResult.FAILED_EXHAUSTED
    repository.record_ms2_attempt.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
        document.processing_cycle_id,
    )
    repository.fail_exhausted_processing.assert_awaited_once_with(
        document.id,
        document.worker_id,
        document.fencing_token,
        document.processing_cycle_id,
    )
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()
    repository.fail_processing.assert_not_awaited()
    repository.schedule_transient_retry.assert_not_awaited()
    repository.release_for_circuit_open.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhaustion_write_exception_propagates_without_http(monkeypatch):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = None
    error = DatabaseUnavailableError("exhaustion write unavailable")
    repository.fail_exhausted_processing.side_effect = error

    with pytest.raises(DatabaseUnavailableError) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    repository.record_ms2_attempt.assert_awaited_once()
    repository.fail_exhausted_processing.assert_awaited_once()
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        MS2HTTPError(503, "transient", "3"),
        MS2HTTPError(404, "unclassified", None),
        MS2HTTPError(400, "unclassified", None),
        MS2HTTPError(404, "permanent_input", None),
    ],
    ids=[
        "transient-http",
        "unclassified-http",
        "known-status-wrong-classification",
        "permanent-classification-unmapped-status",
    ],
)
@pytest.mark.asyncio
async def test_nonpermanent_or_mismatched_http_errors_propagate_unchanged(
    monkeypatch, error
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    ms2.extract.side_effect = error

    with pytest.raises(MS2HTTPError) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    ms2.extract.assert_awaited_once()
    repository.fail_processing.assert_not_awaited()
    repository.fail_exhausted_processing.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()
    repository.schedule_transient_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_ms2_http_error_outside_extract_is_not_treated_as_permanent(
    monkeypatch,
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    error = MS2HTTPError(400, "permanent_input", None)
    repository.record_ms2_attempt.side_effect = error

    with pytest.raises(MS2HTTPError) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    repository.record_ms2_attempt.assert_awaited_once()
    ms2.extract.assert_not_awaited()
    repository.fail_processing.assert_not_awaited()
    repository.fail_exhausted_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_permanent_failure_wins_concurrent_heartbeat_completion(
    monkeypatch,
):
    processor = _load_processor()
    from src.worker.heartbeat import run_with_heartbeat

    periodic_renewal_started = asyncio.Event()
    release_periodic_renewal = asyncio.Event()
    failure_persisted = asyncio.Event()
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = 1
    ms2.extract.side_effect = MS2HTTPError(422, "permanent_input", None)

    async def renew(*_args):
        if repository.renew_processing_lease.await_count == 1:
            return True
        periodic_renewal_started.set()
        await release_periodic_renewal.wait()
        return False

    async def fail_processing(*_args):
        await periodic_renewal_started.wait()
        failure_persisted.set()
        release_periodic_renewal.set()
        return True

    repository.renew_processing_lease.side_effect = renew
    repository.fail_processing.side_effect = fail_processing

    async def fast_heartbeat(
        snapshot,
        *,
        processing_repository,
        operation,
    ):
        return await run_with_heartbeat(
            snapshot,
            processing_repository=processing_repository,
            operation=operation,
            heartbeat_interval_seconds=0.001,
        )

    monkeypatch.setattr(processor, "run_with_heartbeat", fast_heartbeat)
    monkeypatch.setattr(
        processor,
        "read_claimed_pdf",
        AsyncMock(return_value=b"%PDF-test"),
    )

    outcome = await asyncio.wait_for(
        processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        ),
        timeout=1.0,
    )

    assert outcome is processor.ProcessingResult.FAILED_PERMANENT
    assert failure_persisted.is_set()
    assert periodic_renewal_started.is_set()
    repository.fail_processing.assert_awaited_once()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_gridfs_read_error_propagates_without_consuming_attempt(monkeypatch):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    error = NoFile("missing claimed PDF")
    processor.read_claimed_pdf.side_effect = error

    with pytest.raises(NoFile) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    repository.record_ms2_attempt.assert_not_awaited()
    ms2.extract.assert_not_awaited()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_persistence_exception_propagates_without_repeating_write_or_extract(
    monkeypatch,
):
    processor = _load_processor()
    _run_inline(monkeypatch, processor)
    document = _document()
    repository, gridfs, ms2 = _dependencies()
    error = DatabaseUnavailableError("completion acknowledgement uncertain")
    repository.complete_processing.side_effect = error

    with pytest.raises(DatabaseUnavailableError) as raised:
        await processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )

    assert raised.value is error
    ms2.extract.assert_awaited_once()
    repository.complete_processing.assert_awaited_once()


@pytest.mark.asyncio
async def test_lease_loss_during_http_cancels_operation_without_persisting(
    monkeypatch,
):
    processor = _load_processor()
    from src.worker.heartbeat import run_with_heartbeat

    http_started = asyncio.Event()
    http_cancelled = asyncio.Event()
    document = _document()
    renew_count = 0
    repository, gridfs, ms2 = _dependencies()
    repository.record_ms2_attempt.return_value = 1

    async def renew(*_args):
        nonlocal renew_count
        renew_count += 1
        if renew_count == 1:
            return True
        await http_started.wait()
        return False

    async def blocked_extract(*_args):
        http_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            http_cancelled.set()
            raise

    repository.renew_processing_lease.side_effect = renew
    ms2.extract.side_effect = blocked_extract

    async def fast_heartbeat(
        snapshot,
        *,
        processing_repository,
        operation,
    ):
        return await run_with_heartbeat(
            snapshot,
            processing_repository=processing_repository,
            operation=operation,
            heartbeat_interval_seconds=0.001,
        )

    monkeypatch.setattr(processor, "run_with_heartbeat", fast_heartbeat)
    monkeypatch.setattr(
        processor,
        "read_claimed_pdf",
        AsyncMock(return_value=b"%PDF-test"),
    )

    with pytest.raises(ProcessingLeaseLostError):
        await asyncio.wait_for(
            processor.process_claimed_document(
                document,
                processing_repository=repository,
                gridfs_repository=gridfs,
                ms2_client=ms2,
            ),
            timeout=1.0,
        )

    assert http_started.is_set()
    assert http_cancelled.is_set()
    repository.record_ms2_attempt.assert_awaited_once()
    repository.complete_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_cancellation_during_http_propagates_and_skips_persistence(
    monkeypatch,
):
    processor = _load_processor()
    from src.worker.heartbeat import run_with_heartbeat

    http_started = asyncio.Event()
    http_cancelled = asyncio.Event()
    document = _document()
    repository, gridfs, ms2 = _dependencies()

    async def blocked_extract(*_args):
        http_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            http_cancelled.set()
            raise

    ms2.extract.side_effect = blocked_extract

    async def fast_heartbeat(
        snapshot,
        *,
        processing_repository,
        operation,
    ):
        return await run_with_heartbeat(
            snapshot,
            processing_repository=processing_repository,
            operation=operation,
            heartbeat_interval_seconds=0.001,
        )

    monkeypatch.setattr(processor, "run_with_heartbeat", fast_heartbeat)
    monkeypatch.setattr(
        processor,
        "read_claimed_pdf",
        AsyncMock(return_value=b"%PDF-test"),
    )
    task = asyncio.create_task(
        processor.process_claimed_document(
            document,
            processing_repository=repository,
            gridfs_repository=gridfs,
            ms2_client=ms2,
        )
    )
    try:
        await asyncio.wait_for(http_started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert http_cancelled.is_set()
    repository.record_ms2_attempt.assert_awaited_once()
    repository.complete_processing.assert_not_awaited()
