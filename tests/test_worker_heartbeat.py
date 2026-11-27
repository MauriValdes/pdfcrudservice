"""RED tests for the claimed-document heartbeat supervisor."""

import asyncio
from datetime import datetime, timezone
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from bson import ObjectId

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.worker.exceptions import ProcessingLeaseLostError


LEASE_SECONDS = 60.0
TEST_INTERVAL = 0.001


def _document(**overrides):
    values = {
        "_id": str(ObjectId()),
        "filename": "claimed.pdf",
        "checksum": "checksum",
        "size_bytes": 12,
        "status": DocumentStatus.PROCESSING.value,
        "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "worker_id": "worker-test",
        "fencing_token": str(uuid4()),
    }
    values.update(overrides)
    return DocumentModel.model_validate(values)


def _repository(renew=None):
    return SimpleNamespace(
        renew_processing_lease=renew or AsyncMock(return_value=True)
    )


async def _run(document, repository, operation, *, interval=TEST_INTERVAL):
    # Deferred deliberately: the supervisor is the functionality under RED.
    from src.worker.heartbeat import run_with_heartbeat

    return await run_with_heartbeat(
        document,
        processing_repository=repository,
        operation=operation,
        heartbeat_interval_seconds=interval,
    )


async def _eventually(event, *, task=None, timeout=1.0):
    event_waiter = asyncio.create_task(event.wait())
    waiters = {event_waiter}
    if task is not None:
        waiters.add(task)
    try:
        done, _ = await asyncio.wait(
            waiters,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task is not None and task in done:
            await task
        assert event_waiter in done, "event was not set before timeout"
    finally:
        if not event_waiter.done():
            event_waiter.cancel()
        await asyncio.gather(event_waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_initial_renewal_precedes_single_operation_and_returns_original_result():
    document = _document()
    repository = _repository()
    result = object()
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        return result

    returned = await _run(document, repository, operation)

    assert returned is result
    assert calls == 1
    repository.renew_processing_lease.assert_awaited()
    assert repository.renew_processing_lease.await_args_list[0].args == (
        document.id,
        document.worker_id,
        document.fencing_token,
    )


@pytest.mark.asyncio
async def test_renews_periodically_while_operation_is_active_with_captured_identity():
    document = _document()
    expected_identity = (document.id, document.worker_id, document.fencing_token)
    periodic_renewal = asyncio.Event()
    operation_started = asyncio.Event()
    finish_operation = asyncio.Event()
    renew_count = 0

    async def renew(*args):
        nonlocal renew_count
        assert args == expected_identity
        renew_count += 1
        if renew_count >= 2:
            periodic_renewal.set()
        return True

    async def operation():
        operation_started.set()
        await finish_operation.wait()
        return "done"

    repository = _repository(AsyncMock(side_effect=renew))
    task = asyncio.create_task(_run(document, repository, operation))
    try:
        await _eventually(operation_started)
        await _eventually(periodic_renewal)
        document.worker_id = "mutated-worker"
        document.fencing_token = "mutated-token"
        finish_operation.set()
        assert await task == "done"
    finally:
        finish_operation.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert renew_count >= 2
    assert all(call.args == expected_identity for call in repository.renew_processing_lease.await_args_list)


@pytest.mark.asyncio
async def test_initial_ownership_loss_does_not_start_operation():
    document = _document()
    repository = _repository()
    repository.renew_processing_lease.return_value = False
    operation = AsyncMock()

    with pytest.raises(ProcessingLeaseLostError):
        await _run(document, repository, operation)

    operation.assert_not_awaited()
    repository.renew_processing_lease.assert_awaited_once_with(
        document.id, document.worker_id, document.fencing_token
    )


@pytest.mark.asyncio
async def test_initial_database_unavailable_propagates_without_starting_operation():
    document = _document()
    repository = _repository()
    error = DatabaseUnavailableError()
    repository.renew_processing_lease.side_effect = error
    operation = AsyncMock()

    with pytest.raises(DatabaseUnavailableError) as raised:
        await _run(document, repository, operation)

    assert raised.value is error
    operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_periodic_ownership_loss_cancels_and_waits_for_operation():
    document = _document()
    periodic_renewal = asyncio.Event()
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()
    renew_count = 0

    async def renew(*_args):
        nonlocal renew_count
        renew_count += 1
        if renew_count == 1:
            return True
        periodic_renewal.set()
        return False

    async def operation():
        operation_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            operation_cancelled.set()
            raise

    repository = _repository(AsyncMock(side_effect=renew))
    with pytest.raises(ProcessingLeaseLostError):
        await _run(document, repository, operation)

    assert operation_started.is_set()
    assert operation_cancelled.is_set()
    assert repository.renew_processing_lease.await_count == 2


@pytest.mark.asyncio
async def test_periodic_database_unavailable_cancels_operation_and_propagates_same_error():
    document = _document()
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()
    error = DatabaseUnavailableError("heartbeat unavailable")
    repository = _repository()
    repository.renew_processing_lease.side_effect = [True, error]

    async def operation():
        operation_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            operation_cancelled.set()
            raise

    with pytest.raises(DatabaseUnavailableError) as raised:
        await _run(document, repository, operation)

    assert raised.value is error
    assert operation_started.is_set()
    assert operation_cancelled.is_set()
    assert repository.renew_processing_lease.await_count == 2


@pytest.mark.asyncio
async def test_unexpected_periodic_heartbeat_error_cancels_operation_and_propagates():
    document = _document()
    operation_cancelled = asyncio.Event()
    error = RuntimeError("unexpected renew failure")
    repository = _repository()
    repository.renew_processing_lease.side_effect = [True, error]

    async def operation():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            operation_cancelled.set()
            raise

    with pytest.raises(RuntimeError, match="unexpected renew failure") as raised:
        await _run(document, repository, operation)

    assert raised.value is error
    assert operation_cancelled.is_set()


@pytest.mark.asyncio
async def test_operation_exception_is_preserved_and_heartbeat_is_stopped():
    document = _document()
    periodic_renewal = asyncio.Event()
    finish_heartbeat = asyncio.Event()
    renew_count = 0
    error = ValueError("operation failed")

    async def renew(*_args):
        nonlocal renew_count
        renew_count += 1
        if renew_count == 1:
            return True
        periodic_renewal.set()
        await finish_heartbeat.wait()
        return True

    async def operation():
        await periodic_renewal.wait()
        raise error

    repository = _repository(AsyncMock(side_effect=renew))
    task = asyncio.create_task(_run(document, repository, operation))
    try:
        with pytest.raises(ValueError, match="operation failed") as raised:
            await asyncio.wait_for(task, timeout=1.0)
        assert raised.value is error
    finally:
        finish_heartbeat.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert repository.renew_processing_lease.await_count == 2


@pytest.mark.asyncio
async def test_external_cancellation_propagates_and_cleans_up_both_child_tasks():
    document = _document()
    periodic_renewal = asyncio.Event()
    heartbeat_cancelled = asyncio.Event()
    operation_cancelled = asyncio.Event()
    renew_count = 0

    async def renew(*_args):
        nonlocal renew_count
        renew_count += 1
        if renew_count == 1:
            return True
        periodic_renewal.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            heartbeat_cancelled.set()
            raise

    async def operation():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            operation_cancelled.set()
            raise

    repository = _repository(AsyncMock(side_effect=renew))
    task = asyncio.create_task(_run(document, repository, operation))
    await _eventually(periodic_renewal, task=task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert heartbeat_cancelled.is_set()
    assert operation_cancelled.is_set()


@pytest.mark.asyncio
async def test_heartbeat_and_operation_completion_race_preserves_completed_result():
    """An already-finished operation wins; a late renew failure is not its outcome."""
    document = _document()
    periodic_renewal_started = asyncio.Event()
    release_periodic_renewal = asyncio.Event()
    renew_count = 0
    expected_result = object()

    async def renew(*_args):
        nonlocal renew_count
        renew_count += 1
        if renew_count == 1:
            return True
        periodic_renewal_started.set()
        await release_periodic_renewal.wait()
        return False

    async def operation():
        await periodic_renewal_started.wait()
        return expected_result

    repository = _repository(AsyncMock(side_effect=renew))
    task = asyncio.create_task(_run(document, repository, operation))
    try:
        await _eventually(periodic_renewal_started, task=task)
        # Let operation complete while the in-flight heartbeat is held at a
        # barrier, then release the failed renewal to create the race.
        await asyncio.sleep(0)
        release_periodic_renewal.set()
        assert await task is expected_result
    finally:
        release_periodic_renewal.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert repository.renew_processing_lease.await_count == 2


@pytest.mark.asyncio
async def test_renewals_do_not_overlap():
    document = _document()
    periodic_started = asyncio.Event()
    allow_periodic_finish = asyncio.Event()
    active_renewals = 0
    maximum_active_renewals = 0
    renew_count = 0

    async def renew(*_args):
        nonlocal active_renewals, maximum_active_renewals, renew_count
        renew_count += 1
        active_renewals += 1
        maximum_active_renewals = max(maximum_active_renewals, active_renewals)
        try:
            if renew_count > 1:
                periodic_started.set()
                await allow_periodic_finish.wait()
            return True
        finally:
            active_renewals -= 1

    async def operation():
        await periodic_started.wait()
        await asyncio.sleep(TEST_INTERVAL * 3)
        allow_periodic_finish.set()
        return "done"

    repository = _repository(AsyncMock(side_effect=renew))
    try:
        assert await _run(document, repository, operation) == "done"
    finally:
        allow_periodic_finish.set()

    assert maximum_active_renewals == 1


@pytest.mark.parametrize(
    "document",
    [
        None,
        SimpleNamespace(id=None, worker_id="worker", fencing_token="token"),
        SimpleNamespace(id="not-an-object-id", worker_id="worker", fencing_token="token"),
        SimpleNamespace(id=str(ObjectId()), worker_id="", fencing_token="token"),
        SimpleNamespace(id=str(ObjectId()), worker_id=None, fencing_token="token"),
        SimpleNamespace(id=str(ObjectId()), worker_id="worker", fencing_token=""),
        SimpleNamespace(id=str(ObjectId()), worker_id="worker", fencing_token=None),
    ],
    ids=[
        "document-absent",
        "document-id-absent",
        "document-id-invalid",
        "worker-empty",
        "worker-not-string",
        "fencing-token-empty",
        "fencing-token-not-string",
    ],
)
@pytest.mark.asyncio
async def test_rejects_missing_or_invalid_claim_identity_before_renewal(document):
    repository = _repository()

    with pytest.raises((TypeError, ValueError)):
        await _run(document, repository, AsyncMock())

    repository.renew_processing_lease.assert_not_awaited()


@pytest.mark.parametrize(
    "interval",
    [
        0,
        -0.1,
        True,
        False,
        float("nan"),
        float("inf"),
        -float("inf"),
        60,
        60.0,
        61,
    ],
    ids=[
        "zero",
        "negative",
        "true",
        "false",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "equal-lease-int",
        "equal-lease-float",
        "greater-than-lease",
    ],
)
@pytest.mark.asyncio
async def test_rejects_invalid_heartbeat_interval_before_renewal(interval):
    repository = _repository()

    with pytest.raises((TypeError, ValueError)):
        await _run(_document(), repository, AsyncMock(), interval=interval)

    repository.renew_processing_lease.assert_not_awaited()


@pytest.mark.asyncio
async def test_accepts_finite_positive_interval_strictly_below_lease():
    document = _document()
    repository = _repository()
    operation = AsyncMock(return_value="ok")
    interval = math.nextafter(LEASE_SECONDS, 0.0)

    assert await _run(document, repository, operation, interval=interval) == "ok"
    operation.assert_awaited_once()
