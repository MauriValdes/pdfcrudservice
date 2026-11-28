"""RED tests for polling, recovery, and concurrency in the worker runner."""

import asyncio
import importlib
import logging
import math
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.worker.circuit_breaker import CircuitBreaker


def _load_runner():
    return importlib.import_module("src.worker.worker_runner")


def _document(document_id):
    return SimpleNamespace(id=document_id)


def _take(script):
    if not script:
        return None
    value = script.popleft()
    if isinstance(value, BaseException):
        raise value
    return value


class _Repository:
    def __init__(self, *, recovered=(), pending=()):
        self.recovered = deque(recovered)
        self.pending = deque(pending)
        self.calls = []
        self.call_events = []

    async def recover_expired_processing(self, worker_id):
        self.calls.append(("recover", worker_id))
        if self.call_events:
            self.call_events[-1].set()
        return _take(self.recovered)

    async def claim_pending(self, worker_id):
        self.calls.append(("claim", worker_id))
        if self.call_events:
            self.call_events[-1].set()
        return _take(self.pending)


async def _wait(event):
    await asyncio.wait_for(event.wait(), timeout=1)


def _make_worker(module, repository, processor, **kwargs):
    kwargs.setdefault("circuit_breaker", CircuitBreaker())
    return module.ProcessingWorker(
        repository,
        object(),
        object(),
        processor=processor,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_default_capacity_admits_two_and_does_not_acquire_when_full():
    module = _load_runner()
    repository = _Repository(
        recovered=[_document("doc-a"), _document("doc-c")],
        pending=[_document("doc-b")],
    )
    started = []
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def processor(document, **_dependencies):
        started.append(document.id)
        if len(started) == 2:
            two_started.set()
        await release.wait()

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    await _wait(two_started)
    assert started == ["doc-a", "doc-b"]
    assert [kind for kind, _worker_id in repository.calls] == [
        "recover",
        "claim",
    ]

    worker.request_stop()
    release.set()
    await asyncio.wait_for(run_task, timeout=1)
    assert started == ["doc-a", "doc-b"]


@pytest.mark.asyncio
async def test_finished_task_releases_capacity_for_next_claim():
    module = _load_runner()
    repository = _Repository(
        recovered=[_document("doc-a"), _document("doc-c")],
        pending=[_document("doc-b")],
    )
    started = []
    first_started = asyncio.Event()
    third_started = asyncio.Event()
    release_first = asyncio.Event()
    release_others = asyncio.Event()

    async def processor(document, **_dependencies):
        started.append(document.id)
        if document.id == "doc-a":
            first_started.set()
            await release_first.wait()
        elif document.id == "doc-c":
            third_started.set()
            await release_others.wait()
        else:
            await release_others.wait()

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    await _wait(first_started)
    release_first.set()
    await _wait(third_started)
    assert set(started) == {"doc-a", "doc-b", "doc-c"}

    worker.request_stop()
    release_others.set()
    await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_acquisitions_alternate_after_each_success():
    module = _load_runner()
    repository = _Repository(
        recovered=[_document("a"), _document("c")],
        pending=[_document("b"), _document("d")],
    )
    four_started = asyncio.Event()
    started = []

    async def processor(document, **_dependencies):
        started.append(document.id)
        if len(started) == 4:
            four_started.set()

    worker = _make_worker(
        module, repository, processor, max_concurrency=4
    )
    run_task = asyncio.create_task(worker.run())
    await _wait(four_started)
    assert [kind for kind, _ in repository.calls] == [
        "recover",
        "claim",
        "recover",
        "claim",
    ]
    assert started == ["a", "b", "c", "d"]
    worker.request_stop()
    await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_empty_preferred_source_falls_back_to_other_source():
    module = _load_runner()
    repository = _Repository(pending=[_document("pending-doc")])
    started = asyncio.Event()

    async def processor(document, **_dependencies):
        assert document.id == "pending-doc"
        started.set()

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    await _wait(started)
    assert [kind for kind, _ in repository.calls[:2]] == ["recover", "claim"]
    worker.request_stop()
    await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_worker_identity_is_stable_per_run_and_new_for_another_instance():
    module = _load_runner()
    identities_by_run = []

    for _ in range(2):
        repository = _Repository(recovered=[_document("one")])
        finished = asyncio.Event()

        async def processor(_document, **_dependencies):
            finished.set()

        worker = _make_worker(module, repository, processor)
        run_task = asyncio.create_task(worker.run())
        try:
            await _wait(finished)
        finally:
            worker.request_stop()
            if not run_task.done():
                await asyncio.wait_for(run_task, timeout=1)
        run_ids = {worker_id for _kind, worker_id in repository.calls}
        assert len(run_ids) == 1
        identities_by_run.append(run_ids.pop())

    assert identities_by_run[0] != identities_by_run[1]
    assert all(isinstance(worker_id, str) and worker_id for worker_id in identities_by_run)


@pytest.mark.asyncio
async def test_polling_progression_is_bounded_and_first_empty_round_waits_initial():
    module = _load_runner()
    repository = _Repository()
    worker = _make_worker(module, repository, AsyncMock())
    observed = []

    async def controlled_wait(_stop_event, _task_event, timeout_seconds):
        observed.append(timeout_seconds)
        if len(observed) == 5:
            worker.request_stop()

    worker_module_wait = module._wait_for_poll
    module._wait_for_poll = controlled_wait
    try:
        await asyncio.wait_for(worker.run(), timeout=1)
    finally:
        module._wait_for_poll = worker_module_wait

    assert observed == [1, 2, 4, 5, 5]


@pytest.mark.asyncio
async def test_successful_acquisition_resets_poll_interval():
    module = _load_runner()
    repository = _Repository(pending=[None, _document("claimed")])
    worker = _make_worker(module, repository, AsyncMock())
    observed = []

    async def controlled_wait(_stop_event, _task_event, timeout_seconds):
        observed.append(timeout_seconds)
        if len(observed) == 2:
            worker.request_stop()

    original_wait = module._wait_for_poll
    module._wait_for_poll = controlled_wait
    try:
        await asyncio.wait_for(worker.run(), timeout=1)
    finally:
        module._wait_for_poll = original_wait

    assert observed == [1, 1]


@pytest.mark.asyncio
async def test_polling_wait_is_interrupted_by_stop_request():
    module = _load_runner()
    repository = _Repository()
    worker = _make_worker(
        module,
        repository,
        AsyncMock(),
        poll_initial_seconds=5,
        poll_max_seconds=5,
    )
    wait_started = asyncio.Event()
    original_wait = module._wait_for_poll

    async def observed_wait(stop_event, task_event, timeout_seconds):
        wait_started.set()
        return await original_wait(stop_event, task_event, timeout_seconds)

    module._wait_for_poll = observed_wait
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(wait_started)
        worker.request_stop()
        await asyncio.wait_for(run_task, timeout=0.5)
        assert len(repository.calls) == 2
    finally:
        module._wait_for_poll = original_wait
        worker.request_stop()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_polling_wait_is_interrupted_by_task_completion():
    module = _load_runner()
    repository = _Repository(recovered=[_document("active")])
    started = asyncio.Event()
    release = asyncio.Event()
    both_sources_empty = asyncio.Event()
    polling_wait_started = asyncio.Event()
    acquisition_after_completion = asyncio.Event()
    acquisition_count = 0

    async def processor(_document, **_dependencies):
        started.set()
        await release.wait()

    async def recover_expired_processing(worker_id):
        repository.calls.append(("recover", worker_id))
        if len(repository.calls) >= 3:
            both_sources_empty.set()
        return _take(repository.recovered)

    async def claim_pending(worker_id):
        nonlocal acquisition_count
        acquisition_count += 1
        repository.calls.append(("claim", worker_id))
        if acquisition_count >= 2:
            acquisition_after_completion.set()
        return None

    repository.recover_expired_processing = recover_expired_processing
    repository.claim_pending = claim_pending
    worker = _make_worker(
        module,
        repository,
        processor,
        poll_initial_seconds=5,
        poll_max_seconds=5,
    )
    original_wait = module._wait_for_poll

    async def observed_wait(stop_event, task_event, timeout_seconds):
        assert timeout_seconds == 5
        assert both_sources_empty.is_set()
        polling_wait_started.set()
        return await original_wait(stop_event, task_event, timeout_seconds)

    module._wait_for_poll = observed_wait
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(started)
        await _wait(both_sources_empty)
        await _wait(polling_wait_started)
        assert not acquisition_after_completion.is_set()

        release.set()
        await _wait(acquisition_after_completion)
        assert acquisition_count == 2
    finally:
        module._wait_for_poll = original_wait
        worker.request_stop()
        release.set()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_task_completion_resets_backoff_without_successful_reacquisition():
    module = _load_runner()
    repository = _Repository(recovered=[_document("active")])
    started = asyncio.Event()
    release = asyncio.Event()
    third_wait_started = asyncio.Event()
    fourth_wait_started = asyncio.Event()
    observed = []
    worker = None

    async def processor(_document, **_dependencies):
        started.set()
        await release.wait()

    worker = _make_worker(module, repository, processor, poll_max_seconds=5)
    original_wait = module._wait_for_poll

    async def controlled_wait(stop_event, task_event, timeout_seconds):
        observed.append(timeout_seconds)
        if len(observed) < 3:
            return
        if len(observed) == 3:
            third_wait_started.set()
            return await original_wait(stop_event, task_event, timeout_seconds)
        if len(observed) == 4:
            fourth_wait_started.set()
            worker.request_stop()

    module._wait_for_poll = controlled_wait
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(started)
        await _wait(third_wait_started)
        assert observed == [1, 2, 4]
        release.set()
        await _wait(fourth_wait_started)
        assert observed == [1, 2, 4, 1]
        assert len([kind for kind, _ in repository.calls if kind == "recover"]) >= 3
    finally:
        module._wait_for_poll = original_wait
        worker.request_stop()
        release.set()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_duplicate_active_document_suspends_acquisition_and_drains():
    module = _load_runner()
    duplicate = _document("same-id")
    repository = _Repository(recovered=[duplicate], pending=[duplicate])
    processor_started = asyncio.Event()
    release = asyncio.Event()
    duplicate_acquired = asyncio.Event()
    processor_calls = 0
    processor_cancelled = asyncio.Event()

    async def processor(_document, **_dependencies):
        nonlocal processor_calls
        processor_calls += 1
        processor_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            processor_cancelled.set()
            raise

    async def claim_pending(worker_id):
        repository.calls.append(("claim", worker_id))
        duplicate_acquired.set()
        return _take(repository.pending)

    repository.claim_pending = claim_pending
    worker = _make_worker(module, repository, processor, max_concurrency=2)
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(processor_started)
        await _wait(duplicate_acquired)
        assert not run_task.done()
        assert processor_calls == 1
        assert len(repository.calls) == 2
        assert not processor_cancelled.is_set()
        release.set()
        with pytest.raises(RuntimeError, match="same-id"):
            await asyncio.wait_for(run_task, timeout=1)
        assert processor_calls == 1
        assert len(repository.calls) == 2
        assert not processor_cancelled.is_set()
    finally:
        release.set()
        if not run_task.done():
            try:
                await asyncio.wait_for(run_task, timeout=1)
            except RuntimeError:
                pass


@pytest.mark.asyncio
async def test_processor_exception_is_logged_consumed_and_capacity_reused(caplog):
    module = _load_runner()
    repository = _Repository(recovered=[_document("bad"), _document("good")])
    good_started = asyncio.Event()
    release = asyncio.Event()

    async def processor(document, **_dependencies):
        if document.id == "bad":
            raise ValueError("processor failed")
        good_started.set()
        await release.wait()

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(good_started)
        assert "processor failed" in caplog.text
        worker.request_stop()
        release.set()
        await asyncio.wait_for(run_task, timeout=1)
        assert [kind for kind, _ in repository.calls].count("recover") >= 2
    finally:
        worker.request_stop()
        release.set()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_all_processing_outcomes_are_logged_without_extra_writes(caplog):
    module = _load_runner()
    caplog.set_level(logging.INFO, logger=module.logger.name)
    processor_module = importlib.import_module("src.worker.document_processor")
    outcomes = list(processor_module.ProcessingResult)
    documents = [_document(str(i)) for i in range(len(outcomes))]
    repository = _Repository(recovered=documents)
    completed = asyncio.Event()
    returned = []

    async def processor(document, **_dependencies):
        returned.append(document.id)
        if len(returned) == len(outcomes):
            completed.set()
        return outcomes[int(document.id)]

    worker = _make_worker(
        module,
        repository,
        processor,
        max_concurrency=len(outcomes),
    )
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(completed)
        worker.request_stop()
        await asyncio.wait_for(run_task, timeout=1)
        assert len(returned) == len(outcomes)
        assert set(returned) == {str(i) for i in range(len(outcomes))}
        assert all(kind in {"recover", "claim"} for kind, _ in repository.calls)
        assert all(outcome.value in caplog.text for outcome in outcomes)
        outcome_records = [
            record
            for record in caplog.records
            if record.getMessage().startswith("Processing outcome for document ")
        ]
        assert len(outcome_records) == len(outcomes)
        for outcome in outcomes:
            matching_records = [
                record
                for record in outcome_records
                if record.getMessage().endswith(f": {outcome.value}")
            ]
            assert len(matching_records) == 1
            expected_level = (
                logging.INFO
                if outcome is processor_module.ProcessingResult.COMPLETED
                else logging.WARNING
            )
            assert matching_records[0].levelno == expected_level
        assert "private upstream detail" not in caplog.text
        assert "secret" not in caplog.text
    finally:
        worker.request_stop()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_acquisition_exception_is_preserved_and_does_not_fallback():
    module = _load_runner()
    error = RuntimeError("ambiguous acquisition")
    repository = _Repository(recovered=[error])
    worker = _make_worker(module, repository, AsyncMock())

    with pytest.raises(RuntimeError) as raised:
        await asyncio.wait_for(worker.run(), timeout=1)

    assert raised.value is error
    assert [kind for kind, _ in repository.calls] == ["recover"]


@pytest.mark.asyncio
async def test_acquisition_exception_drains_admitted_tasks_before_propagating():
    module = _load_runner()
    error = RuntimeError("claim outcome unknown")
    repository = _Repository(
        recovered=[_document("active")],
        pending=[error],
    )
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def processor(_document, **_dependencies):
        started.set()
        try:
            await release.wait()
        finally:
            finished.set()

    worker = _make_worker(module, repository, processor, max_concurrency=2)
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(started)
        await asyncio.sleep(0)
        assert not run_task.done()
        assert not finished.is_set()
        release.set()
        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(run_task, timeout=1)
        assert raised.value is error
        assert finished.is_set()
    finally:
        release.set()
        if not run_task.done():
            try:
                await asyncio.wait_for(run_task, timeout=1)
            except RuntimeError:
                pass


@pytest.mark.asyncio
async def test_request_stop_before_run_prevents_any_acquisition():
    module = _load_runner()
    repository = _Repository()
    worker = _make_worker(module, repository, AsyncMock())
    worker.request_stop()
    await asyncio.wait_for(worker.run(), timeout=1)
    assert repository.calls == []


@pytest.mark.asyncio
async def test_orderly_stop_drains_active_processor_without_cancelling_it():
    module = _load_runner()
    repository = _Repository(recovered=[_document("active")])
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def processor(_document, **_dependencies):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(started)
        worker.request_stop()
        await asyncio.sleep(0)
        assert not run_task.done()
        assert not cancelled.is_set()
        release.set()
        await asyncio.wait_for(run_task, timeout=1)
        assert not cancelled.is_set()
    finally:
        worker.request_stop()
        release.set()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "next_result",
    [_document("after-stop"), None],
    ids=["claim", "empty"],
)
async def test_stop_during_acquisition_does_not_query_fallback_after_none(
    next_result,
):
    module = _load_runner()
    repository = _Repository()
    acquisition_started = asyncio.Event()
    finish_acquisition = asyncio.Event()
    processor_started = asyncio.Event()

    async def recover_expired_processing(worker_id):
        repository.calls.append(("recover", worker_id))
        acquisition_started.set()
        await finish_acquisition.wait()
        return next_result

    repository.recover_expired_processing = recover_expired_processing

    async def processor(_document, **_dependencies):
        processor_started.set()

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    await _wait(acquisition_started)
    worker.request_stop()
    finish_acquisition.set()
    await asyncio.wait_for(run_task, timeout=1)

    assert [kind for kind, _ in repository.calls] == ["recover"]
    assert processor_started.is_set() is (next_result is not None)


@pytest.mark.asyncio
async def test_external_cancellation_cancels_and_collects_processor_tasks():
    module = _load_runner()
    repository = _Repository(recovered=[_document("active")])
    started = asyncio.Event()
    cancellation_received = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cancellation_count = 0
    cancellation_requested = False

    async def processor(_document, **_dependencies):
        nonlocal cancellation_count
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_count += 1
            cancellation_received.set()
            await cleanup_release.wait()
            cleanup_finished.set()
            raise

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(started)
        run_task.cancel()
        cancellation_requested = True
        await _wait(cancellation_received)
        assert not cleanup_finished.is_set()
        assert not run_task.done()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1)
        assert cleanup_finished.is_set()
        assert cancellation_count == 1
    finally:
        cleanup_release.set()
        if not run_task.done():
            if not cancellation_requested:
                run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_individually_cancelled_processor_releases_capacity():
    module = _load_runner()
    repository = _Repository(recovered=[_document("cancelled"), _document("next")])
    cancellation_observed = asyncio.Event()
    next_started = asyncio.Event()
    release_next = asyncio.Event()

    async def processor(document, **_dependencies):
        if document.id == "cancelled":
            cancellation_observed.set()
            raise asyncio.CancelledError
        next_started.set()
        await release_next.wait()

    worker = _make_worker(
        module, repository, processor, max_concurrency=1
    )
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(cancellation_observed)
        await _wait(next_started)
        assert not run_task.cancelled()
        worker.request_stop()
        release_next.set()
        await asyncio.wait_for(run_task, timeout=1)
    finally:
        worker.request_stop()
        release_next.set()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_processor_receives_all_injected_dependencies():
    module = _load_runner()
    repository = _Repository(recovered=[_document("injected")])
    gridfs_repository = object()
    ms2_client = object()
    circuit_breaker = CircuitBreaker()
    called = asyncio.Event()

    async def process(*_args, **_kwargs):
        called.set()

    processor = AsyncMock(side_effect=process)
    worker = module.ProcessingWorker(
        repository,
        gridfs_repository,
        ms2_client,
        processor=processor,
        circuit_breaker=circuit_breaker,
    )
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(called)
        worker.request_stop()
        await asyncio.wait_for(run_task, timeout=1)
        processor.assert_awaited_once_with(
            _document("injected"),
            processing_repository=repository,
            gridfs_repository=gridfs_repository,
            ms2_client=ms2_client,
            circuit_breaker=circuit_breaker,
            circuit_breaker_busy_retry_seconds=1.0,
        )
    finally:
        worker.request_stop()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_concurrent_processor_tasks_share_breaker_and_receive_poll_interval():
    module = _load_runner()
    repository = _Repository(
        recovered=[_document("first"), _document("second")]
    )
    circuit_breaker = CircuitBreaker()
    poll_interval = 2.75
    all_started = asyncio.Event()
    release = asyncio.Event()
    received = []

    async def processor(document, **dependencies):
        received.append(
            (
                document.id,
                dependencies["circuit_breaker"],
                dependencies["circuit_breaker_busy_retry_seconds"],
            )
        )
        if len(received) == 2:
            all_started.set()
        await release.wait()

    worker = _make_worker(
        module,
        repository,
        processor,
        circuit_breaker=circuit_breaker,
        max_concurrency=2,
        poll_initial_seconds=poll_interval,
    )
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(all_started)
        assert {entry[0] for entry in received} == {"first", "second"}
        assert all(entry[1] is circuit_breaker for entry in received)
        assert all(entry[2] == poll_interval for entry in received)
        worker.request_stop()
        release.set()
        await asyncio.wait_for(run_task, timeout=1)
    finally:
        worker.request_stop()
        release.set()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_circuit_breaker_release_outcome_is_logged_as_non_success(caplog):
    module = _load_runner()
    processor_module = importlib.import_module("src.worker.document_processor")
    outcome = processor_module.ProcessingResult.CIRCUIT_BREAKER_RELEASED
    repository = _Repository(recovered=[_document("released")])
    processor_called = asyncio.Event()

    async def processor(_document, **_dependencies):
        processor_called.set()
        return outcome

    worker = _make_worker(module, repository, processor)
    run_task = asyncio.create_task(worker.run())
    try:
        await _wait(processor_called)
        worker.request_stop()
        await asyncio.wait_for(run_task, timeout=1)
    finally:
        worker.request_stop()
        if not run_task.done():
            await asyncio.wait_for(run_task, timeout=1)

    assert f": {outcome.value}" in caplog.text
    matching = [
        record for record in caplog.records
        if record.getMessage().endswith(f": {outcome.value}")
    ]
    assert len(matching) == 1
    assert matching[0].levelno == logging.WARNING


def test_processing_worker_requires_an_explicit_circuit_breaker():
    module = _load_runner()

    with pytest.raises(TypeError):
        module.ProcessingWorker(_Repository(), object(), object())


@pytest.mark.asyncio
async def test_second_run_of_same_instance_is_rejected_after_completion():
    module = _load_runner()
    repository = _Repository()
    worker = _make_worker(module, repository, AsyncMock())
    worker.request_stop()
    await worker.run()
    with pytest.raises(RuntimeError):
        await worker.run()


@pytest.mark.parametrize(
    "capacity",
    [0, -1, True, False, 1.5, "2", None],
)
def test_invalid_capacity_is_rejected(capacity):
    module = _load_runner()
    with pytest.raises((TypeError, ValueError)):
        _make_worker(
            module,
            _Repository(),
            AsyncMock(),
            max_concurrency=capacity,
        )


@pytest.mark.parametrize(
    "interval",
    [0, -1, True, False, float("nan"), float("inf"), -float("inf"), "1"],
)
def test_invalid_poll_intervals_are_rejected(interval):
    module = _load_runner()
    with pytest.raises((TypeError, ValueError)):
        _make_worker(
            module,
            _Repository(),
            AsyncMock(),
            poll_initial_seconds=interval,
        )


@pytest.mark.parametrize(
    ("initial", "maximum"),
    [(2, 1), (5, 5 - 1e-9)],
)
def test_poll_maximum_below_initial_is_rejected(initial, maximum):
    module = _load_runner()
    with pytest.raises(ValueError):
        _make_worker(
            module,
            _Repository(),
            AsyncMock(),
            poll_initial_seconds=initial,
            poll_max_seconds=maximum,
        )


def test_nonfinite_poll_maximum_is_rejected():
    module = _load_runner()
    with pytest.raises((TypeError, ValueError)):
        _make_worker(
            module,
            _Repository(),
            AsyncMock(),
            poll_max_seconds=math.inf,
        )
