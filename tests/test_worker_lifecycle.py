"""RED contract tests for the worker-owned lifecycle."""

import asyncio
import importlib
import math
from unittest.mock import AsyncMock, Mock

import pytest


def _lifecycle_api():
    """Import lazily so this RED suite fails while the lifecycle is absent."""
    return importlib.import_module("src.worker.worker_lifecycle")


class FakeMongo:
    def __init__(self, events=None, *, connected=True, indexes_ready=True):
        self.events = events if events is not None else []
        self.connected = connected
        self.indexes_ready = indexes_ready
        self.db = object()
        self.connect = AsyncMock(side_effect=self._connect)
        self.close = AsyncMock(side_effect=self._close)

    async def _connect(self, *args, **kwargs):
        self.events.append("mongo.connect")

    async def _close(self):
        self.events.append("mongo.close")


class FakeHttpClient:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.aclose = AsyncMock(side_effect=self._close)

    async def _close(self):
        self.events.append("http.close")


class FakeWorker:
    def __init__(self, events=None, *, run=None):
        self.events = events if events is not None else []
        self.request_stop = Mock(side_effect=self._request_stop)
        self.run = AsyncMock(side_effect=run or self._run)
        self.cancelled = asyncio.Event()

    def _request_stop(self):
        self.events.append("worker.request_stop")

    async def _run(self):
        self.events.append("worker.run")


def _make_lifecycle(
    monkeypatch,
    *,
    mongo=None,
    http=None,
    worker=None,
    events=None,
    shutdown_grace_seconds=0.05,
):
    module = _lifecycle_api()
    events = events if events is not None else []
    mongo = mongo or FakeMongo(events)
    http = http or FakeHttpClient(events)
    worker = worker or FakeWorker(events)

    mongo_factory = Mock(return_value=mongo)
    http_factory = Mock(return_value=http)
    worker_factory = Mock(return_value=worker)
    processing_repository = object()
    gridfs_repository = object()
    ms2_client = object()

    monkeypatch.setattr(
        module, "ProcessingRepository", Mock(return_value=processing_repository)
    )
    monkeypatch.setattr(
        module, "GridFSRepository", Mock(return_value=gridfs_repository)
    )
    monkeypatch.setattr(module, "MS2Client", Mock(return_value=ms2_client))

    lifecycle = module.WorkerLifecycle(
        mongo_factory=mongo_factory,
        http_client_factory=http_factory,
        worker_factory=worker_factory,
        ms2_base_url="http://ms2.test",
        ms2_timeout_seconds=40.0,
        worker_kwargs={
            "max_concurrency": 2,
            "poll_initial_seconds": 1.0,
            "poll_max_seconds": 5.0,
        },
        shutdown_grace_seconds=shutdown_grace_seconds,
    )
    return (
        lifecycle,
        module,
        mongo,
        http,
        worker,
        mongo_factory,
        http_factory,
        worker_factory,
        processing_repository,
        gridfs_repository,
        ms2_client,
    )


async def _wait_for_event(event):
    await asyncio.wait_for(event.wait(), timeout=1.0)


def test_initialization_order_and_owned_dependencies(monkeypatch):
    events = []
    mongo = FakeMongo(events)
    http = FakeHttpClient(events)
    worker = FakeWorker(events)
    (
        lifecycle,
        module,
        _,
        _,
        _,
        mongo_factory,
        http_factory,
        worker_factory,
        processing_repository,
        gridfs_repository,
        ms2_client,
    ) = _make_lifecycle(
        monkeypatch, mongo=mongo, http=http, worker=worker, events=events
    )
    stop_event = asyncio.Event()

    async def scenario():
        await lifecycle.run(stop_event)

    asyncio.run(scenario())

    mongo_factory.assert_called_once_with()
    mongo.connect.assert_awaited_once_with()
    http_factory.assert_called_once_with()
    module.ProcessingRepository.assert_called_once_with(mongo.db)
    module.GridFSRepository.assert_called_once_with(mongo.db)
    module.MS2Client.assert_called_once_with(
        http, base_url="http://ms2.test", timeout_seconds=40.0
    )
    worker_factory.assert_called_once_with(
        processing_repository,
        gridfs_repository,
        ms2_client,
        max_concurrency=2,
        poll_initial_seconds=1.0,
        poll_max_seconds=5.0,
    )
    assert events.index("mongo.connect") < events.index("worker.run")
    assert events.index("worker.run") < events.index("http.close")
    assert events.index("http.close") < events.index("mongo.close")


@pytest.mark.parametrize(
    ("connected", "indexes_ready"),
    [(False, True), (True, False), (False, False)],
)
def test_unavailable_mongo_or_indexes_prevents_runner_start(
    monkeypatch, connected, indexes_ready
):
    events = []
    mongo = FakeMongo(
        events, connected=connected, indexes_ready=indexes_ready
    )
    lifecycle, _, _, _, worker, *_ = _make_lifecycle(
        monkeypatch, mongo=mongo, events=events
    )

    with pytest.raises(RuntimeError):
        asyncio.run(lifecycle.run(asyncio.Event()))

    worker.run.assert_not_awaited()
    mongo.close.assert_awaited_once()
    assert events[-1] == "mongo.close"


def test_connection_failure_closes_mongo_and_propagates_primary(monkeypatch):
    failure = OSError("connection failed")
    mongo = FakeMongo()
    mongo.connect.side_effect = failure
    lifecycle, _, _, http, worker, *_ = _make_lifecycle(
        monkeypatch, mongo=mongo
    )

    with pytest.raises(OSError) as raised:
        asyncio.run(lifecycle.run(asyncio.Event()))

    assert raised.value is failure
    mongo.close.assert_awaited_once()
    http.aclose.assert_not_awaited()
    worker.run.assert_not_awaited()


@pytest.mark.parametrize(
    ("failing_dependency", "http_created"),
    [
        ("GridFSRepository", False),
        ("MS2Client", True),
        ("worker_factory", True),
    ],
)
def test_partial_initialization_failure_closes_acquired_resources(
    monkeypatch, failing_dependency, http_created
):
    failure = ValueError(f"{failing_dependency} construction failed")
    lifecycle, module, mongo, http, worker, _, http_factory, worker_factory, *_ = (
        _make_lifecycle(monkeypatch)
    )
    if failing_dependency == "worker_factory":
        worker_factory.side_effect = failure
    else:
        getattr(module, failing_dependency).side_effect = failure

    with pytest.raises(ValueError) as raised:
        asyncio.run(lifecycle.run(asyncio.Event()))

    assert raised.value is failure
    if http_created:
        http_factory.assert_called_once_with()
        http.aclose.assert_awaited_once()
    else:
        http_factory.assert_not_called()
        http.aclose.assert_not_awaited()
    mongo.close.assert_awaited_once()
    worker.run.assert_not_awaited()
    if http_created:
        assert http.events.index("http.close") < mongo.events.index("mongo.close")


def test_stop_already_requested_acquires_no_resources(monkeypatch):
    lifecycle, _, mongo, http, worker, mongo_factory, http_factory, *_ = (
        _make_lifecycle(monkeypatch)
    )
    stop_event = asyncio.Event()
    stop_event.set()

    asyncio.run(lifecycle.run(stop_event))

    mongo_factory.assert_not_called()
    http_factory.assert_not_called()
    mongo.close.assert_not_awaited()
    http.aclose.assert_not_awaited()
    worker.run.assert_not_awaited()


def test_stop_during_initialization_skips_runner_and_closes_resources(monkeypatch):
    stop_event = asyncio.Event()
    events = []
    mongo = FakeMongo(events)

    async def connect_then_stop():
        await mongo._connect()
        stop_event.set()

    mongo.connect.side_effect = connect_then_stop
    lifecycle, _, _, http, worker, mongo_factory, http_factory, *_ = _make_lifecycle(
        monkeypatch, mongo=mongo, events=events
    )

    asyncio.run(lifecycle.run(stop_event))

    worker.run.assert_not_awaited()
    mongo_factory.assert_called_once_with()
    mongo.close.assert_awaited_once()
    http_factory.assert_not_called()
    http.aclose.assert_not_awaited()
    assert events == ["mongo.connect", "mongo.close"]


def test_lifecycle_instance_rejects_second_run(monkeypatch):
    lifecycle, _, mongo, http, worker, *_ = _make_lifecycle(monkeypatch)
    asyncio.run(lifecycle.run(asyncio.Event()))

    with pytest.raises(RuntimeError):
        asyncio.run(lifecycle.run(asyncio.Event()))

    assert mongo.connect.await_count == 1
    assert http.aclose.await_count == 1
    assert worker.run.await_count == 1


def test_runner_normal_completion_closes_resources(monkeypatch):
    lifecycle, _, mongo, http, worker, *_ = _make_lifecycle(monkeypatch)

    asyncio.run(lifecycle.run(asyncio.Event()))

    worker.run.assert_awaited_once()
    http.aclose.assert_awaited_once()
    mongo.close.assert_awaited_once()


def test_fatal_runner_error_is_propagated_without_restart(monkeypatch):
    failure = LookupError("fatal runner error")
    worker = FakeWorker(run=Mock(side_effect=failure))
    lifecycle, _, mongo, http, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker
    )

    with pytest.raises(LookupError) as raised:
        asyncio.run(lifecycle.run(asyncio.Event()))

    assert raised.value is failure
    assert worker.run.await_count == 1
    http.aclose.assert_awaited_once()
    mongo.close.assert_awaited_once()


def test_simultaneous_runner_failure_and_stop_preserves_runner_error(monkeypatch):
    failure = LookupError("fatal runner error")
    stop_event = asyncio.Event()

    async def fail_and_signal_stop():
        stop_event.set()
        raise failure

    worker = FakeWorker(run=fail_and_signal_stop)
    lifecycle, _, _, _, _, *_ = _make_lifecycle(monkeypatch, worker=worker)

    with pytest.raises(LookupError) as raised:
        asyncio.run(lifecycle.run(stop_event))

    assert raised.value is failure
    assert worker.run.await_count == 1


def test_orderly_stop_requests_drain_without_cancelling_runner(monkeypatch):
    entered = asyncio.Event()
    stop_requested = asyncio.Event()
    finish = asyncio.Event()

    async def drain():
        entered.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            worker.cancelled.set()
            raise

    worker = FakeWorker(run=drain)
    worker.request_stop.side_effect = lambda: stop_requested.set()
    lifecycle, _, mongo, http, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker
    )
    stop_event = asyncio.Event()

    async def scenario():
        task = asyncio.create_task(lifecycle.run(stop_event))
        try:
            await _wait_for_event(entered)
            stop_event.set()
            await _wait_for_event(stop_requested)
            assert worker.request_stop.call_count == 1
            assert worker.run.await_count == 1
            assert not http.aclose.await_count
            assert not mongo.close.await_count
        finally:
            finish.set()
            await task

    asyncio.run(scenario())

    assert worker.run.await_count == 1
    assert worker.cancelled.is_set() is False
    http.aclose.assert_awaited_once()
    mongo.close.assert_awaited_once()


def test_graceful_drain_completes_inside_deadline_without_cancel(monkeypatch):
    entered = asyncio.Event()
    stop_requested = asyncio.Event()
    finish = asyncio.Event()

    async def drain():
        entered.set()
        await finish.wait()

    worker = FakeWorker(run=drain)
    worker.request_stop.side_effect = lambda: stop_requested.set()
    lifecycle, _, _, http, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker, shutdown_grace_seconds=0.2
    )
    stop_event = asyncio.Event()

    async def scenario():
        task = asyncio.create_task(lifecycle.run(stop_event))
        try:
            await _wait_for_event(entered)
            stop_event.set()
            await _wait_for_event(stop_requested)
        finally:
            finish.set()
            await task

    asyncio.run(scenario())

    worker.run.assert_awaited_once()
    http.aclose.assert_awaited_once()


def test_shutdown_timeout_cancels_once_and_raises_timeout_error(monkeypatch):
    worker = FakeWorker()
    runner_started = asyncio.Event()
    cancellation_received = asyncio.Event()

    async def wait_for_cancellation():
        runner_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            worker.cancelled.set()
            cancellation_received.set()
            raise

    worker.run = AsyncMock(side_effect=wait_for_cancellation)
    lifecycle, _, _, http, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker, shutdown_grace_seconds=0.01
    )
    stop_event = asyncio.Event()

    async def scenario():
        task = asyncio.create_task(lifecycle.run(stop_event))
        try:
            await _wait_for_event(runner_started)
            stop_event.set()
            await _wait_for_event(cancellation_received)
            with pytest.raises(TimeoutError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert worker.run.await_count == 1
    assert worker.cancelled.is_set()
    http.aclose.assert_awaited_once()


def test_resources_remain_open_until_runner_cancellation_cleanup_finishes(
    monkeypatch,
):
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    runner_started = asyncio.Event()
    stop_requested = asyncio.Event()
    events = []

    async def runner_cleanup():
        runner_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("runner.cleanup.start")
            cleaning.set()
            await finish_cleanup.wait()
            events.append("runner.cleanup.end")
            raise

    worker = FakeWorker(events, run=runner_cleanup)
    worker.request_stop.side_effect = lambda: stop_requested.set()
    lifecycle, _, _, http, _, *_ = _make_lifecycle(
        monkeypatch,
        worker=worker,
        events=events,
        shutdown_grace_seconds=0.01,
    )
    stop_event = asyncio.Event()

    async def scenario():
        task = asyncio.create_task(lifecycle.run(stop_event))
        try:
            await _wait_for_event(runner_started)
            stop_event.set()
            await _wait_for_event(stop_requested)
            await _wait_for_event(cleaning)
            assert not http.aclose.await_count
            finish_cleanup.set()
            with pytest.raises(TimeoutError):
                await task
        finally:
            finish_cleanup.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert events.index("runner.cleanup.end") < events.index("http.close")


def test_external_cancellation_is_preserved_and_runner_cleanup_is_awaited(
    monkeypatch,
):
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    runner_started = asyncio.Event()
    events = []

    async def runner_cleanup():
        runner_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("runner.cleanup.start")
            cleaning.set()
            await finish_cleanup.wait()
            events.append("runner.cleanup.end")
            raise

    worker = FakeWorker(events, run=runner_cleanup)
    lifecycle, _, _, http, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker, events=events
    )

    async def scenario():
        task = asyncio.create_task(lifecycle.run(asyncio.Event()))
        try:
            await _wait_for_event(runner_started)
            task.cancel()
            await _wait_for_event(cleaning)
            assert not http.aclose.await_count
            finish_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish_cleanup.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    worker.request_stop.assert_called_once()
    assert events.index("runner.cleanup.end") < events.index("http.close")


def test_repeated_external_cancellation_does_not_recancel_runner(monkeypatch):
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    runner_started = asyncio.Event()
    worker = FakeWorker()
    cancel_count = 0

    async def runner_cleanup():
        nonlocal cancel_count
        runner_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_count += 1
            cleaning.set()
            await finish_cleanup.wait()
            raise

    worker.run = AsyncMock(side_effect=runner_cleanup)
    lifecycle, _, _, http, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker
    )

    async def scenario():
        task = asyncio.create_task(lifecycle.run(asyncio.Event()))
        try:
            await _wait_for_event(runner_started)
            task.cancel()
            await _wait_for_event(cleaning)
            task.cancel()
            task.cancel()
            assert task.cancelling() == 3
            done, _ = await asyncio.wait({task}, timeout=0.01)
            assert not done
            assert not http.aclose.await_count
        finally:
            finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert cancel_count == 1
    http.aclose.assert_awaited_once()


def test_repeated_external_cancellation_during_http_close_finishes_cleanup(
    monkeypatch,
):
    runner_started = asyncio.Event()
    http_close_started = asyncio.Event()
    finish_http_close = asyncio.Event()
    events = []

    async def finish_runner():
        runner_started.set()

    worker = FakeWorker(events, run=finish_runner)
    http = FakeHttpClient(events)

    async def blocked_http_close():
        events.append("http.close.start")
        http_close_started.set()
        await finish_http_close.wait()
        events.append("http.close.end")

    http.aclose.side_effect = blocked_http_close
    lifecycle, _, mongo, _, _, _, http_factory, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker, http=http, events=events
    )

    async def scenario():
        task = asyncio.create_task(lifecycle.run(asyncio.Event()))
        try:
            await _wait_for_event(runner_started)
            await _wait_for_event(http_close_started)
            task.cancel()
            task.cancel()
            assert task.cancelling() == 2
            done, _ = await asyncio.wait({task}, timeout=0.01)
            assert not done
            assert http.aclose.await_count == 1
            assert mongo.close.await_count == 0
        finally:
            finish_http_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    http_factory.assert_called_once_with()
    assert http.aclose.await_count == 1
    assert mongo.close.await_count == 1
    assert events.index("http.close.end") < events.index("mongo.close")


def test_external_cancellation_during_mongo_connect_closes_only_mongo(
    monkeypatch,
):
    connect_started = asyncio.Event()
    never_release = asyncio.Event()
    mongo = FakeMongo()

    async def blocked_connect():
        connect_started.set()
        await never_release.wait()

    mongo.connect.side_effect = blocked_connect
    lifecycle, _, _, http, worker, _, http_factory, *_ = _make_lifecycle(
        monkeypatch, mongo=mongo
    )

    async def scenario():
        task = asyncio.create_task(lifecycle.run(asyncio.Event()))
        try:
            await _wait_for_event(connect_started)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            never_release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    mongo.close.assert_awaited_once()
    http_factory.assert_not_called()
    http.aclose.assert_not_awaited()
    worker.run.assert_not_awaited()


def test_http_closes_before_mongo_and_both_close_failures_are_attempted(
    monkeypatch,
):
    events = []
    mongo = FakeMongo(events)
    http = FakeHttpClient(events)

    async def fail_http_close():
        events.append("http.close")
        raise RuntimeError("http close")

    async def fail_mongo_close():
        events.append("mongo.close")
        raise OSError("mongo close")

    http.aclose.side_effect = fail_http_close
    mongo.close.side_effect = fail_mongo_close
    lifecycle, _, _, _, _, *_ = _make_lifecycle(
        monkeypatch, mongo=mongo, http=http, events=events
    )

    with pytest.raises(RuntimeError, match="http close"):
        asyncio.run(lifecycle.run(asyncio.Event()))

    assert events[-2:] == ["http.close", "mongo.close"]
    http.aclose.assert_awaited_once()
    mongo.close.assert_awaited_once()


def test_primary_error_survives_cleanup_failure(monkeypatch):
    primary = ValueError("primary failure")
    lifecycle, _, mongo, http, worker, *_ = _make_lifecycle(monkeypatch)
    worker.run.side_effect = primary
    http.aclose.side_effect = RuntimeError("secondary close failure")

    with pytest.raises(ValueError) as raised:
        asyncio.run(lifecycle.run(asyncio.Event()))

    assert raised.value is primary
    http.aclose.assert_awaited_once()
    mongo.close.assert_awaited_once()


def test_first_close_error_propagates_when_there_is_no_primary(monkeypatch):
    lifecycle, _, mongo, http, _, *_ = _make_lifecycle(monkeypatch)
    http_error = RuntimeError("http close failure")
    mongo.close.side_effect = OSError("mongo close failure")
    http.aclose.side_effect = http_error

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(lifecycle.run(asyncio.Event()))

    assert raised.value is http_error
    http.aclose.assert_awaited_once()
    mongo.close.assert_awaited_once()


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        False,
        float("inf"),
        float("-inf"),
        float("nan"),
        "1",
        None,
        object(),
    ],
)
def test_invalid_shutdown_grace_is_rejected(monkeypatch, value):
    module = _lifecycle_api()

    with pytest.raises((TypeError, ValueError)):
        module.WorkerLifecycle(
            mongo_factory=Mock(),
            http_client_factory=Mock(),
            worker_factory=Mock(),
            ms2_base_url="http://ms2.test",
            shutdown_grace_seconds=value,
        )


def test_shutdown_grace_accepts_finite_positive_real(monkeypatch):
    module = _lifecycle_api()
    lifecycle = module.WorkerLifecycle(
        mongo_factory=Mock(),
        http_client_factory=Mock(),
        worker_factory=Mock(),
        ms2_base_url="http://ms2.test",
        shutdown_grace_seconds=0.25,
    )

    assert math.isfinite(lifecycle.shutdown_grace_seconds)
    assert lifecycle.shutdown_grace_seconds == 0.25


def test_external_cancellation_during_stop_waiter_cleanup_is_preserved(
    monkeypatch,
):
    runner_started = asyncio.Event()
    finish_runner = asyncio.Event()
    stop_waiter_started = asyncio.Event()
    stop_waiter_cleanup_started = asyncio.Event()
    finish_stop_waiter_cleanup = asyncio.Event()
    events = []

    class BlockingStopEvent:
        def is_set(self):
            return False

        async def wait(self):
            stop_waiter_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stop_waiter_cleanup_started.set()
                await finish_stop_waiter_cleanup.wait()
                raise

    async def runner():
        runner_started.set()
        await finish_runner.wait()

    worker = FakeWorker(events, run=runner)
    lifecycle, _, mongo, http, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker, events=events
    )

    async def scenario():
        task = asyncio.create_task(lifecycle.run(BlockingStopEvent()))
        try:
            await _wait_for_event(runner_started)
            await _wait_for_event(stop_waiter_started)
            finish_runner.set()
            await _wait_for_event(stop_waiter_cleanup_started)
            task.cancel()
            finish_stop_waiter_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish_runner.set()
            finish_stop_waiter_cleanup.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    http.aclose.assert_awaited_once()
    mongo.close.assert_awaited_once()


def test_runner_failure_racing_external_cancellation_is_consumed(
    monkeypatch,
):
    module = _lifecycle_api()
    failure = RuntimeError("runner failed before cancellation")
    runner_finished = asyncio.Event()
    wait_result_ready = asyncio.Event()
    allow_wait_to_return = asyncio.Event()
    runner_task_holder = {}

    async def fail_runner():
        runner_task_holder["task"] = asyncio.current_task()
        runner_finished.set()
        raise failure

    worker = FakeWorker(run=fail_runner)
    lifecycle, _, _, _, _, *_ = _make_lifecycle(
        monkeypatch, worker=worker
    )
    original_wait = asyncio.wait

    async def gated_wait(tasks, *args, **kwargs):
        done, pending = await original_wait(tasks, *args, **kwargs)
        if runner_finished.is_set():
            wait_result_ready.set()
            await allow_wait_to_return.wait()
        return done, pending

    monkeypatch.setattr(module.asyncio, "wait", gated_wait)

    async def scenario():
        task = asyncio.create_task(lifecycle.run(asyncio.Event()))
        try:
            await _wait_for_event(wait_result_ready)
            runner_task = runner_task_holder["task"]
            assert runner_task.done()
            task.cancel()
            allow_wait_to_return.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            # CPython's flag verifies the exception was retrieved without
            # retrieving it here and masking an unconsumed task exception.
            assert runner_task._log_traceback is False
        finally:
            allow_wait_to_return.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
