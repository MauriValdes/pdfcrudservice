"""RED tests for the worker process entrypoint and portable signal handling."""

import asyncio
import importlib
import importlib.util
import runpy
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def _entrypoint_api():
    """Import lazily so RED is caused by the not-yet-created entrypoint."""
    return importlib.import_module("src.worker.entrypoint")


def _config(*, enabled=True):
    return SimpleNamespace(
        enabled=enabled,
        max_concurrency=2,
        poll_initial_seconds=1.0,
        poll_max_seconds=5.0,
        ms2_timeout_seconds=40.0,
        shutdown_grace_seconds=60.0,
        ms2_base_url="http://ms2.test",
    )


class FakeSignalModule:
    SIGINT = signal.SIGINT
    SIGTERM = signal.SIGTERM

    def __init__(self, *, fail_on=None):
        self.handlers = {
            self.SIGINT: "previous-int",
            self.SIGTERM: "previous-term",
        }
        self.calls = []
        self.fail_on = fail_on

    def getsignal(self, signum):
        return self.handlers[signum]

    def signal(self, signum, handler):
        self.calls.append((signum, handler))
        if signum == self.fail_on:
            raise OSError("signal registration failed")
        previous = self.handlers[signum]
        self.handlers[signum] = handler
        return previous


class FakeLoop:
    def __init__(self):
        self.scheduled = []

    def call_soon_threadsafe(self, callback):
        self.scheduled.append(callback)


def test_disabled_worker_is_rejected_without_acquiring_or_registering(
    monkeypatch,
):
    module = _entrypoint_api()
    lifecycle_factory = Mock()
    mongo_factory = Mock()
    http_client_factory = Mock()
    worker_factory = Mock()
    signals = FakeSignalModule()

    with pytest.raises(RuntimeError, match="disabled|enabled"):
        asyncio.run(
            module.run_worker(
                _config(enabled=False),
                lifecycle_factory=lifecycle_factory,
                mongo_factory=mongo_factory,
                http_client_factory=http_client_factory,
                worker_factory=worker_factory,
                signal_module=signals,
            )
        )

    lifecycle_factory.assert_not_called()
    mongo_factory.assert_not_called()
    http_client_factory.assert_not_called()
    worker_factory.assert_not_called()
    assert signals.calls == []


def test_enabled_entrypoint_passes_factories_and_all_configured_values():
    module = _entrypoint_api()
    config = _config()
    config.max_concurrency = 3
    config.poll_initial_seconds = 0.5
    config.poll_max_seconds = 4.0
    config.ms2_timeout_seconds = 42.5
    config.shutdown_grace_seconds = 75.0
    config.ms2_base_url = "https://worker-ms2.test/api/v2"
    lifecycle = Mock()
    lifecycle.run = AsyncMock()
    lifecycle_factory = Mock(return_value=lifecycle)
    mongo_factory = Mock(name="MongoDB")
    http_client_factory = Mock(name="AsyncClient")
    worker_factory = Mock(name="ProcessingWorker")
    signals = FakeSignalModule()

    asyncio.run(
        module.run_worker(
            config,
            lifecycle_factory=lifecycle_factory,
            mongo_factory=mongo_factory,
            http_client_factory=http_client_factory,
            worker_factory=worker_factory,
            signal_module=signals,
        )
    )

    lifecycle_factory.assert_called_once_with(
        mongo_factory=mongo_factory,
        http_client_factory=http_client_factory,
        worker_factory=worker_factory,
        ms2_base_url="https://worker-ms2.test/api/v2",
        ms2_timeout_seconds=42.5,
        worker_kwargs={
            "max_concurrency": 3,
            "poll_initial_seconds": 0.5,
            "poll_max_seconds": 4.0,
        },
        shutdown_grace_seconds=75.0,
    )
    lifecycle.run.assert_awaited_once()
    assert lifecycle.run.await_args.args[0].is_set() is False
    mongo_factory.assert_not_called()
    http_client_factory.assert_not_called()
    worker_factory.assert_not_called()


def test_enabled_entrypoint_propagates_fatal_lifecycle_error_and_restores(
    monkeypatch,
):
    module = _entrypoint_api()
    failure = RuntimeError("fatal lifecycle failure")
    lifecycle = Mock()
    lifecycle.run = AsyncMock(side_effect=failure)
    lifecycle_factory = Mock(return_value=lifecycle)
    signals = FakeSignalModule()

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            module.run_worker(
                _config(),
                lifecycle_factory=lifecycle_factory,
                mongo_factory=Mock(),
                http_client_factory=Mock(),
                worker_factory=Mock(),
                signal_module=signals,
            )
        )

    assert raised.value is failure
    assert signals.handlers[signals.SIGINT] == "previous-int"
    assert signals.handlers[signals.SIGTERM] == "previous-term"


def test_enabled_entrypoint_restores_signal_handlers_after_cancellation():
    module = _entrypoint_api()
    lifecycle = Mock()
    lifecycle.run = AsyncMock(side_effect=asyncio.CancelledError())
    signals = FakeSignalModule()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            module.run_worker(
                _config(),
                lifecycle_factory=Mock(return_value=lifecycle),
                mongo_factory=Mock(),
                http_client_factory=Mock(),
                worker_factory=Mock(),
                signal_module=signals,
            )
        )

    assert signals.handlers[signals.SIGINT] == "previous-int"
    assert signals.handlers[signals.SIGTERM] == "previous-term"


def test_signal_handlers_request_stop_via_loop_and_restore_previous_handlers():
    module = _entrypoint_api()
    signals = FakeSignalModule()
    loop = FakeLoop()
    stop_event = asyncio.Event()

    restore = module.install_signal_handlers(
        loop, stop_event, signal_module=signals
    )

    assert [signum for signum, _ in signals.calls] == [
        signals.SIGINT,
        signals.SIGTERM,
    ]
    handlers = dict(signals.calls)
    handlers[signals.SIGINT](signals.SIGINT, None)
    handlers[signals.SIGTERM](signals.SIGTERM, None)
    for callback in loop.scheduled:
        callback()
    assert stop_event.is_set()

    restore()

    assert signals.handlers[signals.SIGINT] == "previous-int"
    assert signals.handlers[signals.SIGTERM] == "previous-term"


def test_signal_registration_failure_restores_partially_installed_handler():
    module = _entrypoint_api()
    signals = FakeSignalModule(fail_on=FakeSignalModule.SIGTERM)
    loop = FakeLoop()

    with pytest.raises(OSError, match="registration failed"):
        module.install_signal_handlers(
            loop, asyncio.Event(), signal_module=signals
        )

    assert signals.handlers[signals.SIGINT] == "previous-int"
    assert signals.handlers[signals.SIGTERM] == "previous-term"


def test_run_worker_does_not_start_lifecycle_if_signal_installation_fails():
    module = _entrypoint_api()
    signals = FakeSignalModule(fail_on=FakeSignalModule.SIGTERM)
    lifecycle = Mock()
    lifecycle.run = AsyncMock()
    lifecycle_factory = Mock(return_value=lifecycle)

    with pytest.raises(OSError, match="registration failed"):
        asyncio.run(
            module.run_worker(
                _config(),
                lifecycle_factory=lifecycle_factory,
                mongo_factory=Mock(),
                http_client_factory=Mock(),
                worker_factory=Mock(),
                signal_module=signals,
            )
        )

    lifecycle.run.assert_not_awaited()
    assert signals.handlers[signals.SIGINT] == "previous-int"
    assert signals.handlers[signals.SIGTERM] == "previous-term"


def test_signal_handlers_are_idempotent_for_repeated_requests():
    module = _entrypoint_api()
    signals = FakeSignalModule()
    loop = FakeLoop()
    stop_event = asyncio.Event()
    restore = module.install_signal_handlers(
        loop, stop_event, signal_module=signals
    )
    handler = dict(signals.calls)[signals.SIGTERM]

    handler(signals.SIGTERM, None)
    handler(signals.SIGTERM, None)
    for callback in loop.scheduled:
        callback()

    assert stop_event.is_set()
    restore()


def test_importing_entrypoint_does_not_construct_resources_or_register_signals(
    monkeypatch,
):
    from src.database import mongodb
    from src.worker import ms2_client, worker_lifecycle, worker_runner

    entrypoint_name = "src.worker.entrypoint"
    main_name = "src.worker.__main__"
    worker_package = importlib.import_module("src.worker")
    sentinel = object()
    names = (entrypoint_name, main_name)
    saved_modules = {name: sys.modules.get(name, sentinel) for name in names}
    saved_attributes = {
        "entrypoint": getattr(worker_package, "entrypoint", sentinel),
        "__main__": getattr(worker_package, "__main__", sentinel),
    }

    def evict_module(name):
        sys.modules.pop(name, None)
        attribute = name.rsplit(".", maxsplit=1)[-1]
        if hasattr(worker_package, attribute):
            delattr(worker_package, attribute)

    try:
        for name in names:
            evict_module(name)

        with monkeypatch.context() as patch:
            mongo_factory = Mock()
            client_factory = Mock()
            lifecycle_factory = Mock()
            worker_factory = Mock()
            patch.setattr(mongodb, "MongoDB", mongo_factory)
            patch.setattr(ms2_client.httpx, "AsyncClient", client_factory)
            patch.setattr(
                worker_lifecycle, "WorkerLifecycle", lifecycle_factory
            )
            patch.setattr(worker_runner, "ProcessingWorker", worker_factory)

            signal_module = importlib.import_module("signal")
            signal_spy = Mock(wraps=signal_module.signal)
            patch.setattr(signal_module, "signal", signal_spy)
            asyncio_run_spy = Mock(wraps=asyncio.run)
            patch.setattr(asyncio, "run", asyncio_run_spy)

            entrypoint = importlib.import_module(entrypoint_name)
            main_spy = Mock()
            patch.setattr(entrypoint, "main", main_spy)
            ordinary_main = importlib.import_module(main_name)

            assert callable(entrypoint.main)
            assert callable(ordinary_main.main)
            main_spy.assert_not_called()
            signal_spy.assert_not_called()
            asyncio_run_spy.assert_not_called()
            mongo_factory.assert_not_called()
            client_factory.assert_not_called()
            lifecycle_factory.assert_not_called()
            worker_factory.assert_not_called()
    finally:
        for name, saved in saved_modules.items():
            if saved is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        for attribute, saved in saved_attributes.items():
            if saved is sentinel:
                if hasattr(worker_package, attribute):
                    delattr(worker_package, attribute)
            else:
                setattr(worker_package, attribute, saved)


def test_main_delegates_run_worker_through_asyncio_run(monkeypatch):
    module = _entrypoint_api()
    config = _config()
    settings_factory = Mock(return_value=config)
    run_worker = AsyncMock()
    original_asyncio_run = asyncio.run
    run_spy = Mock(side_effect=original_asyncio_run)
    monkeypatch.setattr(module, "WorkerSettings", settings_factory)
    monkeypatch.setattr(module, "run_worker", run_worker)
    monkeypatch.setattr(module.asyncio, "run", run_spy)

    module.main()

    settings_factory.assert_called_once_with()
    run_spy.assert_called_once()
    run_worker.assert_awaited_once_with(config)


def test_dunder_main_calls_main_only_when_executed_as_main(monkeypatch):
    entrypoint = _entrypoint_api()
    main_spy = Mock()
    monkeypatch.setattr(entrypoint, "main", main_spy)
    module_name = "src.worker.__main__"
    worker_package = importlib.import_module("src.worker")
    sentinel = object()
    saved_module = sys.modules.get(module_name, sentinel)
    saved_attribute = getattr(worker_package, "__main__", sentinel)

    try:
        sys.modules.pop(module_name, None)
        if hasattr(worker_package, "__main__"):
            delattr(worker_package, "__main__")

        imported_module = importlib.import_module(module_name)
        main_spy.assert_not_called()
        assert callable(imported_module.main)

        sys.modules.pop(module_name, None)
        if hasattr(worker_package, "__main__"):
            delattr(worker_package, "__main__")
        runpy.run_module(module_name, run_name="__main__")
        main_spy.assert_called_once_with()
    finally:
        if saved_module is sentinel:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_module
        if saved_attribute is sentinel:
            if hasattr(worker_package, "__main__"):
                delattr(worker_package, "__main__")
        else:
            setattr(worker_package, "__main__", saved_attribute)
