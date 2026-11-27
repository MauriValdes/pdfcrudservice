"""Process entrypoint and portable signal handling for the PDF worker."""

import asyncio
import signal
from typing import Callable

import httpx

from src.database.mongodb import MongoDB
from src.worker.worker_config import WorkerSettings
from src.worker.worker_lifecycle import WorkerLifecycle
from src.worker.worker_runner import ProcessingWorker


def install_signal_handlers(
    loop,
    stop_event: asyncio.Event,
    *,
    signal_module=signal,
) -> Callable[[], None]:
    """Install stop-request handlers and return an idempotent restoration hook."""
    installed: list[tuple[int, object]] = []
    restored = False

    def restore_handlers() -> None:
        nonlocal restored
        if restored:
            return
        restored = True

        first_error: BaseException | None = None
        for signum, previous_handler in reversed(installed):
            try:
                signal_module.signal(signum, previous_handler)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def request_stop(_signum, _frame) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    try:
        for name in ("SIGINT", "SIGTERM"):
            signum = getattr(signal_module, name, None)
            if signum is None:
                continue
            previous_handler = signal_module.signal(signum, request_stop)
            installed.append((signum, previous_handler))
    except BaseException:
        try:
            restore_handlers()
        except BaseException:
            # Preserve the signal-installation failure as the primary error.
            pass
        raise

    return restore_handlers


async def run_worker(
    config: WorkerSettings | None = None,
    *,
    lifecycle_factory=None,
    mongo_factory=None,
    http_client_factory=None,
    worker_factory=None,
    signal_module=signal,
) -> None:
    """Validate settings, assemble the lifecycle, then run it once."""
    if config is None:
        config = WorkerSettings()
    if not config.enabled:
        raise RuntimeError("PDF worker is disabled; set PDF_WORKER_ENABLED=true")

    # Resolve injectable defaults at invocation time, without constructing resources.
    if lifecycle_factory is None:
        lifecycle_factory = WorkerLifecycle
    if mongo_factory is None:
        mongo_factory = MongoDB
    if http_client_factory is None:
        http_client_factory = httpx.AsyncClient
    if worker_factory is None:
        worker_factory = ProcessingWorker

    lifecycle = lifecycle_factory(
        mongo_factory=mongo_factory,
        http_client_factory=http_client_factory,
        worker_factory=worker_factory,
        ms2_base_url=config.ms2_base_url,
        ms2_timeout_seconds=config.ms2_timeout_seconds,
        worker_kwargs={
            "max_concurrency": config.max_concurrency,
            "poll_initial_seconds": config.poll_initial_seconds,
            "poll_max_seconds": config.poll_max_seconds,
        },
        shutdown_grace_seconds=config.shutdown_grace_seconds,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    restore_handlers = install_signal_handlers(
        loop,
        stop_event,
        signal_module=signal_module,
    )
    try:
        await lifecycle.run(stop_event)
    except BaseException:
        try:
            restore_handlers()
        except BaseException:
            # A restoration failure must not replace a fatal error/cancellation.
            pass
        raise
    else:
        restore_handlers()


def main() -> None:
    """Run the async worker entrypoint under a fresh event loop."""
    config = WorkerSettings()
    asyncio.run(run_worker(config))
