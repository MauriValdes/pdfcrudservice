"""Owned resources, supervision, and orderly shutdown for the PDF worker."""

import asyncio
from contextlib import AsyncExitStack
import logging
import math
from numbers import Real
from typing import Any, Callable

import httpx

from src.database.mongodb import MongoDB
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import ProcessingRepository
from src.worker.circuit_breaker import CircuitBreaker
from src.worker.ms2_client import MS2Client
from src.worker.worker_runner import ProcessingWorker

logger = logging.getLogger(__name__)


class _StopBeforeRunner(Exception):
    """Internal control flow for a stop requested during initialization."""


class WorkerLifecycle:
    """Own a worker's MongoDB/HTTP resources and supervise its single run.

    The grace period bounds only orderly draining. An operation that suppresses
    cancellation can prevent the runner from finishing, so this class cannot
    guarantee an absolute shutdown deadline. Cancellation also cannot undo
    remote effects. The lifecycle performs no compensating writes or lease
    releases; leases and fencing remain the recovery mechanism.
    """

    def __init__(
        self,
        *,
        mongo_factory: Callable[[], MongoDB],
        http_client_factory: Callable[[], httpx.AsyncClient],
        worker_factory: Callable[..., ProcessingWorker],
        circuit_breaker_factory: Callable[[], CircuitBreaker] = CircuitBreaker,
        ms2_base_url: str,
        ms2_timeout_seconds: float = 40.0,
        worker_kwargs: dict | None = None,
        shutdown_grace_seconds: float = 60.0,
    ):
        if isinstance(shutdown_grace_seconds, bool) or not isinstance(
            shutdown_grace_seconds, Real
        ):
            raise TypeError("shutdown_grace_seconds must be a real number")
        try:
            grace = float(shutdown_grace_seconds)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "shutdown_grace_seconds must be finite and positive"
            ) from error
        if not math.isfinite(grace) or grace <= 0:
            raise ValueError(
                "shutdown_grace_seconds must be finite and positive"
            )

        self._mongo_factory = mongo_factory
        self._http_client_factory = http_client_factory
        self._worker_factory = worker_factory
        self._circuit_breaker_factory = circuit_breaker_factory
        self._ms2_base_url = ms2_base_url
        self._ms2_timeout_seconds = ms2_timeout_seconds
        self._worker_kwargs = dict(worker_kwargs or {})
        self.shutdown_grace_seconds = grace
        self._started = False

    async def run(self, stop_event: asyncio.Event) -> None:
        """Initialize owned dependencies, run once, drain, then close resources."""
        if self._started:
            raise RuntimeError("WorkerLifecycle instances can only run once")
        # Set before the first await so concurrent calls cannot both acquire.
        self._started = True

        if stop_event.is_set():
            return

        stack = AsyncExitStack()
        await stack.__aenter__()
        cleanup_errors: list[tuple[str, BaseException]] = []
        mongo = None
        http_client = None
        worker = None
        runner_task: asyncio.Task | None = None
        stop_waiter: asyncio.Task | None = None
        stop_requested = False
        runner_cancel_requested = False
        primary_error: BaseException | None = None
        primary_traceback = None
        external_cancellation: asyncio.CancelledError | None = None

        def request_stop_once() -> None:
            nonlocal stop_requested
            if worker is not None and not stop_requested:
                # Mark first, including if request_stop itself raises.
                stop_requested = True
                worker.request_stop()

        async def cancel_and_join_runner() -> None:
            nonlocal runner_cancel_requested
            if runner_task is None:
                return
            if runner_task.done():
                try:
                    runner_task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as error:
                    if error is not primary_error:
                        self._record_cleanup_error(
                            cleanup_errors, "worker.runner", error
                        )
                return
            if worker is not None:
                try:
                    request_stop_once()
                except BaseException as error:
                    self._record_cleanup_error(
                        cleanup_errors, "worker.request_stop", error
                    )
            if not runner_cancel_requested:
                runner_cancel_requested = True
                runner_task.cancel()
            error, interrupted_by_external_cancel = (
                await self._await_task_shielded(runner_task)
            )
            if error is not None and not isinstance(
                error, asyncio.CancelledError
            ):
                self._record_cleanup_error(
                    cleanup_errors, "worker.runner", error
                )
            if interrupted_by_external_cancel:
                # The original cancellation is already being preserved by run().
                return

        try:
            mongo = self._mongo_factory()
            stack.push_async_callback(
                self._close_resource,
                cleanup_errors,
                "mongodb",
                mongo.close,
            )
            await mongo.connect()
            if not mongo.connected or not mongo.indexes_ready:
                raise RuntimeError(
                    "MongoDB connection or required indexes are not ready"
                )

            # Do not acquire any further resources if shutdown arrived during
            # connect or index initialization.
            if stop_event.is_set():
                raise _StopBeforeRunner

            processing_repository = ProcessingRepository(mongo.db)
            gridfs_repository = GridFSRepository(mongo.db)

            http_client = self._http_client_factory()
            stack.push_async_callback(
                self._close_resource,
                cleanup_errors,
                "http",
                http_client.aclose,
            )
            ms2_client = MS2Client(
                http_client,
                base_url=self._ms2_base_url,
                timeout_seconds=self._ms2_timeout_seconds,
            )
            circuit_breaker = self._circuit_breaker_factory()
            worker = self._worker_factory(
                processing_repository,
                gridfs_repository,
                ms2_client,
                circuit_breaker=circuit_breaker,
                **self._worker_kwargs,
            )

            if not stop_event.is_set():
                runner_task = asyncio.create_task(worker.run())
                stop_waiter = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait(
                    {runner_task, stop_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # If both become ready together, always consume runner.result().
                if runner_task in done or runner_task.done():
                    runner_task.result()
                else:
                    request_stop_once()
                    done, _ = await asyncio.wait(
                        {runner_task},
                        timeout=self.shutdown_grace_seconds,
                    )
                    if runner_task not in done and not runner_task.done():
                        runner_cancel_requested = True
                        runner_task.cancel()
                        runner_error, interrupted = (
                            await self._await_task_shielded(runner_task)
                        )
                        if interrupted:
                            raise asyncio.CancelledError
                        if runner_error is not None and not isinstance(
                            runner_error, asyncio.CancelledError
                        ):
                            raise runner_error
                        raise TimeoutError(
                            "Worker runner exceeded shutdown grace period"
                        )
                    runner_task.result()
        except _StopBeforeRunner:
            pass
        except BaseException as error:
            primary_error = error
            primary_traceback = error.__traceback__
            if isinstance(error, asyncio.CancelledError):
                external_cancellation = error
            # Any exceptional exit must settle the runner before resources close.
            await cancel_and_join_runner()
        finally:
            if stop_waiter is not None:
                interrupted = await self._cancel_and_consume(stop_waiter)
                if interrupted and external_cancellation is None:
                    external_cancellation = asyncio.CancelledError()

            close_task = asyncio.create_task(stack.aclose())
            close_error, interrupted = await self._await_task_shielded(close_task)
            if close_error is not None:
                self._record_cleanup_error(
                    cleanup_errors, "resource_stack", close_error
                )
            if interrupted and external_cancellation is None:
                external_cancellation = asyncio.CancelledError()

        # A pre-existing operation/cancellation remains primary over cleanup.
        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)
        if external_cancellation is not None:
            raise external_cancellation
        if cleanup_errors:
            raise cleanup_errors[0][1]

    async def _close_resource(
        self,
        errors: list[tuple[str, BaseException]],
        resource_name: str,
        close: Callable[[], Any],
    ) -> None:
        try:
            await close()
        except BaseException as error:
            self._record_cleanup_error(errors, resource_name, error)

    @staticmethod
    def _record_cleanup_error(
        errors: list[tuple[str, BaseException]],
        resource_name: str,
        error: BaseException,
    ) -> None:
        errors.append((resource_name, error))
        # Do not log exception text or traceback: either may contain sensitive data.
        logger.warning(
            "Worker cleanup failed: resource=%s exception_type=%s",
            resource_name,
            type(error).__name__,
        )

    @staticmethod
    async def _await_task_shielded(
        task: asyncio.Task,
    ) -> tuple[BaseException | None, bool]:
        """Wait for a task despite repeated caller cancellation; consume result."""
        current = asyncio.current_task()
        cancellation_count = current.cancelling() if current is not None else 0
        interrupted = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if current is not None and current.cancelling() > cancellation_count:
                    interrupted = True
                    cancellation_count = current.cancelling()
                if task.done():
                    break
            except BaseException:
                # Retrieve the task's exact exception below.
                break
        try:
            task.result()
        except BaseException as error:
            return error, interrupted
        return None, interrupted

    @classmethod
    async def _cancel_and_consume(cls, task: asyncio.Task) -> bool:
        if not task.done():
            task.cancel()
        _, interrupted = await cls._await_task_shielded(task)
        return interrupted
