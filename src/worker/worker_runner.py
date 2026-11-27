"""Polling and concurrency control for claimed document processing."""

import asyncio
import logging
import math
from numbers import Real
from typing import Callable
from uuid import uuid4

from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import ProcessingRepository
from src.worker.document_processor import (
    ProcessingResult,
    process_claimed_document,
)
from src.worker.ms2_client import MS2Client

logger = logging.getLogger(__name__)


async def _wait_for_poll(
    stop_event: asyncio.Event,
    task_event: asyncio.Event,
    timeout_seconds: float | None,
) -> None:
    """Wait for stop, a completed processor task, or the polling timeout."""
    stop_waiter = asyncio.create_task(stop_event.wait())
    task_waiter = asyncio.create_task(task_event.wait())
    waiters = (stop_waiter, task_waiter)

    try:
        await asyncio.wait(
            waiters,
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


class ProcessingWorker:
    """Acquire work and supervise a bounded set of processor tasks.

    Each instance is single-use. Its caller owns all injected dependencies.
    """

    def __init__(
        self,
        processing_repository: ProcessingRepository,
        gridfs_repository: GridFSRepository,
        ms2_client: MS2Client,
        *,
        processor: Callable = process_claimed_document,
        max_concurrency: int = 2,
        poll_initial_seconds: float = 1.0,
        poll_max_seconds: float = 5.0,
    ) -> None:
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")

        initial_interval = self._validated_interval(
            poll_initial_seconds,
            "poll_initial_seconds",
        )
        maximum_interval = self._validated_interval(
            poll_max_seconds,
            "poll_max_seconds",
        )
        if maximum_interval < initial_interval:
            raise ValueError(
                "poll_max_seconds must be greater than or equal to "
                "poll_initial_seconds"
            )

        self._processing_repository = processing_repository
        self._gridfs_repository = gridfs_repository
        self._ms2_client = ms2_client
        self._processor = processor
        self._max_concurrency = max_concurrency
        self._poll_initial_seconds = initial_interval
        self._poll_max_seconds = maximum_interval

        self._stop_event = asyncio.Event()
        self._task_event = asyncio.Event()
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._next_source = "recover"
        self._poll_interval = initial_interval
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def _validated_interval(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        try:
            interval = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{name} must be finite and positive") from error
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return interval

    def request_stop(self) -> None:
        """Idempotently request an orderly stop from this worker's event loop."""
        if self._loop is not None:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError as error:
                raise RuntimeError(
                    "request_stop() must be called from the worker event loop"
                ) from error
            if current_loop is not self._loop:
                raise RuntimeError(
                    "request_stop() must be called from the worker event loop"
                )
        self._stop_event.set()

    async def run(self) -> None:
        """Run the polling loop once, then drain or cancel owned tasks."""
        if self._started:
            raise RuntimeError("ProcessingWorker instances can only run once")
        self._started = True
        self._loop = asyncio.get_running_loop()
        worker_id = str(uuid4())

        try:
            await self._run_loop(worker_id)
        except asyncio.CancelledError:
            self._cancel_pending_processors()
            await self._drain_tasks()
            raise
        except Exception:
            # Acquisition failures and duplicate claims stop admissions. Drain
            # already admitted work before preserving the original exception.
            interrupted = await self._drain_tasks()
            if interrupted:
                raise asyncio.CancelledError
            raise
        else:
            # Orderly stop waits for processors; a cancellation arriving
            # during this shielded drain switches to cancellation semantics.
            interrupted = await self._drain_tasks()
            if interrupted:
                self._cancel_pending_processors()
                await self._drain_tasks()
                raise asyncio.CancelledError

    async def _run_loop(self, worker_id: str) -> None:
        while True:
            self._collect_finished_tasks()

            if self._stop_event.is_set():
                return

            if len(self._active_tasks) >= self._max_concurrency:
                self._prepare_task_wait()
                if self._stop_event.is_set():
                    continue
                if self._has_finished_tasks():
                    continue
                await _wait_for_poll(
                    self._stop_event,
                    self._task_event,
                    None,
                )
                continue

            try:
                document = await self._acquire_one(worker_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                raise

            completed_during_acquisition = self._collect_finished_tasks()

            if document is None:
                if self._stop_event.is_set():
                    continue
                if completed_during_acquisition:
                    continue
                self._prepare_task_wait()
                if self._stop_event.is_set():
                    continue
                if self._has_finished_tasks():
                    continue
                await _wait_for_poll(
                    self._stop_event,
                    self._task_event,
                    self._poll_interval,
                )
                self._poll_interval = min(
                    self._poll_interval * 2,
                    self._poll_max_seconds,
                )
                continue

            self._poll_interval = self._poll_initial_seconds
            document_id = getattr(document, "id", None)
            if not isinstance(document_id, str) or not document_id:
                raise RuntimeError("Acquisition returned a document without a valid id")
            if document_id in self._active_tasks:
                raise RuntimeError(
                    f"Duplicate active document acquisition: {document_id}"
                )

            task = asyncio.create_task(self._process(document))
            self._active_tasks[document_id] = task
            task.add_done_callback(
                lambda finished_task, claimed_id=document_id: (
                    self._notify_task_completion(claimed_id, finished_task)
                )
            )

    async def _acquire_one(self, worker_id: str):
        preferred = self._next_source
        preferred_method = self._acquisition_method(preferred)
        document = await preferred_method(worker_id)
        if document is not None:
            self._next_source = self._other_source(preferred)
            return document

        if self._stop_event.is_set():
            return None

        fallback = self._other_source(preferred)
        fallback_method = self._acquisition_method(fallback)
        document = await fallback_method(worker_id)
        if document is not None:
            # The next preference is always the source other than the one that
            # actually supplied this claim.
            self._next_source = self._other_source(fallback)
        return document

    def _acquisition_method(self, source: str):
        if source == "recover":
            return self._processing_repository.recover_expired_processing
        return self._processing_repository.claim_pending

    @staticmethod
    def _other_source(source: str) -> str:
        return "claim" if source == "recover" else "recover"

    async def _process(self, document):
        try:
            return await self._processor(
                document,
                processing_repository=self._processing_repository,
                gridfs_repository=self._gridfs_repository,
                ms2_client=self._ms2_client,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            document_id = getattr(document, "id", "<unknown>")
            # Do not include exception text or traceback; downstream errors
            # can contain response bodies or other sensitive material.
            logger.error(
                "processor failed for document %s (%s)",
                document_id,
                type(error).__name__,
            )
            raise

    def _notify_task_completion(
        self,
        document_id: str,
        task: asyncio.Task,
    ) -> None:
        # A task may be collected before its scheduled done callback runs.
        # Ignore that stale notification rather than waking the poll loop.
        if self._active_tasks.get(document_id) is task:
            self._task_event.set()

    def _has_finished_tasks(self) -> bool:
        return any(task.done() for task in self._active_tasks.values())

    def _prepare_task_wait(self) -> None:
        # Clear only after the caller has collected completed tasks. There is
        # no await between clearing and the following done/stop checks.
        self._task_event.clear()

    def _collect_finished_tasks(self) -> bool:
        completed_any = False
        for document_id, task in tuple(self._active_tasks.items()):
            if not task.done():
                continue

            completed_any = True
            del self._active_tasks[document_id]
            self._poll_interval = self._poll_initial_seconds
            try:
                outcome = task.result()
            except asyncio.CancelledError:
                logger.info(
                    "Processor task cancelled for document %s",
                    document_id,
                )
            except Exception:
                # _process logs a sanitized exception before the task is marked
                # complete; retrieving it here prevents an unobserved failure.
                pass
            else:
                if isinstance(outcome, ProcessingResult):
                    log_outcome = (
                        logger.info
                        if outcome is ProcessingResult.COMPLETED
                        else logger.warning
                    )
                    log_outcome(
                        "Processing outcome for document %s: %s",
                        document_id,
                        outcome.value,
                    )
                else:
                    logger.warning(
                        "Processor returned an unknown outcome for document %s (%s)",
                        document_id,
                        type(outcome).__name__,
                    )

        if not self._has_finished_tasks():
            self._task_event.clear()
        return completed_any

    def _cancel_pending_processors(self) -> None:
        for task in tuple(self._active_tasks.values()):
            if not task.done() and task.cancelling() == 0:
                task.cancel()

    async def _drain_tasks(self) -> bool:
        """Wait for current processors without letting cancellation cascade.

        Returns whether this run was cancelled while draining. Processor tasks
        are cancelled at most once; repeated cancellation of the runner cannot
        cancel them again through a gather operation.
        """
        tasks = tuple(self._active_tasks.values())
        if not tasks:
            self._collect_finished_tasks()
            return False

        drain_task = asyncio.ensure_future(
            asyncio.gather(*tasks, return_exceptions=True)
        )
        interrupted = False
        while not drain_task.done():
            try:
                await asyncio.shield(drain_task)
            except asyncio.CancelledError:
                interrupted = True
                self._cancel_pending_processors()

        # Retrieve any exception from the auxiliary drain task and consume all
        # processor results through the normal task collector.
        await drain_task
        self._collect_finished_tasks()
        return interrupted
