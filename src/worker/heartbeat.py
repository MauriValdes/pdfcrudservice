"""Lease heartbeat supervision for a claimed document operation."""

import asyncio
import math
from numbers import Real
from typing import Awaitable, Callable, TypeVar

from bson import ObjectId

from src.models.document import DocumentModel
from src.repository.processing_repository import ProcessingRepository
from src.worker.exceptions import ProcessingLeaseLostError

T = TypeVar("T")

_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0
_PROCESSING_LEASE_SECONDS = 60.0


async def run_with_heartbeat(
    document: DocumentModel,
    *,
    processing_repository: ProcessingRepository,
    operation: Callable[[], Awaitable[T]],
    heartbeat_interval_seconds: float = _DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> T:
    """Run one operation while renewing its claimed document lease.

    Cancelling the local operation does not prove that remote effects or writes
    were not applied. Operations must cooperate with cancellation; this
    supervisor cannot guarantee a bounded shutdown for one that suppresses it.
    """
    document_id = getattr(document, "id", None)
    worker_id = getattr(document, "worker_id", None)
    fencing_token = getattr(document, "fencing_token", None)

    if not isinstance(document_id, str):
        raise TypeError("document.id must be a string")
    if not ObjectId.is_valid(document_id):
        raise ValueError("document.id must be a valid ObjectId")
    if not isinstance(worker_id, str):
        raise TypeError("document.worker_id must be a string")
    if not worker_id:
        raise ValueError("document.worker_id must not be empty")
    if not isinstance(fencing_token, str):
        raise TypeError("document.fencing_token must be a string")
    if not fencing_token:
        raise ValueError("document.fencing_token must not be empty")

    if isinstance(heartbeat_interval_seconds, bool) or not isinstance(
        heartbeat_interval_seconds, Real
    ):
        raise TypeError("heartbeat interval must be a real number")
    try:
        interval = float(heartbeat_interval_seconds)
    except OverflowError as error:
        raise ValueError("heartbeat interval must be finite") from error
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("heartbeat interval must be finite and positive")
    if interval >= _PROCESSING_LEASE_SECONDS:
        raise ValueError("heartbeat interval must be shorter than the lease")

    is_current_owner = await processing_repository.renew_processing_lease(
        document_id,
        worker_id,
        fencing_token,
    )
    if not is_current_owner:
        raise ProcessingLeaseLostError(
            f"Processing lease lost for document {document_id}"
        )

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(interval)
            is_current_owner = (
                await processing_repository.renew_processing_lease(
                    document_id,
                    worker_id,
                    fencing_token,
                )
            )
            if not is_current_owner:
                raise ProcessingLeaseLostError(
                    f"Processing lease lost for document {document_id}"
                )

    async def invoke_operation() -> T:
        return await operation()

    operation_task = asyncio.create_task(invoke_operation())
    heartbeat_task = asyncio.create_task(heartbeat())
    tasks = (operation_task, heartbeat_task)

    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Prefer an operation that has already completed, including when both
        # tasks completed in the same event-loop turn.
        if operation_task.done():
            return await operation_task

        # A normally-returning heartbeat is unexpected: it must supervise
        # until cancelled. Awaiting it also propagates its original exception.
        await heartbeat_task
        raise RuntimeError("Heartbeat stopped while the operation was active")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

        # return_exceptions retrieves failures from both completed and
        # cancelled tasks without replacing the active result/error.
        await asyncio.gather(*tasks, return_exceptions=True)
