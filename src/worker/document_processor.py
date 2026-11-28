"""Basic orchestration for processing one already-claimed document."""

import logging
from datetime import datetime, timezone
from enum import Enum

from bson import ObjectId

from src.models.document import DocumentModel
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import (
    ProcessingRepository,
    RetryTimestampExpiredError,
)
from src.worker.error_classifier import DocumentAction, classify_ms2_error
from src.worker.heartbeat import run_with_heartbeat
from src.worker.ms2_client import (
    MS2Client,
    MS2HTTPError,
    MS2InvalidResponseError,
    MS2TransportError,
)
from src.worker.pdf_reader import read_claimed_pdf
from src.worker.retry_policy import RetryExhausted, RetryScheduled, plan_retry


_logger = logging.getLogger(__name__)
_MS2_ERRORS = (MS2HTTPError, MS2TransportError, MS2InvalidResponseError)


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC instant."""
    return datetime.now(timezone.utc)


class ProcessingResult(Enum):
    """Non-persistent outcome of one processing invocation."""

    COMPLETED = "completed"
    ATTEMPT_NOT_AUTHORIZED = "attempt_not_authorized"
    RESULT_NOT_CONFIRMED = "result_not_confirmed"
    FAILED_PERMANENT = "failed_permanent"
    FAILED_EXHAUSTED = "failed_exhausted"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED_TECHNICAL = "failed_technical"


async def process_claimed_document(
    document: DocumentModel,
    *,
    processing_repository: ProcessingRepository,
    gridfs_repository: GridFSRepository,
    ms2_client: MS2Client,
) -> ProcessingResult:
    """Process one claimed document with at most one authorized MS2 extraction.

    The caller is responsible for preventing concurrent invocations for the
    same claim. When deciding exhaustion, no other processing for the claim or
    acceptable authorized result may be pending for this acquisition.
    Cancellation cannot establish whether remote effects occurred.
    """
    snapshot = document.model_copy(deep=True)

    document_id = snapshot.id
    worker_id = snapshot.worker_id
    fencing_token = snapshot.fencing_token
    processing_cycle_id = snapshot.processing_cycle_id
    file_id = snapshot.file_id
    filename = snapshot.filename

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
    if not isinstance(processing_cycle_id, str):
        raise TypeError("document.processing_cycle_id must be a string")
    if not processing_cycle_id:
        raise ValueError("document.processing_cycle_id must not be empty")
    if not isinstance(file_id, ObjectId):
        raise ValueError("document.file_id must be an ObjectId")
    if not isinstance(filename, str):
        raise TypeError("document.filename must be a string")
    if not filename:
        raise ValueError("document.filename must not be empty")

    async def operation() -> ProcessingResult:
        pdf_bytes = await read_claimed_pdf(
            snapshot,
            processing_repository=processing_repository,
            gridfs_repository=gridfs_repository,
        )

        attempt_number = await processing_repository.record_ms2_attempt(
            document_id,
            worker_id,
            fencing_token,
            processing_cycle_id,
        )
        if attempt_number is None:
            confirmed = await processing_repository.fail_exhausted_processing(
                document_id,
                worker_id,
                fencing_token,
                processing_cycle_id,
            )

            if confirmed:
                return ProcessingResult.FAILED_EXHAUSTED

            return ProcessingResult.ATTEMPT_NOT_AUTHORIZED

        try:
            result = await ms2_client.extract(
                pdf_bytes,
                filename,
            )
        except _MS2_ERRORS as error:
            category = classify_ms2_error(error)
            if category is None:
                raise

            if category.alert_required:
                fields = (
                    f"event_id=worker.ms2.technical_failure "
                    f"document_id={document_id} error_code={category.code}"
                )
                if category.http_status is not None:
                    fields += f" http_status={category.http_status}"
                _logger.error(fields)

            if category.document_action in (
                DocumentAction.FAIL_INPUT,
                DocumentAction.FAIL_TECHNICAL,
            ):
                confirmed = await processing_repository.fail_processing(
                    document_id,
                    worker_id,
                    fencing_token,
                    processing_cycle_id,
                    attempt_number,
                    category.code,
                )
                if not confirmed:
                    return ProcessingResult.RESULT_NOT_CONFIRMED
                if category.document_action is DocumentAction.FAIL_INPUT:
                    return ProcessingResult.FAILED_PERMANENT
                return ProcessingResult.FAILED_TECHNICAL

            if category.document_action is DocumentAction.RETRY:
                now = _utc_now()
                retry_plan = plan_retry(
                    attempt_count=attempt_number,
                    cycle_started_at=snapshot.processing_cycle_started_at,
                    now=now,
                    retry_after=(
                        error.retry_after
                        if isinstance(error, MS2HTTPError)
                        else None
                    ),
                )
                if isinstance(retry_plan, RetryExhausted):
                    confirmed = await processing_repository.fail_exhausted_processing(
                        document_id,
                        worker_id,
                        fencing_token,
                        processing_cycle_id,
                    )
                    return (
                        ProcessingResult.FAILED_EXHAUSTED
                        if confirmed
                        else ProcessingResult.RESULT_NOT_CONFIRMED
                    )

                if isinstance(retry_plan, RetryScheduled):
                    try:
                        scheduled = (
                            await processing_repository.schedule_transient_retry(
                                document_id,
                                worker_id,
                                fencing_token,
                                processing_cycle_id,
                                attempt_number,
                                category.code,
                                retry_plan.next_retry_at,
                            )
                        )
                    except RetryTimestampExpiredError:
                        scheduled = False

                    if scheduled:
                        return ProcessingResult.RETRY_SCHEDULED

                    confirmed = await processing_repository.fail_exhausted_processing(
                        document_id,
                        worker_id,
                        fencing_token,
                        processing_cycle_id,
                    )
                    return (
                        ProcessingResult.FAILED_EXHAUSTED
                        if confirmed
                        else ProcessingResult.RESULT_NOT_CONFIRMED
                    )

                raise TypeError("plan_retry returned an unsupported result")

            raise TypeError("MS2 classifier returned an unsupported action")

        confirmed = await processing_repository.complete_processing(
            document_id,
            worker_id,
            fencing_token,
            processing_cycle_id,
            attempt_number,
            result.content,
            result.page_count,
        )
        if confirmed:
            return ProcessingResult.COMPLETED

        return ProcessingResult.RESULT_NOT_CONFIRMED

    return await run_with_heartbeat(
        snapshot,
        processing_repository=processing_repository,
        operation=operation,
    )
