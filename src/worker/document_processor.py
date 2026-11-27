"""Basic orchestration for processing one already-claimed document."""

from enum import Enum

from bson import ObjectId

from src.models.document import DocumentModel
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import ProcessingRepository
from src.worker.heartbeat import run_with_heartbeat
from src.worker.ms2_client import MS2Client
from src.worker.pdf_reader import read_claimed_pdf


class ProcessingResult(Enum):
    """Non-persistent outcome of one processing invocation."""

    COMPLETED = "completed"
    ATTEMPT_NOT_AUTHORIZED = "attempt_not_authorized"
    RESULT_NOT_CONFIRMED = "result_not_confirmed"


async def process_claimed_document(
    document: DocumentModel,
    *,
    processing_repository: ProcessingRepository,
    gridfs_repository: GridFSRepository,
    ms2_client: MS2Client,
) -> ProcessingResult:
    """Process one claimed document with at most one authorized MS2 extraction.

    The caller is responsible for preventing concurrent invocations for the
    same claim. Cancellation cannot establish whether remote effects occurred.
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
            return ProcessingResult.ATTEMPT_NOT_AUTHORIZED

        result = await ms2_client.extract(
            pdf_bytes,
            filename,
        )

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
