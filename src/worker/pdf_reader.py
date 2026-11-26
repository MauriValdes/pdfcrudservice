"""Lectura de PDFs GridFS asociados a documentos reclamados por el worker."""

from bson import ObjectId

from src.models.document import DocumentModel
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import ProcessingRepository
from src.worker.exceptions import ProcessingLeaseLostError


async def _require_current_owner(
    document: DocumentModel,
    processing_repository: ProcessingRepository,
) -> None:
    """Renueva y valida atómicamente la propiedad del documento."""
    is_current_owner = await processing_repository.renew_processing_lease(
        document.id,
        document.worker_id,
        document.fencing_token,
    )
    if not is_current_owner:
        raise ProcessingLeaseLostError(
            f"Processing lease lost for document {document.id}"
        )


async def read_claimed_pdf(
    document: DocumentModel,
    *,
    processing_repository: ProcessingRepository,
    gridfs_repository: GridFSRepository,
) -> bytes:
    """Lee el PDF original de un documento reclamado que sigue siendo propio.

    La propiedad se renueva antes de acceder a GridFS y se verifica nuevamente
    antes de entregar los bytes al llamador. No elimina ni modifica el archivo.
    """
    file_id = document.file_id
    if not isinstance(file_id, ObjectId):
        raise ValueError("El documento reclamado debe tener un file_id ObjectId")

    await _require_current_owner(document, processing_repository)
    pdf_bytes = await gridfs_repository.read(file_id)
    await _require_current_owner(document, processing_repository)
    return pdf_bytes
