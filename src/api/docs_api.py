import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from src.models.document import DocumentStatus
from src.config.config import settings
from src.database.mongodb import db_connection
from src.repository.document_repository import DocumentRepository
from src.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


# Inyección de Dependencias
def get_document_service() -> DocumentService:
    repository = DocumentRepository(db_connection.db)
    return DocumentService(
        repository=repository,
    )


# 1. CREATE: Subir y registrar el procesamiento de un PDF
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentResponse,
)
async def upload_pdf(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    """Registra un PDF nuevo en la base de datos con status PENDING.

    La extracción de texto y el procesamiento pesado se coordina
    desde el MS2 (Extract Service) o workers externos.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser un PDF"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo está vacío"
        )

    # Rechazamos temprano archivos que superen el límite permitido
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"El archivo supera el tamaño máximo de {settings.max_file_size_mb} MB"
        )

    try:
        # Guardar metadatos del PDF en la base de datos (estado PENDING)
        task = await service.submit_pdf(file_bytes=file_bytes, filename=file.filename)
        return task
    except ValueError as e:
        logger.warning(f"Validación rechazada: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error interno al registrar el PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error interno al registrar el PDF: {str(e)}"
        )


# 2. READ: Obtener todos los documentos
@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=List[DocumentResponse],
)
async def get_all_documents(
    service: DocumentService = Depends(get_document_service),
):
    """Obtiene la lista completa de documentos almacenados."""
    try:
        documents = await service.get_all_documents()
        return documents
    except Exception as e:
        logger.error(f"Error al obtener documentos: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener documentos: {str(e)}"
        )


# 3. READ: Obtener un documento por ID
@router.get(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    response_model=DocumentResponse,
)
async def get_document_by_id(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """Obtiene un documento por su ID.

    Returns 404 Not Found si el documento no existe.
    """
    try:
        doc = await service.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Documento no encontrado"
            )
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener documento {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener documento: {str(e)}"
        )


# 4. UPDATE: Actualizar un documento (PATCH parcial)
@router.patch(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    response_model=DocumentResponse,
)
async def update_document(
    document_id: str,
    update_data: DocumentUpdate,
    service: DocumentService = Depends(get_document_service),
):
    """Actualiza parcialmente un documento (status, content, page_count, error_message).

    Solo los campos proporcionados serán actualizados en la base de datos.
    """
    try:
        doc = await service.update_document(document_id, update_data)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Documento no encontrado"
            )
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al actualizar documento {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar documento: {str(e)}"
        )


# 5. DELETE: Borrar un documento por ID
@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document_by_id(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """Elimina un documento por su ID.

    Returns 204 No Content en éxito.
    Returns 404 Not Found si el documento no existe.
    """
    try:
        await service.delete_document(document_id)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar documento {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar documento: {str(e)}"
        )