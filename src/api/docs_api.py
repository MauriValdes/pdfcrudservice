import logging
from typing import List

from bson import ObjectId
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from src.api.pdf_validation import PdfValidationExecutor, validate_pdf_upload
from src.schemas.document import DocumentReceipt, DocumentUpdate, DocumentResponse
from src.core.exceptions import AppError, DatabaseUnavailableError
from src.database.mongodb import db_connection
from src.repository.document_repository import DocumentRepository, InvalidObjectIdError
from src.services.document_service import (
    DocumentService,
    UploadFinalizationRejectedError,
    UploadLeaseLostError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


# Inyección de Dependencias
def get_document_service() -> DocumentService:
    # Fast-fail ante indisponibilidad de MongoDB:
    # - db is None: la conexión nunca llegó a establecerse (evita el TypeError
    #   de indexar None y el 500 inesperado).
    # - connected False: el supervisor ya detectó una caída y la reconexión
    #   está en curso; responder 503 inmediato en lugar de esperar los
    #   timeouts internos de Motor (~30s).
    if db_connection.db is None or not db_connection.connected:
        raise DatabaseUnavailableError()
    repository = DocumentRepository(db_connection.db)
    return DocumentService(
        repository=repository,
    )


# 1. CREATE: Recibir y almacenar un PDF para procesamiento asíncrono
@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentReceipt,
    responses={
        status.HTTP_200_OK: {
            "model": DocumentReceipt,
            "description": "Checksum duplicado; se devuelve el documento existente.",
            "headers": {
                "Location": {
                    "description": "URL relativa del estado del documento.",
                    "schema": {"type": "string"},
                }
            },
        },
        status.HTTP_202_ACCEPTED: {
            "description": "PDF persistido y carga finalizada de forma durable.",
            "headers": {
                "Location": {
                    "description": "URL relativa del estado del documento.",
                    "schema": {"type": "string"},
                }
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "MIME inválido o archivo vacío.",
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "description": "El archivo supera el tamaño máximo permitido.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "El contenido no es un PDF válido.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Error inesperado durante la recepción.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "No se pudo confirmar la recepción durable.",
        },
    },
)
async def upload_pdf(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentReceipt:
    """Valida y almacena el PDF; la extracción se procesa posteriormente."""
    executor = getattr(request.app.state, "pdf_validation_executor", None)
    if not isinstance(executor, PdfValidationExecutor) or executor.closed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio temporalmente no disponible",
        )

    try:
        file_bytes = await validate_pdf_upload(file, executor=executor)
    except HTTPException:
        raise
    except AppError:
        raise
    except RuntimeError as e:
        if (
            executor.closed
            and str(e) == "PDF validation executor is shutting down"
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servicio temporalmente no disponible",
            ) from None
        logger.exception("Error inesperado al validar el PDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error inesperado durante la recepción del documento",
        ) from None
    except Exception:
        logger.exception("Error inesperado al validar el PDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error inesperado durante la recepción del documento",
        ) from None

    try:
        document, classification = await service.receive_pdf(
            file_bytes=file_bytes,
            filename=file.filename,
        )
        if classification not in ("accepted", "existing"):
            raise ValueError("Clasificación de recepción desconocida.")

        status_url = f"/documents/{document.id}"
        receipt = DocumentReceipt(
            document_id=document.id,
            status=document.status,
            created_at=document.created_at,
            status_url=status_url,
        )
        response.headers["Location"] = status_url
        response.status_code = (
            status.HTTP_202_ACCEPTED
            if classification == "accepted"
            else status.HTTP_200_OK
        )
        return receipt
    except HTTPException:
        raise
    except AppError:
        raise
    except (UploadLeaseLostError, UploadFinalizationRejectedError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio temporalmente no disponible",
        ) from None
    except Exception:
        logger.exception("Error inesperado durante la recepción asíncrona")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error inesperado durante la recepción del documento",
        ) from None


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
    except AppError:
        # Indisponibilidad de MongoDB (503): el manejador central la traduce.
        raise
    except Exception as e:
        logger.error(f"Error al obtener documentos: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener documentos"
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
    Returns 400 Bad Request si el identificador no es un ObjectId válido.
    """
    if not ObjectId.is_valid(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identificador de documento inválido"
        )
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
    except AppError:
        # Indisponibilidad de MongoDB (503): el manejador central la traduce.
        raise
    except Exception as e:
        logger.error(f"Error al obtener documento {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener documento"
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
    except InvalidObjectIdError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except AppError:
        # Indisponibilidad de MongoDB (503): el manejador central la traduce.
        raise
    except Exception as e:
        logger.error(f"Error al actualizar documento {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al actualizar documento"
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
    except InvalidObjectIdError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except AppError:
        # Indisponibilidad de MongoDB (503): el manejador central la traduce.
        raise
    except Exception as e:
        logger.error(f"Error al eliminar documento {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al eliminar documento"
        )
