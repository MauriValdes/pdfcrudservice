import logging
from datetime import datetime
from typing import List

import httpx
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from src.models.document import DocumentStatus
from src.config.config import settings
from src.database.mongodb import db_connection
from src.repository.document_repository import DocumentRepository, InvalidObjectIdError
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
    """Registra un PDF nuevo y procesa su extracción de texto de forma síncrona.

    Flujo completo:
    1. Validación del archivo.
    2. Registro en MongoDB con estado PENDING.
    3. Comunicación HTTP con MS2 para extraer texto y página count.
    4. Actualización del documento a COMPLETED o FAILED.
    5. Devolución al cliente del documento con estado final.

    El cliente recibe siempre el resultado final (COMPLETED o FAILED),
    nunca PENDING.
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

    # Identificador del documento registrado; None si el registro falla
    # (por ejemplo, cuando submit_pdf detecta un documento duplicado).
    document_id: str | None = None

    try:
        # Paso 1: Guardar metadatos del PDF en la base de datos (estado PENDING)
        task = await service.submit_pdf(file_bytes=file_bytes, filename=file.filename)
        document_id = task.id

        # Paso 2: Orquestar extracción con MS2 (síncrono)
        await service.process_pdf_extraction(
            document_id=task.id,
            pdf_bytes=file_bytes,
            filename=file.filename,
        )

        # Paso 3: Recuperar el documento actualizado desde MongoDB
        # (submit_pdf retorna PENDING, process_pdf_extraction actualiza a COMPLETED/FAILED)
        updated_doc = await service.get_document_by_id(task.id)

        if not updated_doc:
            # Esto debería suceder solo en condiciones extremas (competencia de carrera en BD)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo recuperar el documento después del procesamiento"
            )

        return updated_doc

    except HTTPException:
        raise
    except ValueError as e:
        # Errores de negocio o de validación de respuesta MS2.
        # Incluye el documento duplicado: el servicio convierte
        # DuplicateKeyError en ValueError("El documento ya existe").
        # Si ya se actualizó a FAILED en process_pdf_extraction, retornamos ese estado
        try:
            failed_doc = await service.get_document_by_id(document_id) if document_id else None
            if failed_doc and failed_doc.status == "failed":
                return failed_doc
        except Exception:
            pass
        # Si no hay documento FAILED, lanzamos el error HTTP
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error inesperado durante el procesamiento síncrono: {e}", exc_info=True)
        # Si falla la persistencia final, intentamos retornar el estado actual del documento
        try:
            failed_doc = await service.get_document_by_id(document_id) if document_id else None
            if failed_doc and failed_doc.status == "failed":
                return failed_doc
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error inesperado durante el procesamiento"
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
    except Exception as e:
        logger.error(f"Error al eliminar documento {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al eliminar documento"
        )