import hashlib
from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from src.repository.document_repository import DocumentRepository
from datetime import datetime, timezone


class DocumentService:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository

    async def submit_pdf(self, file_bytes: bytes, filename: str) -> DocumentResponse:
        """Guarda los metadatos básicos del PDF en la base de datos.

        Persiste el documento con estado PENDING y retorna el DTO de respuesta.
        La extracción de texto y el procesamiento pesado se coordina
        desde el MS2 (Extract Service) o workers externos.
        """
        # 1. Calcular checksum del PDF
        checksum = hashlib.sha256(file_bytes).hexdigest()

        # 2. Verificar si el documento ya existe por su checksum
        existing_doc = await self.repository.find_by_checksum(checksum)
        if existing_doc is not None:
            raise ValueError("El documento ya existe")

        # 3. Construir el documento de metadata con estado PENDING
        now = datetime.now(timezone.utc)
        doc_dict = DocumentCreate(
            filename=filename,
            checksum=checksum,
            size_bytes=len(file_bytes),
        ).model_dump()

        doc_orm = DocumentModel(
            filename=doc_dict["filename"],
            checksum=doc_dict["checksum"],
            size_bytes=doc_dict["size_bytes"],
            status=DocumentStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        # Guardar en MongoDB
        result = await self.repository.collection.insert_one(
            doc_orm.model_dump(by_alias=True, exclude={"id"})
        )

        # 4. Retornar el DTO de respuesta
        return DocumentResponse(
            id=str(result.inserted_id),
            filename=doc_orm.filename,
            checksum=doc_orm.checksum,
            size_bytes=doc_orm.size_bytes,
            status=doc_orm.status.value,
            created_at=doc_orm.created_at,
            updated_at=doc_orm.updated_at,
        )

    async def get_all_documents(self) -> list[DocumentResponse]:
        """Obtiene la lista completa de documentos como DTOs."""
        return await self.repository.find_all()

    async def get_document_by_id(self, doc_id: str) -> DocumentResponse | None:
        """Obtiene un documento por su ID (None si no existe o el ID es inválido)."""
        doc = await self.repository.find_by_id(doc_id)
        if not doc:
            return None

        # Handle both dict (from MongoDB) and DocumentResponse objects
        if isinstance(doc, dict):
            id_str = str(doc.get("_id", "")) if doc.get("_id") else ""
            filename = doc.get("filename", "")
            checksum = doc.get("checksum", "")
            size_bytes = doc.get("size_bytes", 0)
            status = doc.get("status", "pending")
            content = doc.get("content")
            page_count = doc.get("page_count")
            error_message = doc.get("error_message")
            created_at = doc.get("created_at", datetime.now())
            updated_at = doc.get("updated_at", datetime.now())
        else:
            # DocumentResponse or similar ORM object
            id_str = str(doc.id) if doc.id else ""
            filename = doc.filename
            checksum = doc.checksum
            size_bytes = doc.size_bytes
            status = doc.status if isinstance(doc.status, str) else getattr(doc.status, "value", "pending")
            content = doc.content
            page_count = doc.page_count
            error_message = doc.error_message
            created_at = doc.created_at
            updated_at = doc.updated_at

        return DocumentResponse(
            id=id_str,
            filename=filename,
            checksum=checksum,
            size_bytes=size_bytes,
            status=status,
            content=content,
            page_count=page_count,
            error_message=error_message,
            created_at=created_at,
            updated_at=updated_at,
        )

    async def update_document(
        self, document_id: str, update_data: DocumentUpdate
    ) -> DocumentResponse | None:
        """Actualiza un documento con los nuevos valores.

        Soporta actualizaciones parciales: solo los campos proporcionados
        serán actualizados en la base de datos.
        """
        # Convertir DocumentUpdate a diccionario, incluyendo solo los campos
        # que fueron proporcionados (excluyendo los valores None por defecto)
        update_dict = update_data.model_dump(exclude_unset=True)

        if not update_dict:
            return None

        return await self.repository.update_document(document_id, update_data)

    async def delete_document(self, document_id: str) -> bool:
        """Elimina un documento por su ID.

        Raises:
            ValueError: Si el documento no existe.
        """
        deleted = await self.repository.delete_by_id(document_id)
        if not deleted:
            raise ValueError("El documento no existe")
        return True