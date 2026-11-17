from bson import ObjectId
from datetime import datetime
import inspect

from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
)

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from src.config.config import settings

# Errores de conectividad de PyMongo/Motor que indican indisponibilidad de
# MongoDB (servidor inalcanzable, conexión perdida o red interrumpida).
# NO se incluyen DuplicateKeyError ni OperationFailure: son errores de
# datos/índice y deben conservar su tratamiento actual (400 en duplicados).
# ExecutionTimeout queda fuera de esta etapa por decisión de alcance.
_UNAVAILABLE_ERRORS = (
    ServerSelectionTimeoutError,
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
)


async def _await_translated(awaitable):
    """Espera una operación Motor y traduce los errores de conectividad."""
    try:
        return await awaitable
    except _UNAVAILABLE_ERRORS as e:
        raise DatabaseUnavailableError() from e


class _UnavailableAwareCollection:
    """Fachada sobre la colección Motor que traduce errores de conectividad
    de PyMongo/Motor a DatabaseUnavailableError.

    Necesaria porque DocumentService accede directamente a
    `repository.collection` (p. ej. insert_one en submit_pdf); el servicio no
    captura esos errores y, sin esta traducción, llegarían crudos a la API
    como 500. Las operaciones que retornan cursores (find) se devuelven
    crudas: su consumo (to_list) se traduce en los métodos del repositorio.
    """

    def __init__(self, collection):
        self._collection = collection

    def __getattr__(self, name):
        attr = getattr(self._collection, name)

        if not callable(attr):
            return attr

        def translating(*args, **kwargs):
            result = attr(*args, **kwargs)
            # Operaciones awaitables de Motor (insert_one, find_one, ...)
            if inspect.isawaitable(result):
                return _await_translated(result)
            # Cursores y demás objetos se devuelven sin envolver
            return result

        return translating


class InvalidObjectIdError(ValueError):
    """El identificador proporcionado no tiene formato de ObjectId válido de MongoDB."""


class DocumentRepository:
    def __init__(self, db):
        self.db = db
        self.collection = _UnavailableAwareCollection(db[settings.mongo_collection])

    async def find_by_checksum(self, checksum: str) -> dict | None:
        """Busca un documento en la base de datos por su hash único."""
        try:
            document = await self.collection.find_one({"checksum": checksum})
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e
        return document

    async def find_by_id(self, document_id: str) -> dict | None:
        """Busca un documento por su ObjectId. Devuelve None si el ID es inválido o no existe."""
        if not ObjectId.is_valid(document_id):
            return None
        try:
            return await self.collection.find_one({"_id": ObjectId(document_id)})
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e

    async def save_document(self, document_data: DocumentCreate) -> str:
        """Guarda un documento nuevo en MongoDB.

        Convierte los datos del esquema Create a un DocumentModel y lo persiste.
        """
        from datetime import datetime

        now = datetime.now()
        doc_dict = document_data.model_dump()
        doc_orm = DocumentModel(
            filename=doc_dict["filename"],
            checksum=doc_dict["checksum"],
            size_bytes=doc_dict["size_bytes"],
            created_at=now,
            updated_at=now,
        )
        result = None
        try:
            result = await self.collection.insert_one(doc_orm.model_dump(by_alias=True, exclude={"id"}))
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e
        return str(result.inserted_id)

    async def find_all(self) -> list[DocumentResponse]:
        """Obtiene todos los documentos almacenados en la colección y los convierte a DTOs."""
        try:
            cursor = self.collection.find({})
            documents = await cursor.to_list(length=None)
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e
        responses = []
        for doc in documents:
            # Convertir _id a string para el DTO
            doc_id = str(doc.get("_id", "")) if doc.get("_id") else ""
            response = DocumentResponse(
                id=doc_id,
                filename=doc.get("filename", ""),
                checksum=doc.get("checksum", ""),
                size_bytes=doc.get("size_bytes", 0),
                status=doc.get("status", "pending"),
                content=doc.get("content"),
                page_count=doc.get("page_count"),
                error_message=doc.get("error_message"),
                created_at=doc.get("created_at", datetime.now()),
                updated_at=doc.get("updated_at", datetime.now()),
            )
            responses.append(response)
        return responses

    async def delete_by_id(self, document_id: str) -> int:
        """Elimina un documento por su ID de MongoDB y retorna la cantidad de eliminados."""
        if not ObjectId.is_valid(document_id):
            raise InvalidObjectIdError("Identificador de documento inválido")
        try:
            result = await self.collection.delete_one({"_id": ObjectId(document_id)})
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e
        return result.deleted_count

    async def update_document(self, document_id: str, update_data: DocumentUpdate) -> DocumentResponse | None:
        """Actualiza un documento con los nuevos datos proporcionados."""
        if not ObjectId.is_valid(document_id):
            raise InvalidObjectIdError("Identificador de documento inválido")

        from datetime import datetime

        # Filtrar solo los campos que tienen valor (no None)
        update_dict = {k: v for k, v in update_data.model_dump(exclude_unset=True).items() if v is not None}
        update_dict["updated_at"] = datetime.now()

        if not update_dict:
            return None

        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": update_dict},
            )
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e

        if result.modified_count == 0:
            return None

        # Devolver el documento actualizado
        try:
            updated_doc = await self.collection.find_one({"_id": ObjectId(document_id)})
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e
        if not updated_doc:
            return None

        return DocumentResponse(
            id=str(updated_doc.get("_id", "")) if updated_doc.get("_id") else "",
            filename=updated_doc.get("filename", ""),
            checksum=updated_doc.get("checksum", ""),
            size_bytes=updated_doc.get("size_bytes", 0),
            status=updated_doc.get("status", "pending"),
            content=updated_doc.get("content"),
            page_count=updated_doc.get("page_count"),
            error_message=updated_doc.get("error_message"),
            created_at=updated_doc.get("created_at", datetime.now()),
            updated_at=updated_doc.get("updated_at", datetime.now()),
        )