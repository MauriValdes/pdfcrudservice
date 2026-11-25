from bson import ObjectId
from datetime import datetime, timezone
from typing import Mapping, Optional, Any
import inspect

from pymongo import ReturnDocument
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    DuplicateKeyError,
    NetworkTimeout,
    ServerSelectionTimeoutError,
)

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from src.config.config import settings
from src.repository.gridfs_repository import GridFSRepository

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

    async def reserve_document(
        self, document_data: DocumentCreate
    ) -> tuple[DocumentModel, bool]:
        """Reserva atómicamente un checksum creando un documento uploading.

        Retorna el documento reservado y ``True`` si la inserción tuvo éxito.
        Si el checksum ya existe, retorna el documento existente y ``False``
        sin modificarlo.
        """
        now = datetime.now(timezone.utc)
        document = DocumentModel(
            filename=document_data.filename,
            checksum=document_data.checksum,
            size_bytes=document_data.size_bytes,
            status=DocumentStatus.UPLOADING,
            created_at=now,
            updated_at=now,
            file_id=None,
            upload_owner_id=None,
            upload_lease_until=None,
            upload_fencing_token=None,
        )
        document_data_for_insert = document.model_dump(
            by_alias=True,
            exclude={"id"},
        )

        try:
            result = await self.collection.insert_one(document_data_for_insert)
        except DuplicateKeyError as duplicate_error:
            try:
                existing_document = await self.collection.find_one(
                    {"checksum": document_data.checksum}
                )
            except _UNAVAILABLE_ERRORS as e:
                raise DatabaseUnavailableError() from e

            if existing_document is None:
                raise duplicate_error

            existing_data = dict(existing_document)
            existing_id = existing_data.pop("_id", None)
            if existing_id is not None:
                existing_data["id"] = str(existing_id)
            return DocumentModel.model_validate(existing_data), False
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e

        document.id = str(result.inserted_id)
        return document, True

    async def acquire_upload_ownership(
        self,
        checksum: str,
        upload_owner_id: str,
        upload_fencing_token: str,
        upload_lease_until: datetime,
    ) -> tuple[DocumentModel | None, bool]:
        """Intenta adquirir atómicamente la propiedad de una carga uploading."""
        now = datetime.now(timezone.utc)
        lease_offset = (
            upload_lease_until.utcoffset()
            if upload_lease_until.tzinfo is not None
            else None
        )
        if (
            lease_offset != timezone.utc.utcoffset(now)
            or upload_lease_until <= now
        ):
            raise ValueError(
                "upload_lease_until debe ser una fecha UTC consciente y futura."
            )

        query = {
            "checksum": checksum,
            "status": DocumentStatus.UPLOADING.value,
            "file_id": None,
            "$or": [
                {"upload_owner_id": {"$exists": False}},
                {"upload_owner_id": None},
                {"upload_lease_until": {"$lt": now}},
            ],
        }
        update = {
            "$set": {
                "upload_owner_id": upload_owner_id,
                "upload_lease_until": upload_lease_until,
                "upload_fencing_token": upload_fencing_token,
                "updated_at": now,
            }
        }

        try:
            document_data = await self.collection.find_one_and_update(
                query,
                update,
                return_document=ReturnDocument.AFTER,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e

        acquired = document_data is not None
        if document_data is None:
            try:
                document_data = await self.collection.find_one(
                    {"checksum": checksum}
                )
            except _UNAVAILABLE_ERRORS as e:
                raise DatabaseUnavailableError() from e

        if document_data is None:
            return None, False

        existing_data = dict(document_data)
        document_id = existing_data.pop("_id", None)
        if document_id is not None:
            existing_data["id"] = str(document_id)
        return DocumentModel.model_validate(existing_data), acquired

    async def renew_upload_lease(
        self,
        document_id: str,
        upload_owner_id: str,
        upload_fencing_token: str,
        upload_lease_until: datetime,
    ) -> bool:
        """Renueva el lease si el llamante conserva la propiedad vigente.

        Retorna ``False`` si no se cumplieron las condiciones de renovación;
        en ese caso, el llamante debe dejar de actuar como propietario.
        """
        if not ObjectId.is_valid(document_id):
            raise ValueError("Identificador de documento inválido")

        now = datetime.now(timezone.utc)
        lease_offset = (
            upload_lease_until.utcoffset()
            if upload_lease_until.tzinfo is not None
            else None
        )
        if (
            lease_offset != timezone.utc.utcoffset(now)
            or upload_lease_until <= now
        ):
            raise ValueError(
                "upload_lease_until debe ser una fecha UTC consciente y futura."
            )

        query = {
            "_id": ObjectId(document_id),
            "status": DocumentStatus.UPLOADING.value,
            "upload_owner_id": upload_owner_id,
            "upload_fencing_token": upload_fencing_token,
            "upload_lease_until": {
                "$gte": now,
                "$lt": upload_lease_until,
            },
        }
        update = {
            "$set": {
                "upload_lease_until": upload_lease_until,
                "updated_at": now,
            }
        }

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e

        return result.matched_count == 1

    async def finalize_upload(
        self,
        document_id: str,
        upload_owner_id: str,
        upload_fencing_token: str,
        file_id: ObjectId,
    ) -> bool:
        """Asocia el PDF ya confirmado en GridFS y pasa a pending atómicamente.

        Retorna ``False`` si no se cumplen las condiciones de finalización.
        Un error de conectividad puede dejar incierto el resultado de escritura.
        """
        if not ObjectId.is_valid(document_id):
            raise ValueError("Identificador de documento inválido")
        if not isinstance(file_id, ObjectId):
            raise ValueError("file_id debe ser una instancia de ObjectId")

        now = datetime.now(timezone.utc)
        query = {
            "_id": ObjectId(document_id),
            "status": DocumentStatus.UPLOADING.value,
            "upload_owner_id": upload_owner_id,
            "upload_fencing_token": upload_fencing_token,
            "upload_lease_until": {"$gte": now},
            "$or": [
                {"file_id": {"$exists": False}},
                {"file_id": None},
            ],
        }
        update = {
            "$set": {
                "file_id": file_id,
                "status": DocumentStatus.PENDING.value,
                "updated_at": now,
            }
        }

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as e:
            raise DatabaseUnavailableError() from e

        return result.matched_count == 1

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

    async def delete_by_checksum(self, checksum: str) -> int:
        """Elimina exclusivamente el documento con el checksum exacto indicado."""
        try:
            result = await self.collection.delete_one({"checksum": checksum})
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

    def _gridfs_repository(self) -> GridFSRepository:
        """Crea y retorna un GridFSRepository reutilizando la base MongoDB existente.

        Este método está pensado para ser usado por el servicio o endpoints
        que necesiten almacenar PDFs originales en GridFS sin crear un nuevo
        cliente MongoDB ni configurar Write Concern nuevamente.

        El bucket GridFS se crea de forma perezosa sobre ``self.db``, que es
        la misma base que usa ``DocumentRepository``, por lo que ambas
        operaciones (documentos y GridFS) comparten configuración de conexión
        y Write Concern (w=1, journal=True).
        """
        return GridFSRepository(self.db)
