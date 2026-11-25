"""Adaptador de acceso a GridFS para almacenar los PDF originales."""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from bson import ObjectId
from gridfs.errors import NoFile
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridFSFileInfo:
    """Metadatos de un archivo almacenado en GridFS."""

    file_id: ObjectId
    filename: str
    length: int
    metadata: Optional[Mapping[str, Any]]
    upload_date: Optional[datetime]
    content_type: Optional[str]


class GridFSRepository:
    """Repositorio de archivos PDF almacenados en GridFS."""

    def __init__(self, db):
        self.db = db

    def _bucket(self) -> AsyncIOMotorGridFSBucket:
        """Crea un bucket GridFS sobre la base Motor existente."""
        return AsyncIOMotorGridFSBucket(self.db)

    async def save(
        self,
        file_id: ObjectId,
        filename: str,
        data: bytes,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ObjectId:
        """Guarda un archivo con un identificador preasignado.

        El file_id debe ser un ObjectId nuevo y único para cada intento de
        carga. No se realizan consultas previas para comprobar su disponibilidad.

        El archivo solo se considera guardado cuando ``close()`` finaliza
        correctamente. Ante cualquier error o cancelación se intenta abortar
        la carga y se propaga la excepción original.
        """
        bucket = self._bucket()
        grid_in = await bucket.open_upload_stream_with_id(
            file_id,
            filename,
            metadata=metadata,
        )

        try:
            await grid_in.write(data)
        except BaseException:
            # write() falló antes de close().
            # No hay documento en fs.files aún (close no se ejecutó),
            # abort() limpia chunks parciales (best-effort).
            try:
                await grid_in.abort()
            except BaseException:
                logger.exception(
                    "No se pudo abortar la carga GridFS para file_id=%s",
                    file_id,
                )
            raise

        # write() fue exitoso, ahora cerrar
        try:
            await grid_in.close()
        except BaseException:
            # close() lanzó excepción real.
            # abort() como limpieza best-effort.
            try:
                await grid_in.abort()
            except BaseException:
                logger.exception(
                    "No se pudo abortar la carga GridFS para file_id=%s",
                    file_id,
                )
            raise

        # Ambos pasos completaron exitosamente
        return file_id

    async def stat(self, file_id: ObjectId) -> Optional[GridFSFileInfo]:
        """Devuelve los metadatos del archivo o ``None`` si no existe."""
        bucket = self._bucket()

        try:
            grid_out = await bucket.open_download_stream(file_id)
        except NoFile:
            return None

        return GridFSFileInfo(
            file_id=file_id,
            filename=grid_out.filename,
            length=grid_out.length,
            metadata=grid_out.metadata,
            upload_date=grid_out.upload_date,
            content_type=grid_out.content_type,
        )

    async def read(self, file_id: ObjectId) -> bytes:
        """Devuelve el contenido completo del archivo.

        Si el archivo no existe, propaga ``gridfs.errors.NoFile``.
        """
        bucket = self._bucket()
        grid_out = await bucket.open_download_stream(file_id)
        return await grid_out.read()

    async def delete(self, file_id: ObjectId) -> bool:
        """Elimina el archivo y sus chunks.

        Devuelve ``True`` si existía y fue eliminado, ``False`` si no existía.
        """
        bucket = self._bucket()

        try:
            await bucket.delete(file_id)
        except NoFile:
            return False

        return True
