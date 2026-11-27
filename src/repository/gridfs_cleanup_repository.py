"""Primitivas internas para borrar y verificar recursos GridFS exactos.

El llamador debe proporcionar un GridFSRepository configurado para la misma
base y el bucket ``fs`` utilizado aquí. La operación requiere autorización y
auditoría durable externas, ausencia de escrituras GridFS activas, identidad
inmutable del recurso y seguridad frente a referencias compartidas.
"""

from bson import ObjectId

from src.repository.gridfs_repository import GridFSRepository


class GridFSCleanupRepository:
    """Borra un único ObjectId GridFS y verifica metadata y chunks residuales."""

    def __init__(self, db, gridfs_repository=None):
        self.db = db
        if gridfs_repository is None:
            gridfs_repository = GridFSRepository(db)
        elif isinstance(gridfs_repository, GridFSRepository):
            repository_db = gridfs_repository.db
            if (
                repository_db.name != db.name
                or repository_db.client is not db.client
            ):
                raise ValueError(
                    "GridFSRepository debe usar la misma base y cliente"
                )
        self.gridfs_repository = gridfs_repository

    async def delete_and_verify_absent(self, file_id: ObjectId) -> bool:
        """Delete once, then check exact metadata and chunk records.

        Exceptions and cancellation propagate; neither is evidence of physical
        absence. This primitive does not itself authorize deletion.
        """
        if not isinstance(file_id, ObjectId):
            raise ValueError("Identificador GridFS inválido")

        await self.gridfs_repository.delete(file_id)
        file_metadata = await self.db["fs.files"].find_one(
            {"_id": file_id},
            {"_id": 1},
        )
        file_chunk = await self.db["fs.chunks"].find_one(
            {"files_id": file_id},
            {"_id": 1},
        )
        return file_metadata is None and file_chunk is None
