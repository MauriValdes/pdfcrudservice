import logging

from motor.motor_asyncio import AsyncIOMotorClient

from src.config.config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self, database_name: str = None):
        """
        Conecta a MongoDB. Si se especifica database_name, usa esa base de datos (para tests).
        Si no se especifica, usa la de desarrollo por defecto.
        """
        try:
            self.client = AsyncIOMotorClient(settings.mongo_url)

            # Selección dinámica de la base de datos
            if database_name:
                self.db = self.client[database_name]
            else:
                self.db = self.client[settings.mongo_db_dev]

            # Verificar conexión
            await self.client.admin.command('ping')
            logger.info(f"Conectado a MongoDB en la base de datos: {self.db.name}")

            # Crear índice único sobre 'checksum' si no existe
            await self._ensure_unique_index_on_checksum()
        except Exception as e:
            logger.error(f"Error al conectar con MongoDB: {e}")
            raise

    async def _ensure_unique_index_on_checksum(self):
        """
        Crea un índice único sobre el campo 'checksum' si no existe.
        La operación es idempotente: si el índice ya existe con la misma configuración,
        MongoDB no lanza error.
        """
        try:
            await self.db[settings.mongo_collection].create_index(
                [("checksum", 1)],
                name="checksum_unique",
                unique=True,
            )
            logger.info("Índice único 'checksum_unique' creado o ya existente.")
        except Exception as e:
            logger.error(f"Error al crear índice único en 'checksum': {e}")
            raise

    async def close(self):
        if self.client:
            self.client.close()
            logger.info("Conexión con MongoDB cerrada")


db_connection = MongoDB()
