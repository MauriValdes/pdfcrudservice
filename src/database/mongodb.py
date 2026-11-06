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
        except Exception as e:
            logger.error(f"Error al conectar con MongoDB: {e}")
            raise

    async def close(self):
        if self.client:
            self.client.close()
            logger.info("Conexión con MongoDB cerrada")


db_connection = MongoDB()
