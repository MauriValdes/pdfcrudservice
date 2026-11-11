from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_url: str = Field(default="mongodb://localhost:27017", alias="MONGO_URL")
    # 🗄️ Base de datos de desarrollo (usada por defecto al conectar)
    mongo_db_dev: str = "pdf_database_dev"
    max_file_size_mb: int = 5
    mongo_collection: str = "processed_pdfs"
    # 📡 Servicio de extracción (MS2)
    extract_service_url: str = Field(default="http://localhost:8002", alias="EXTRACT_SERVICE_URL")
    extract_service_timeout: int = Field(default=30, alias="EXTRACT_SERVICE_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8',
        extra='ignore' 
    )
settings = Settings()