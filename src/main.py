"""Punto de entrada del microservicio MS3 (PDF CRUD Service).

Orden de inicialización (deliberado):

1. Se importa la configuración (solo pydantic-settings) y el módulo de
   logging centralizado (solo biblioteca estándar).
2. Se configura el logging con configure_logging() ANTES de importar
   FastAPI y el resto de dependencias de la aplicación. Esto es necesario
   porque logging.config.dictConfig() cierra los handlers registrados en el
   proceso: si se ejecutara más tarde, cerraría los handlers que hubiera
   registrado cualquier integración externa (APM, OpenTelemetry, etc.).
3. Recién entonces se importan FastAPI y los módulos de la aplicación, que
   ya quedan con el logging configurado.

A partir de este punto, toda la salida de logs del servicio es JSON por
stdout/stderr y no existe ningún mensaje de texto plano.
"""

import logging
from contextlib import asynccontextmanager

from src.config.config import settings
from src.core.logging_config import configure_logging

# Nombre del servicio: identifica el origen de cada registro.
SERVICE_NAME = "ms3-crud"

# 1) Logging centralizado: formato JSON, stdout/stderr y nivel desde Settings.
configure_logging(SERVICE_NAME, level=settings.log_level)

# 2) Dependencias de la aplicación (importadas con el logging ya configurado).
from fastapi import FastAPI  # noqa: E402

from src.api.docs_api import router as docs_router  # noqa: E402
from src.core.exceptions import register_exception_handlers  # noqa: E402
from src.database.mongodb import db_connection  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Iniciando conexión con MongoDB...")
    await db_connection.connect()

    yield

    logger.info("Cerrando conexiones...")
    await db_connection.close()


logger.info("Servicio MS3 (pdfcrudservice) iniciado")

app = FastAPI(
    title="PDF Extract Text API",
    description="API para procesar PDFs y almacenar resultados en MongoDB",
    version="0.1.0",
    lifespan=lifespan
)

# Manejo centralizado de excepciones de dominio (AppError -> JSON HTTP)
register_exception_handlers(app)

app.include_router(docs_router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Servidor funcionando y base de datos vinculada"
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}
