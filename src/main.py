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
import asyncio
from contextlib import asynccontextmanager
from fastapi import HTTPException, status  # noqa: E402

from src.config.config import settings
from src.core.logging_config import configure_logging

# Nombre del servicio: identifica el origen de cada registro.
SERVICE_NAME = "ms3-crud"

# 1) Logging centralizado: formato JSON, stdout/stderr y nivel desde Settings.
configure_logging(SERVICE_NAME, level=settings.log_level)

# 2) Dependencias de la aplicación (importadas con el logging ya configurado).
from fastapi import FastAPI  # noqa: E402

from src.api.docs_api import router as docs_router  # noqa: E402
from src.api.pdf_validation import PdfValidationExecutor  # noqa: E402
from src.core.exceptions import register_exception_handlers  # noqa: E402
from src.database.mongodb import db_connection, retry_connect  # noqa: E402

logger = logging.getLogger(__name__)

# Log de inicio de servicio: debe aparecer cuando el módulo se carga/recarga.
# Este es necesario para que los tests que hacen importlib.reload(src.main)
# verifiquen el arranque correcto.
logger.info("Servicio MS3 (pdfcrudservice) iniciado")

# 3) Definición del lifespan con supervisión de conexión MongoDB.
#    La tarea de conexión y supervisión se inicia en background para no
#    bloquear el arranque. Si MongoDB no está disponible, FastAPI inicia
#    normalmente y las tareas intentarán recuperarse automáticamente.


@asynccontextmanager
async def lifespan(app: FastAPI):
    pdf_validation_executor = PdfValidationExecutor()
    app.state.pdf_validation_executor = pdf_validation_executor
    connection_task = None

    try:
        # Iniciar tarea de conexión y supervisión de MongoDB en background.
        # Esto permite que FastAPI inicie aunque MongoDB no esté disponible.
        # La tarea intentará conectarse con backoff exponencial y, tras éxito,
        # iniciará supervisión continua con ping periódico.
        logger.info("Iniciando conexión con MongoDB...")
        connection_task = asyncio.create_task(retry_connect())
        logger.info("Tarea de conexión MongoDB iniciada en background")
        yield
    finally:
        # Rechazar nuevas validaciones antes de esperar a que termine la activa.
        app.state.pdf_validation_executor = None
        pdf_validation_executor.begin_shutdown()
        try:
            await pdf_validation_executor.aclose()
        finally:
            # Mantener la limpieza MongoDB existente incluso si falla el cierre
            # del executor PDF.
            if connection_task is not None:
                # Cancelar la tarea de conexión/supervisión
                connection_task.cancel()
            # Detener tarea de supervisión (si existe)
            await db_connection.stop_supervision()
            # Esperar a que la tarea de conexión se detenga
            if connection_task is not None:
                try:
                    await connection_task
                except asyncio.CancelledError:
                    logger.info(
                        "Tarea de conexión MongoDB cancelada durante apagado"
                    )
            # Cerrar conexión MongoDB gracefully
            await db_connection.disconnect()
            logger.info("Cerrando conexiones...")


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
    """Liveness check: verifica que el proceso FastAPI esté vivo.

    Este endpoint es independiente de MongoDB: responde 200 aunque la BD
    esté caída, ya que su propósito es que el orquestador (K8s, Docker)
    sepa que el container sigue corriendo. No realiza operaciones de BD.
    """
    return {"status": "ok", "service": "ms3-crud", "timestamp": "live"}


@app.get("/ready")
async def readiness_check():
    """Readiness check: verifica que el servicio esté listo para recibir peticiones.

    Retorna 200 cuando MongoDB está disponible Y sus índices críticos
    están inicializados. Retorna 503 cuando MongoDB no está disponible
    o sus índices no están listos.

    Este endpoint es usado por orquestadores para decidir enrutar tráfico.
    """
    # Usar el estado consolidado de indexes_ready que combina
    # _connected y _indexes_created en mongodb.py
    if db_connection.indexes_ready:
        # Índices listos y conexión activa
        return {"status": "ready", "database": "mongodb_connected"}
    # Si no estamos listos por ninguna razón (no conectado, índices faltantes, etc.)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"database": "mongodb_unavailable", "reason": "Índices no listos o MongoDB no disponible"}
    )
