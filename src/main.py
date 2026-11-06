from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.database.mongodb import db_connection
from src.api.docs_api import router as docs_router
from src.core.exceptions import register_exception_handlers

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Iniciando conexión con MongoDB... 📡")
    await db_connection.connect()

    yield

    print("Cerrando conexiones... 🔌")
    await db_connection.close()

print("¡Hola desde main.py!")

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
