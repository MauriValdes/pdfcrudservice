# PDF CRUD Service

Microservicio REST desarrollado con Python y FastAPI para la gestión de documentos PDF y sus metadatos. Permite registrar documentos, calcular su checksum SHA-256, detectar duplicados y realizar operaciones CRUD sobre los registros almacenados en MongoDB.

El servicio forma parte de una arquitectura de microservicios para el procesamiento de documentos PDF.

---

## Objetivos

- Registrar documentos PDF.
- Validar el formato, el contenido y el tamaño de los archivos.
- Calcular el checksum SHA-256 para identificar documentos.
- Evitar registros duplicados.
- Administrar los metadatos y estados de procesamiento.
- Exponer una API REST para realizar operaciones CRUD.
- Permitir el escalado horizontal mediante Docker Compose y Traefik.

---

## Stack tecnológico

- **Lenguaje:** Python 3.12+
- **Framework:** FastAPI
- **Servidor ASGI:** Uvicorn
- **Gestión de dependencias:** uv
- **Validación de datos:** Pydantic
- **Base de datos:** MongoDB
- **Testing:** Pytest y HTTPX
- **Contenedores:** Docker y Docker Compose
- **API Gateway:** Traefik

---

## Arquitectura

El servicio implementa una arquitectura por capas, separando las responsabilidades de cada componente.

| Capa | Responsabilidad |
|---|---|
| API | Definición de endpoints REST y validaciones HTTP |
| Services | Lógica de negocio y gestión de documentos |
| Repository | Operaciones de acceso y persistencia de datos |
| Models | Modelos de dominio |
| Schemas | Esquemas de entrada y salida |
| Database | Conexión con MongoDB |
| Config | Configuración mediante variables de entorno |
| Core | Excepciones y manejo centralizado de errores |

---

## Funcionalidades

- Registro de documentos PDF.
- Validación de archivos y límite de tamaño.
- Generación de checksum SHA-256.
- Detección de documentos duplicados.
- Consulta de todos los documentos.
- Consulta individual por identificador.
- Actualización parcial de metadatos y estados.
- Eliminación de documentos.

### Estados de los documentos

- `PENDING`: documento registrado, pendiente de procesamiento.
- `PROCESSING`: documento en proceso.
- `COMPLETED`: procesamiento finalizado.
- `FAILED`: procesamiento fallido.

---

## Endpoints REST

| Método | Endpoint | Descripción |
|---|---|---|
| POST | `/documents/` | Registrar un documento PDF |
| GET | `/documents` | Obtener todos los documentos |
| GET | `/documents/{document_id}` | Obtener un documento por ID |
| PATCH | `/documents/{document_id}` | Actualizar parcialmente un documento |
| DELETE | `/documents/{document_id}` | Eliminar un documento |

La documentación interactiva se encuentra disponible mediante Swagger UI en `/docs`.

---

## Flujo actual de funcionamiento

1. El cliente envía un archivo PDF.
2. La API valida el formato, el contenido y el tamaño.
3. Se calcula el checksum SHA-256.
4. Se verifica que el documento no esté registrado previamente.
5. Se crea el registro en MongoDB con estado `PENDING`.
6. El cliente recibe la respuesta con los datos del documento registrado.

**Integración con MS2:** la comunicación con el microservicio de extracción PDF (MS2) forma parte de la integración prevista para la arquitectura completa. En el estado actual, este servicio administra los registros y sus metadatos, pero todavía no realiza la extracción mediante MS2.

---

## Requisitos

- Python 3.12 o superior.
- Git.
- Docker.
- Docker Compose.
- uv.

---

## Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/MauriValdes/pdfcrudservice.git
cd pdfcrudservice
```

### 2. Instalar dependencias

```bash
uv sync
```

### 3. Configurar variables de entorno

```bash
cp ../pdfinfraestructure/.env.example ../pdfinfraestructure/.env
```

Editar el archivo central con las credenciales y URLs locales. La aplicación
lee variables del entorno y no carga `.env` por sí misma.

Para ejecución local, `MONGO_URL` debe apuntar al MongoDB publicado en
`localhost`, y `EXTRACT_SERVICE_URL` al MS2 accesible desde el host. Los Compose
individual e integrado sustituyen esas direcciones por las internas adecuadas:
el Compose individual usa `MONGO_URL_DOCKER` (o construye la URI autenticada
desde las variables de MongoDB) y `EXTRACT_SERVICE_URL_SINGLE_DOCKER`; el
integrado inyecta la URI MongoDB hacia `db` y `EXTRACT_SERVICE_URL_DOCKER` hacia
Traefik interno. La plantilla central es la referencia para esos valores.

### 4. Iniciar MongoDB

Desde `pdfcrudservice/`:

```bash
docker compose --env-file ../pdfinfraestructure/.env -f docker-compose.yml up -d db
```

Desde la raíz del workspace:

```bash
docker compose --env-file ./pdfinfraestructure/.env --project-directory ./pdfcrudservice -f ./pdfcrudservice/docker-compose.yml up -d db
```

### 5. Ejecutar la aplicación

```bash
uv run --env-file ../pdfinfraestructure/.env uvicorn src.main:app --reload --port 8001
```

La aplicación estará disponible en:

- API: http://localhost:8001
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

---

## Ejecución con Docker Compose

El archivo central de variables está en `pdfinfraestructure/.env`. Compose no lo
descubre automáticamente desde `pdfcrudservice/`, por eso se indica con
`--env-file` en todos los comandos.

En modo de instancia única, `APP_PORT` solo configura el puerto publicado de
MS3 en el host; el listener del contenedor permanece en `8000`. No controla el
puerto del Compose integrado.

Para iniciar la infraestructura con cinco réplicas del servicio, desde
`pdfcrudservice/`:

```bash
docker compose --env-file ../pdfinfraestructure/.env -f docker-compose.yml up -d --build
```

El mismo comando desde la raíz del workspace:

```bash
docker compose --env-file ./pdfinfraestructure/.env --project-directory ./pdfcrudservice -f ./pdfcrudservice/docker-compose.yml up -d --build
```

El acceso externo se realiza mediante Traefik:

- API: http://localhost
- Dashboard de Traefik: http://localhost:8081

### Modo de instancia única

Para realizar pruebas comparativas con una sola réplica, desde
`pdfcrudservice/`:

```bash
docker compose --env-file ../pdfinfraestructure/.env -f docker-compose.yml -f docker-compose.single.yml up -d --build
```

O desde la raíz del workspace:

```bash
docker compose --env-file ./pdfinfraestructure/.env --project-directory ./pdfcrudservice -f ./pdfcrudservice/docker-compose.yml -f ./pdfcrudservice/docker-compose.single.yml up -d --build
```

La API estará disponible directamente en:

http://localhost:8001

---

## Pruebas automatizadas

Ejecutar la suite de pruebas mediante:

```bash
uv run pytest
```

---

## Metodología y buenas prácticas

El desarrollo contempla:

- Test Driven Development (TDD).
- Principios SOLID.
- KISS, DRY y YAGNI.
- Separación de responsabilidades.
- Arquitectura por capas.
- Configuración externa mediante variables de entorno.
- Aplicación de principios de Twelve-Factor App.

---

## Equipo de desarrollo

- Mauricio Valdés
- Fausto Basile

---

## Contexto académico

Proyecto desarrollado con fines académicos para la asignatura **Desarrollo de Software** de la **Universidad Tecnológica Nacional (UTN)**.
