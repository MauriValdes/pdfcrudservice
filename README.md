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

MS3 utiliza dos usuarios independientes: `pdf_ms3_app` tiene `readWrite`
exclusivamente sobre `pdf_database_dev`; `pdf_ms3_test` tiene `readWrite`
exclusivamente sobre `pdf_database_test`. Las pruebas de integración deben
usarse con `MONGO_TEST_URL`, nunca con la URI de aplicación.

La plantilla `../pdfinfraestructure/.env.example` contiene las URI por modo.
Localmente, MS3 usa `MONGO_URL` y las pruebas usan `MONGO_TEST_URL`, con
`localhost:27018`. En Docker Compose, MS3 usa `MONGO_URL_DOCKER` y el proceso
de pruebas usa `MONGO_TEST_URL_DOCKER`, con el nombre interno `db:27017`.
Cada URI selecciona su propia base y autentica contra ella mediante
`authSource`: `pdf_database_dev` para la aplicación y `pdf_database_test`
para testing.

Ejemplos de formato (usar las URI de la plantilla central):

```text
MONGO_URL=mongodb://pdf_ms3_app:<PASSWORD_URL_ENCODED>@localhost:27018/pdf_database_dev?authSource=pdf_database_dev
MONGO_TEST_URL=mongodb://pdf_ms3_test:<PASSWORD_URL_ENCODED>@localhost:27018/pdf_database_test?authSource=pdf_database_test
MONGO_URL_DOCKER=mongodb://pdf_ms3_app:<PASSWORD_URL_ENCODED>@db:27017/pdf_database_dev?authSource=pdf_database_dev
MONGO_TEST_URL_DOCKER=mongodb://pdf_ms3_test:<PASSWORD_URL_ENCODED>@db:27017/pdf_database_test?authSource=pdf_database_test
```

Las contraseñas con caracteres especiales deben codificarse para URL
(percent-encoding) antes de incluirlas en una URI. `MONGO_INITDB_ROOT_USERNAME`
y `MONGO_INITDB_ROOT_PASSWORD` quedan reservadas para inicialización y
administración de MongoDB; MS3 y sus pruebas no deben utilizarlas.
`EXTRACT_SERVICE_URL` apunta al MS2 accesible desde el host; Compose configura
la dirección interna correspondiente.

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

### Lease de carga de PDF

MS3 permite configurar la propiedad temporal de carga mediante estas variables
opcionales en el archivo central `../pdfinfraestructure/.env`:

| Variable | Propósito | Unidad | Default de `Settings` |
|---|---|---|---:|
| `UPLOAD_LEASE_SECONDS` | Duración del lease de carga | segundos | `120` |
| `UPLOAD_LEASE_RENEW_INTERVAL_SECONDS` | Intervalo de renovación del lease de carga | segundos | `30` |

Ambos valores deben ser enteros positivos y el intervalo de renovación debe ser
menor que la duración del lease. Si se omiten, Docker Compose no los inyecta y
MS3 utiliza los defaults de `Settings`. Si se definen explícitamente con valores
inválidos, se conservan y la validación de `Settings` impide el inicio de MS3.

Para configurarlos, agregar sus valores al `.env` central y ejecutar Compose
con `--env-file ../pdfinfraestructure/.env`, como en los comandos anteriores.
No se debe crear un archivo `.env` dentro de `pdfcrudservice`.

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
