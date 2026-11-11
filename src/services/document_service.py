import hashlib
import httpx

from pymongo.errors import DuplicateKeyError

from src.config.config import settings
from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from src.repository.document_repository import DocumentRepository
from datetime import datetime, timezone


class DocumentService:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
        # Configuración de comunicación con MS2 (Extract Service)
        self.extract_service_url = settings.extract_service_url
        self.extract_service_timeout = settings.extract_service_timeout

    async def submit_pdf(self, file_bytes: bytes, filename: str) -> DocumentResponse:
        """Guarda los metadatos básicos del PDF en la base de datos.

        Persiste el documento con estado PENDING y retorna el DTO de respuesta.
        La extracción de texto y el procesamiento pesado se coordina
        desde el MS2 (Extract Service) o workers externos.
        """
        # 1. Calcular checksum del PDF
        checksum = hashlib.sha256(file_bytes).hexdigest()

        # 2. Verificar si el documento ya existe por su checksum
        existing_doc = await self.repository.find_by_checksum(checksum)
        if existing_doc is not None:
            raise ValueError("El documento ya existe")

        # 3. Construir el documento de metadata con estado PENDING
        now = datetime.now(timezone.utc)
        doc_dict = DocumentCreate(
            filename=filename,
            checksum=checksum,
            size_bytes=len(file_bytes),
        ).model_dump()

        doc_orm = DocumentModel(
            filename=doc_dict["filename"],
            checksum=doc_dict["checksum"],
            size_bytes=doc_dict["size_bytes"],
            status=DocumentStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        # Guardar en MongoDB
        try:
            result = await self.repository.collection.insert_one(
                doc_orm.model_dump(by_alias=True, exclude={"id"})
            )
        except DuplicateKeyError:
            raise ValueError("El documento ya existe")

        # 4. Retornar el DTO de respuesta
        return DocumentResponse(
            id=str(result.inserted_id),
            filename=doc_orm.filename,
            checksum=doc_orm.checksum,
            size_bytes=doc_orm.size_bytes,
            status=doc_orm.status.value,
            created_at=doc_orm.created_at,
            updated_at=doc_orm.updated_at,
        )

    async def get_all_documents(self) -> list[DocumentResponse]:
        """Obtiene la lista completa de documentos como DTOs."""
        return await self.repository.find_all()

    async def get_document_by_id(self, doc_id: str) -> DocumentResponse | None:
        """Obtiene un documento por su ID (None si no existe o el ID es inválido)."""
        doc = await self.repository.find_by_id(doc_id)
        if not doc:
            return None

        # Handle both dict (from MongoDB) and DocumentResponse objects
        if isinstance(doc, dict):
            id_str = str(doc.get("_id", "")) if doc.get("_id") else ""
            filename = doc.get("filename", "")
            checksum = doc.get("checksum", "")
            size_bytes = doc.get("size_bytes", 0)
            status = doc.get("status", "pending")
            content = doc.get("content")
            page_count = doc.get("page_count")
            error_message = doc.get("error_message")
            created_at = doc.get("created_at", datetime.now())
            updated_at = doc.get("updated_at", datetime.now())
        else:
            # DocumentResponse or similar ORM object
            id_str = str(doc.id) if doc.id else ""
            filename = doc.filename
            checksum = doc.checksum
            size_bytes = doc.size_bytes
            status = doc.status if isinstance(doc.status, str) else getattr(doc.status, "value", "pending")
            content = doc.content
            page_count = doc.page_count
            error_message = doc.error_message
            created_at = doc.created_at
            updated_at = doc.updated_at

        return DocumentResponse(
            id=id_str,
            filename=filename,
            checksum=checksum,
            size_bytes=size_bytes,
            status=status,
            content=content,
            page_count=page_count,
            error_message=error_message,
            created_at=created_at,
            updated_at=updated_at,
        )

    async def update_document(
        self, document_id: str, update_data: DocumentUpdate
    ) -> DocumentResponse | None:
        """Actualiza un documento con los nuevos valores.

        Soporta actualizaciones parciales: solo los campos proporcionados
        serán actualizados en la base de datos.
        """
        # Convertir DocumentUpdate a diccionario, incluyendo solo los campos
        # que fueron proporcionados (excluyendo los valores None por defecto)
        update_dict = update_data.model_dump(exclude_unset=True)

        if not update_dict:
            return None

        return await self.repository.update_document(document_id, update_data)

    # --- NUEVO: Comunicación HTTP con MS2 - Extracción de texto ---

    async def call_extraction_service(
        self, pdf_bytes: bytes, filename: str
    ) -> dict:
        """Realiza una petición HTTP al servicio MS2 para extraer texto de un PDF.

        Envía el archivo en formato multipart/form-data con campo 'file'.
        Valida la respuesta y retorna el diccionario parseado.

        Raises:
            httpx.TimeoutException: Si la conexión excede el timeout configurado.
            httpx.ConnectError: Si el servicio MS2 no está disponible.
            ValueError: Si la respuesta de MS2 es inválida (códigos de error,
                        campos faltantes o tipos incorrectos).
        """
        url = self.extract_service_url.rstrip("/") + "/extract"
        timeout = self.extract_service_timeout

        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            response = await client.post(url, files=files)

        # Verificar código de estado HTTP
        if response.status_code != 200:
            if response.status_code >= 500:
                # Errores de servidor MS2: mensaje siempre genérico, sin exponer
                # el detalle interno que el servicio haya enviado en el cuerpo.
                raise ValueError(
                    f"MS2 retornó error HTTP {response.status_code}: "
                    "servicio de extracción no disponible"
                )
            # Errores de cliente (4xx): conservamos el detalle útil de MS2
            error_detail = "Unknown error from extraction service"
            try:
                error_json = response.json()
                error_detail = error_json.get(
                    "detail", error_detail
                )
            except Exception:
                pass
            raise ValueError(
                f"MS2 retornó error HTTP {response.status_code}: {error_detail}"
            )

        # Parsear respuesta JSON
        try:
            extract_data = response.json()
        except Exception as e:
            raise ValueError(f"MS2 respondió con cuerpo no JSON: {str(e)}")

        # Validar campos obligatorios y tipos
        content = extract_data.get("content")
        page_count = extract_data.get("page_count")

        # content debe ser string (puede estar vacío - PDFs escaneados son válidos)
        if not isinstance(content, str):
            raise ValueError(
                f"El campo 'content' debe ser string, recibido: {type(content).__name__}"
            )

        # page_count debe ser entero >= 0 (explicitamente rechazamos booleanos,
        # ya que en Python isinstance(True, int) retorna True)
        if (
            not isinstance(page_count, int)
            or isinstance(page_count, bool)
            or page_count < 0
        ):
            raise ValueError(
                f"El campo 'page_count' debe ser entero >= 0, recibido: {page_count}"
            )

        return {
            "content": content,
            "page_count": page_count,
        }

    # --- NUEVO: Orquestación de extracción de PDF ---

    async def process_pdf_extraction(
        self, document_id: str, pdf_bytes: bytes, filename: str
    ) -> None:
        """Orquesta el procesamiento de un PDF: llama a MS2 y actualiza el estado.

        Flujo:
        1. Llamar a MS2 /extract para obtener content y page_count.
        2. Si tiene éxito: actualizar MongoDB a COMPLETED con los resultados.
        3. Si falla: actualizar MongoDB a FAILED con el mensaje de error.
        4. Cualquier error en la actualización de MongoDB se propaga (no se oculta).

        Importante: Este método NO devuelve nada al cliente. El cliente debe
        consultar el documento en MongoDB para obtener el estado final.
        """
        # 1. Llamar a MS2 para extraer texto
        try:
            extract_result = await self.call_extraction_service(pdf_bytes, filename)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            # Error de comunicación con MS2
            await self.repository.update_document(
                document_id,
                DocumentUpdate(status="failed", error_message=str(e)),
            )
            raise
        except ValueError as e:
            # Error de validación de la respuesta de MS2
            await self.repository.update_document(
                document_id,
                DocumentUpdate(status="failed", error_message=str(e)),
            )
            raise

        # 2. Actualizar MongoDB a COMPLETED con los resultados de la extracción
        try:
            await self.repository.update_document(
                document_id,
                DocumentUpdate(
                    status="completed",
                    content=extract_result["content"],
                    page_count=extract_result["page_count"],
                ),
            )
        except Exception as e:
            # Si falla la actualización de MongoDB, volvemos a lanzar la excepción
            # El documento puede quedar en estado PENDING o tener un estado parcial
            # No intente recuperar automáticamente - eso corresponde a una etapa futura
            raise

    # --- FIN de la sección de comunicación HTTP con MS2 ---

    async def delete_document(self, document_id: str) -> bool:
        """Elimina un documento por su ID.

        Raises:
            ValueError: Si el documento no existe.
        """
        deleted = await self.repository.delete_by_id(document_id)
        if not deleted:
            raise ValueError("El documento no existe")
        return True