import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Mapping, Optional
from uuid import uuid4

from bson import ObjectId
import httpx
from pymongo.errors import DuplicateKeyError

from src.config.config import settings
from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentCreate, DocumentUpdate, DocumentResponse
from src.repository.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


def _upload_utc_now() -> datetime:
    """Reloj de pared UTC aislado para la coordinación de cargas."""
    return datetime.now(timezone.utc)


def _upload_monotonic_now() -> float:
    """Reloj monotónico usado para deadlines e intervalos del lease."""
    return time.monotonic()


async def _wait_for_upload_signal(
    event: asyncio.Event,
    timeout: float,
) -> bool:
    """Espera una señal hasta el timeout; False indica vencimiento."""
    if event.is_set():
        return True
    if timeout <= 0:
        return False
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


def _normalize_upload_utc(value: datetime) -> datetime:
    """Normaliza una fecha a UTC sin perder su precisión."""
    if not isinstance(value, datetime):
        raise ValueError("La fecha del lease debe ser datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_upload_expiry(value: datetime) -> datetime:
    """Normaliza un vencimiento confirmado y trunca a precisión BSON."""
    value = _normalize_upload_utc(value)
    return value.replace(microsecond=(value.microsecond // 1000) * 1000)


ReservationClassification = Literal[
    "created",
    "existing",
    "existing_uploading",
]


class UploadLeaseLostError(Exception):
    """La carga dejó de tener autorización para continuar."""


class UploadFinalizationRejectedError(Exception):
    """La carga GridFS terminó, pero MongoDB rechazó su finalización."""


class DocumentService:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
        # Configuración de comunicación con MS2 (Extract Service)
        self.extract_service_url = settings.extract_service_url
        self.extract_service_timeout = settings.extract_service_timeout

    async def reserve_or_classify(
        self,
        document_data: DocumentCreate,
    ) -> tuple[DocumentModel, ReservationClassification]:
        """Reserva el checksum y clasifica el documento devuelto."""
        document, created = await self.repository.reserve_document(document_data)

        if created:
            return document, "created"
        if document.status == DocumentStatus.UPLOADING:
            return document, "existing_uploading"
        return document, "existing"

    async def receive_pdf(
        self,
        *,
        file_bytes: bytes,
        filename: str,
    ) -> tuple[DocumentModel, Literal["accepted", "existing"]]:
        """Orquesta la recepción durable sin iniciar la extracción."""
        checksum = hashlib.sha256(file_bytes).hexdigest()
        document_data = DocumentCreate(
            filename=filename,
            checksum=checksum,
            size_bytes=len(file_bytes),
        )
        document, classification = await self.reserve_or_classify(document_data)

        if classification == "existing":
            return document, "existing"
        if classification not in ("created", "existing_uploading"):
            raise ValueError("Clasificación de reserva desconocida.")

        acquired_document, ownership_acquired = (
            await self.acquire_upload_ownership(
                document,
                settings.upload_lease_seconds,
            )
        )
        if not ownership_acquired:
            if acquired_document is None:
                raise UploadLeaseLostError(
                    "No se pudo recuperar el documento de la reserva."
                )
            return acquired_document, "existing"

        if (
            not isinstance(acquired_document, DocumentModel)
            or acquired_document.status != DocumentStatus.UPLOADING
            or acquired_document.file_id is not None
            or not acquired_document.id
            or not acquired_document.upload_owner_id
            or not acquired_document.upload_fencing_token
        ):
            raise UploadLeaseLostError(
                "La adquisición no devolvió una reserva uploading válida."
            )

        file_id = ObjectId()
        metadata = {
            "document_id": acquired_document.id,
            "checksum": acquired_document.checksum,
            "upload_owner_id": acquired_document.upload_owner_id,
            "upload_fencing_token": acquired_document.upload_fencing_token,
        }
        finalized = await self.save_and_finalize_upload(
            acquired_document,
            file_id,
            filename,
            file_bytes,
            metadata,
            ownership_acquired=ownership_acquired,
        )
        if not finalized:
            raise UploadFinalizationRejectedError(
                "MongoDB rechazó la finalización de la carga."
            )

        accepted_document = acquired_document.model_copy(
            update={
                "status": DocumentStatus.PENDING,
                "file_id": file_id,
            }
        )
        return accepted_document, "accepted"

    async def acquire_upload_ownership(
        self,
        document: DocumentModel,
        lease_duration_seconds: int,
    ) -> tuple[DocumentModel | None, bool]:
        """Intenta adquirir la propiedad temporal de una reserva de carga."""
        if lease_duration_seconds <= 0:
            raise ValueError("lease_duration_seconds debe ser mayor que cero.")

        upload_owner_id = str(uuid4())
        upload_fencing_token = str(uuid4())
        upload_lease_until = datetime.now(timezone.utc) + timedelta(
            seconds=lease_duration_seconds
        )

        return await self.repository.acquire_upload_ownership(
            document.checksum,
            upload_owner_id,
            upload_fencing_token,
            upload_lease_until,
        )

    async def renew_upload_lease(
        self,
        document: DocumentModel,
    ) -> datetime | None:
        """Renueva una vez el lease de carga si se conserva su propiedad."""
        if (
            not document.id
            or not document.upload_owner_id
            or not document.upload_fencing_token
        ):
            raise ValueError(
                "El documento debe tener id, propietario y token de carga."
            )

        upload_lease_until = _upload_utc_now() + timedelta(
            seconds=settings.upload_lease_seconds
        )
        renewed = await self.repository.renew_upload_lease(
            document.id,
            document.upload_owner_id,
            document.upload_fencing_token,
            upload_lease_until,
        )
        return upload_lease_until if renewed else None

    async def save_original_pdf_with_lease(
        self,
        document: DocumentModel,
        file_id: ObjectId,
        filename: str,
        data: bytes,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        ownership_acquired: bool,
    ) -> ObjectId:
        """Guarda el PDF mientras coordina la renovación de su lease."""
        if ownership_acquired is not True:
            raise UploadLeaseLostError("La propiedad inicial no fue adquirida.")
        if document.status != DocumentStatus.UPLOADING:
            raise ValueError("El documento debe estar en estado uploading.")
        if document.file_id is not None:
            raise ValueError("El documento no debe tener un file_id asociado.")
        if not ObjectId.is_valid(document.id):
            raise ValueError("El id del documento debe ser un ObjectId válido.")
        if not isinstance(document.upload_owner_id, str) or not document.upload_owner_id:
            raise ValueError("El documento debe tener upload_owner_id.")
        if (
            not isinstance(document.upload_fencing_token, str)
            or not document.upload_fencing_token
        ):
            raise ValueError("El documento debe tener upload_fencing_token.")
        if not isinstance(document.upload_lease_until, datetime):
            raise ValueError("El documento debe tener upload_lease_until válido.")
        if not isinstance(file_id, ObjectId):
            raise ValueError("file_id debe ser una instancia de ObjectId.")

        confirmed_expiry = _normalize_upload_expiry(document.upload_lease_until)
        deadline_anchor = _upload_monotonic_now()
        utc_now = _normalize_upload_utc(_upload_utc_now())
        if confirmed_expiry <= utc_now:
            raise UploadLeaseLostError("El lease inicial ya venció.")

        confirmed_deadline = deadline_anchor + (
            confirmed_expiry - utc_now
        ).total_seconds()
        if confirmed_deadline <= _upload_monotonic_now():
            raise UploadLeaseLostError("El lease inicial ya venció.")

        renew_interval = float(settings.upload_lease_renew_interval_seconds)
        if renew_interval <= 0:
            raise ValueError("El intervalo de renovación debe ser positivo.")

        terminal_error: BaseException | None = None
        secondary_error_ids: set[int] = set()
        writer_done = False
        writer_observed = False
        # También actúa como señal de parada para despertar al coordinador.
        activity_event = asyncio.Event()
        renewal_task: asyncio.Task | None = None
        renewal_observed = False
        next_renewal_at = _upload_monotonic_now() + renew_interval
        if confirmed_deadline - _upload_monotonic_now() <= renew_interval:
            next_renewal_at = _upload_monotonic_now()

        def set_terminal(error: BaseException) -> None:
            nonlocal terminal_error
            if terminal_error is None:
                terminal_error = error
                activity_event.set()
            elif error is not terminal_error and id(error) not in secondary_error_ids:
                secondary_error_ids.add(id(error))
                logger.warning(
                    "Secondary upload coordination error ignored (%s)",
                    type(error).__name__,
                )

        def should_continue() -> bool:
            if terminal_error is not None:
                return False
            if _upload_monotonic_now() >= confirmed_deadline:
                set_terminal(
                    UploadLeaseLostError("El lease de carga venció.")
                )
                return False
            return True

        def observe_writer(task: asyncio.Task) -> None:
            nonlocal writer_done, writer_observed
            if writer_observed:
                return
            writer_observed = True
            writer_done = True
            try:
                task.result()
            except BaseException as error:
                set_terminal(error)
            activity_event.set()

        def observe_renewal(task: asyncio.Task) -> None:
            nonlocal confirmed_expiry, confirmed_deadline
            nonlocal next_renewal_at, renewal_observed
            if renewal_observed:
                return
            renewal_observed = True
            try:
                renewed_value = task.result()
            except BaseException as error:
                try:
                    observed_monotonic = _upload_monotonic_now()
                    if (
                        terminal_error is None
                        and observed_monotonic >= confirmed_deadline
                    ):
                        set_terminal(
                            UploadLeaseLostError(
                                "La renovación se observó después del vencimiento."
                            )
                        )
                except BaseException as clock_error:
                    set_terminal(clock_error)
                set_terminal(error)
                return

            try:
                if terminal_error is not None:
                    return
                observed_monotonic = _upload_monotonic_now()
                if observed_monotonic >= confirmed_deadline:
                    set_terminal(
                        UploadLeaseLostError(
                            "La renovación se observó después del vencimiento."
                        )
                    )
                    return
                if renewed_value is None:
                    set_terminal(
                        UploadLeaseLostError("Se rechazó la renovación del lease.")
                    )
                    return
                renewed_expiry = _normalize_upload_expiry(renewed_value)
                utc_now = _normalize_upload_utc(_upload_utc_now())
                if renewed_expiry <= utc_now or renewed_expiry <= confirmed_expiry:
                    raise ValueError("Renovación vencida o no extensiva.")
                renewed_deadline = observed_monotonic + (
                    renewed_expiry - utc_now
                ).total_seconds()
                if renewed_deadline <= _upload_monotonic_now():
                    raise ValueError("El nuevo deadline ya venció.")
            except (TypeError, ValueError, OverflowError):
                set_terminal(
                    UploadLeaseLostError(
                        "La renovación devolvió un vencimiento inválido."
                    )
                )
            except BaseException as error:
                set_terminal(error)
            else:
                confirmed_expiry = renewed_expiry
                confirmed_deadline = renewed_deadline
                next_renewal_at = observed_monotonic + renew_interval
            finally:
                activity_event.set()

        writer_task = asyncio.create_task(
            self.save_original_pdf(
                file_id=file_id,
                filename=filename,
                data=data,
                metadata=metadata,
                should_continue=should_continue,
            )
        )
        writer_task.add_done_callback(observe_writer)

        async def drain_task(task: asyncio.Task) -> None:
            """Drena una operación sin propagarle cancelaciones del caller."""
            nonlocal terminal_error
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as cancellation:
                    current_task = asyncio.current_task()
                    externally_cancelled = (
                        current_task is not None and current_task.cancelling() > 0
                    )
                    if externally_cancelled or not (
                        task.done() and task.cancelled()
                    ):
                        set_terminal(cancellation)
                        if not writer_task.done():
                            writer_task.cancel()
                    if task.done():
                        break
                except BaseException:
                    break

            try:
                task.result()
            except BaseException as error:
                set_terminal(error)

        try:
            while True:
                if writer_task.done() and not writer_observed:
                    observe_writer(writer_task)
                if renewal_task is not None and renewal_task.done():
                    if not renewal_observed:
                        observe_renewal(renewal_task)
                    renewal_task = None

                if terminal_error is not None:
                    if not writer_task.done():
                        writer_task.cancel()
                    break

                now_monotonic = _upload_monotonic_now()
                if now_monotonic >= confirmed_deadline:
                    set_terminal(
                        UploadLeaseLostError("El lease de carga venció.")
                    )
                    continue

                if writer_done and renewal_task is None:
                    if _upload_monotonic_now() >= confirmed_deadline:
                        set_terminal(
                            UploadLeaseLostError(
                                "El lease venció antes de retornar."
                            )
                        )
                        continue
                    return file_id

                if renewal_task is not None:
                    timeout = confirmed_deadline - now_monotonic
                    activity_event.clear()
                    if renewal_task.done():
                        continue
                    signaled = await _wait_for_upload_signal(
                        activity_event,
                        timeout,
                    )
                    if (
                        not signaled
                        and _upload_monotonic_now() >= confirmed_deadline
                    ):
                        set_terminal(
                            UploadLeaseLostError(
                                "El lease venció con una renovación pendiente."
                            )
                        )
                    continue

                if now_monotonic >= next_renewal_at:
                    renewal_observed = False
                    renewal_task = asyncio.create_task(
                        self.renew_upload_lease(document)
                    )
                    renewal_task.add_done_callback(observe_renewal)
                    continue

                timeout = min(
                    next_renewal_at,
                    confirmed_deadline,
                ) - now_monotonic
                activity_event.clear()
                if writer_task.done():
                    continue
                await _wait_for_upload_signal(activity_event, timeout)
        except asyncio.CancelledError as cancellation:
            set_terminal(cancellation)
            if not writer_task.done():
                writer_task.cancel()
        except BaseException as error:
            set_terminal(error)
            if not writer_task.done():
                writer_task.cancel()

        if renewal_task is not None:
            # Nunca se cancela una renovación iniciada: puede ser una operación Motor.
            await drain_task(renewal_task)
        await drain_task(writer_task)

        assert terminal_error is not None
        raise terminal_error

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

        return DocumentResponse.for_public_read(
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
            httpx.TransportError: Si falla la comunicación de transporte con MS2.
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
        except httpx.TransportError as e:
            # Cualquier fallo de transporte (timeout, conexión, lectura,
            # escritura o protocolo) deja el documento en un estado terminal.
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

    async def save_original_pdf(
        self,
        file_id: ObjectId,
        filename: str,
        data: bytes,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        should_continue: Callable[[], bool] | None = None,
    ) -> ObjectId:
        """Guarda un PDF original en GridFS usando un identificador preasignado.

        Este método forma parte del flujo de persistencia v1.6 para GridFS:
        - ``file_id`` debe ser un ``ObjectId`` nuevo y único por cada intento de carga.
        - La escritura se confirma con ``await grid_in.close()``.
        - Ante errores o cancelaciones durante ``write()`` o ``close()``, se intenta
          ``abort()`` como limpieza best-effort y se propaga la excepción original.
        - El ``file_id`` únicamente se considera guardado después de completar exitosamente
          el cierre ``close()``; en caso contrario, no se devuelve y la carga se considera
          fallida sin archivo asociado en ``fs.files``.

        Esta operación reutiliza la conexión MongoDB ya inyectada en el repositorio
        (``self.repository.db``), por lo que comparte el Write Concern configurado
        (``w=1, journal=True``) y no crea un nuevo cliente.

        Args:
            file_id: Identificador preasignado (debe ser ``ObjectId()`` nuevo por intento).
            filename: Nombre lógico del archivo PDF.
            data: Contenido del PDF en bytes.
            metadata: Metadatos opcionales que se persistirán en el documento de ``fs.files``.

        Returns:
            ``file_id`` únicamente si el cierre ``close()`` finaliza exitosamente.

        Raises:
            ``gridfs.errors.NoFile`` u otros errores de GridFS si la operación falla
            de manera que impida el guardado.
            ``BaseException`` (incluye ``asyncio.CancelledError``) si la escritura o el
            cierre se interrumpen.
        """
        gridfs_repo = self.repository._gridfs_repository()
        save_arguments = {
            "file_id": file_id,
            "filename": filename,
            "data": data,
            "metadata": metadata,
        }
        if should_continue is not None:
            save_arguments["should_continue"] = should_continue
        return await gridfs_repo.save(**save_arguments)

    async def save_and_finalize_upload(
        self,
        document: DocumentModel,
        file_id: ObjectId,
        filename: str,
        data: bytes,
        metadata: Mapping[str, Any] | None = None,
        *,
        ownership_acquired: bool,
    ) -> bool:
        """Guarda el PDF y solicita su asociación atómica a la reserva.

        La adquisición de propiedad debe haberse confirmado previamente.
        ``file_id`` debe ser un ObjectId nuevo y exclusivo de este intento.
        La finalización vuelve a validar atómicamente la propiedad y vigencia
        del lease en el repositorio; este método no compensa ni reconcilia
        resultados rechazados o inciertos.
        """
        await self.save_original_pdf_with_lease(
            document,
            file_id,
            filename,
            data,
            metadata,
            ownership_acquired=ownership_acquired,
        )
        return await self.repository.finalize_upload(
            document.id,
            document.upload_owner_id,
            document.upload_fencing_token,
            file_id,
        )
