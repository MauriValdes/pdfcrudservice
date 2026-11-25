import ipaddress
import re

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymongo.uri_parser import parse_uri


MONGO_TEST_DATABASE = "pdf_database_test"


def _is_valid_mongo_host(host: str) -> bool:
    """Validate an IP literal or DNS hostname without resolving it."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    try:
        hostname = host.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if not hostname or len(hostname) > 253:
        return False
    return all(
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
        is not None
        for label in hostname.split(".")
    )


class Settings(BaseSettings):
    # Local default without authentication; Docker Compose injects the
    # authenticated URI explicitly for its MongoDB container.
    mongo_url: str = Field(default="mongodb://localhost:27017", alias="MONGO_URL")
    # Deliberately optional for normal application startup. Integration tests
    # must request it through get_mongo_test_url(), which fails closed.
    mongo_test_url: str | None = Field(default=None, alias="MONGO_TEST_URL")
    # 🗄️ Base de datos de desarrollo (usada por defecto al conectar)
    mongo_db_dev: str = "pdf_database_dev"
    max_file_size_mb: int = 5
    mongo_collection: str = "processed_pdfs"
    # ⏱️ Propiedad temporal de cargas PDF (independiente del worker).
    upload_lease_seconds: int = Field(
        default=120,
        gt=0,
        alias="UPLOAD_LEASE_SECONDS",
    )
    upload_lease_renew_interval_seconds: int = Field(
        default=30,
        gt=0,
        alias="UPLOAD_LEASE_RENEW_INTERVAL_SECONDS",
    )
    # 📡 Servicio de extracción (MS2)
    extract_service_url: str = Field(default="http://localhost:8002", alias="EXTRACT_SERVICE_URL")
    extract_service_timeout: int = Field(default=30, alias="EXTRACT_SERVICE_TIMEOUT")
    # 📋 Nivel de logging (12-Factor: se configura por entorno)
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Normaliza el nivel de logging.

        Aquí solo se lee y normaliza el valor: la interpretación del nivel y
        el fallback ante un valor inválido son responsabilidad de
        src.core.logging_config, de modo que un LOG_LEVEL incorrecto nunca
        impida el arranque del servicio.
        """
        return value.strip().upper()

    @model_validator(mode="after")
    def _validate_upload_lease_settings(self) -> "Settings":
        """Exige renovar el lease antes de que venza."""
        if (
            self.upload_lease_renew_interval_seconds
            >= self.upload_lease_seconds
        ):
            raise ValueError(
                "UPLOAD_LEASE_RENEW_INTERVAL_SECONDS debe ser menor que "
                "UPLOAD_LEASE_SECONDS."
            )
        return self

    def get_mongo_test_url(self) -> str:
        """Return a validated testing URI without ever falling back to MONGO_URL."""
        uri = self.mongo_test_url
        if uri is None or not uri.strip():
            raise ValueError("MONGO_TEST_URL es obligatoria para las pruebas de integración.")

        uri = uri.strip()
        try:
            scheme = uri.partition("://")[0].lower()
            if scheme != "mongodb":
                raise ValueError
            test_uri = parse_uri(uri)
            if test_uri["database"] != MONGO_TEST_DATABASE:
                raise ValueError
            hosts = test_uri["nodelist"]
            fqdn = test_uri.get("fqdn")
            if not hosts and not fqdn:
                raise ValueError
            if any(
                not _is_valid_mongo_host(host) or port < 1 or port > 65535
                for host, port in hosts
            ):
                raise ValueError
            if fqdn and not _is_valid_mongo_host(fqdn):
                raise ValueError

            def identity(parsed_uri, parsed_scheme):
                hosts = tuple(
                    sorted(
                        (host.lower(), port)
                        for host, port in parsed_uri["nodelist"]
                    )
                )
                fqdn = parsed_uri.get("fqdn")
                options = tuple(
                    sorted(
                        (key.lower(), repr(value))
                        for key, value in parsed_uri["options"].items()
                    )
                )
                return (
                    parsed_scheme,
                    hosts,
                    fqdn.lower() if fqdn else None,
                    parsed_uri.get("username"),
                    parsed_uri.get("password"),
                    parsed_uri.get("database"),
                    options,
                )

            if uri == self.mongo_url.strip():
                raise ValueError
            try:
                production_scheme = self.mongo_url.partition("://")[0].lower()
                if production_scheme != "mongodb":
                    raise ValueError
                production_uri = parse_uri(self.mongo_url.strip())
            except Exception:
                raise ValueError
            if identity(test_uri, scheme) == identity(
                production_uri, production_scheme
            ):
                raise ValueError
        except Exception:
            raise ValueError(
                "MONGO_TEST_URL debe ser una URI MongoDB válida con base "
                f"{MONGO_TEST_DATABASE}, distinta de MONGO_URL."
            ) from None

        return uri

    model_config = SettingsConfigDict(
        extra='ignore' 
    )
settings = Settings()
