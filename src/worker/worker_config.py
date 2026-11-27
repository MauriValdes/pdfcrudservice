"""Environment-backed settings specific to the PDF worker process."""

from typing import Annotated

from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.config import settings
from src.worker.ms2_client import validate_ms2_base_url


PositiveInt = Annotated[int, Field(gt=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class WorkerSettings(BaseSettings):
    """Validated worker configuration; shared application settings stay shared."""

    enabled: bool = False
    max_concurrency: PositiveInt = 2
    poll_initial_seconds: PositiveFiniteFloat = 1.0
    poll_max_seconds: PositiveFiniteFloat = 5.0
    ms2_timeout_seconds: PositiveFiniteFloat = 40.0
    shutdown_grace_seconds: PositiveFiniteFloat = 60.0

    _validated_ms2_base_url: str = PrivateAttr()

    model_config = SettingsConfigDict(
        env_prefix="PDF_WORKER_",
        extra="ignore",
    )

    @field_validator(
        "max_concurrency",
        "poll_initial_seconds",
        "poll_max_seconds",
        "ms2_timeout_seconds",
        "shutdown_grace_seconds",
        mode="before",
    )
    @classmethod
    def _reject_boolean_numbers(cls, value):
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid numeric settings")
        return value

    @model_validator(mode="after")
    def _validate_worker_relationships_and_url(self):
        if self.poll_max_seconds < self.poll_initial_seconds:
            raise ValueError(
                "poll_max_seconds must be greater than or equal to "
                "poll_initial_seconds"
            )

        try:
            # Read the shared setting at instance validation time, not import time.
            self._validated_ms2_base_url = validate_ms2_base_url(
                settings.extract_service_url
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

        return self

    @property
    def ms2_base_url(self) -> str:
        """Return the validated, normalized URL shared with the application."""
        return self._validated_ms2_base_url
