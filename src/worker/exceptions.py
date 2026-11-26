"""Excepciones propias del worker de procesamiento."""


class ProcessingLeaseLostError(RuntimeError):
    """El worker dejó de ser propietario vigente del documento."""
