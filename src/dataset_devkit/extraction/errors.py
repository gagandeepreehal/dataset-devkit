"""Classified recording-pipeline failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_devkit.sanity import SanityObservation


class ExtractionError(ValueError):
    """Base for expected, classified recording failures."""


class StructuralExtractionError(ExtractionError):
    """Raised when an MCAP cannot satisfy the version-one extraction contract."""


class NonstructuralSanityError(ExtractionError):
    """Raised after deterministic nonstructural checks configured as errors."""

    def __init__(
        self,
        observations: tuple[SanityObservation, ...],
        *,
        error_observations: tuple[SanityObservation, ...] | None = None,
    ) -> None:
        self.observations = observations
        self.error_observations = error_observations or observations
        codes = ", ".join(item.code for item in self.error_observations)
        super().__init__(f"nonstructural sanity policy failed: {codes}")
