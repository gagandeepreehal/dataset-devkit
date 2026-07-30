"""Reusable validation for filesystem-safe public identifiers."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field


def validate_safe_segment(value: str) -> str:
    """Require one nonblank, traversal-safe, cross-platform path segment."""
    unsafe_characters = set('/\\\0<>:"|?*')
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or any(character in unsafe_characters or ord(character) < 32 for character in value)
    ):
        raise ValueError("value must be a nonempty safe path segment")
    return value


SafeSegment = Annotated[str, Field(min_length=1), AfterValidator(validate_safe_segment)]
