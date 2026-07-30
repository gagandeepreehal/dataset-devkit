"""Reusable validation for filesystem-safe public identifiers."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field

_WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def validate_safe_segment(value: str) -> str:
    """Require one nonblank, traversal-safe, cross-platform path segment."""
    unsafe_characters = set('/\\\0<>:"|?*')
    stem = value.split(".", 1)[0].upper()
    if (
        not value
        or value != value.strip()
        or value.endswith(".")
        or value in {".", ".."}
        or stem in _WINDOWS_RESERVED_STEMS
        or any(character in unsafe_characters or ord(character) < 32 for character in value)
    ):
        raise ValueError("value must be a nonempty safe path segment")
    return value


SafeSegment = Annotated[
    str,
    Field(
        min_length=1,
        json_schema_extra={
            "pattern": (
                r"^(?!\.{1,2}$)(?!\s)(?!.*(?:\.|\s)$)"
                r"(?!(?:[Cc][Oo][Nn]|[Pp][Rr][Nn]|[Aa][Uu][Xx]|[Nn][Uu][Ll]|"
                r"[Cc][Oo][Mm][1-9]|[Ll][Pp][Tt][1-9])(?:\.|$))"
                r"[^\u0000-\u001f/\\<>:\"|?*]+$"
            )
        },
    ),
    AfterValidator(validate_safe_segment),
]
