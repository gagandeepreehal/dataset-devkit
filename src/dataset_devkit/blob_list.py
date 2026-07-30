"""Exact, container-relative Azure MCAP blob-list parsing."""

from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath

_REQUIRED_PREFIX = "mcap-h265/"


class BlobListError(ValueError):
    """Raised when a blob-list entry is unsafe or outside the supported source prefix."""


def validate_blob_path(value: str, *, line_number: int | None = None) -> str:
    """Validate and return one exact container-relative MCAP blob name."""
    path = PurePosixPath(value)
    invalid = (
        not value.startswith(_REQUIRED_PREFIX)
        or not value.endswith(".mcap")
        or path.is_absolute()
        or "\\" in value
        or "%" in value
        or "?" in value
        or "#" in value
        or value != posixpath.normpath(value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    )
    if invalid:
        location = f" at line {line_number}" if line_number is not None else ""
        raise BlobListError(f"invalid MCAP blob path{location}: {value!r}")
    return value


def parse_blob_list(path: Path) -> tuple[str, ...]:
    """Parse exact blob names, ignoring blank and comment-only lines."""
    values: list[str] = []
    first_lines: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline=None) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            value = raw_line.rstrip("\n")
            if not value.strip() or value.lstrip().startswith("#"):
                continue
            validate_blob_path(value, line_number=line_number)
            if value in first_lines:
                raise BlobListError(
                    f"duplicate blob path at line {line_number} "
                    f"(first seen at line {first_lines[value]}): {value!r}"
                )
            first_lines[value] = line_number
            values.append(value)
    return tuple(values)
