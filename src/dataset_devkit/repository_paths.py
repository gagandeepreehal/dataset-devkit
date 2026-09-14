"""Safe Hugging Face repository paths for MCAP recordings."""

from __future__ import annotations

import posixpath
from pathlib import PurePosixPath

from dataset_devkit.identifiers import validate_safe_segment


class RepositoryPathError(ValueError):
    """Raised when a repository recording path is unsafe or unsupported."""


_SUPPORTED_FILE_SUFFIXES = {".mp4", ".parquet", ".json", ".jsonl", ".tar"}


def validate_repo_file_path(value: str, *, line_number: int | None = None) -> str:
    """Validate one normalized, supported file path under the public data prefix."""
    path = PurePosixPath(value)
    invalid = (
        not value.startswith("data/")
        or path.suffix not in _SUPPORTED_FILE_SUFFIXES
        or path.is_absolute()
        or "\\" in value
        or any(marker in value for marker in ("%", "?", "#"))
        or value != posixpath.normpath(value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    )
    if not invalid:
        try:
            for part in value.split("/"):
                validate_safe_segment(part)
        except ValueError:
            invalid = True
    if invalid:
        location = f" at line {line_number}" if line_number is not None else ""
        raise RepositoryPathError(f"invalid repository file path{location}: {value!r}")
    return value


def validate_repo_mcap_path(value: str, *, line_number: int | None = None) -> str:
    path = PurePosixPath(value)
    invalid = (
        not value.startswith("data/")
        or not value.endswith(".mcap")
        or path.is_absolute()
        or "\\" in value
        or any(marker in value for marker in ("%", "?", "#"))
        or value != posixpath.normpath(value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    )
    if invalid:
        location = f" at line {line_number}" if line_number is not None else ""
        raise RepositoryPathError(f"invalid MCAP repository path{location}: {value!r}")
    return value
