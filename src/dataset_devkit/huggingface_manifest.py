"""Strict streaming parser for repository MCAP manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_mcap_path

_MAX_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024
_MAX_ROWS = 250_000


class ManifestError(ValueError):
    """Raised when a Hugging Face repository manifest is invalid."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    repo_path: str
    size: int
    sha256: str


def parse_manifest(path: Path) -> tuple[ManifestEntry, ...]:
    try:
        if path.stat().st_size > _MAX_BYTES:
            raise ManifestError("manifest exceeds the maximum size")
    except OSError as error:
        raise ManifestError("manifest cannot be read") from error
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    try:
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if line_number > _MAX_ROWS:
                    raise ManifestError("manifest contains too many rows")
                if len(raw.rstrip(b"\r\n")) > _MAX_LINE_BYTES:
                    raise ManifestError(f"manifest line {line_number} is too large")
                if not raw.strip() or raw.lstrip().startswith(b"#"):
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ManifestError(f"manifest row {line_number} must be an object")
                repo_path = value.get("repo_path")
                size = value.get("source_size")
                digest = value.get("sha256")
                if not isinstance(repo_path, str):
                    raise ManifestError(f"manifest row {line_number} has an invalid repo_path")
                try:
                    validate_repo_mcap_path(repo_path, line_number=line_number)
                except RepositoryPathError as error:
                    raise ManifestError(str(error)) from error
                if repo_path in seen:
                    raise ManifestError(f"duplicate repository path at line {line_number}")
                if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                    raise ManifestError(f"manifest row {line_number} has an invalid source_size")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ManifestError(f"manifest row {line_number} has an invalid sha256")
                seen.add(repo_path)
                entries.append(ManifestEntry(repo_path, size, digest))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError("manifest is not valid UTF-8 JSONL") from error
    if not entries:
        raise ManifestError("manifest contains no MCAP recordings")
    return tuple(entries)
