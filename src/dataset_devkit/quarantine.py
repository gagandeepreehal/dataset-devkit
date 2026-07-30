"""Safe, canonical quarantine reporting without source-cache mutation."""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.uncertainty import bounded_freeze

FailureCategory = Literal["structural", "sanity", "unexpected"]
ArtifactHandling = Literal["no_owned_artifacts", "preserved_in_place"]
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class QuarantineReport:
    recording_id: str
    source_path: str
    status: Literal["quarantined"]
    category: FailureCategory
    exception_type: str
    exception_message: str
    stage: str
    deterministic_details: dict[str, Any] = field(default_factory=dict)
    observed_context: tuple[dict[str, Any], ...] = ()
    source_config_hash: str | None = None
    extraction_config_hash: str | None = None
    artifact_handling: ArtifactHandling = "no_owned_artifacts"
    schema_version: Literal["1.0"] = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "deterministic_details",
            bounded_freeze(
                self.deterministic_details, root_path="deterministic_details"
            ),
        )
        object.__setattr__(
            self,
            "observed_context",
            bounded_freeze(self.observed_context, root_path="observed_context"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recording_id": self.recording_id,
            "source_path": self.source_path,
            "status": self.status,
            "category": self.category,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "stage": self.stage,
            "deterministic_details": _json_value(self.deterministic_details),
            "observed_context": _json_value(self.observed_context),
            "source_config_hash": self.source_config_hash,
            "extraction_config_hash": self.extraction_config_hash,
            "artifact_handling": self.artifact_handling,
        }


@dataclass(frozen=True)
class QuarantineArtifact:
    report: QuarantineReport
    path: Path


def _open_directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise StructuralExtractionError("quarantine directory must be an absolute trusted path")
    current = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            if not component or component == ".":
                continue
            created = False
            try:
                os.mkdir(component, 0o700, dir_fd=current)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(current)
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            except OSError as error:
                raise StructuralExtractionError(
                    "unsafe quarantine directory ancestor or symlink"
                ) from error
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise StructuralExtractionError("unsafe quarantine directory ancestor")
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def _write_all(file_descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(file_descriptor, content[offset:])
        if written <= 0:
            raise OSError("short quarantine report write")
        offset += written


def _read_all(file_descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _unlink_if_identity(
    directory_fd: int,
    filename: str,
    expected: tuple[int, int] | None,
) -> None:
    if expected is None:
        return
    try:
        current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) == expected:
        with suppress(OSError):
            os.unlink(filename, dir_fd=directory_fd)


def write_quarantine_report(directory: Path, report: QuarantineReport) -> QuarantineArtifact:
    """Durably publish complete canonical bytes below a no-follow root."""
    content = (
        json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    directory_fd = _open_directory(directory)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", report.recording_id).strip("._-") or "recording"
    try:
        while True:
            token = uuid.uuid4().hex
            filename = f"{slug}-{token}.quarantine.json"
            temporary_name = f".{slug}-{token}.quarantine.tmp"
            try:
                file_descriptor = os.open(
                    temporary_name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            identity: tuple[int, int] | None = None
            published = False
            collision = False
            try:
                opened = os.fstat(file_descriptor)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise StructuralExtractionError("unsafe quarantine report inode")
                identity = opened.st_dev, opened.st_ino
                _write_all(file_descriptor, content)
                os.fsync(file_descriptor)
                os.lseek(file_descriptor, 0, os.SEEK_SET)
                if _read_all(file_descriptor) != content:
                    raise StructuralExtractionError(
                        "quarantine report verification failed"
                    )
                current = os.stat(
                    temporary_name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or (current.st_dev, current.st_ino) != identity
                ):
                    raise StructuralExtractionError(
                        "quarantine temporary report identity changed"
                    )
                try:
                    os.link(
                        temporary_name,
                        filename,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    collision = True
                else:
                    published = True
                    os.unlink(temporary_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                    final = os.stat(
                        filename, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if (
                        not stat.S_ISREG(final.st_mode)
                        or final.st_nlink != 1
                        or (final.st_dev, final.st_ino) != identity
                    ):
                        raise StructuralExtractionError(
                            "quarantine report identity changed"
                        )
            except Exception:
                if published:
                    _unlink_if_identity(directory_fd, filename, identity)
                _unlink_if_identity(directory_fd, temporary_name, identity)
                raise
            finally:
                os.close(file_descriptor)
            if collision:
                temporary = os.stat(
                    temporary_name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(temporary.st_mode)
                    or temporary.st_nlink != 1
                    or (temporary.st_dev, temporary.st_ino) != identity
                ):
                    raise StructuralExtractionError(
                        "quarantine collision cleanup identity changed"
                    )
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
                continue
            return QuarantineArtifact(report, directory / filename)
    finally:
        os.close(directory_fd)
