"""Safe, canonical quarantine reporting without source-cache mutation."""

from __future__ import annotations

import fcntl
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
from dataset_devkit.provenance import canonical_json

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
            try:
                child_stat = os.fstat(child)
            except Exception:
                os.close(child)
                raise
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise StructuralExtractionError("unsafe quarantine directory ancestor")
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def _directory_path_identities(path: Path) -> tuple[tuple[int, int], ...]:
    if not path.is_absolute() or ".." in path.parts:
        raise StructuralExtractionError("quarantine directory must be an absolute trusted path")
    current = os.open("/", _DIRECTORY_FLAGS)
    root_stat = os.fstat(current)
    identities = [(root_stat.st_dev, root_stat.st_ino)]
    try:
        for component in path.parts[1:]:
            if not component or component == ".":
                continue
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            except OSError as error:
                raise StructuralExtractionError(
                    "quarantine directory path identity changed"
                ) from error
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise StructuralExtractionError(
                    "quarantine directory path identity changed"
                )
            identities.append((child_stat.st_dev, child_stat.st_ino))
            os.close(current)
            current = child
        return tuple(identities)
    finally:
        os.close(current)


def _revalidate_directory_identity(
    path: Path,
    directory_fd: int,
    expected_path: tuple[tuple[int, int], ...],
) -> None:
    opened = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not expected_path
        or (opened.st_dev, opened.st_ino) != expected_path[-1]
    ):
        raise StructuralExtractionError("quarantine directory path identity changed")
    if _directory_path_identities(path) != expected_path:
        raise StructuralExtractionError("quarantine directory path identity changed")


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


def _owned_file_identity(
    directory_fd: int,
    filename: str,
    file_descriptor: int,
    *,
    expected: tuple[int, int] | None = None,
) -> tuple[int, int]:
    opened = os.fstat(file_descriptor)
    listed = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    identity = opened.st_dev, opened.st_ino
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not stat.S_ISREG(listed.st_mode)
        or listed.st_nlink != 1
        or identity != (listed.st_dev, listed.st_ino)
        or (expected is not None and identity != expected)
    ):
        raise StructuralExtractionError("quarantine manifest file identity changed")
    return identity


def _read_owned_manifest(
    directory_fd: int,
    manifest_name: str,
    expected: tuple[int, int],
) -> bytes:
    read_fd = os.open(manifest_name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=directory_fd)
    try:
        _owned_file_identity(
            directory_fd,
            manifest_name,
            read_fd,
            expected=expected,
        )
        content = _read_all(read_fd)
        _owned_file_identity(
            directory_fd,
            manifest_name,
            read_fd,
            expected=expected,
        )
        return content
    finally:
        os.close(read_fd)


def _revalidate_manifest_identity(
    directory_fd: int,
    manifest_name: str,
    expected: tuple[int, int],
) -> None:
    read_fd = os.open(manifest_name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=directory_fd)
    try:
        _owned_file_identity(
            directory_fd,
            manifest_name,
            read_fd,
            expected=expected,
        )
    finally:
        os.close(read_fd)


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


def write_rejection_manifest(
    directory: Path,
    manifest_name: str,
    reports: tuple[QuarantineReport, ...],
) -> Path:
    """Merge durable reports into one locked, canonical JSONL rejection inventory."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", manifest_name) is None:
        raise StructuralExtractionError("unsafe quarantine manifest name")
    directory_fd = _open_directory(directory)
    locked = False
    temporary: str | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        directory_stat = os.fstat(directory_fd)
        directory_identity = directory_stat.st_dev, directory_stat.st_ino
        directory_path_identities = _directory_path_identities(directory)
        if (
            not directory_path_identities
            or directory_path_identities[-1] != directory_identity
        ):
            raise StructuralExtractionError("quarantine directory path identity changed")
        _revalidate_directory_identity(directory, directory_fd, directory_path_identities)
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        locked = True
        _revalidate_directory_identity(directory, directory_fd, directory_path_identities)
        existing: list[dict[str, Any]] = []
        manifest_identity: tuple[int, int] | None = None
        try:
            manifest_stat = os.stat(
                manifest_name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_nlink != 1:
                raise StructuralExtractionError("unsafe existing quarantine manifest")
            manifest_identity = manifest_stat.st_dev, manifest_stat.st_ino
            for line in _read_owned_manifest(
                directory_fd,
                manifest_name,
                manifest_identity,
            ).decode("utf-8").splitlines():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("rejection manifest row must be an object")
                existing.append(value)
        rows = existing + [report.as_dict() for report in reports]
        unique = {canonical_json(row): row for row in rows}
        content = "".join(f"{key}\n" for key in sorted(unique)).encode("utf-8")
        temporary = f".{manifest_name}.{uuid.uuid4().hex}.tmp"
        write_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            temporary_identity = _owned_file_identity(
                directory_fd,
                temporary,
                write_fd,
            )
            _write_all(write_fd, content)
            os.fsync(write_fd)
            _owned_file_identity(
                directory_fd,
                temporary,
                write_fd,
                expected=temporary_identity,
            )
        finally:
            os.close(write_fd)
        if manifest_identity is None:
            try:
                os.stat(manifest_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise StructuralExtractionError(
                    "quarantine manifest destination identity changed"
                )
        else:
            _revalidate_manifest_identity(
                directory_fd,
                manifest_name,
                manifest_identity,
            )
        _revalidate_directory_identity(directory, directory_fd, directory_path_identities)
        os.replace(temporary, manifest_name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        final = os.stat(manifest_name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != temporary_identity
        ):
            raise StructuralExtractionError("quarantine manifest publication identity changed")
        temporary = None
        os.fsync(directory_fd)
        _revalidate_directory_identity(directory, directory_fd, directory_path_identities)
        return directory / manifest_name
    except StructuralExtractionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise StructuralExtractionError("failed to update quarantine rejection manifest") from error
    finally:
        if temporary is not None:
            _unlink_if_identity(directory_fd, temporary, temporary_identity)
        if locked:
            with suppress(OSError):
                fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)
