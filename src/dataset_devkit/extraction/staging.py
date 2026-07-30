"""Narrow POSIX-safe atomic JPEG staging."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import StagedImage

_SAFE_RECORDING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_Identity = tuple[int, int]


@dataclass
class StagingInvocation:
    staging_root: Path
    directory_name: str
    path: Path
    directory_identity: _Identity
    owned_files: dict[str, _Identity] = field(default_factory=dict)


def _recording_slug(recording_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", recording_id).strip("._-")
    return slug or "recording"


def _camera_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "camera"


def _identity(file_stat: os.stat_result) -> _Identity:
    return file_stat.st_dev, file_stat.st_ino


def _open_directory_chain(path: Path, *, create: bool) -> tuple[int, tuple[_Identity, ...]]:
    if not path.is_absolute() or ".." in path.parts:
        raise StructuralExtractionError("staging directory must be an absolute trusted path")
    current_fd = os.open("/", _DIRECTORY_FLAGS)
    identities = [_identity(os.fstat(current_fd))]
    try:
        for component in path.parts[1:]:
            if not component or component == ".":
                continue
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except OSError as error:
                raise StructuralExtractionError(
                    "unsafe staging directory ancestor or symlink"
                ) from error
            child_stat = os.fstat(child_fd)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child_fd)
                raise StructuralExtractionError("unsafe staging directory ancestor")
            identities.append(_identity(child_stat))
            os.close(current_fd)
            current_fd = child_fd
        return current_fd, tuple(identities)
    except Exception:
        os.close(current_fd)
        raise


def _assert_directory_chain_unchanged(
    path: Path, expected_identities: tuple[_Identity, ...]
) -> None:
    try:
        check_fd, actual_identities = _open_directory_chain(path, create=False)
    except StructuralExtractionError as error:
        raise StructuralExtractionError("staging directory ancestor changed") from error
    try:
        if actual_identities != expected_identities:
            raise StructuralExtractionError("staging directory ancestor identity changed")
    finally:
        os.close(check_fd)


def _write_all(file_fd: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(file_fd, content[offset:])
        if written <= 0:
            raise OSError("short write while staging JPEG")
        offset += written


def _read_all(file_fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _unlink_relative(directory_fd: int, filename: str) -> None:
    with suppress(FileNotFoundError):
        os.unlink(filename, dir_fd=directory_fd)
        os.fsync(directory_fd)


def _unlink_if_identity(
    directory_fd: int, filename: str, expected_identity: _Identity | None
) -> None:
    if expected_identity is None:
        return
    try:
        current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISREG(current.st_mode) and _identity(current) == expected_identity:
        _unlink_relative(directory_fd, filename)


def create_staging_invocation(staging_root: Path, recording_id: str) -> StagingInvocation:
    """Exclusively create one collision-isolated staging directory."""
    prefix = _recording_slug(recording_id)
    root_fd, _ = _open_directory_chain(staging_root, create=True)
    try:
        while True:
            directory_name = f"{prefix}-{uuid.uuid4().hex}"
            try:
                os.mkdir(directory_name, mode=0o700, dir_fd=root_fd)
                break
            except FileExistsError:
                continue
        directory_stat = os.stat(directory_name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise StructuralExtractionError("new staging invocation is not a directory")
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    path = staging_root / directory_name
    return StagingInvocation(staging_root, directory_name, path, _identity(directory_stat))


def rollback_staging_invocation(invocation: StagingInvocation) -> None:
    """Remove only invocation-owned inodes, then its still-identical empty directory."""
    root_fd, _ = _open_directory_chain(invocation.staging_root, create=False)
    try:
        try:
            directory_stat = os.stat(
                invocation.directory_name, dir_fd=root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or _identity(directory_stat) != invocation.directory_identity
        ):
            raise StructuralExtractionError("staging invocation directory identity changed")
        directory_fd = os.open(invocation.directory_name, _DIRECTORY_FLAGS, dir_fd=root_fd)
        try:
            for filename, expected_identity in tuple(invocation.owned_files.items()):
                try:
                    current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if (
                    stat.S_ISREG(current.st_mode)
                    and current.st_nlink == 1
                    and _identity(current) == expected_identity
                ):
                    _unlink_relative(directory_fd, filename)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        current_dir = os.stat(
            invocation.directory_name, dir_fd=root_fd, follow_symlinks=False
        )
        if _identity(current_dir) != invocation.directory_identity:
            raise StructuralExtractionError("staging invocation directory identity changed")
        try:
            os.rmdir(invocation.directory_name, dir_fd=root_fd)
        except OSError as error:
            raise StructuralExtractionError(
                "staging invocation rollback left unowned or unsafe entries"
            ) from error
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def stage_jpeg(
    staging_root: Path,
    recording_id: str,
    camera_index: int,
    camera_name: str,
    timestamp_ns: int,
    image: Image.Image,
    expected_dimensions: tuple[int, int],
    *,
    batch_ordinal: int | None = None,
    invocation: StagingInvocation | None = None,
) -> StagedImage:
    """Atomically persist and bind verification to one quality-95 JPEG byte sequence."""
    if _SAFE_RECORDING.fullmatch(recording_id) is None:
        raise StructuralExtractionError("unsafe recording identifier for staging")
    if invocation is not None and (
        invocation.staging_root != staging_root
        or invocation.directory_name != recording_id
    ):
        raise StructuralExtractionError("staging invocation does not match target directory")
    recording_dir = staging_root / recording_id
    ordinal_prefix = "" if batch_ordinal is None else f"{batch_ordinal:09d}-"
    filename = (
        f"{ordinal_prefix}{camera_index:03d}-{_camera_slug(camera_name)}-{timestamp_ns}.jpg"
    )
    temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
    encoded_stream = BytesIO()
    try:
        image.convert("RGB").save(encoded_stream, format="JPEG", quality=95)
    except Exception as error:
        raise StructuralExtractionError("failed to encode staged JPEG") from error
    encoded = encoded_stream.getvalue()
    expected_digest = hashlib.sha256(encoded).digest()
    directory_fd, directory_identities = _open_directory_chain(recording_dir, create=True)
    published = False
    temporary_identity: _Identity | None = None
    try:
        try:
            existing = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
        ):
            raise StructuralExtractionError("unsafe existing staging target")
        if existing is not None:
            raise StructuralExtractionError("staging target already exists; refusing to clobber")

        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            temporary_stat = os.fstat(temporary_fd)
            if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_nlink != 1:
                raise StructuralExtractionError("unsafe temporary staging file")
            temporary_identity = _identity(temporary_stat)
            _write_all(temporary_fd, encoded)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)

        _assert_directory_chain_unchanged(recording_dir, directory_identities)
        os.link(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        _assert_directory_chain_unchanged(recording_dir, directory_identities)

        target_fd = os.open(filename, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=directory_fd)
        try:
            opened_stat = os.fstat(target_fd)
            if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                raise StructuralExtractionError("unsafe staged JPEG identity")
            actual = _read_all(target_fd)
        finally:
            os.close(target_fd)
        current_stat = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if _identity(current_stat) != _identity(opened_stat) or current_stat.st_nlink != 1:
            raise StructuralExtractionError("staged JPEG identity changed during verification")
        actual_digest = hashlib.sha256(actual).digest()
        if not hmac.compare_digest(actual_digest, expected_digest) or actual != encoded:
            raise StructuralExtractionError("staged JPEG content changed during verification")
        _assert_directory_chain_unchanged(recording_dir, directory_identities)

        if invocation is not None:
            if _identity(os.fstat(directory_fd)) != invocation.directory_identity:
                raise StructuralExtractionError("staging invocation directory identity changed")
            invocation.owned_files[filename] = _identity(current_stat)

        with Image.open(BytesIO(actual)) as reopened:
            reopened.load()
            if reopened.format != "JPEG" or reopened.mode != "RGB":
                raise StructuralExtractionError("staged image did not reopen as RGB JPEG")
            if reopened.size != expected_dimensions:
                raise StructuralExtractionError(
                    "staged JPEG dimensions differ from camera dimensions"
                )
        _assert_directory_chain_unchanged(recording_dir, directory_identities)
    except StructuralExtractionError:
        if published:
            _unlink_if_identity(directory_fd, filename, temporary_identity)
        raise
    except Exception as error:
        if published:
            _unlink_if_identity(directory_fd, filename, temporary_identity)
        raise StructuralExtractionError("staged JPEG verification failed") from error
    finally:
        _unlink_if_identity(directory_fd, temporary_name, temporary_identity)
        os.close(directory_fd)

    path = recording_dir / filename
    return StagedImage(
        camera_index,
        camera_name,
        timestamp_ns,
        path,
        expected_dimensions[0],
        expected_dimensions[1],
    )
