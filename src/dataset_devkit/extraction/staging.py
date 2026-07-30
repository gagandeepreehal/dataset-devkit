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


@dataclass(frozen=True)
class TombstoneRecord:
    """Immutable inode-bound state for safe committed-drop cleanup retries."""

    invocation_root: Path
    directory_device: int
    directory_inode: int
    directory_chain_identities: tuple[_Identity, ...]
    tombstone_name: str
    original_name: str
    device: int
    inode: int
    expected_regular: bool = True
    expected_single_link: bool = True


@dataclass(frozen=True)
class TombstoneCleanupResult:
    cleaned: tuple[TombstoneRecord, ...]
    remaining: tuple[TombstoneRecord, ...]
    mismatched: tuple[TombstoneRecord, ...]
    directory_fsynced: bool


class StagedImageCleanupError(StructuralExtractionError):
    """A committed drop has owned tombstones that require cleanup/retry."""

    def __init__(self, tombstones: tuple[TombstoneRecord, ...]) -> None:
        self.tombstones = tombstones
        self.owned_tombstones = tombstones
        super().__init__(
            "committed staged-image drop requires tombstone cleanup; "
            f"{len(tombstones)} owned tombstone(s) remain"
        )


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


def _open_verified_owned_images(
    staging_root: Path, images: tuple[StagedImage, ...]
) -> tuple[int, tuple[tuple[str, _Identity], ...], tuple[_Identity, ...]]:
    directory_fd, directory_identities = _open_directory_chain(
        staging_root, create=False
    )
    verified: list[tuple[str, _Identity]] = []
    try:
        for image in images:
            if (
                image.path.parent != staging_root
                or image.device is None
                or image.inode is None
            ):
                raise StructuralExtractionError(
                    "staged image is not owned by this extraction invocation"
                )
            try:
                current = os.stat(
                    image.path.name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as error:
                raise StructuralExtractionError("staged image is unavailable") from error
            expected = (image.device, image.inode)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or _identity(current) != expected
            ):
                raise StructuralExtractionError(
                    "staged image ownership or identity changed"
                )
            verified.append((image.path.name, expected))
        return directory_fd, tuple(verified), directory_identities
    except Exception:
        os.close(directory_fd)
        raise


def verify_owned_staged_images(
    staging_root: Path, images: tuple[StagedImage, ...]
) -> None:
    """Verify every image through a component-wise no-follow invocation directory."""
    directory_fd, _, _ = _open_verified_owned_images(staging_root, images)
    os.close(directory_fd)


def remove_owned_staged_images(
    staging_root: Path, images: tuple[StagedImage, ...]
) -> None:
    """Transactionally hide verified images, then clean committed tombstones."""
    directory_fd, verified, directory_identities = _open_verified_owned_images(
        staging_root, images
    )
    directory_identity = _identity(os.fstat(directory_fd))
    moved: list[tuple[TombstoneRecord, bool]] = []
    try:
        try:
            for filename, expected in verified:
                current = os.stat(
                    filename, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or _identity(current) != expected
                ):
                    raise StructuralExtractionError(
                        "refusing to move changed or linked staged image"
                    )
                tombstone = f".{filename}.drop-{uuid.uuid4().hex}"
                record = TombstoneRecord(
                    invocation_root=staging_root,
                    directory_device=directory_identity[0],
                    directory_inode=directory_identity[1],
                    directory_chain_identities=directory_identities,
                    tombstone_name=tombstone,
                    original_name=filename,
                    device=expected[0],
                    inode=expected[1],
                )
                os.link(
                    filename,
                    tombstone,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                moved.append((record, False))
                tombstone_stat = os.stat(
                    tombstone, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(tombstone_stat.st_mode)
                    or tombstone_stat.st_nlink != 2
                    or _identity(tombstone_stat) != expected
                ):
                    raise StructuralExtractionError(
                        "staged image tombstone identity changed"
                    )
                source_after_link = os.stat(
                    filename, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(source_after_link.st_mode)
                    or source_after_link.st_nlink != 2
                    or _identity(source_after_link) != expected
                ):
                    raise StructuralExtractionError(
                        "staged image source identity changed after tombstone link"
                    )
                os.unlink(filename, dir_fd=directory_fd)
                moved[-1] = (record, True)
                tombstone_stat = os.stat(
                    tombstone, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(tombstone_stat.st_mode)
                    or tombstone_stat.st_nlink != 1
                    or _identity(tombstone_stat) != expected
                ):
                    raise StructuralExtractionError(
                        "staged image tombstone did not become exclusively owned"
                    )
            os.fsync(directory_fd)
        except Exception as transaction_error:
            rollback_error: Exception | None = None
            for record, source_unlinked in reversed(moved):
                try:
                    expected = record.device, record.inode
                    tombstone_stat = os.stat(
                        record.tombstone_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    expected_links = 1 if source_unlinked else 2
                    if not _stat_matches(tombstone_stat, expected, expected_links):
                        raise StructuralExtractionError(
                            "cannot roll back changed staged image tombstone"
                        )
                    if source_unlinked:
                        try:
                            os.stat(
                                record.original_name,
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            pass
                        else:
                            raise StructuralExtractionError(
                                "cannot roll back over a replaced staged image name"
                            )
                        os.link(
                            record.tombstone_name,
                            record.original_name,
                            src_dir_fd=directory_fd,
                            dst_dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                    else:
                        source = os.stat(
                            record.original_name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        if not _stat_matches(source, expected, 2):
                            raise StructuralExtractionError(
                                "cannot roll back changed staged image source"
                            )
                    os.unlink(record.tombstone_name, dir_fd=directory_fd)
                    restored = os.stat(
                        record.original_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if not _stat_matches(restored, expected, 1):
                        raise StructuralExtractionError(
                            "rolled-back staged image identity changed"
                        )
                except Exception as error:
                    rollback_error = rollback_error or error
            try:
                os.fsync(directory_fd)
            except Exception as error:
                rollback_error = rollback_error or error
            if rollback_error is not None:
                raise StructuralExtractionError(
                    "staged image drop transaction rollback failed"
                ) from rollback_error
            raise StructuralExtractionError(
                "staged image drop transaction rolled back before commit"
            ) from transaction_error

        records = tuple(record for record, _ in moved)
        remaining = list(records)
        try:
            for record in records:
                current = os.stat(
                    record.tombstone_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if not _stat_matches(current, (record.device, record.inode), 1):
                    raise StructuralExtractionError(
                        "committed staged image tombstone identity changed"
                    )
                os.unlink(record.tombstone_name, dir_fd=directory_fd)
                remaining.remove(record)
            os.fsync(directory_fd)
        except Exception as cleanup_error:
            raise StagedImageCleanupError(tuple(remaining)) from cleanup_error
    finally:
        os.close(directory_fd)


def _stat_matches(current: os.stat_result, expected: _Identity, links: int) -> bool:
    return (
        stat.S_ISREG(current.st_mode)
        and current.st_nlink == links
        and _identity(current) == expected
    )


def retry_owned_tombstone_cleanup(
    records: tuple[TombstoneRecord, ...],
) -> TombstoneCleanupResult:
    """Safely retry inode-bound tombstone cleanup without following replacements."""
    cleaned: list[TombstoneRecord] = []
    mismatched: list[TombstoneRecord] = []
    fsynced = True
    groups: dict[
        tuple[Path, tuple[_Identity, ...]], list[TombstoneRecord]
    ] = {}
    for record in records:
        groups.setdefault(
            (
                record.invocation_root,
                record.directory_chain_identities,
            ),
            [],
        ).append(record)
    for (root, expected_identities), group in groups.items():
        try:
            directory_fd, identities = _open_directory_chain(root, create=False)
        except StructuralExtractionError:
            mismatched.extend(group)
            fsynced = False
            continue
        try:
            if identities != expected_identities:
                mismatched.extend(group)
                fsynced = False
                continue
            group_cleaned = False
            for record in group:
                if (
                    Path(record.tombstone_name).name != record.tombstone_name
                    or Path(record.original_name).name != record.original_name
                ):
                    mismatched.append(record)
                    continue
                try:
                    file_fd = os.open(
                        record.tombstone_name,
                        os.O_RDONLY | _FILE_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError:
                    mismatched.append(record)
                    continue
                try:
                    opened = os.fstat(file_fd)
                    current = os.stat(
                        record.tombstone_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    expected = record.device, record.inode
                    if (
                        not _stat_matches(opened, expected, 1)
                        or not _stat_matches(current, expected, 1)
                    ):
                        mismatched.append(record)
                        continue
                    os.unlink(record.tombstone_name, dir_fd=directory_fd)
                    cleaned.append(record)
                    group_cleaned = True
                except OSError:
                    mismatched.append(record)
                finally:
                    os.close(file_fd)
            if group_cleaned:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    fsynced = False
        finally:
            os.close(directory_fd)
    remaining = tuple(record for record in records if record not in cleaned)
    mismatch_tuple = tuple(record for record in records if record in mismatched)
    return TombstoneCleanupResult(tuple(cleaned), remaining, mismatch_tuple, fsynced)


def owned_tombstone_record_matches(record: TombstoneRecord) -> bool:
    """Check one cleanup record through its trusted no-follow directory context."""
    try:
        directory_fd, identities = _open_directory_chain(
            record.invocation_root, create=False
        )
    except StructuralExtractionError:
        return False
    try:
        if (
            not identities
            or identities != record.directory_chain_identities
            or Path(record.tombstone_name).name != record.tombstone_name
        ):
            return False
        try:
            file_fd = os.open(
                record.tombstone_name,
                os.O_RDONLY | _FILE_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except OSError:
            return False
        try:
            opened = os.fstat(file_fd)
            current = os.stat(
                record.tombstone_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            expected = record.device, record.inode
            return _stat_matches(opened, expected, 1) and _stat_matches(
                current, expected, 1
            )
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


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
        current_stat.st_dev,
        current_stat.st_ino,
    )
