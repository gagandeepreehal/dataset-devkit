"""Safe JSON extraction-result cache with verified copied JPEG materialization."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from ctypes import CDLL, c_char_p, c_int, get_errno
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from dataset_devkit.extraction.models import RecordingExtractionResult, StagedImage
from dataset_devkit.extraction.staging import (
    create_staging_invocation,
    rollback_staging_invocation,
    staged_directory_metadata,
)
from dataset_devkit.provenance import SourceFingerprint, canonical_json
from dataset_devkit.publication import cleanup_pinned_directory

_ADAPTER = TypeAdapter(RecordingExtractionResult)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW
_Identity = tuple[int, int]


def _identity(value: os.stat_result) -> _Identity:
    return value.st_dev, value.st_ino


@dataclass(frozen=True)
class CacheStoreResult:
    """Non-executable metadata describing a completed cache store."""

    path: Path
    created: bool
    refreshed: bool
    image_count: int


@dataclass(frozen=True)
class _CachedImage:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _CachedGeneration:
    result: RecordingExtractionResult
    images: tuple[_CachedImage, ...]
    generation_identity: _Identity
    images_identity: _Identity


@dataclass
class _CacheRootLease:
    """Pinned authority from the filesystem root through extraction-results."""

    path: Path
    descriptors: tuple[int, ...]
    components: tuple[str, ...]
    identities: tuple[_Identity, ...]

    @property
    def root_fd(self) -> int:
        return self.descriptors[-1]

    @classmethod
    def open(cls, cache_dir: Path, *, create: bool) -> _CacheRootLease:
        configured = cache_dir.absolute()
        if not configured.is_absolute() or ".." in configured.parts:
            raise ValueError("cache directory must be an absolute traversal-safe path")
        path = configured / "extraction-results"
        descriptors = [os.open("/", _DIRECTORY_FLAGS)]
        components: list[str] = []
        identities = [_identity(os.fstat(descriptors[0]))]
        try:
            for component in path.parts[1:]:
                parent_fd = descriptors[-1]
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(component, 0o700, dir_fd=parent_fd)
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
                listed = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                opened = os.fstat(child_fd)
                if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != _identity(opened):
                    os.close(child_fd)
                    raise ValueError("cache directory identity changed while opening")
                components.append(component)
                descriptors.append(child_fd)
                identities.append(_identity(opened))
            return cls(path, tuple(descriptors), tuple(components), tuple(identities))
        except Exception:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            raise

    def assert_bound(self) -> None:
        for descriptor, expected in zip(self.descriptors, self.identities, strict=True):
            if _identity(os.fstat(descriptor)) != expected:
                raise ValueError("pinned cache directory identity changed")
        for index, component in enumerate(self.components):
            listed = os.stat(
                component,
                dir_fd=self.descriptors[index],
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != self.identities[index + 1]:
                raise ValueError("configured cache directory binding changed")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)


def _open_child_directory(parent_fd: int, name: str, *, create: bool) -> tuple[int, _Identity]:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("cache directory component is unsafe")
    if create:
        with suppress(FileExistsError):
            os.mkdir(name, 0o700, dir_fd=parent_fd)
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != _identity(opened):
        os.close(descriptor)
        raise ValueError("cache directory component identity changed")
    return descriptor, _identity(opened)


def _directory_identity(path: Path) -> tuple[int, int]:
    current = path.lstat()
    if not stat.S_ISDIR(current.st_mode):
        raise ValueError("cache generation must be a directory")
    return current.st_dev, current.st_ino


def _exchange_directories(parent_fd: int, left: str, right: str) -> None:
    """Atomically exchange two sibling directories without a missing-generation gap."""
    library = CDLL(None, use_errno=True)
    encoded_left = os.fsencode(left)
    encoded_right = os.fsencode(right)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
        result = rename(parent_fd, encoded_left, parent_fd, encoded_right, 0x00000002)
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
        result = rename(parent_fd, encoded_left, parent_fd, encoded_right, 0x2)
    else:
        raise OSError("atomic directory exchange is unavailable on this platform")
    if result != 0:
        error_number = get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _rename_exclusive(parent_fd: int, source: str, destination: str) -> None:
    library = CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
        result = rename(parent_fd, encoded_source, parent_fd, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
        result = rename(parent_fd, encoded_source, parent_fd, encoded_destination, 0x1)
    else:
        raise OSError("exclusive cache publication is unavailable on this platform")
    if result != 0:
        error_number = get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _assert_child_bound(parent_fd: int, name: str, expected: _Identity) -> None:
    listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != expected:
        raise ValueError("cache directory entry binding changed")


def _fsync_tree_fd(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
        listed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(listed.st_mode):
            child_fd, _ = _open_child_directory(directory_fd, name, create=False)
            try:
                _fsync_tree_fd(child_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(listed.st_mode):
            descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise ValueError("cache generation contains an unsafe entry")
    os.fsync(directory_fd)


def _seal_generation_files(directory_fd: int) -> None:
    """Make published cache payloads read-only while retaining bounded refresh cleanup."""
    images_fd, _ = _open_child_directory(directory_fd, "images", create=False)
    try:
        for name in os.listdir(images_fd):
            current = os.stat(name, dir_fd=images_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise ValueError("cache image must be a single-link regular file")
            os.chmod(name, 0o400, dir_fd=images_fd, follow_symlinks=False)
        os.fsync(images_fd)
    finally:
        os.close(images_fd)
    for name in ("result.json", "manifest.json"):
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise ValueError("cache metadata must be a single-link regular file")
        os.chmod(name, 0o400, dir_fd=directory_fd, follow_symlinks=False)
    os.fsync(directory_fd)


@contextmanager
def _cache_lock(parent_fd: int, name: str) -> Iterator[None]:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | _NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (listed.st_dev, listed.st_ino)
        ):
            raise ValueError("cache lock must be an owned single-link regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _read_owned_at(directory_fd: int, relative: str) -> tuple[bytes, os.stat_result]:
    parts = relative.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("cache artifact path is unsafe")
    current_fd = os.dup(directory_fd)
    try:
        for component in parts[:-1]:
            child_fd, _ = _open_child_directory(current_fd, component, create=False)
            os.close(current_fd)
            current_fd = child_fd
        descriptor = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=current_fd)
        try:
            opened = os.fstat(descriptor)
            listed = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _identity(opened) != _identity(listed)
            ):
                raise ValueError("cache artifact must be an owned single-link regular file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                _identity(opened),
            ) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                _identity(after),
            ):
                raise ValueError("cache artifact changed while reading")
            return b"".join(chunks), after
        finally:
            os.close(descriptor)
    finally:
        os.close(current_fd)


def _open_owned_at(directory_fd: int, relative: str) -> tuple[int, os.stat_result]:
    """Open a pinned, single-link regular cache artifact below ``directory_fd``."""
    parts = relative.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("cache artifact path is unsafe")
    current_fd = os.dup(directory_fd)
    try:
        for component in parts[:-1]:
            child_fd, _ = _open_child_directory(current_fd, component, create=False)
            os.close(current_fd)
            current_fd = child_fd
        descriptor = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=current_fd)
        try:
            opened = os.fstat(descriptor)
            listed = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _identity(opened) != _identity(listed)
            ):
                raise ValueError(
                    "cache artifact must be an owned single-link regular file"
                )
        except Exception:
            os.close(descriptor)
            raise
        return descriptor, opened
    finally:
        os.close(current_fd)


def _verify_streamed_source(
    descriptor: int,
    before: os.stat_result,
    *,
    expected_size: int,
    expected_sha256: str,
    destination_fd: int | None = None,
) -> tuple[os.stat_result, str]:
    """Verify one source artifact while optionally streaming it to a destination FD."""
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        if destination_fd is not None:
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_fd, chunk[offset:])
                if written <= 0:
                    raise OSError("short cache materialization write")
                offset += written
    after = os.fstat(descriptor)
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        _identity(before),
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        _identity(after),
    ):
        raise ValueError("cache artifact changed while streaming")
    actual_sha256 = digest.hexdigest()
    if size != expected_size or actual_sha256 != expected_sha256:
        raise ValueError("cache image differs from its manifest")
    return after, actual_sha256


def _verify_image_at(
    generation_fd: int, image: _CachedImage
) -> os.stat_result:
    descriptor, before = _open_owned_at(generation_fd, image.path)
    try:
        current, _ = _verify_streamed_source(
            descriptor,
            before,
            expected_size=image.size,
            expected_sha256=image.sha256,
        )
        return current
    finally:
        os.close(descriptor)


def _copy_verified_image_at(
    generation_fd: int,
    image: _CachedImage,
    destination_directory_fd: int,
    destination_name: str,
) -> os.stat_result:
    """Stream one verified cache image into a new, exclusively owned file."""
    source_fd, before = _open_owned_at(generation_fd, image.path)
    destination_fd = -1
    destination_identity: _Identity | None = None
    try:
        destination_fd = os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
            dir_fd=destination_directory_fd,
        )
        destination_identity = _identity(os.fstat(destination_fd))
        _verify_streamed_source(
            source_fd,
            before,
            expected_size=image.size,
            expected_sha256=image.sha256,
            destination_fd=destination_fd,
        )
        os.fsync(destination_fd)
        return os.fstat(destination_fd)
    except Exception:
        if destination_identity is not None:
            with suppress(OSError):
                listed = os.stat(
                    destination_name,
                    dir_fd=destination_directory_fd,
                    follow_symlinks=False,
                )
                if _identity(listed) == destination_identity:
                    os.unlink(destination_name, dir_fd=destination_directory_fd)
        raise
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)


def _regular_names_fd(directory_fd: int, prefix: str = "") -> list[str]:
    names: list[str] = []
    for name in sorted(os.listdir(directory_fd)):
        relative = f"{prefix}/{name}" if prefix else name
        listed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(listed.st_mode):
            child_fd, _ = _open_child_directory(directory_fd, name, create=False)
            try:
                names.extend(_regular_names_fd(child_fd, relative))
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(listed.st_mode) and listed.st_nlink == 1:
            names.append(relative)
        else:
            raise ValueError("cache generation contains an unsafe entry")
    return names


def _write_owned_at(directory_fd: int, relative: str, content: bytes) -> os.stat_result:
    descriptor = os.open(
        relative,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("short cache artifact write")
            offset += written
        os.fsync(descriptor)
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _read_owned(path: Path) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("cache artifact must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def identity(item: os.stat_result) -> tuple[int, int, int, int]:
        return item.st_dev, item.st_ino, item.st_size, item.st_nlink
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise ValueError("cache artifact identity changed")
    return b"".join(chunks), after


def _verified_image(image: StagedImage) -> bytes:
    content, current = _read_owned(image.path)
    if image.device is not None and image.device != current.st_dev:
        raise ValueError("staged image device differs")
    if image.inode is not None and image.inode != current.st_ino:
        raise ValueError("staged image inode differs")
    if image.size is not None and image.size != len(content):
        raise ValueError("staged image size differs")
    if image.sha256 is not None and image.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("staged image digest differs")
    return content


class ExtractionResultCache:
    """Persist and reconstruct trusted extraction evidence without executable formats."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.absolute()
        self.root = self.cache_dir / "extraction-results"

    def path_for(self, source: SourceFingerprint, config_hash: str) -> Path:
        if len(config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in config_hash
        ):
            raise ValueError("extraction config hash must be lowercase SHA-256")
        return self.root / source.digest / config_hash

    def _open_source(
        self, lease: _CacheRootLease, source: SourceFingerprint, *, create: bool
    ) -> tuple[int, _Identity]:
        return _open_child_directory(lease.root_fd, source.digest, create=create)

    def _inspect_pinned(
        self,
        lease: _CacheRootLease,
        source_fd: int,
        source_identity: _Identity,
        source: SourceFingerprint,
        config_hash: str,
        *,
        verify_images: bool,
    ) -> _CachedGeneration | None:
        generation_fd = -1
        try:
            generation_fd, generation_identity = _open_child_directory(
                source_fd, config_hash, create=False
            )
            names = _regular_names_fd(generation_fd)
            manifest_bytes, _ = _read_owned_at(generation_fd, "manifest.json")
            result_bytes, _ = _read_owned_at(generation_fd, "result.json")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                return None
            images = manifest.get("images")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("source") != source.to_dict()
                or manifest.get("extraction_config_hash") != config_hash
                or manifest.get("result_sha256") != hashlib.sha256(result_bytes).hexdigest()
                or not isinstance(images, list)
            ):
                return None
            expected_names = ["manifest.json", "result.json"]
            cached_images: list[_CachedImage] = []
            for index, item in enumerate(images):
                if not isinstance(item, dict):
                    return None
                relative = item.get("path")
                expected_relative = f"images/{index:08d}.jpg"
                if relative != expected_relative:
                    return None
                size = item.get("size")
                digest = item.get("sha256")
                if (
                    not isinstance(size, int)
                    or size < 0
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    return None
                expected_names.append(expected_relative)
                cached_images.append(_CachedImage(expected_relative, size, digest))
            if names != sorted(expected_names):
                return None
            result = _ADAPTER.validate_python(json.loads(result_bytes))
            if len(result.samples) != len(images):
                return None
            images_fd, images_identity = _open_child_directory(
                generation_fd, "images", create=False
            )
            os.close(images_fd)
            if verify_images:
                for image in cached_images:
                    _verify_image_at(generation_fd, image)
            lease.assert_bound()
            _assert_child_bound(lease.root_fd, source.digest, source_identity)
            _assert_child_bound(source_fd, config_hash, generation_identity)
            return _CachedGeneration(
                result=result,
                images=tuple(cached_images),
                generation_identity=generation_identity,
                images_identity=images_identity,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            KeyError,
            IndexError,
        ):
            return None
        finally:
            if generation_fd >= 0:
                os.close(generation_fd)

    def contains(self, source: SourceFingerprint, config_hash: str) -> bool:
        """Return whether a complete generation verifies without exposing its evidence."""
        self.path_for(source, config_hash)
        try:
            lease = _CacheRootLease.open(self.cache_dir, create=False)
            try:
                source_fd, source_identity = self._open_source(lease, source, create=False)
                try:
                    return (
                        self._inspect_pinned(
                            lease,
                            source_fd,
                            source_identity,
                            source,
                            config_hash,
                            verify_images=True,
                        )
                        is not None
                    )
                finally:
                    os.close(source_fd)
            finally:
                lease.close()
        except (OSError, ValueError):
            return False

    def materialize(
        self,
        source: SourceFingerprint,
        config_hash: str,
        source_path: Path,
        working_root: Path,
        recording_id: str,
    ) -> RecordingExtractionResult | None:
        """Copy a verified immutable generation into a unique caller-owned invocation."""
        self.path_for(source, config_hash)
        invocation = None
        try:
            lease = _CacheRootLease.open(self.cache_dir, create=False)
            try:
                source_fd, source_identity = self._open_source(lease, source, create=False)
                try:
                    cached = self._inspect_pinned(
                        lease,
                        source_fd,
                        source_identity,
                        source,
                        config_hash,
                        verify_images=False,
                    )
                    if cached is None:
                        return None
                    generation_fd, generation_identity = _open_child_directory(
                        source_fd, config_hash, create=False
                    )
                    try:
                        if generation_identity != cached.generation_identity:
                            raise ValueError("cache generation changed before materialization")
                        invocation = create_staging_invocation(working_root, recording_id)
                        destination_fd = os.open(invocation.path, _DIRECTORY_FLAGS)
                        try:
                            for index, image in enumerate(cached.images):
                                filename = f"{index:08d}.jpg"
                                current = _copy_verified_image_at(
                                    generation_fd,
                                    image,
                                    destination_fd,
                                    filename,
                                )
                                invocation.owned_files[filename] = _identity(current)
                        finally:
                            os.close(destination_fd)
                        lease.assert_bound()
                        _assert_child_bound(lease.root_fd, source.digest, source_identity)
                        _assert_child_bound(source_fd, config_hash, generation_identity)
                    finally:
                        os.close(generation_fd)
                finally:
                    os.close(source_fd)
            finally:
                lease.close()
        except (OSError, ValueError):
            if invocation is not None:
                rollback_staging_invocation(invocation)
            return None
        try:
            if invocation is None:
                raise RuntimeError("cache materialization invocation was not created")
            directory_device, directory_inode, chain = staged_directory_metadata(invocation.path)
            samples = []
            for index, (sample, image) in enumerate(
                zip(cached.result.samples, cached.images, strict=True)
            ):
                filename = f"{index:08d}.jpg"
                path = invocation.path / filename
                current = path.stat()
                samples.append(
                    replace(
                        sample,
                        staged_image=replace(
                            sample.staged_image,
                            path=path,
                            device=current.st_dev,
                            inode=current.st_ino,
                            size=image.size,
                            sha256=image.sha256,
                            invocation_root=invocation.path,
                            root_relative_path=filename,
                            directory_device=directory_device,
                            directory_inode=directory_inode,
                            directory_chain_identities=chain,
                        ),
                    )
                )
            return replace(
                cached.result,
                source_path=source_path.absolute(),
                staging_root=invocation.path,
                samples=tuple(samples),
            )
        except Exception:
            rollback_staging_invocation(invocation)
            raise

    def store(
        self,
        source: SourceFingerprint,
        config_hash: str,
        result: RecordingExtractionResult,
        *,
        force_refresh: bool = False,
    ) -> CacheStoreResult:
        self.path_for(source, config_hash)
        lease = _CacheRootLease.open(self.cache_dir, create=True)
        try:
            source_fd, source_identity = self._open_source(lease, source, create=True)
            try:
                with _cache_lock(source_fd, f".{config_hash}.lock"):
                    lease.assert_bound()
                    _assert_child_bound(lease.root_fd, source.digest, source_identity)
                    cached = self._inspect_pinned(
                        lease,
                        source_fd,
                        source_identity,
                        source,
                        config_hash,
                        verify_images=True,
                    )
                    if cached is not None and not force_refresh:
                        return CacheStoreResult(
                            path=self.path_for(source, config_hash),
                            created=False,
                            refreshed=False,
                            image_count=len(cached.images),
                        )
                    return self._store_locked(
                        lease, source_fd, source_identity, source, config_hash, result
                    )
            finally:
                os.close(source_fd)
        finally:
            lease.close()

    def _store_locked(
        self,
        lease: _CacheRootLease,
        source_fd: int,
        source_identity: _Identity,
        source: SourceFingerprint,
        config_hash: str,
        result: RecordingExtractionResult,
    ) -> CacheStoreResult:
        try:
            existing = os.stat(config_hash, dir_fd=source_fd, follow_symlinks=False)
            if not stat.S_ISDIR(existing.st_mode):
                raise ValueError("cache generation must be a directory")
            existing_identity: _Identity | None = _identity(existing)
        except FileNotFoundError:
            existing_identity = None
        temporary_name = f".{config_hash}.staging-{uuid.uuid4().hex}"
        os.mkdir(temporary_name, 0o700, dir_fd=source_fd)
        temporary_fd, temporary_identity = _open_child_directory(
            source_fd, temporary_name, create=False
        )
        published = False
        try:
            os.mkdir("images", 0o700, dir_fd=temporary_fd)
            images_fd, _ = _open_child_directory(temporary_fd, "images", create=False)
            image_rows: list[dict[str, object]] = []
            cached_samples = []
            try:
                for index, sample in enumerate(result.samples):
                    content = _verified_image(sample.staged_image)
                    filename = f"{index:08d}.jpg"
                    current = _write_owned_at(images_fd, filename, content)
                    digest = hashlib.sha256(content).hexdigest()
                    relative = f"images/{filename}"
                    image_rows.append({"path": relative, "size": len(content), "sha256": digest})
                    cached_samples.append(
                        replace(
                            sample,
                            staged_image=replace(
                                sample.staged_image,
                                path=self.path_for(source, config_hash) / relative,
                                device=current.st_dev,
                                inode=current.st_ino,
                                size=len(content),
                                sha256=digest,
                            ),
                        )
                    )
            finally:
                os.close(images_fd)
            cached_result = replace(
                result,
                staging_root=self.path_for(source, config_hash),
                samples=tuple(cached_samples),
            )
            result_bytes = (canonical_json(_jsonable(cached_result)) + "\n").encode()
            _write_owned_at(temporary_fd, "result.json", result_bytes)
            manifest = {
                "schema_version": 1,
                "source": source.to_dict(),
                "extraction_config_hash": config_hash,
                "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "images": image_rows,
            }
            _write_owned_at(
                temporary_fd,
                "manifest.json",
                (canonical_json(manifest) + "\n").encode(),
            )
            _fsync_tree_fd(temporary_fd)
            _seal_generation_files(temporary_fd)
            lease.assert_bound()
            _assert_child_bound(lease.root_fd, source.digest, source_identity)
            _assert_child_bound(source_fd, temporary_name, temporary_identity)
            if existing_identity is None:
                _rename_exclusive(source_fd, temporary_name, config_hash)
            else:
                self._publish_refresh(
                    source_fd,
                    temporary_name,
                    temporary_fd,
                    temporary_identity,
                    config_hash,
                    existing_identity,
                )
            published = True
            lease.assert_bound()
            loaded = self._inspect_pinned(
                lease,
                source_fd,
                source_identity,
                source,
                config_hash,
                verify_images=True,
            )
            if loaded is None:
                raise ValueError("stored extraction result did not revalidate")
            return CacheStoreResult(
                path=self.path_for(source, config_hash),
                created=existing_identity is None,
                refreshed=existing_identity is not None,
                image_count=len(loaded.images),
            )
        finally:
            try:
                if not published:
                    try:
                        final_stat = os.stat(
                            config_hash, dir_fd=source_fd, follow_symlinks=False
                        )
                        reached_final = _identity(final_stat) == temporary_identity
                    except FileNotFoundError:
                        reached_final = False
                    if not reached_final:
                        with suppress(OSError, ValueError):
                            cleanup_pinned_directory(
                                source_fd,
                                temporary_name,
                                temporary_fd,
                                temporary_identity,
                            )
            finally:
                os.close(temporary_fd)

    def _publish_refresh(
        self,
        source_fd: int,
        temporary_name: str,
        temporary_fd: int,
        temporary_identity: _Identity,
        final_name: str,
        expected_identity: _Identity,
    ) -> None:
        predecessor_fd = -1
        exchanged = False
        try:
            _assert_child_bound(source_fd, temporary_name, temporary_identity)
            _assert_child_bound(source_fd, final_name, expected_identity)
            predecessor_fd, opened_identity = _open_child_directory(
                source_fd, final_name, create=False
            )
            if opened_identity != expected_identity:
                raise ValueError("opened cache generation identity differs")
            _exchange_directories(source_fd, temporary_name, final_name)
            exchanged = True
            os.fsync(source_fd)
            _assert_child_bound(source_fd, final_name, temporary_identity)
            _assert_child_bound(source_fd, temporary_name, expected_identity)
        finally:
            if not exchanged:
                try:
                    refreshed = os.stat(
                        final_name, dir_fd=source_fd, follow_symlinks=False
                    )
                    predecessor = os.stat(
                        temporary_name, dir_fd=source_fd, follow_symlinks=False
                    )
                    exchanged = (
                        _identity(refreshed) == temporary_identity
                        and _identity(predecessor) == expected_identity
                    )
                except FileNotFoundError:
                    pass
            if exchanged and predecessor_fd >= 0:
                with suppress(OSError, ValueError):
                    cleanup_pinned_directory(
                        source_fd,
                        temporary_name,
                        predecessor_fd,
                        expected_identity,
                    )
            if predecessor_fd >= 0:
                os.close(predecessor_fd)
