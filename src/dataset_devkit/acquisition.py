"""Azure Blob acquisition with resumable, fingerprinted local caching."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Protocol, cast

from azure.core import MatchConditions
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from dataset_devkit.blob_list import validate_blob_path
from dataset_devkit.config import AzureConfig, GlobalConfig
from dataset_devkit.provenance import (
    AcquisitionManifest,
    ArtifactIdentity,
    ExtractionManifest,
    IntegrityVerification,
    SourceFingerprint,
    canonical_hash,
    canonical_json,
    extraction_config_hash,
)


class AcquisitionError(RuntimeError):
    """Raised when Azure acquisition cannot safely produce a cache artifact."""


class BlobChangedError(AcquisitionError):
    """Raised when source properties change during a download."""


class IntegrityError(AcquisitionError):
    """Raised when downloaded bytes fail an available integrity check."""


class _ContentSettingsProtocol(Protocol):
    @property
    def content_md5(self) -> bytes | bytearray | None: ...


class BlobPropertiesProtocol(Protocol):
    @property
    def size(self) -> int: ...

    @property
    def etag(self) -> str: ...

    @property
    def content_settings(self) -> _ContentSettingsProtocol: ...


class DownloadProtocol(Protocol):
    def readinto(self, stream: IO[bytes]) -> int: ...


class BlobClientProtocol(Protocol):
    def get_blob_properties(self) -> BlobPropertiesProtocol: ...

    def download_blob(self, *, offset: int, **kwargs: object) -> DownloadProtocol: ...


class BlobServiceClientProtocol(Protocol):
    def get_blob_client(self, *, container: str, blob: str) -> BlobClientProtocol: ...


@dataclass(frozen=True)
class CachePaths:
    directory: Path
    final: Path
    manifest: Path
    extraction_manifest: Path
    partial: Path
    partial_sidecar: Path


@dataclass(frozen=True)
class AcquisitionResult:
    artifact_path: Path
    manifest_path: Path
    extraction_manifest_path: Path
    manifest: AcquisitionManifest


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    link_count: int


@dataclass(frozen=True)
class _VerifiedFile:
    identity: _FileIdentity
    sha256: str
    md5: bytes


@dataclass(frozen=True)
class _DirectoryIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class _RecordingCache:
    root_path: Path
    root_fd: int
    azure_fd: int
    recording_fd: int
    recording_name: str
    root_identity: _DirectoryIdentity
    azure_identity: _DirectoryIdentity
    recording_identity: _DirectoryIdentity

    def validate(self) -> None:
        try:
            root_stat = self.root_path.lstat()
            azure_stat = os.stat("azure", dir_fd=self.root_fd, follow_symlinks=False)
            recording_stat = os.stat(
                self.recording_name,
                dir_fd=self.azure_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise AcquisitionError("trusted cache directory chain changed") from error
        if (
            _directory_identity(root_stat) != self.root_identity
            or _directory_identity(azure_stat) != self.azure_identity
            or _directory_identity(recording_stat) != self.recording_identity
        ):
            raise AcquisitionError("trusted cache directory chain changed")


def _identity(file_stat: os.stat_result) -> _FileIdentity | None:
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        return None
    return _FileIdentity(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        link_count=file_stat.st_nlink,
    )


def _directory_identity(file_stat: os.stat_result) -> _DirectoryIdentity | None:
    if not stat.S_ISDIR(file_stat.st_mode):
        return None
    return _DirectoryIdentity(file_stat.st_dev, file_stat.st_ino)


def _open_or_create_directory_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileExistsError:
        pass
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    if _directory_identity(os.fstat(descriptor)) is None:
        os.close(descriptor)
        raise AcquisitionError(f"cache component is not a directory: {name}")
    return descriptor


@contextmanager
def _open_recording_cache(root_path: Path, recording_name: str) -> Iterator[_RecordingCache]:
    root_path.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root_path, flags)
    azure_fd = -1
    recording_fd = -1
    try:
        root_identity = _directory_identity(os.fstat(root_fd))
        if root_identity is None:
            raise AcquisitionError("configured cache root is not a directory")
        azure_fd = _open_or_create_directory_at(root_fd, "azure")
        recording_fd = _open_or_create_directory_at(azure_fd, recording_name)
        azure_identity = _directory_identity(os.fstat(azure_fd))
        recording_identity = _directory_identity(os.fstat(recording_fd))
        assert azure_identity is not None and recording_identity is not None
        cache = _RecordingCache(
            root_path=root_path,
            root_fd=root_fd,
            azure_fd=azure_fd,
            recording_fd=recording_fd,
            recording_name=recording_name,
            root_identity=root_identity,
            azure_identity=azure_identity,
            recording_identity=recording_identity,
        )
        cache.validate()
        yield cache
    finally:
        if recording_fd >= 0:
            os.close(recording_fd)
        if azure_fd >= 0:
            os.close(azure_fd)
        os.close(root_fd)


def _owned_file_identity_at(directory_fd: int, name: str) -> _FileIdentity | None:
    try:
        return _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
    except FileNotFoundError:
        return None


def _verify_owned_file_at(directory_fd: int, name: str) -> _VerifiedFile:
    sha256_digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    before = _identity(os.fstat(descriptor))
    if before is None:
        os.close(descriptor)
        raise AcquisitionError(f"cache leaf is not an owned regular file: {name}")
    with os.fdopen(descriptor, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha256_digest.update(chunk)
            md5_digest.update(chunk)
        after = _identity(os.fstat(stream.fileno()))
    if after != before or _owned_file_identity_at(directory_fd, name) != before:
        raise AcquisitionError(f"cache leaf identity changed during verification: {name}")
    return _VerifiedFile(before, sha256_digest.hexdigest(), md5_digest.digest())


def _unlink_at(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    os.fsync(directory_fd)


def _atomic_write_text_at(directory_fd: int, name: str, content: str) -> None:
    temporary_name = f".{name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if _owned_file_identity_at(directory_fd, temporary_name) is None:
            raise AcquisitionError("temporary cache leaf is not exclusively owned")
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)


def _create_owned_empty_file_at(directory_fd: int, name: str) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        if _identity(os.fstat(descriptor)) is None:
            raise AcquisitionError(f"new cache leaf is not exclusively owned: {name}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _load_json_at(directory_fd: int, name: str) -> object:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    before = _identity(os.fstat(descriptor))
    if before is None:
        os.close(descriptor)
        raise ValueError("cache JSON leaf is not an owned regular file")
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        value = json.load(stream)
        after = _identity(os.fstat(stream.fileno()))
    if after != before or _owned_file_identity_at(directory_fd, name) != before:
        raise ValueError("cache JSON leaf identity changed while reading")
    return value


def _load_acquisition_manifest_at(
    directory_fd: int, name: str
) -> AcquisitionManifest | None:
    try:
        return AcquisitionManifest.from_dict(_load_json_at(directory_fd, name))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_acquisition_manifest_at(
    directory_fd: int, name: str, manifest: AcquisitionManifest
) -> None:
    _atomic_write_text_at(
        directory_fd,
        name,
        canonical_json(manifest.to_dict()) + "\n",
    )


def _load_extraction_manifest_at(
    directory_fd: int, name: str
) -> ExtractionManifest | None:
    try:
        return ExtractionManifest.from_dict(_load_json_at(directory_fd, name))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_extraction_manifest_at(
    directory_fd: int, name: str, manifest: ExtractionManifest
) -> None:
    _atomic_write_text_at(
        directory_fd,
        name,
        canonical_json(manifest.to_dict()) + "\n",
    )


def _content_md5(properties: BlobPropertiesProtocol) -> bytes | None:
    value = properties.content_settings.content_md5
    return bytes(value) if value is not None else None


@contextmanager
def _recording_lock_at(directory_fd: int) -> Iterator[None]:
    name = ".acquisition.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    lock_identity = _identity(os.fstat(descriptor))
    if lock_identity is None:
        os.close(descriptor)
        raise AcquisitionError("acquisition lock must be an owned regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if _owned_file_identity_at(directory_fd, name) != lock_identity:
            raise AcquisitionError("acquisition lock identity changed")
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class AzureBlobAcquirer:
    """Acquire exact Azure blobs into a verified, provenance-bearing cache."""

    def __init__(
        self,
        *,
        azure: AzureConfig,
        cache_dir: Path,
        extraction_config_hash: str,
        service_client: BlobServiceClientProtocol | None = None,
    ) -> None:
        self.azure = azure
        self.cache_dir = cache_dir.resolve()
        self.extraction_config_hash = extraction_config_hash
        if service_client is None:
            credential = DefaultAzureCredential()
            service_client = cast(
                BlobServiceClientProtocol,
                BlobServiceClient(account_url=azure.account_url, credential=credential),
            )
        self.service_client = service_client

    @classmethod
    def from_config(
        cls,
        config: GlobalConfig,
        *,
        service_client: BlobServiceClientProtocol | None = None,
    ) -> AzureBlobAcquirer:
        """Build the acquisition service from validated global configuration."""
        return cls(
            azure=config.azure,
            cache_dir=config.paths.cache_dir,
            extraction_config_hash=extraction_config_hash(config),
            service_client=service_client,
        )

    def fingerprint_for(
        self, blob_path: str, properties: BlobPropertiesProtocol
    ) -> SourceFingerprint:
        """Create the complete cache identity for current Azure properties."""
        validate_blob_path(blob_path)
        size = properties.size
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise AcquisitionError("Azure returned an invalid blob size")
        return SourceFingerprint(
            account_url=self.azure.account_url,
            container=self.azure.container,
            blob_path=blob_path,
            etag=str(properties.etag),
            size=size,
        )

    def paths_for(self, source: SourceFingerprint) -> CachePaths:
        """Return a hash-derived cache layout that cannot contain blob path traversal."""
        directory = self._recording_directory(source.blob_path)
        stem = f"artifact-{source.digest}"
        return CachePaths(
            directory=directory,
            final=directory / f"{stem}.mcap",
            manifest=directory / f"{stem}.manifest.json",
            extraction_manifest=directory / f"{stem}.extraction.manifest.json",
            partial=directory / "download.partial",
            partial_sidecar=directory / "download.partial.json",
        )

    def _recording_directory(self, blob_path: str) -> Path:
        recording_identity = {
            "account_url": self.azure.account_url,
            "container": self.azure.container,
            "blob_path": blob_path,
        }
        directory = self.cache_dir / "azure" / canonical_hash(recording_identity)
        if not directory.resolve().is_relative_to(self.cache_dir):
            raise AcquisitionError("cache layout would escape the configured cache directory")
        return directory

    def _properties(self, client: BlobClientProtocol, blob_path: str) -> BlobPropertiesProtocol:
        try:
            return client.get_blob_properties()
        except Exception as error:
            raise AcquisitionError(f"failed to read Azure properties for {blob_path!r}") from error

    def _validate_owned_source(self, source: SourceFingerprint) -> None:
        validate_blob_path(source.blob_path)
        if (
            source.account_url != self.azure.account_url
            or source.container != self.azure.container
        ):
            raise AcquisitionError("source fingerprint does not belong to this Azure cache")

    def record_extraction_complete(
        self,
        source: SourceFingerprint,
        completed_extraction_config_hash: str,
    ) -> Path:
        """Record completed extraction inside the source's trusted, locked cache."""
        self._validate_owned_source(source)
        paths = self.paths_for(source)
        recording_directory = self._recording_directory(source.blob_path)
        manifest = ExtractionManifest.from_dict(
            ExtractionManifest(
                source=source,
                extraction_config_hash=completed_extraction_config_hash,
            ).to_dict()
        )
        with (
            _open_recording_cache(self.cache_dir, recording_directory.name) as cache,
            _recording_lock_at(cache.recording_fd),
        ):
            self._record_extraction_complete_locked(cache, paths, manifest)
        return paths.extraction_manifest

    def _record_extraction_complete_locked(
        self,
        cache: _RecordingCache,
        paths: CachePaths,
        manifest: ExtractionManifest,
    ) -> None:
        cache.validate()
        try:
            _write_extraction_manifest_at(
                cache.recording_fd,
                paths.extraction_manifest.name,
                manifest,
            )
            if (
                _owned_file_identity_at(
                    cache.recording_fd,
                    paths.extraction_manifest.name,
                )
                is None
            ):
                raise AcquisitionError(
                    "extraction manifest is not an owned regular cache file"
                )
            cache.validate()
        except (OSError, AcquisitionError):
            _unlink_at(cache.recording_fd, paths.extraction_manifest.name)
            raise

    def extraction_cache_reusable(
        self,
        source: SourceFingerprint,
        expected_extraction_config_hash: str,
    ) -> bool:
        """Check completed extraction inside the source's trusted, locked cache."""
        try:
            self._validate_owned_source(source)
            paths = self.paths_for(source)
            recording_directory = self._recording_directory(source.blob_path)
            with (
                _open_recording_cache(self.cache_dir, recording_directory.name) as cache,
                _recording_lock_at(cache.recording_fd),
            ):
                return self._extraction_cache_reusable_locked(
                    cache,
                    paths,
                    source,
                    expected_extraction_config_hash,
                )
        except (OSError, AcquisitionError):
            return False

    def _extraction_cache_reusable_locked(
        self,
        cache: _RecordingCache,
        paths: CachePaths,
        source: SourceFingerprint,
        expected_extraction_config_hash: str,
    ) -> bool:
        cache.validate()
        manifest = _load_extraction_manifest_at(
            cache.recording_fd,
            paths.extraction_manifest.name,
        )
        cache.validate()
        return bool(
            manifest is not None
            and manifest.source == source
            and manifest.extraction_config_hash == expected_extraction_config_hash
        )

    def _load_partial_source(
        self, directory_fd: int, name: str
    ) -> SourceFingerprint | None:
        if _owned_file_identity_at(directory_fd, name) is None:
            return None
        try:
            return SourceFingerprint.from_dict(_load_json_at(directory_fd, name))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _discard_partial(self, directory_fd: int, paths: CachePaths) -> None:
        _unlink_at(directory_fd, paths.partial.name)
        _unlink_at(directory_fd, paths.partial_sidecar.name)

    def _cached_manifest(
        self,
        paths: CachePaths,
        source: SourceFingerprint,
        properties: BlobPropertiesProtocol,
        directory_fd: int,
    ) -> AcquisitionManifest | None:
        try:
            verified_final = (
                _verify_owned_file_at(directory_fd, paths.final.name)
                if _owned_file_identity_at(directory_fd, paths.final.name) is not None
                else None
            )
        except (OSError, AcquisitionError):
            verified_final = None
        manifest = (
            _load_acquisition_manifest_at(directory_fd, paths.manifest.name)
            if _owned_file_identity_at(directory_fd, paths.manifest.name) is not None
            and verified_final is not None
            else None
        )
        expected_relative = paths.final.relative_to(self.cache_dir).as_posix()
        valid = bool(
            manifest is not None
            and manifest.source == source
            and manifest.artifact.cache_relative_path == expected_relative
            and manifest.artifact.size == source.size
            and verified_final is not None
            and verified_final.identity.size == source.size
            and verified_final.sha256 == manifest.artifact.sha256
        )
        expected_md5 = _content_md5(properties)
        if valid and expected_md5 is not None:
            assert verified_final is not None
            valid = verified_final.md5 == expected_md5
        if not valid:
            _unlink_at(directory_fd, paths.final.name)
            _unlink_at(directory_fd, paths.manifest.name)
            return None
        assert manifest is not None
        return manifest

    def _verify_download(
        self,
        paths: CachePaths,
        source: SourceFingerprint,
        before: BlobPropertiesProtocol,
        after: BlobPropertiesProtocol,
        directory_fd: int,
    ) -> tuple[IntegrityVerification, _VerifiedFile]:
        after_source = self.fingerprint_for(source.blob_path, after)
        before_md5 = _content_md5(before)
        after_md5 = _content_md5(after)
        if after_source != source or before_md5 != after_md5:
            raise BlobChangedError(f"Azure blob changed while downloading {source.blob_path!r}")
        try:
            verified = _verify_owned_file_at(directory_fd, paths.partial.name)
        except (OSError, AcquisitionError) as error:
            raise IntegrityError(
                f"download partial is not an owned regular file: {source.blob_path!r}"
            ) from error
        if verified.identity.size != source.size:
            raise IntegrityError(
                f"downloaded size mismatch for {source.blob_path!r}: "
                f"expected {source.size}, got {verified.identity.size}"
            )
        if after_md5 is not None:
            if verified.md5 != after_md5:
                raise IntegrityError(f"content MD5 mismatch for {source.blob_path!r}")
            return (
                IntegrityVerification(
                    method="content_md5",
                    verified=True,
                    content_md5=base64.b64encode(after_md5).decode("ascii"),
                ),
                verified,
            )
        return (
            IntegrityVerification(method="size_etag", verified=True, content_md5=None),
            verified,
        )

    def acquire(self, blob_path: str) -> AcquisitionResult:
        """Acquire one exact blob, resuming only an identity-compatible partial."""
        validate_blob_path(blob_path)
        client = self.service_client.get_blob_client(
            container=self.azure.container, blob=blob_path
        )
        before = self._properties(client, blob_path)
        source = self.fingerprint_for(blob_path, before)
        paths = self.paths_for(source)
        recording_directory = self._recording_directory(blob_path)
        with (
            _open_recording_cache(self.cache_dir, recording_directory.name) as cache,
            _recording_lock_at(cache.recording_fd),
        ):
            cache.validate()
            return self._acquire_locked(
                blob_path,
                client,
                before,
                source,
                paths,
                cache,
            )

    def _acquire_locked(
        self,
        blob_path: str,
        client: BlobClientProtocol,
        before: BlobPropertiesProtocol,
        source: SourceFingerprint,
        paths: CachePaths,
        cache: _RecordingCache,
    ) -> AcquisitionResult:
        cache.validate()

        directory_fd = cache.recording_fd
        cached = self._cached_manifest(paths, source, before, directory_fd)
        if cached is not None:
            cache_hit = replace(
                cached,
                status="cache_hit",
                requested_extraction_config_hash=self.extraction_config_hash,
            )
            _write_acquisition_manifest_at(directory_fd, paths.manifest.name, cache_hit)
            cache.validate()
            return AcquisitionResult(
                paths.final,
                paths.manifest,
                paths.extraction_manifest,
                cache_hit,
            )

        partial_identity = _owned_file_identity_at(directory_fd, paths.partial.name)
        partial_size = partial_identity.size if partial_identity is not None else None
        sidecar_is_regular = (
            _owned_file_identity_at(directory_fd, paths.partial_sidecar.name) is not None
        )
        partial_source = (
            self._load_partial_source(directory_fd, paths.partial_sidecar.name)
            if sidecar_is_regular
            else None
        )
        compatible = bool(
            partial_source == source
            and partial_size is not None
            and partial_size <= source.size
            and (partial_size == 0 or _content_md5(before) is not None)
        )
        if not compatible:
            self._discard_partial(directory_fd, paths)
            _atomic_write_text_at(
                directory_fd,
                paths.partial_sidecar.name,
                canonical_json(source.to_dict()) + "\n",
            )
            _create_owned_empty_file_at(directory_fd, paths.partial.name)
        current_partial = _owned_file_identity_at(directory_fd, paths.partial.name)
        offset = current_partial.size if current_partial is not None else 0
        resumed = offset > 0

        try:
            if offset < source.size:
                flags = os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(
                    paths.partial.name,
                    flags,
                    0o600,
                    dir_fd=directory_fd,
                )
                partial_stat = os.fstat(descriptor)
                if not stat.S_ISREG(partial_stat.st_mode) or partial_stat.st_nlink != 1:
                    os.close(descriptor)
                    raise AcquisitionError("download partial must be a regular file")
                with os.fdopen(descriptor, "ab") as stream:
                    download = client.download_blob(
                        offset=offset,
                        etag=source.etag,
                        match_condition=MatchConditions.IfNotModified,
                        validate_content=True,
                    )
                    download.readinto(stream)
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception as error:
            raise AcquisitionError(f"Azure download failed for {blob_path!r}") from error

        after = self._properties(client, blob_path)
        try:
            integrity, verified_partial = self._verify_download(
                paths,
                source,
                before,
                after,
                directory_fd,
            )
        except IntegrityError:
            self._discard_partial(directory_fd, paths)
            raise

        artifact = ArtifactIdentity(
            cache_relative_path=paths.final.relative_to(self.cache_dir).as_posix(),
            size=source.size,
            sha256=verified_partial.sha256,
        )
        manifest = AcquisitionManifest(
            source=source,
            status="resumed" if resumed else "downloaded",
            artifact=artifact,
            integrity=integrity,
            requested_extraction_config_hash=self.extraction_config_hash,
        )
        try:
            cache.validate()
            if (
                _owned_file_identity_at(directory_fd, paths.partial.name)
                != verified_partial.identity
            ):
                raise AcquisitionError("refusing to finalize an unsafe download partial")
            os.replace(
                paths.partial.name,
                paths.final.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
            if (
                _owned_file_identity_at(directory_fd, paths.final.name)
                != verified_partial.identity
            ):
                raise AcquisitionError("final cache artifact has the wrong file identity")
            verified_final = _verify_owned_file_at(directory_fd, paths.final.name)
            if verified_final != verified_partial:
                raise AcquisitionError("final cache artifact failed integrity re-verification")
            _write_acquisition_manifest_at(directory_fd, paths.manifest.name, manifest)
            if _owned_file_identity_at(directory_fd, paths.manifest.name) is None:
                raise AcquisitionError("acquisition manifest is not an owned regular file")
            cache.validate()
        except (OSError, AcquisitionError) as error:
            _unlink_at(directory_fd, paths.final.name)
            _unlink_at(directory_fd, paths.manifest.name)
            message = f"failed to finalize cache artifact for {blob_path!r}"
            raise AcquisitionError(message) from error
        finally:
            _unlink_at(directory_fd, paths.partial_sidecar.name)
        return AcquisitionResult(
            paths.final,
            paths.manifest,
            paths.extraction_manifest,
            manifest,
        )
