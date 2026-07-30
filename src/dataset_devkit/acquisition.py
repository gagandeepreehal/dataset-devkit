"""Azure Blob acquisition with resumable, fingerprinted local caching."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
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
    IntegrityVerification,
    SourceFingerprint,
    canonical_hash,
    canonical_json,
    extraction_config_hash,
    load_manifest,
    write_manifest,
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
    partial: Path
    partial_sidecar: Path


@dataclass(frozen=True)
class AcquisitionResult:
    artifact_path: Path
    manifest_path: Path
    manifest: AcquisitionManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> bytes:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def _content_md5(properties: BlobPropertiesProtocol) -> bytes | None:
    value = properties.content_settings.content_md5
    return bytes(value) if value is not None else None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
        recording_identity = {
            "account_url": source.account_url,
            "container": source.container,
            "blob_path": source.blob_path,
        }
        directory = self.cache_dir / "azure" / canonical_hash(recording_identity)
        if not directory.resolve().is_relative_to(self.cache_dir):
            raise AcquisitionError("cache layout would escape the configured cache directory")
        stem = f"artifact-{source.digest}"
        return CachePaths(
            directory=directory,
            final=directory / f"{stem}.mcap",
            manifest=directory / f"{stem}.manifest.json",
            partial=directory / "download.partial",
            partial_sidecar=directory / "download.partial.json",
        )

    def _properties(self, client: BlobClientProtocol, blob_path: str) -> BlobPropertiesProtocol:
        try:
            return client.get_blob_properties()
        except Exception as error:
            raise AcquisitionError(f"failed to read Azure properties for {blob_path!r}") from error

    def _load_partial_source(self, path: Path) -> SourceFingerprint | None:
        try:
            return SourceFingerprint.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _discard_partial(self, paths: CachePaths) -> None:
        paths.partial.unlink(missing_ok=True)
        paths.partial_sidecar.unlink(missing_ok=True)

    def _cached_manifest(
        self,
        paths: CachePaths,
        source: SourceFingerprint,
        properties: BlobPropertiesProtocol,
    ) -> AcquisitionManifest | None:
        manifest = load_manifest(paths.manifest)
        expected_relative = paths.final.relative_to(self.cache_dir).as_posix()
        valid = bool(
            manifest is not None
            and manifest.source == source
            and manifest.artifact.cache_relative_path == expected_relative
            and manifest.artifact.size == source.size
            and paths.final.is_file()
            and paths.final.stat().st_size == source.size
            and _sha256_file(paths.final) == manifest.artifact.sha256
        )
        expected_md5 = _content_md5(properties)
        if valid and expected_md5 is not None:
            valid = _md5_file(paths.final) == expected_md5
        if not valid:
            paths.final.unlink(missing_ok=True)
            paths.manifest.unlink(missing_ok=True)
            return None
        assert manifest is not None
        return manifest

    def _verify_download(
        self,
        paths: CachePaths,
        source: SourceFingerprint,
        before: BlobPropertiesProtocol,
        after: BlobPropertiesProtocol,
    ) -> IntegrityVerification:
        after_source = self.fingerprint_for(source.blob_path, after)
        before_md5 = _content_md5(before)
        after_md5 = _content_md5(after)
        if after_source != source or before_md5 != after_md5:
            raise BlobChangedError(f"Azure blob changed while downloading {source.blob_path!r}")
        actual_size = paths.partial.stat().st_size
        if actual_size != source.size:
            raise IntegrityError(
                f"downloaded size mismatch for {source.blob_path!r}: "
                f"expected {source.size}, got {actual_size}"
            )
        if after_md5 is not None:
            if _md5_file(paths.partial) != after_md5:
                raise IntegrityError(f"content MD5 mismatch for {source.blob_path!r}")
            return IntegrityVerification(
                method="content_md5",
                verified=True,
                content_md5=base64.b64encode(after_md5).decode("ascii"),
            )
        return IntegrityVerification(method="size_etag", verified=True, content_md5=None)

    def acquire(self, blob_path: str) -> AcquisitionResult:
        """Acquire one exact blob, resuming only an identity-compatible partial."""
        validate_blob_path(blob_path)
        client = self.service_client.get_blob_client(
            container=self.azure.container, blob=blob_path
        )
        before = self._properties(client, blob_path)
        source = self.fingerprint_for(blob_path, before)
        paths = self.paths_for(source)
        paths.directory.mkdir(parents=True, exist_ok=True)

        cached = self._cached_manifest(paths, source, before)
        if cached is not None:
            cache_hit = replace(
                cached,
                status="cache_hit",
                extraction_config_hash=self.extraction_config_hash,
            )
            write_manifest(paths.manifest, cache_hit)
            return AcquisitionResult(paths.final, paths.manifest, cache_hit)

        partial_source = self._load_partial_source(paths.partial_sidecar)
        compatible = bool(
            partial_source == source
            and paths.partial.is_file()
            and paths.partial.stat().st_size <= source.size
        )
        if not compatible:
            self._discard_partial(paths)
            _atomic_write_text(paths.partial_sidecar, canonical_json(source.to_dict()) + "\n")
        offset = paths.partial.stat().st_size if paths.partial.exists() else 0
        resumed = offset > 0

        try:
            if offset < source.size:
                with paths.partial.open("ab") as stream:
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
            integrity = self._verify_download(paths, source, before, after)
        except IntegrityError:
            self._discard_partial(paths)
            raise

        artifact = ArtifactIdentity(
            cache_relative_path=paths.final.relative_to(self.cache_dir).as_posix(),
            size=source.size,
            sha256=_sha256_file(paths.partial),
        )
        manifest = AcquisitionManifest(
            source=source,
            status="resumed" if resumed else "downloaded",
            artifact=artifact,
            integrity=integrity,
            extraction_config_hash=self.extraction_config_hash,
        )
        try:
            os.replace(paths.partial, paths.final)
            write_manifest(paths.manifest, manifest)
        except OSError as error:
            paths.final.unlink(missing_ok=True)
            message = f"failed to finalize cache artifact for {blob_path!r}"
            raise AcquisitionError(message) from error
        finally:
            paths.partial_sidecar.unlink(missing_ok=True)
        return AcquisitionResult(paths.final, paths.manifest, manifest)
