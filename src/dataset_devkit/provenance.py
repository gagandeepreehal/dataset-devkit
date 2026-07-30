"""Deterministic acquisition fingerprints and provenance manifests."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from dataset_devkit.config import GlobalConfig

DownloadStatus = Literal["downloaded", "resumed", "cache_hit"]
IntegrityMethod = Literal["content_md5", "size_etag"]


def canonical_json(value: object) -> str:
    """Return the stable UTF-8 JSON representation used by all provenance hashes."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: object) -> str:
    """Hash a value's canonical UTF-8 JSON representation with SHA-256."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceFingerprint:
    account_url: str
    container: str
    blob_path: str
    etag: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> SourceFingerprint:
        if not isinstance(value, dict) or set(value) != {
            "account_url",
            "container",
            "blob_path",
            "etag",
            "size",
        }:
            raise ValueError("invalid source fingerprint")
        account_url = value["account_url"]
        container = value["container"]
        blob_path = value["blob_path"]
        etag = value["etag"]
        size = value["size"]
        if (
            not isinstance(account_url, str)
            or not isinstance(container, str)
            or not isinstance(blob_path, str)
            or not isinstance(etag, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ValueError("invalid source fingerprint values")
        return cls(account_url, container, blob_path, etag, size)

    @property
    def digest(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def cache_key(self) -> str:
        return self.digest


@dataclass(frozen=True)
class ArtifactIdentity:
    cache_relative_path: str
    size: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object) -> ArtifactIdentity:
        if not isinstance(value, dict) or set(value) != {"cache_relative_path", "size", "sha256"}:
            raise ValueError("invalid artifact identity")
        relative_path = value["cache_relative_path"]
        size = value["size"]
        sha256 = value["sha256"]
        if (
            not isinstance(relative_path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError("invalid artifact identity values")
        return cls(relative_path, size, sha256)


@dataclass(frozen=True)
class IntegrityVerification:
    method: IntegrityMethod
    verified: bool
    content_md5: str | None

    @classmethod
    def from_dict(cls, value: object) -> IntegrityVerification:
        if not isinstance(value, dict) or set(value) != {"method", "verified", "content_md5"}:
            raise ValueError("invalid integrity verification")
        method = value["method"]
        verified = value["verified"]
        content_md5 = value["content_md5"]
        if (
            method not in {"content_md5", "size_etag"}
            or not isinstance(verified, bool)
            or (content_md5 is not None and not isinstance(content_md5, str))
        ):
            raise ValueError("invalid integrity verification values")
        if not verified:
            raise ValueError("integrity verification must be successful")
        if method == "size_etag":
            if content_md5 is not None:
                raise ValueError("size_etag integrity cannot claim content MD5")
        else:
            if not isinstance(content_md5, str):
                raise ValueError("content_md5 integrity requires an MD5 value")
            try:
                decoded_md5 = base64.b64decode(content_md5, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("invalid content_md5 integrity value") from error
            if (
                len(decoded_md5) != 16
                or base64.b64encode(decoded_md5).decode("ascii") != content_md5
            ):
                raise ValueError("invalid content_md5 integrity value")
        return cls(cast(IntegrityMethod, method), verified, content_md5)


@dataclass(frozen=True)
class AcquisitionManifest:
    source: SourceFingerprint
    status: DownloadStatus
    artifact: ArtifactIdentity
    integrity: IntegrityVerification
    requested_extraction_config_hash: str
    manifest_version: Literal[2] = 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> AcquisitionManifest:
        if not isinstance(value, dict) or set(value) != {
            "manifest_version",
            "source",
            "status",
            "artifact",
            "integrity",
            "requested_extraction_config_hash",
        }:
            raise ValueError("invalid acquisition manifest")
        if value["manifest_version"] != 2 or value["status"] not in {
            "downloaded",
            "resumed",
            "cache_hit",
        }:
            raise ValueError("unsupported acquisition manifest")
        config_hash = value["requested_extraction_config_hash"]
        if not isinstance(config_hash, str) or len(config_hash) != 64:
            raise ValueError("invalid extraction config hash")
        return cls(
            source=SourceFingerprint.from_dict(value["source"]),
            status=cast(DownloadStatus, value["status"]),
            artifact=ArtifactIdentity.from_dict(value["artifact"]),
            integrity=IntegrityVerification.from_dict(value["integrity"]),
            requested_extraction_config_hash=config_hash,
        )


@dataclass(frozen=True)
class ExtractionManifest:
    """Proof recorded only after extraction output has completed."""

    source: SourceFingerprint
    extraction_config_hash: str
    manifest_version: Literal[1] = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ExtractionManifest:
        if not isinstance(value, dict) or set(value) != {
            "manifest_version",
            "source",
            "extraction_config_hash",
        }:
            raise ValueError("invalid extraction manifest")
        config_hash = value["extraction_config_hash"]
        if (
            value["manifest_version"] != 1
            or not isinstance(config_hash, str)
            or len(config_hash) != 64
        ):
            raise ValueError("unsupported extraction manifest")
        return cls(
            source=SourceFingerprint.from_dict(value["source"]),
            extraction_config_hash=config_hash,
        )


def extraction_config_hash(config: GlobalConfig) -> str:
    """Hash only resolved configuration that affects recording extraction."""
    extraction_config = {
        "schema_version": config.schema_version,
        "topics": config.topics.model_dump(mode="json"),
        "downsampling": config.downsampling.model_dump(mode="json"),
        "image": config.image.model_dump(mode="json"),
        "gnss": config.gnss.model_dump(mode="json"),
        "frame_validity": config.frame_validity.model_dump(mode="json"),
        "sanity_checks": config.sanity_checks.model_dump(mode="json"),
    }
    return canonical_hash(extraction_config)


def _write_canonical_manifest(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_manifest(path: Path, manifest: AcquisitionManifest) -> None:
    """Atomically write a canonical acquisition manifest."""
    _write_canonical_manifest(path, manifest.to_dict())


def _load_manifest_value(path: Path) -> object:
    file_stat = path.lstat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError("manifest is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened_stat = os.fstat(descriptor)
    if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("manifest is not a regular file")
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_manifest(path: Path) -> AcquisitionManifest | None:
    """Load a manifest; missing, malformed, and incompatible files are cache misses."""
    try:
        value = _load_manifest_value(path)
        return AcquisitionManifest.from_dict(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
