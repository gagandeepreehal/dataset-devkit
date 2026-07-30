"""Deterministic acquisition fingerprints and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import os
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
        return cls(cast(IntegrityMethod, method), verified, content_md5)


@dataclass(frozen=True)
class AcquisitionManifest:
    source: SourceFingerprint
    status: DownloadStatus
    artifact: ArtifactIdentity
    integrity: IntegrityVerification
    extraction_config_hash: str
    manifest_version: Literal[1] = 1

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
            "extraction_config_hash",
        }:
            raise ValueError("invalid acquisition manifest")
        if value["manifest_version"] != 1 or value["status"] not in {
            "downloaded",
            "resumed",
            "cache_hit",
        }:
            raise ValueError("unsupported acquisition manifest")
        config_hash = value["extraction_config_hash"]
        if not isinstance(config_hash, str) or len(config_hash) != 64:
            raise ValueError("invalid extraction config hash")
        return cls(
            source=SourceFingerprint.from_dict(value["source"]),
            status=cast(DownloadStatus, value["status"]),
            artifact=ArtifactIdentity.from_dict(value["artifact"]),
            integrity=IntegrityVerification.from_dict(value["integrity"]),
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


def write_manifest(path: Path, manifest: AcquisitionManifest) -> None:
    """Atomically write a canonical acquisition manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(manifest.to_dict()))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_manifest(path: Path) -> AcquisitionManifest | None:
    """Load a manifest; missing, malformed, and incompatible files are cache misses."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return AcquisitionManifest.from_dict(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None


def extraction_cache_reusable(
    manifest_path: Path,
    source: SourceFingerprint,
    expected_extraction_config_hash: str,
) -> bool:
    """Decide whether extraction output provenance exactly matches this request."""
    manifest = load_manifest(manifest_path)
    return bool(
        manifest is not None
        and manifest.source == source
        and manifest.extraction_config_hash == expected_extraction_config_hash
    )
