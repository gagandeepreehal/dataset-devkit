"""Deterministic acquisition fingerprints and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from dataset_devkit.config import GlobalConfig, GlobalConfigV1, normalized_source
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_mcap_path

DownloadStatus = Literal["downloaded", "cache_hit"]


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
    repo_id: str
    revision: str
    repo_path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        invalid = (
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
                self.repo_id,
            )
            is None
            or len(self.revision) != 40
            or any(character not in "0123456789abcdef" for character in self.revision)
            or not self.repo_path
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
            or not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size <= 0
        )
        try:
            validate_repo_mcap_path(self.repo_path)
        except RepositoryPathError:
            invalid = True
        if invalid:
            raise ValueError("invalid source fingerprint values")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> SourceFingerprint:
        if not isinstance(value, dict) or set(value) != {
            "repo_id",
            "revision",
            "repo_path",
            "sha256",
            "size",
        }:
            raise ValueError("invalid source fingerprint")
        repo_id = value["repo_id"]
        revision = value["revision"]
        repo_path = value["repo_path"]
        sha256 = value["sha256"]
        size = value["size"]
        if (
            not isinstance(repo_id, str)
            or not isinstance(revision, str)
            or len(revision) != 40
            or not isinstance(repo_path, str)
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise ValueError("invalid source fingerprint values")
        return cls(repo_id, revision, repo_path, sha256, size)

    @property
    def digest(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def cache_key(self) -> str:
        return self.digest


@dataclass(frozen=True)
class DecodedSourceFingerprint:
    repo_id: str
    revision: str
    recording_id: str
    catalog_row_sha256: str
    artifact_set_sha256: str
    total_size: int
    source_type: Literal["decoded_hf"] = "decoded_hf"
    privacy_classification: Literal["privacy_transformed"] = "privacy_transformed"

    def __post_init__(self) -> None:
        invalid = (
            not isinstance(self.repo_id, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
                self.repo_id,
            )
            is None
            or not isinstance(self.revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None
            or not isinstance(self.catalog_row_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.catalog_row_sha256) is None
            or not isinstance(self.artifact_set_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.artifact_set_sha256) is None
            or not isinstance(self.total_size, int)
            or isinstance(self.total_size, bool)
            or self.total_size <= 0
        )
        try:
            validate_safe_segment(self.recording_id)
        except ValueError:
            invalid = True
        if invalid:
            raise ValueError("invalid decoded source fingerprint values")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> DecodedSourceFingerprint:
        expected = {
            "repo_id",
            "revision",
            "recording_id",
            "catalog_row_sha256",
            "artifact_set_sha256",
            "total_size",
            "source_type",
            "privacy_classification",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid decoded source fingerprint")
        if (
            value["source_type"] != "decoded_hf"
            or value["privacy_classification"] != "privacy_transformed"
        ):
            raise ValueError("invalid decoded source fingerprint classification")
        try:
            return cls(
                repo_id=cast(str, value["repo_id"]),
                revision=cast(str, value["revision"]),
                recording_id=cast(str, value["recording_id"]),
                catalog_row_sha256=cast(str, value["catalog_row_sha256"]),
                artifact_set_sha256=cast(str, value["artifact_set_sha256"]),
                total_size=cast(int, value["total_size"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid decoded source fingerprint values") from error

    @property
    def digest(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def cache_key(self) -> str:
        return self.digest


type RecordingFingerprint = SourceFingerprint | DecodedSourceFingerprint


def fingerprint_to_dict(value: RecordingFingerprint) -> dict[str, object]:
    """Serialize an exact discriminated recording fingerprint."""
    if isinstance(value, SourceFingerprint):
        return {
            "source_type": "mcap_hf",
            "privacy_classification": "restricted_raw",
            **value.to_dict(),
        }
    return value.to_dict()


def fingerprint_from_dict(value: object) -> RecordingFingerprint:
    """Parse a strict recording fingerprint without accepting hybrid shapes."""
    if not isinstance(value, dict):
        raise ValueError("invalid recording fingerprint")
    source_type = value.get("source_type")
    if source_type == "decoded_hf":
        return DecodedSourceFingerprint.from_dict(value)
    if source_type == "mcap_hf":
        expected = {
            "source_type",
            "privacy_classification",
            "repo_id",
            "revision",
            "repo_path",
            "sha256",
            "size",
        }
        if set(value) != expected or value.get("privacy_classification") != "restricted_raw":
            raise ValueError("invalid MCAP source fingerprint")
        legacy = {
            key: value[key]
            for key in ("repo_id", "revision", "repo_path", "sha256", "size")
        }
        return SourceFingerprint.from_dict(legacy)
    raise ValueError("invalid recording fingerprint source type")


def fingerprint_locator(value: RecordingFingerprint) -> str:
    """Return the backend-neutral recording locator safe for diagnostics."""
    if isinstance(value, SourceFingerprint):
        return value.repo_path
    return value.recording_id


@dataclass(frozen=True)
class ArtifactIdentity:
    cache_relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.cache_relative_path)
        if (
            path.is_absolute()
            or path.as_posix() != self.cache_relative_path
            or any(part in {"", ".", ".."} for part in self.cache_relative_path.split("/"))
            or not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size <= 0
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("invalid artifact identity values")

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
            or size <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError("invalid artifact identity values")
        return cls(relative_path, size, sha256)


@dataclass(frozen=True)
class AcquisitionManifest:
    source: SourceFingerprint
    status: DownloadStatus
    artifact: ArtifactIdentity
    requested_extraction_config_hash: str
    manifest_version: Literal[1] = 1

    def __post_init__(self) -> None:
        if self.artifact.size != self.source.size or self.artifact.sha256 != self.source.sha256:
            raise ValueError("artifact identity does not match source fingerprint")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> AcquisitionManifest:
        if not isinstance(value, dict) or set(value) != {
            "manifest_version",
            "source",
            "status",
            "artifact",
            "requested_extraction_config_hash",
        }:
            raise ValueError("invalid acquisition manifest")
        if value["manifest_version"] != 1 or value["status"] not in {
            "downloaded",
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
        "source_extraction": (
            {
                "topics": config.topics.model_dump(mode="json"),
                "mcap_compatibility": config.mcap_compatibility.model_dump(mode="json"),
            }
            if isinstance(config, GlobalConfigV1)
            else normalized_source(config).model_dump(
                mode="json", exclude={"repo_id", "revision", "privacy_classification"}
            )
        ),
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


def load_extraction_manifest(path: Path) -> ExtractionManifest | None:
    """Load trusted extraction evidence; invalid or linked files are cache misses."""
    try:
        value = _load_manifest_value(path)
        return ExtractionManifest.from_dict(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
