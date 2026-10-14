"""Immutable models for the integrated INDUS-AD repository contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal

from dataset_devkit.config import CameraName
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.provenance import IndusAdSourceFingerprint
from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_file_path

ArtifactKind = Literal[
    "video",
    "camera_frames",
    "calibration",
    "gnss",
    "camera_annotations",
    "dense_index",
    "semantic_archive",
    "depth_archive",
]


def _validate_blob_id(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("blob_id must be a lowercase Git object ID")


def _validate_optional_sha256(value: str | None) -> None:
    if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    path: str
    size: int
    blob_id: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            validate_repo_file_path(self.path)
        except RepositoryPathError as error:
            raise ValueError("invalid repository file path") from error
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size <= 0:
            raise ValueError("repository file size must be a positive integer")
        _validate_blob_id(self.blob_id)
        _validate_optional_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class IndusAdArtifact:
    path: str
    size: int
    blob_id: str
    kind: ArtifactKind
    camera: CameraName | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        RepositoryFile(self.path, self.size, self.blob_id, self.sha256)
        if self.kind == "video" and self.camera is None:
            raise ValueError("video artifacts require a camera")
        if self.kind != "video" and self.camera is not None and self.kind != "camera_frames":
            raise ValueError("camera is only valid for video or camera-frame artifacts")


@dataclass(frozen=True, slots=True)
class IndusAdRecordingEntry:
    recording_id: str
    split: str
    videos: tuple[IndusAdArtifact, ...]
    artifacts: tuple[IndusAdArtifact, ...]
    fingerprint: IndusAdSourceFingerprint

    def __post_init__(self) -> None:
        validate_safe_segment(self.recording_id)
        validate_safe_segment(self.split)
        if self.fingerprint.recording_id != self.recording_id:
            raise ValueError("recording fingerprint identity differs")
        if not self.videos:
            raise ValueError("recording requires selected videos")
        paths = [item.path for item in (*self.videos, *self.artifacts)]
        if len(paths) != len(set(paths)):
            raise ValueError("recording contains duplicate artifact paths")
        cameras = [item.camera for item in self.videos]
        if len(cameras) != len(set(cameras)) or any(
            item.kind != "video" for item in self.videos
        ):
            raise ValueError("recording videos must have unique cameras")

    def artifacts_of(self, kind: ArtifactKind) -> tuple[IndusAdArtifact, ...]:
        return tuple(item for item in self.artifacts if item.kind == kind)


@dataclass(frozen=True, slots=True)
class IndusAdSelectionPlan:
    repo_id: str
    revision: str
    contract: Literal["indus_ad_integrated_v1"]
    viewer_privacy_counts: Mapping[str, int]
    annotation_schema_sha256: str
    recordings: tuple[IndusAdRecordingEntry, ...]

    def __post_init__(self) -> None:
        if self.contract != "indus_ad_integrated_v1":
            raise ValueError("unsupported INDUS-AD contract")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("invalid INDUS-AD revision")
        if re.fullmatch(r"[0-9a-f]{64}", self.annotation_schema_sha256) is None:
            raise ValueError("invalid annotation schema digest")
        expected_keys = {"face_plate_blurred", "missing"}
        if set(self.viewer_privacy_counts) != expected_keys or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.viewer_privacy_counts.values()
        ):
            raise ValueError("invalid Viewer privacy counts")
        object.__setattr__(
            self,
            "viewer_privacy_counts",
            MappingProxyType(dict(self.viewer_privacy_counts)),
        )
        recording_ids = [item.recording_id for item in self.recordings]
        if not self.recordings or len(recording_ids) != len(set(recording_ids)):
            raise ValueError("selection must contain unique recordings")


def repository_basename(path: str) -> str:
    """Return a validated repository basename for diagnostics."""
    return PurePosixPath(path).name
