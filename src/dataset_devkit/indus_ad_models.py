"""Immutable models for the integrated INDUS-AD repository contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal, cast

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


WeakLabelProvenance = Literal["machine_generated_weak_pseudo_label"]
AvailabilityState = Literal["available", "missing", "not_selected", "unavailable_source"]


@dataclass(frozen=True, slots=True)
class SemanticFraction:
    class_id: int
    fraction: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.class_id, int)
            or isinstance(self.class_id, bool)
            or not -32768 <= self.class_id <= 32767
            or not math.isfinite(self.fraction)
            or not 0.0 <= self.fraction <= 1.0
        ):
            raise ValueError("invalid semantic fraction")


@dataclass(frozen=True, slots=True)
class WeatherWeakLabel:
    probabilities: Mapping[str, float | None]
    label: str | None
    confidence: float | None
    margin: float | None
    uncertain: bool | None

    def __post_init__(self) -> None:
        values = dict(self.probabilities)
        expected = {
            "clear",
            "overcast",
            "rain",
            "fog",
            "low_light",
            "uncertain_mixed",
        }
        if set(values) != expected:
            raise ValueError("weather probabilities have unexpected keys")
        present = [value for value in values.values() if value is not None]
        if present and (
            len(present) != len(values)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in present)
            or not math.isclose(sum(present), 1.0, abs_tol=1e-5)
        ):
            raise ValueError("weather probabilities must be finite and sum to one")
        for value in (self.confidence, self.margin):
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError("invalid weather confidence or margin")
        if self.label is not None and self.label not in expected:
            raise ValueError("invalid weather label")
        object.__setattr__(self, "probabilities", MappingProxyType(values))


@dataclass(frozen=True, slots=True)
class ImageQualitySummary:
    values: Mapping[str, float | None]

    def __post_init__(self) -> None:
        expected = {
            "luminance_mean",
            "luminance_p05",
            "luminance_p95",
            "contrast_std",
            "saturation_mean",
            "laplacian_variance",
            "dark_fraction",
            "overexposed_fraction",
        }
        values = dict(self.values)
        if set(values) != expected:
            raise ValueError("image-quality summary has unexpected keys")
        if any(value is not None and not math.isfinite(value) for value in values.values()):
            raise ValueError("image-quality values must be finite")
        for name in ("dark_fraction", "overexposed_fraction"):
            value = values[name]
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("image-quality fractions must be within zero and one")
        object.__setattr__(self, "values", MappingProxyType(values))


@dataclass(frozen=True, slots=True)
class CameraAnnotation:
    recording_id: str
    camera_name: CameraName
    selected_time_of_day_ns: int | None
    recording_offset_ns: int
    frame_index: int
    frame_id: str
    sample_kind: str
    window_id: str
    decoded_width: int
    decoded_height: int
    semantic_fractions: tuple[SemanticFraction, ...]
    category_fractions: Mapping[str, float | None]
    depth_summary_m: Mapping[str, float | None]
    weather: WeatherWeakLabel
    image_quality: ImageQualitySummary
    provenance: WeakLabelProvenance = "machine_generated_weak_pseudo_label"

    def __post_init__(self) -> None:
        validate_safe_segment(self.recording_id)
        validate_safe_segment(self.frame_id)
        validate_safe_segment(self.sample_kind)
        validate_safe_segment(self.window_id)
        if (
            self.selected_time_of_day_ns is not None
            and (
                not isinstance(self.selected_time_of_day_ns, int)
                or isinstance(self.selected_time_of_day_ns, bool)
                or not 0 <= self.selected_time_of_day_ns < 86_400_000_000_000
            )
        ):
            raise ValueError("invalid selected time of day")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.recording_offset_ns, self.frame_index)
        ):
            raise ValueError("invalid camera annotation offset or frame index")
        if self.decoded_width <= 0 or self.decoded_height <= 0:
            raise ValueError("decoded dimensions must be positive")
        if sum(item.fraction for item in self.semantic_fractions) > 1.0 + 1e-6:
            raise ValueError("semantic fractions exceed one")
        class_ids = [item.class_id for item in self.semantic_fractions]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("semantic fractions contain duplicate classes")
        categories = dict(self.category_fractions)
        if any(
            value is not None
            and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
            for value in categories.values()
        ):
            raise ValueError("category fractions must be finite fractions")
        depth = dict(self.depth_summary_m)
        if set(depth) != {"p10", "p50", "p90"}:
            raise ValueError("depth summary has unexpected keys")
        present_depth = [depth[name] for name in ("p10", "p50", "p90")]
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in present_depth
        ):
            raise ValueError("depth summary must be finite and nonnegative")
        if all(value is not None for value in present_depth) and not (
            cast(float, present_depth[0])
            <= cast(float, present_depth[1])
            <= cast(float, present_depth[2])
        ):
            raise ValueError("depth percentiles are inconsistent")
        object.__setattr__(self, "category_fractions", MappingProxyType(categories))
        object.__setattr__(self, "depth_summary_m", MappingProxyType(depth))

    @property
    def identity(self) -> tuple[str, str]:
        return (self.recording_id, self.frame_id)

    @property
    def source_frame_identity(self) -> tuple[str, CameraName, int, int]:
        return (
            self.recording_id,
            self.camera_name,
            self.frame_index,
            self.recording_offset_ns,
        )


@dataclass(frozen=True, slots=True)
class GnssScenarioAnnotation:
    recording_id: str
    window_index: int
    window_start_offset_ns: int
    window_end_offset_ns: int
    motion_label: str
    validity_tags: tuple[str, ...]
    quality_tags: tuple[str, ...]
    anomaly_candidate: bool
    anomaly_reasons: tuple[str, ...]
    confidence: float
    schema_version: Literal["indus_ad_gnss_annotations_v1"]
    run_id: str
    provenance: WeakLabelProvenance = "machine_generated_weak_pseudo_label"

    def __post_init__(self) -> None:
        validate_safe_segment(self.recording_id)
        validate_safe_segment(self.motion_label)
        validate_safe_segment(self.run_id)
        for values in (self.validity_tags, self.quality_tags, self.anomaly_reasons):
            for value in values:
                validate_safe_segment(value)
            if len(values) != len(set(values)):
                raise ValueError("GNSS annotation tags must be unique")
        if (
            self.window_index < 0
            or self.window_start_offset_ns < 0
            or self.window_end_offset_ns <= self.window_start_offset_ns
        ):
            raise ValueError("invalid GNSS annotation window")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid GNSS annotation confidence")


@dataclass(frozen=True, slots=True)
class DenseAssetRef:
    recording_id: str
    camera_name: CameraName
    frame_index: int
    recording_offset_ns: int
    frame_id: str
    kind: Literal["semantic", "depth"]
    dtype: str
    width: int
    height: int
    member_path: str
    payload_bytes: int
    payload_sha256: str
    shard_path: str

    def __post_init__(self) -> None:
        validate_safe_segment(self.recording_id)
        validate_safe_segment(self.frame_id)
        validate_safe_segment(self.dtype)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.frame_index, self.recording_offset_ns)
        ):
            raise ValueError("invalid dense frame identity")
        if self.width <= 0 or self.height <= 0 or self.payload_bytes <= 0:
            raise ValueError("dense shape and payload size must be positive")
        _validate_optional_sha256(self.payload_sha256)
        try:
            validate_repo_file_path(self.shard_path)
        except RepositoryPathError as error:
            raise ValueError("invalid dense shard path") from error
        member = PurePosixPath(self.member_path)
        if (
            member.is_absolute()
            or member.as_posix() != self.member_path
            or "\\" in self.member_path
            or any(part in {"", ".", ".."} for part in self.member_path.split("/"))
        ):
            raise ValueError("invalid dense member path")

    @property
    def identity(self) -> tuple[str, CameraName, int, int, str]:
        return (
            self.recording_id,
            self.camera_name,
            self.frame_index,
            self.recording_offset_ns,
            self.kind,
        )


@dataclass(frozen=True, slots=True)
class AnnotationAvailability:
    state: AvailabilityState
    reason_code: str

    def __post_init__(self) -> None:
        validate_safe_segment(self.reason_code)


@dataclass(frozen=True, slots=True)
class IndusAdAnnotationBundle:
    camera: tuple[CameraAnnotation, ...] = ()
    gnss: tuple[GnssScenarioAnnotation, ...] = ()
    dense_refs: tuple[DenseAssetRef, ...] = ()
    availability: Mapping[str, AnnotationAvailability] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "availability", MappingProxyType(dict(self.availability)))
