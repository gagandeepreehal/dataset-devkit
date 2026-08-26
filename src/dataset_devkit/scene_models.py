"""Immutable Task 5 scene-graph records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from dataset_devkit.extraction.models import CameraCalibration, EgoPose, StagedImage
from dataset_devkit.provenance import SourceFingerprint


def _jsonable(value: object) -> object:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class AnnotationRecord:
    token: str
    line_number: int
    repo_path: str
    timestamp_ns: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationMatch:
    annotation_token: str
    line_number: int
    matched: bool
    sample_timestamp_ns: int | None
    signed_error_ns: int | None
    absolute_error_ns: int | None
    reason: Literal["matched", "different_recording", "no_valid_samples", "outside_tolerance"]


@dataclass(frozen=True)
class AnnotationWindow:
    token: str
    annotation_tokens: tuple[str, ...]
    first_timestamp_ns: int
    last_timestamp_ns: int
    first_sample_timestamp_ns: int
    last_sample_timestamp_ns: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class SceneRecord:
    token: str
    name: str
    ordinal: int
    kind: Literal["automatic", "annotation"]
    first_sample_token: str
    last_sample_token: str
    nbr_samples: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    labels: tuple[str, ...]
    annotation_refs: tuple[str, ...]
    annotation_window_ref: str
    source_repo_path: str


@dataclass(frozen=True)
class SampleRecord:
    token: str
    scene_token: str
    timestamp_ns: int
    grid_timestamp_ns: int
    batch_timestamp_ns: int
    prev: str
    next: str


@dataclass(frozen=True)
class SampleDataRecord:
    token: str
    sample_token: str
    scene_token: str
    channel: str
    camera_index: int
    channel_ordinal: int
    timestamp_ns: int
    filename: str
    staged_image: StagedImage
    calibration: CameraCalibration | None
    ego_pose: EgoPose
    prev: str
    next: str
    grid_signed_sync_error_ns: int = 0
    camera_signed_sync_error_ns: int = 0
    gnss_source_validity: tuple[bool, ...] = ()


@dataclass(frozen=True)
class UnassignedSample:
    timestamp_ns: int
    reason: Literal[
        "candidate_too_short",
        "inter_scene_skip",
        "annotation_mode_excluded",
        "annotation_range_excluded",
    ]
    details: str


@dataclass(frozen=True)
class SourceSampleRecord:
    timestamp_ns: int
    batch_timestamp_ns: int
    expected_channels: tuple[str, ...]
    valid_run_id: int
    present_channels: tuple[str, ...] = ()
    source_gnss_valid: bool = True
    grid_signed_sync_error_ns: int = 0


@dataclass(frozen=True)
class RecordingSceneResult:
    source: SourceFingerprint
    source_samples: tuple[SourceSampleRecord, ...]
    scenes: tuple[SceneRecord, ...]
    samples: tuple[SampleRecord, ...]
    sample_data: tuple[SampleDataRecord, ...]
    annotations: tuple[AnnotationRecord, ...]
    annotation_matches: tuple[AnnotationMatch, ...]
    annotation_windows: tuple[AnnotationWindow, ...]
    unassigned: tuple[UnassignedSample, ...]
    build_mode: Literal["automatic", "annotation_only", "hybrid"]
    min_scene_duration_ns: int
    max_scene_duration_ns: int
    min_scene_samples: int
    annotation_match_tolerance_ns: int
    annotation_before_ns: int
    annotation_after_ns: int
    max_sample_gap_ns: int
    inter_scene_skip_ns: int
    annotation_window_merge_semantics: Literal["same_run_overlap_or_touch_v1"]
    dataset_namespace: UUID
    build_config_token: str
    feature_evidence_token: str = ""

    def to_dict(self) -> dict[str, object]:
        return _jsonable(self)  # type: ignore[return-value]
