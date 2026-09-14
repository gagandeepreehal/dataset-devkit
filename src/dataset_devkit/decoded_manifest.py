"""Strict control-plane validation for privacy-transformed decoded V2 releases."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pyarrow.parquet as pq

from dataset_devkit.config import (
    DEFAULT_DECODED_CAMERAS,
    CameraName,
    DecodedHfSourceConfig,
)
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.provenance import DecodedSourceFingerprint, canonical_hash
from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_file_path

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATE_TOKEN = re.compile(r"(?:^|/)(?:date=)?\d{4}-\d{2}-\d{2}(?:/|$)")
_ALL_CAMERAS = {*DEFAULT_DECODED_CAMERAS, "cam_front_tilted"}
_ARTIFACT_KINDS = {
    "auxiliary_messages",
    "calibration",
    "camera_frames",
    "camera_messages",
    "camera_videos",
    "depth_index",
    "frame_labels",
    "gap_events",
    "gnss",
    "keyframes",
    "recording_metadata",
    "semantic_index",
    "sync_groups",
}
_MODALITY_ARTIFACT = {
    "gnss": "gnss",
    "calibration": "calibration",
    "camera_labels": "frame_labels",
    "semantic": "semantic_index",
    "depth": "depth_index",
}
_CATALOG_COLUMNS = {
    "recording_id",
    "split",
    "default_cameras",
    "available_cameras",
    "video_paths",
    "video_sizes",
    "video_sha256",
    "frame_counts",
    "duration_ns",
    "start_time_of_day_ns",
    "end_time_of_day_ns",
    "start_offset_ns",
    "end_offset_ns",
    "artifact_paths",
    "artifact_sizes",
    "artifact_sha256",
    "has_gnss",
    "has_calibration",
    "has_camera_labels",
    "has_semantic",
    "has_depth",
    "published_horizontal_resolution_m",
    "has_cam_front",
    "has_cam_front_left",
    "has_cam_front_right",
    "has_cam_front_tilted",
    "has_cam_rear",
    "has_cam_rear_left",
    "has_cam_rear_right",
}


class DecodedManifestError(ValueError):
    """Raised when decoded V2 public control data is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class PublicArtifact:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DecodedArtifact(PublicArtifact):
    kind: str


@dataclass(frozen=True, slots=True)
class DecodedVideo(PublicArtifact):
    camera: CameraName
    frame_count: int
    start_offset_ns: int
    end_offset_ns: int


@dataclass(frozen=True, slots=True)
class PrivacyContract:
    model_lock_sha256: str
    privacy_config_sha256: str
    absolute_dates_removed: bool
    exact_recording_offsets_retained: bool
    local_enu_precision: Literal["retained"]
    global_horizontal_resolution_m: float


@dataclass(frozen=True, slots=True)
class DecodedRecordingEntry:
    recording_id: str
    split: str
    duration_ns: int
    videos: tuple[DecodedVideo, ...]
    artifacts: tuple[DecodedArtifact, ...]
    fingerprint: DecodedSourceFingerprint

    def artifact(self, kind: str) -> DecodedArtifact:
        matches = tuple(item for item in self.artifacts if item.kind == kind)
        if len(matches) != 1:
            raise DecodedManifestError(f"recording requires exactly one {kind!r} artifact")
        return matches[0]


@dataclass(frozen=True, slots=True)
class DecodedSelectionPlan:
    repo_id: str
    revision: str
    privacy: PrivacyContract
    recordings: tuple[DecodedRecordingEntry, ...]


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DecodedManifestError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise DecodedManifestError(f"invalid {label}")
    return value


def _read_parquet(path: Path, expected_columns: set[str], label: str) -> list[dict[str, object]]:
    try:
        parquet = pq.ParquetFile(path)
        if set(parquet.schema_arrow.names) != expected_columns:
            raise DecodedManifestError(f"{label} has unexpected columns")
        return cast(list[dict[str, object]], parquet.read().to_pylist())
    except DecodedManifestError:
        raise
    except Exception as error:
        raise DecodedManifestError(f"invalid {label}") from error


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecodedManifestError(f"invalid {label} SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DecodedManifestError(f"{label} must be a positive integer")
    return value


def _offset(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DecodedManifestError(f"{label} must be a nonnegative integer")
    return value


def _public_path(value: object) -> str:
    if not isinstance(value, str):
        raise DecodedManifestError("artifact path must be a string")
    try:
        validate_repo_file_path(value)
    except RepositoryPathError as error:
        raise DecodedManifestError(f"invalid public artifact path: {value!r}") from error
    if _DATE_TOKEN.search(value) is not None:
        raise DecodedManifestError(f"public artifact path contains a date token: {value!r}")
    return value


def _privacy_contract(root: Path) -> PrivacyContract:
    processing = _read_json(root / "data/manifests/processing-config.json", "processing config")
    models = _read_json(root / "data/manifests/model-receipt.json", "model receipt")
    audit_rows = _read_parquet(
        root / "data/manifests/schema-audit.parquet",
        {"audit_json"},
        "schema audit",
    )
    if len(audit_rows) != 1 or not isinstance(audit_rows[0].get("audit_json"), str):
        raise DecodedManifestError("schema audit must contain exactly one JSON row")
    try:
        audit = json.loads(cast(str, audit_rows[0]["audit_json"]))
    except json.JSONDecodeError as error:
        raise DecodedManifestError("schema audit JSON is invalid") from error
    if not isinstance(audit, dict):
        raise DecodedManifestError("schema audit JSON is invalid")
    if (
        processing.get("global_horizontal_resolution_m") != 1.0
        or processing.get("local_enu_precision") != "retained"
        or audit.get("absolute_dates_removed") is not True
        or audit.get("exact_recording_offsets_retained") is not True
        or audit.get("published_horizontal_resolution_m") != 1.0
    ):
        raise DecodedManifestError("public privacy contract assertions do not match")
    return PrivacyContract(
        _sha256(models.get("model_lock_sha256"), "model lock"),
        _sha256(models.get("privacy_config_sha256"), "privacy config"),
        True,
        True,
        "retained",
        1.0,
    )


def _output_index(root: Path) -> dict[str, PublicArtifact]:
    rows = _read_parquet(
        root / "data/manifests/output-files.parquet",
        {"path", "size", "sha256"},
        "output manifest",
    )
    result: dict[str, PublicArtifact] = {}
    for row in rows:
        path = _public_path(row.get("path"))
        if path in result:
            raise DecodedManifestError("output manifest contains a duplicate path")
        result[path] = PublicArtifact(
            path,
            _positive_int(row.get("size"), "output size"),
            _sha256(row.get("sha256"), "output"),
        )
    return result


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DecodedManifestError(f"catalog {label} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _string_list(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or len(value) != len(set(cast(list[str], value)))
    ):
        raise DecodedManifestError(f"catalog {label} must contain unique strings")
    return cast(list[str], value)


def _bind_output(
    artifact: PublicArtifact, output_index: dict[str, PublicArtifact]
) -> PublicArtifact:
    if output_index.get(artifact.path) != artifact:
        raise DecodedManifestError(
            f"selected artifact differs from output manifest: {artifact.path!r}"
        )
    return artifact


def _parse_row(
    row: dict[str, object],
    source: DecodedHfSourceConfig,
    output_index: dict[str, PublicArtifact],
) -> DecodedRecordingEntry:
    if set(row) != _CATALOG_COLUMNS:
        raise DecodedManifestError("recording catalog row has unexpected columns")
    recording_id = row.get("recording_id")
    split = row.get("split")
    try:
        if not isinstance(recording_id, str) or validate_safe_segment(recording_id) != recording_id:
            raise ValueError
        if not isinstance(split, str) or validate_safe_segment(split) != split:
            raise ValueError
    except ValueError as error:
        raise DecodedManifestError("recording catalog identity is invalid") from error

    defaults = _string_list(row.get("default_cameras"), "default cameras")
    available = _string_list(row.get("available_cameras"), "available cameras")
    if defaults != list(DEFAULT_DECODED_CAMERAS):
        raise DecodedManifestError("recording catalog default cameras differ")
    if available != sorted(available) or not set(defaults).issubset(available):
        raise DecodedManifestError("recording catalog is missing a default camera")
    if not set(available).issubset(_ALL_CAMERAS):
        raise DecodedManifestError("recording catalog contains an unknown camera")

    video_paths = _mapping(row.get("video_paths"), "video paths")
    video_sizes = _mapping(row.get("video_sizes"), "video sizes")
    video_hashes = _mapping(row.get("video_sha256"), "video SHA-256")
    frame_counts = _mapping(row.get("frame_counts"), "frame counts")
    if not all(set(mapping) == set(available) for mapping in (
        video_paths, video_sizes, video_hashes, frame_counts
    )):
        raise DecodedManifestError("recording catalog video mappings differ")
    for camera in _ALL_CAMERAS:
        if row.get(f"has_{camera}") is not (camera in available):
            raise DecodedManifestError("recording catalog camera availability differs")

    start_offset = _offset(row.get("start_offset_ns"), "start offset")
    end_offset = _offset(row.get("end_offset_ns"), "end offset")
    duration = _positive_int(row.get("duration_ns"), "duration")
    if end_offset <= start_offset or duration != end_offset - start_offset:
        raise DecodedManifestError("recording catalog duration or offsets differ")
    for label in ("start_time_of_day_ns", "end_time_of_day_ns"):
        value = row.get(label)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < 86_400_000_000_000
        ):
            raise DecodedManifestError("recording catalog time of day is invalid")
    resolution = row.get("published_horizontal_resolution_m")
    if not isinstance(resolution, float) or not math.isfinite(resolution) or resolution != 1.0:
        raise DecodedManifestError("recording catalog resolution is invalid")

    videos: list[DecodedVideo] = []
    for camera in source.cameras:
        if camera not in available:
            raise DecodedManifestError(f"selected camera is unavailable: {camera}")
        video = DecodedVideo(
            _public_path(video_paths[camera]),
            _positive_int(video_sizes[camera], "video size"),
            _sha256(video_hashes[camera], "video"),
            camera,
            _positive_int(frame_counts[camera], "frame count"),
            start_offset,
            end_offset,
        )
        _bind_output(PublicArtifact(video.path, video.size, video.sha256), output_index)
        videos.append(video)

    artifact_paths = _mapping(row.get("artifact_paths"), "artifact paths")
    artifact_sizes = _mapping(row.get("artifact_sizes"), "artifact sizes")
    artifact_hashes = _mapping(row.get("artifact_sha256"), "artifact SHA-256")
    if not (set(artifact_paths) == set(artifact_sizes) == set(artifact_hashes)):
        raise DecodedManifestError("recording catalog artifact mappings differ")
    if not set(artifact_paths).issubset(_ARTIFACT_KINDS):
        raise DecodedManifestError("recording catalog contains an unknown artifact kind")
    all_artifacts: dict[str, tuple[DecodedArtifact, ...]] = {}
    for kind in sorted(artifact_paths):
        paths = _string_list(artifact_paths[kind], f"{kind} paths")
        sizes = artifact_sizes[kind]
        hashes = artifact_hashes[kind]
        if not isinstance(sizes, list) or not isinstance(hashes, list):
            raise DecodedManifestError("recording catalog artifact arrays are invalid")
        if not len(paths) == len(sizes) == len(hashes):
            raise DecodedManifestError("recording catalog artifact arrays differ")
        items = tuple(
            DecodedArtifact(
                _public_path(path),
                _positive_int(size, "artifact size"),
                _sha256(digest, "artifact"),
                kind,
            )
            for path, size, digest in zip(paths, sizes, hashes, strict=True)
        )
        if tuple(item.path for item in items) != tuple(sorted(item.path for item in items)):
            raise DecodedManifestError("recording catalog artifacts are not sorted")
        all_artifacts[kind] = items

    expected_flags = {
        "gnss": "has_gnss",
        "calibration": "has_calibration",
        "frame_labels": "has_camera_labels",
        "semantic_index": "has_semantic",
        "depth_index": "has_depth",
    }
    for kind, flag in expected_flags.items():
        if row.get(flag) is not bool(all_artifacts.get(kind)):
            raise DecodedManifestError("recording catalog modality availability differs")

    required_kinds = {"camera_frames"}
    required_kinds.update(
        artifact_kind
        for modality, artifact_kind in _MODALITY_ARTIFACT.items()
        if modality in source.modalities
    )
    selected_artifacts: list[DecodedArtifact] = []
    for kind in sorted(required_kinds):
        items = all_artifacts.get(kind, ())
        if len(items) != 1:
            raise DecodedManifestError(f"recording requires exactly one {kind!r} artifact")
        item = items[0]
        _bind_output(PublicArtifact(item.path, item.size, item.sha256), output_index)
        selected_artifacts.append(item)

    identity = [
        {
            "kind": f"video:{item.camera}",
            "path": item.path,
            "size": item.size,
            "sha256": item.sha256,
        }
        for item in videos
    ]
    identity.extend(
        {"kind": item.kind, "path": item.path, "size": item.size, "sha256": item.sha256}
        for item in selected_artifacts
    )
    identity.sort(key=lambda item: (cast(str, item["kind"]), cast(str, item["path"])))
    fingerprint = DecodedSourceFingerprint(
        source.repo_id,
        source.revision,
        recording_id,
        canonical_hash(row),
        canonical_hash(identity),
        sum(cast(int, item["size"]) for item in identity),
    )
    return DecodedRecordingEntry(
        recording_id,
        split,
        duration,
        tuple(videos),
        tuple(selected_artifacts),
        fingerprint,
    )


def parse_decoded_control_plane(
    root: Path, source: DecodedHfSourceConfig
) -> DecodedSelectionPlan:
    """Validate the complete public control plane and select payload identities."""
    privacy = _privacy_contract(root)
    output_index = _output_index(root)
    rows = _read_parquet(root / source.recordings_path, _CATALOG_COLUMNS, "recording catalog")
    recording_ids = [row.get("recording_id") for row in rows]
    if len(recording_ids) != len(set(recording_ids)):
        raise DecodedManifestError("recording catalog contains a duplicate recording")
    entries = tuple(_parse_row(row, source, output_index) for row in rows)
    selected = tuple(
        item
        for item in entries
        if (not source.recording_ids or item.recording_id in source.recording_ids)
        and (not source.splits or item.split in source.splits)
    )
    if not selected:
        raise DecodedManifestError("decoded selection matched no recordings")
    if source.recording_ids and {item.recording_id for item in selected} != set(
        source.recording_ids
    ):
        raise DecodedManifestError("decoded selection is missing requested recordings")
    return DecodedSelectionPlan(source.repo_id, source.revision, privacy, selected)
