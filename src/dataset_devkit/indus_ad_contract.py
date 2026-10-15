"""Fail-closed validation for the integrated INDUS-AD Hub layout."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from dataset_devkit.config import CameraName, IndusAdHfSourceConfig
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.indus_ad_models import (
    ArtifactKind,
    IndusAdArtifact,
    IndusAdRecordingEntry,
    IndusAdSelectionPlan,
    RepositoryFile,
)
from dataset_devkit.provenance import IndusAdSourceFingerprint, canonical_hash
from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_file_path

VIDEO_VIEWER_PATH = "data/viewer/videos/part-00000.parquet"
CAMERA_ANNOTATION_VIEWER_PATH = "data/viewer/camera-annotations/part-00000.parquet"
METADATA_PATH = "data/metadata.jsonl"
VIDEO_INTEGRITY_PATH = "data/manifests/video-integrity.jsonl"
VIDEO_PRIVACY_PATH = "data/manifests/video-privacy-release.json"

_CAMERAS: tuple[CameraName, ...] = (
    "cam_front",
    "cam_front_left",
    "cam_front_right",
    "cam_front_tilted",
    "cam_rear",
    "cam_rear_left",
    "cam_rear_right",
)
_REQUIRED_CONTROLS = {
    METADATA_PATH,
    VIDEO_VIEWER_PATH,
    CAMERA_ANNOTATION_VIEWER_PATH,
    VIDEO_INTEGRITY_PATH,
    VIDEO_PRIVACY_PATH,
}
_ALLOWED_DATA_PREFIXES = (
    "data/annotations/camera/",
    "data/manifests/",
    "data/tables/",
    "data/videos/",
    "data/viewer/",
)
_DATE_TOKEN = re.compile(r"(?:^|/)(?:date=)?\d{4}-\d{2}-\d{2}(?:/|$)")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class IndusAdContractError(ValueError):
    """Raised when the integrated repository differs from its closed contract."""


def video_viewer_schema() -> pa.Schema:
    video = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    fields: list[pa.Field] = [pa.field("recording_id", pa.string())]
    fields.extend(pa.field(f"{camera}_video", video) for camera in _CAMERAS)
    fields.extend(pa.field(f"{camera}_video_privacy_status", pa.string()) for camera in _CAMERAS)
    return pa.schema(fields)


def camera_annotation_schema() -> pa.Schema:
    semantic = pa.list_(
        pa.struct(
            [
                pa.field("class_id", pa.int16()),
                pa.field("fraction", pa.float32()),
            ]
        )
    )
    fields = [
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("camera_name", pa.string(), nullable=False),
        pa.field("selected_time_of_day_ns", pa.int64()),
        pa.field("recording_offset_ns", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("frame_id", pa.string(), nullable=False),
        pa.field("sample_kind", pa.string(), nullable=False),
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("decoded_width", pa.int32()),
        pa.field("decoded_height", pa.int32()),
        pa.field("semantic_fractions", semantic),
    ]
    fields.extend(
        pa.field(name, pa.float32())
        for name in (
            "drivable_fraction",
            "lane_marking_fraction",
            "person_rider_fraction",
            "two_wheeler_fraction",
            "four_wheeler_fraction",
            "building_fraction",
            "vegetation_fraction",
            "sky_fraction",
            "depth_p10",
            "depth_p50",
            "depth_p90",
            "weather_clear_probability",
            "weather_overcast_probability",
            "weather_rain_probability",
            "weather_fog_probability",
            "weather_low_light_probability",
            "weather_uncertain_mixed_probability",
        )
    )
    fields.extend(
        [
            pa.field("weather_label", pa.string()),
            pa.field("weather_confidence", pa.float32()),
            pa.field("weather_margin", pa.float32()),
            pa.field("weather_uncertain", pa.bool_()),
        ]
    )
    fields.extend(
        pa.field(name, pa.float32())
        for name in (
            "luminance_mean",
            "luminance_p05",
            "luminance_p95",
            "contrast_std",
            "saturation_mean",
            "laplacian_variance",
            "dark_fraction",
            "overexposed_fraction",
        )
    )
    return pa.schema(fields)


def dense_index_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("artifact_kind", pa.string()),
            pa.field("camera_name", pa.string()),
            pa.field("dtype", pa.string()),
            pa.field("frame_index", pa.int64()),
            pa.field("height", pa.int64()),
            pa.field("member_path", pa.string()),
            pa.field("payload_bytes", pa.int64()),
            pa.field("payload_sha256", pa.string()),
            pa.field("recording_id", pa.string()),
            pa.field("recording_offset_ns", pa.int64()),
            pa.field("shard_path", pa.string()),
            pa.field("width", pa.int64()),
        ]
    )


def _read_bytes(root: Path, repo_path: str, tree: Mapping[str, RepositoryFile]) -> bytes:
    identity = tree.get(repo_path)
    if identity is None:
        raise IndusAdContractError(f"required repository file is missing: {repo_path}")
    path = root / repo_path
    try:
        data = path.read_bytes()
    except OSError as error:
        raise IndusAdContractError(f"required control file is unavailable: {repo_path}") from error
    if len(data) != identity.size:
        raise IndusAdContractError(f"tree size differs for {repo_path}")
    return data


def _load_json(root: Path, repo_path: str, tree: Mapping[str, RepositoryFile]) -> dict[str, object]:
    try:
        value = json.loads(_read_bytes(root, repo_path, tree))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IndusAdContractError(f"invalid JSON control: {repo_path}") from error
    if not isinstance(value, dict):
        raise IndusAdContractError(f"JSON control must be an object: {repo_path}")
    return value


def _load_jsonl(
    root: Path, repo_path: str, tree: Mapping[str, RepositoryFile]
) -> list[dict[str, object]]:
    try:
        lines = _read_bytes(root, repo_path, tree).decode("utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IndusAdContractError(f"invalid JSONL control: {repo_path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise IndusAdContractError(f"JSONL control contains invalid rows: {repo_path}")
    return cast(list[dict[str, object]], rows)


def _load_parquet(
    root: Path,
    repo_path: str,
    expected_schema: pa.Schema,
    tree: Mapping[str, RepositoryFile],
) -> pa.Table:
    _read_bytes(root, repo_path, tree)
    try:
        parquet = pq.ParquetFile(root / repo_path)
        actual = parquet.schema_arrow.remove_metadata()
        if not actual.equals(expected_schema, check_metadata=False):
            raise IndusAdContractError(f"unexpected Parquet schema: {repo_path}")
        return parquet.read()
    except IndusAdContractError:
        raise
    except Exception as error:
        raise IndusAdContractError(f"invalid Parquet control: {repo_path}") from error


def _validated_tree(files: Sequence[RepositoryFile]) -> dict[str, RepositoryFile]:
    result: dict[str, RepositoryFile] = {}
    for item in files:
        if item.path in result:
            raise IndusAdContractError("repository tree contains a duplicate path")
        if (
            item.path.startswith("data/")
            and item.path != METADATA_PATH
            and not item.path.startswith(_ALLOWED_DATA_PREFIXES)
        ):
            raise IndusAdContractError(f"unknown data path family: {item.path}")
        if _DATE_TOKEN.search(item.path):
            raise IndusAdContractError(f"repository path contains an absolute date: {item.path}")
        result[item.path] = item
    missing = _REQUIRED_CONTROLS - set(result)
    if missing:
        raise IndusAdContractError(f"repository tree lacks required controls: {sorted(missing)!r}")
    return result


def _recording_id(row: Mapping[str, object], label: str) -> str:
    value = row.get("recording_id")
    try:
        if not isinstance(value, str) or validate_safe_segment(value) != value:
            raise ValueError
    except ValueError as error:
        raise IndusAdContractError(f"{label} recording identity is invalid") from error
    return value


def _index_rows(rows: Iterable[dict[str, object]], label: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        recording_id = _recording_id(row, label)
        if recording_id in result:
            raise IndusAdContractError(f"{label} contains a duplicate recording")
        result[recording_id] = row
    return result


def _repo_video_path(value: object, source: IndusAdHfSourceConfig) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise IndusAdContractError("video path must be a string or null")
    if value.startswith("hf://"):
        parsed = urlsplit(value)
        expected_prefix = f"datasets/{source.repo_id}@"
        if parsed.netloc != "datasets" or not parsed.path.startswith(f"/{source.repo_id}@"):
            raise IndusAdContractError("Viewer video URL targets a foreign repository")
        marker = "/data/"
        if marker not in parsed.path:
            raise IndusAdContractError("Viewer video URL lacks a data path")
        value = "data/" + parsed.path.split(marker, 1)[1]
        del expected_prefix
    elif value.startswith("videos/"):
        value = f"data/{value}"
    try:
        validate_repo_file_path(value)
    except RepositoryPathError as error:
        raise IndusAdContractError("invalid video repository path") from error
    if not value.startswith("data/videos/") or not value.endswith(".mp4"):
        raise IndusAdContractError("video path is outside the allowed family")
    return value


def _viewer_video_path(value: object, source: IndusAdHfSourceConfig) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"bytes", "path"}:
        raise IndusAdContractError("Viewer video cell has an invalid shape")
    if value["bytes"] is not None:
        raise IndusAdContractError("Viewer must not embed video bytes")
    return _repo_video_path(value["path"], source)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_privacy_manifest(
    root: Path,
    source: IndusAdHfSourceConfig,
    tree: Mapping[str, RepositoryFile],
) -> dict[str, object]:
    value = _load_json(root, VIDEO_PRIVACY_PATH, tree)
    count_fields = (
        "recording_count",
        "video_count",
        "face_plate_blurred_video_count",
        "missing_camera_slot_count",
        "not_face_plate_blurred_video_count",
    )
    if (
        value.get("repo_id") != source.repo_id
        or value.get("schema_version") != 2
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or cast(int, value.get(field)) < 0
            for field in count_fields
        )
        or value.get("recording_count") == 0
        or value.get("video_count") != value.get("face_plate_blurred_video_count")
        or value.get("not_face_plate_blurred_video_count") != 0
    ):
        raise IndusAdContractError("privacy release manifest has unexpected identity or counts")
    files = value.get("files")
    if not isinstance(files, list):
        raise IndusAdContractError("privacy release manifest lacks file bindings")
    bindings: dict[str, tuple[int, str]] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"bytes", "path", "sha256"}:
            raise IndusAdContractError("privacy release file binding is invalid")
        path, size, digest = item["path"], item["bytes"], item["sha256"]
        if (
            not isinstance(path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or path in bindings
        ):
            raise IndusAdContractError("privacy release file binding is invalid")
        bindings[path] = (size, digest)
    for path in (METADATA_PATH, VIDEO_INTEGRITY_PATH, VIDEO_VIEWER_PATH):
        data = _read_bytes(root, path, tree)
        if bindings.get(path) != (len(data), _sha256_bytes(data)):
            raise IndusAdContractError(f"privacy release digest differs for {path}")
    return value


def _artifact(
    path: str,
    kind: str,
    tree: Mapping[str, RepositoryFile],
    *,
    camera: CameraName | None = None,
    sha256: str | None = None,
) -> IndusAdArtifact:
    identity = tree.get(path)
    if identity is None:
        raise IndusAdContractError(f"selected artifact is absent from tree: {path}")
    return IndusAdArtifact(
        identity.path,
        identity.size,
        identity.blob_id,
        cast(ArtifactKind, kind),
        camera,
        sha256 or identity.sha256,
    )


def _one_recording_table(
    tree: Mapping[str, RepositoryFile],
    prefix: str,
    recording_id: str,
    *,
    camera: CameraName | None = None,
) -> str:
    matches = [
        path
        for path in tree
        if path.startswith(prefix)
        and PurePosixPath(path).stem.endswith(recording_id)
        and (camera is None or f"/camera={camera}/" in path)
    ]
    if len(matches) != 1:
        raise IndusAdContractError(f"recording requires exactly one {prefix.rstrip('/')} table")
    return matches[0]


def _schema_digest(schema: pa.Schema) -> str:
    fields = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]
    return canonical_hash(fields)


def parse_indus_ad_contract(
    root: Path,
    source: IndusAdHfSourceConfig,
    files: Sequence[RepositoryFile],
) -> IndusAdSelectionPlan:
    """Validate pinned controls and return an exact selected-artifact plan."""
    tree = _validated_tree(files)
    privacy = _validate_privacy_manifest(root, source, tree)
    metadata = _index_rows(_load_jsonl(root, METADATA_PATH, tree), "metadata")
    integrity = _index_rows(_load_jsonl(root, VIDEO_INTEGRITY_PATH, tree), "integrity")
    viewer_table = _load_parquet(root, VIDEO_VIEWER_PATH, video_viewer_schema(), tree)
    viewer = _index_rows(cast(list[dict[str, object]], viewer_table.to_pylist()), "videos Viewer")
    annotation_table = _load_parquet(
        root,
        CAMERA_ANNOTATION_VIEWER_PATH,
        camera_annotation_schema(),
        tree,
    )
    dense_paths = sorted(
        path
        for path in tree
        if path.startswith("data/annotations/camera/dense-index/") and path.endswith(".parquet")
    )
    if not dense_paths:
        raise IndusAdContractError("repository lacks dense indexes")
    for path in dense_paths:
        _load_parquet(root, path, dense_index_schema(), tree)
    if not (set(metadata) == set(integrity) == set(viewer)):
        raise IndusAdContractError("privacy controls contain different recording identities")
    if len(metadata) != privacy.get("recording_count"):
        raise IndusAdContractError("privacy manifest recording count differs")

    recording_tables: dict[str, dict[str, RepositoryFile]] = {
        recording_id: {} for recording_id in metadata
    }
    identity_lengths = {len(recording_id) for recording_id in metadata}
    for path, identity in tree.items():
        if not path.startswith("data/tables/"):
            continue
        stem = PurePosixPath(path).stem
        for length in identity_lengths:
            candidate = stem[-length:]
            if candidate in recording_tables:
                recording_tables[candidate][path] = identity

    privacy_counts = {"face_plate_blurred": 0, "missing": 0}
    entries: list[IndusAdRecordingEntry] = []
    for recording_id in sorted(metadata):
        meta = metadata[recording_id]
        video_row = viewer[recording_id]
        integrity_row = integrity[recording_id]
        videos: list[IndusAdArtifact] = []
        for camera in _CAMERAS:
            viewer_path = _viewer_video_path(video_row.get(f"{camera}_video"), source)
            metadata_path = _repo_video_path(meta.get(f"{camera}_video_file_name"), source)
            integrity_path = _repo_video_path(
                integrity_row.get(f"{camera}_video_file_name"), source
            )
            viewer_status = video_row.get(f"{camera}_video_privacy_status")
            metadata_status = meta.get(f"{camera}_video_privacy_status")
            if viewer_path is None:
                if (
                    camera != "cam_front_tilted"
                    or metadata_path is not None
                    or integrity_path is not None
                    or viewer_status != "missing"
                    or metadata_status != "missing"
                ):
                    raise IndusAdContractError("missing-camera privacy controls disagree")
                privacy_counts["missing"] += 1
                continue
            if (
                viewer_path != metadata_path
                or viewer_path != integrity_path
                or viewer_status != "face_plate_blurred"
                or metadata_status != "face_plate_blurred"
            ):
                raise IndusAdContractError("Viewer privacy status or video identity differs")
            size = integrity_row.get(f"{camera}_video_byte_size")
            digest = integrity_row.get(f"{camera}_video_sha256")
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
            ):
                raise IndusAdContractError("video integrity identity is invalid")
            tree_identity = tree.get(viewer_path)
            if tree_identity is None or tree_identity.size != size:
                raise IndusAdContractError("video tree size differs from integrity control")
            if camera in source.cameras:
                videos.append(_artifact(viewer_path, "video", tree, camera=camera, sha256=digest))
            privacy_counts["face_plate_blurred"] += 1
        if any(camera not in {video.camera for video in videos} for camera in source.cameras):
            raise IndusAdContractError("selected camera is absent from a recording")

        selected_by_id = not source.recording_ids or recording_id in source.recording_ids
        selected_by_split = not source.splits or "train" in source.splits
        if not (selected_by_id and selected_by_split):
            continue
        recording_tree = recording_tables[recording_id]
        artifacts: list[IndusAdArtifact] = []
        for camera in source.cameras:
            path = _one_recording_table(
                recording_tree,
                "data/tables/camera-frames/",
                recording_id,
                camera=camera,
            )
            artifacts.append(_artifact(path, "camera_frames", tree, camera=camera))
        artifacts.extend(
            [
                _artifact(
                    _one_recording_table(recording_tree, "data/tables/calibrations/", recording_id),
                    "calibration",
                    tree,
                ),
                _artifact(
                    _one_recording_table(recording_tree, "data/tables/gnss/", recording_id),
                    "gnss",
                    tree,
                ),
                _artifact(CAMERA_ANNOTATION_VIEWER_PATH, "camera_annotations", tree),
            ]
        )
        artifacts.extend(_artifact(path, "dense_index", tree) for path in dense_paths)
        selected_identity: list[dict[str, object]] = [
            {
                "path": item.path,
                "size": item.size,
                "blob_id": item.blob_id,
                "kind": item.kind,
                "camera": item.camera,
            }
            for item in (*videos, *artifacts)
        ]
        selected_identity.sort(key=lambda item: cast(str, item["path"]))
        annotation_digest = _schema_digest(annotation_table.schema.remove_metadata())
        fingerprint = IndusAdSourceFingerprint(
            repo_id=source.repo_id,
            revision=source.revision,
            contract=source.contract,
            recording_id=recording_id,
            catalog_sha256=canonical_hash(
                {"metadata": meta, "viewer": video_row, "integrity": integrity_row}
            ),
            selected_artifacts_sha256=canonical_hash(selected_identity),
            annotation_schema_sha256=annotation_digest,
            total_size=sum(cast(int, item["size"]) for item in selected_identity),
        )
        entries.append(
            IndusAdRecordingEntry(
                recording_id,
                "train",
                tuple(videos),
                tuple(artifacts),
                fingerprint,
            )
        )

    expected_counts = {
        "face_plate_blurred": privacy.get("face_plate_blurred_video_count"),
        "missing": privacy.get("missing_camera_slot_count"),
    }
    if privacy_counts != expected_counts:
        raise IndusAdContractError("privacy manifest and Viewer counts differ")
    if not entries:
        raise IndusAdContractError("integrated selection matched no recordings")
    if source.recording_ids and {item.recording_id for item in entries} != set(
        source.recording_ids
    ):
        raise IndusAdContractError("integrated selection is missing requested recordings")
    annotation_digest = _schema_digest(annotation_table.schema.remove_metadata())
    return IndusAdSelectionPlan(
        source.repo_id,
        source.revision,
        source.contract,
        privacy_counts,
        annotation_digest,
        tuple(entries),
    )
