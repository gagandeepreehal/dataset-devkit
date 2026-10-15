"""Strict parsers for machine-generated INDUS-AD weak pseudo-labels."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq

from dataset_devkit.config import CameraName
from dataset_devkit.indus_ad_contract import camera_annotation_schema, dense_index_schema
from dataset_devkit.indus_ad_models import (
    AnnotationAvailability,
    CameraAnnotation,
    DenseAssetRef,
    GnssScenarioAnnotation,
    ImageQualitySummary,
    IndusAdAnnotationBundle,
    SemanticFraction,
    WeatherWeakLabel,
)

_CAMERAS = {
    "cam_front",
    "cam_front_left",
    "cam_front_right",
    "cam_front_tilted",
    "cam_rear",
    "cam_rear_left",
    "cam_rear_right",
}
_FORBIDDEN_FIELD_TOKEN = re.compile(
    r"(?:^|_)(?:date|epoch|timestamp|latitude|longitude|lat|lon|coordinate|"
    r"geometry|endpoint|osm|road_name|gps_week|gps_day|day_of_year)(?:_|$)",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(?:^|[^0-9])\d{4}-\d{2}-\d{2}(?:[^0-9]|$)")


class IndusAdAnnotationError(ValueError):
    """Raised when weak-label data violates its closed public contract."""


def scan_forbidden_metadata(value: object, location: str = "root") -> None:
    """Recursively reject absolute-time, coordinate, and OSM metadata."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            text = str(key)
            if _FORBIDDEN_FIELD_TOKEN.search(text):
                raise IndusAdAnnotationError(f"forbidden metadata at {location}.{text}")
            scan_forbidden_metadata(nested, f"{location}.{text}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            scan_forbidden_metadata(nested, f"{location}[{index}]")
        return
    if isinstance(value, str):
        if _ISO_DATE.search(value):
            raise IndusAdAnnotationError(f"forbidden date material at {location}")
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                nested = json.loads(stripped)
            except json.JSONDecodeError:
                return
            scan_forbidden_metadata(nested, location)


def _validate_schema(path: Path, expected: pa.Schema, label: str) -> pq.ParquetFile:
    try:
        parquet = pq.ParquetFile(path)
    except Exception as error:
        raise IndusAdAnnotationError(f"invalid {label} Parquet") from error
    schema = parquet.schema_arrow.remove_metadata()
    if not schema.equals(expected, check_metadata=False):
        raise IndusAdAnnotationError(f"unexpected {label} schema")
    if parquet.schema_arrow.metadata:
        scan_forbidden_metadata(
            {
                key.decode("utf-8", errors="strict"): value.decode("utf-8", errors="strict")
                for key, value in parquet.schema_arrow.metadata.items()
            },
            f"{label}.schema_metadata",
        )
    return parquet


def _rows(parquet: pq.ParquetFile) -> Iterable[dict[str, object]]:
    for batch in parquet.iter_batches(batch_size=1024):
        yield from cast(list[dict[str, object]], batch.to_pylist())


def _optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise IndusAdAnnotationError(f"{label} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise IndusAdAnnotationError(f"{label} must be finite")
    return result


def _required_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise IndusAdAnnotationError(f"{label} must be an integer")
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise IndusAdAnnotationError(f"{label} must be a nonblank string")
    return value


def _camera_annotation(
    row: dict[str, object], expected_recordings: Mapping[str, int]
) -> CameraAnnotation:
    scan_forbidden_metadata(row, "camera_annotation")
    recording_id = _required_string(row["recording_id"], "recording_id")
    duration = expected_recordings.get(recording_id)
    if duration is None:
        raise IndusAdAnnotationError("camera annotation belongs to a foreign recording")
    camera = _required_string(row["camera_name"], "camera_name")
    if camera not in _CAMERAS:
        raise IndusAdAnnotationError("camera annotation has an unknown camera")
    offset = _required_int(row["recording_offset_ns"], "recording_offset_ns")
    if not 0 <= offset <= duration:
        raise IndusAdAnnotationError("camera annotation offset is outside the recording")
    raw_semantic = row["semantic_fractions"]
    if not isinstance(raw_semantic, list):
        raise IndusAdAnnotationError("semantic_fractions must be a list")
    semantic = tuple(
        SemanticFraction(
            _required_int(item.get("class_id"), "semantic class_id"),
            cast(float, _optional_float(item.get("fraction"), "semantic fraction")),
        )
        for item in raw_semantic
        if isinstance(item, dict)
    )
    if len(semantic) != len(raw_semantic):
        raise IndusAdAnnotationError("semantic_fractions contains an invalid item")
    categories = {
        name.removesuffix("_fraction"): _optional_float(row[name], name)
        for name in (
            "drivable_fraction",
            "lane_marking_fraction",
            "person_rider_fraction",
            "two_wheeler_fraction",
            "four_wheeler_fraction",
            "building_fraction",
            "vegetation_fraction",
            "sky_fraction",
        )
    }
    depth = {
        "p10": _optional_float(row["depth_p10"], "depth_p10"),
        "p50": _optional_float(row["depth_p50"], "depth_p50"),
        "p90": _optional_float(row["depth_p90"], "depth_p90"),
    }
    weather = WeatherWeakLabel(
        probabilities={
            name: _optional_float(row[f"weather_{name}_probability"], name)
            for name in (
                "clear",
                "overcast",
                "rain",
                "fog",
                "low_light",
                "uncertain_mixed",
            )
        },
        label=cast(str | None, row["weather_label"]),
        confidence=_optional_float(row["weather_confidence"], "weather_confidence"),
        margin=_optional_float(row["weather_margin"], "weather_margin"),
        uncertain=cast(bool | None, row["weather_uncertain"]),
    )
    quality = ImageQualitySummary(
        {
            name: _optional_float(row[name], name)
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
        }
    )
    try:
        return CameraAnnotation(
            recording_id=recording_id,
            camera_name=cast(CameraName, camera),
            selected_time_of_day_ns=cast(int | None, row["selected_time_of_day_ns"]),
            recording_offset_ns=offset,
            frame_index=_required_int(row["frame_index"], "frame_index"),
            frame_id=_required_string(row["frame_id"], "frame_id"),
            sample_kind=_required_string(row["sample_kind"], "sample_kind"),
            window_id=_required_string(row["window_id"], "window_id"),
            decoded_width=_required_int(row["decoded_width"], "decoded_width"),
            decoded_height=_required_int(row["decoded_height"], "decoded_height"),
            semantic_fractions=semantic,
            category_fractions=categories,
            depth_summary_m=depth,
            weather=weather,
            image_quality=quality,
        )
    except ValueError as error:
        raise IndusAdAnnotationError("invalid camera weak-label row") from error


def parse_camera_annotations(
    path: Path, *, expected_recordings: Mapping[str, int]
) -> IndusAdAnnotationBundle:
    """Parse the exact camera Viewer schema without weakening pseudo-label semantics."""
    parquet = _validate_schema(path, camera_annotation_schema(), "camera annotations")
    parsed: list[CameraAnnotation] = []
    identities: set[tuple[str, str]] = set()
    source_identities: set[tuple[str, CameraName, int, int]] = set()
    for row in _rows(parquet):
        annotation = _camera_annotation(row, expected_recordings)
        if annotation.identity in identities:
            raise IndusAdAnnotationError("duplicate camera frame identity")
        if annotation.source_frame_identity in source_identities:
            raise IndusAdAnnotationError("duplicate camera source-frame identity")
        identities.add(annotation.identity)
        source_identities.add(annotation.source_frame_identity)
        parsed.append(annotation)
    return IndusAdAnnotationBundle(
        camera=tuple(parsed),
        availability={
            "camera": AnnotationAvailability("available", "camera_annotations_loaded"),
            "gnss": AnnotationAvailability("not_selected", "gnss_not_selected"),
            "dense": AnnotationAvailability("not_selected", "dense_not_selected"),
        },
    )


def parse_dense_indexes(
    paths: Sequence[Path], *, camera_annotations: Sequence[CameraAnnotation]
) -> tuple[DenseAssetRef, ...]:
    """Parse dense member references without opening any referenced archive."""
    frame_bindings = {item.source_frame_identity: item.frame_id for item in camera_annotations}
    if len(frame_bindings) != len(camera_annotations):
        raise IndusAdAnnotationError("camera annotations contain duplicate source bindings")
    result: list[DenseAssetRef] = []
    identities: set[tuple[str, CameraName, int, int, str]] = set()
    for path in paths:
        parquet = _validate_schema(path, dense_index_schema(), "dense index")
        for row in _rows(parquet):
            scan_forbidden_metadata(row, "dense_index")
            camera = _required_string(row["camera_name"], "camera_name")
            if camera not in _CAMERAS:
                raise IndusAdAnnotationError("dense index has an unknown camera")
            binding = (
                _required_string(row["recording_id"], "recording_id"),
                cast(CameraName, camera),
                _required_int(row["frame_index"], "frame_index"),
                _required_int(row["recording_offset_ns"], "recording_offset_ns"),
            )
            frame_id = frame_bindings.get(binding)
            if frame_id is None:
                raise IndusAdAnnotationError("dense index lacks a camera source-frame binding")
            kind = row["artifact_kind"]
            if kind not in {"semantic", "depth"}:
                raise IndusAdAnnotationError("dense index has an unsupported artifact kind")
            try:
                reference = DenseAssetRef(
                    recording_id=binding[0],
                    camera_name=binding[1],
                    frame_index=binding[2],
                    recording_offset_ns=binding[3],
                    frame_id=frame_id,
                    kind=cast(Literal["semantic", "depth"], kind),
                    dtype=_required_string(row["dtype"], "dtype"),
                    width=_required_int(row["width"], "width"),
                    height=_required_int(row["height"], "height"),
                    member_path=_required_string(row["member_path"], "member_path"),
                    payload_bytes=_required_int(row["payload_bytes"], "payload_bytes"),
                    payload_sha256=_required_string(row["payload_sha256"], "payload_sha256"),
                    shard_path=_required_string(row["shard_path"], "shard_path"),
                )
            except ValueError as error:
                raise IndusAdAnnotationError("invalid dense-index row") from error
            if reference.identity in identities:
                raise IndusAdAnnotationError("duplicate dense asset identity")
            identities.add(reference.identity)
            result.append(reference)
    return tuple(result)


def gnss_scenario_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("source_revision", pa.string(), nullable=False),
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("window_index", pa.int64(), nullable=False),
            pa.field("window_start_offset_ns", pa.int64(), nullable=False),
            pa.field("window_end_offset_ns", pa.int64(), nullable=False),
            pa.field("motion_label", pa.string(), nullable=False),
            pa.field("validity_tags", pa.list_(pa.string()), nullable=False),
            pa.field("quality_tags", pa.list_(pa.string()), nullable=False),
            pa.field("anomaly_candidate", pa.bool_(), nullable=False),
            pa.field("anomaly_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("confidence", pa.float32(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
        ]
    )


def parse_gnss_scenarios(
    path: Path,
    *,
    expected_recordings: Mapping[str, int],
    expected_source_revision: str,
    selected_recording_ids: frozenset[str] | None = None,
) -> tuple[GnssScenarioAnnotation, ...]:
    """Parse privacy-minimized recording-relative GNSS weak labels."""
    parquet = _validate_schema(path, gnss_scenario_schema(), "GNSS annotations")
    result: list[GnssScenarioAnnotation] = []
    identities: set[tuple[str, int]] = set()
    for row in _rows(parquet):
        scan_forbidden_metadata(row, "gnss_annotation")
        if (
            row["schema_version"] != "indus_ad_gnss_annotations_v1"
            or row["source_revision"] != expected_source_revision
        ):
            raise IndusAdAnnotationError("GNSS annotation contract identity differs")
        recording_id = _required_string(row["recording_id"], "recording_id")
        if selected_recording_ids is not None and recording_id not in selected_recording_ids:
            continue
        duration = expected_recordings.get(recording_id)
        if duration is None:
            raise IndusAdAnnotationError("GNSS annotation belongs to a foreign recording")
        end = _required_int(row["window_end_offset_ns"], "window_end_offset_ns")
        if end > duration:
            raise IndusAdAnnotationError("GNSS annotation window exceeds recording")
        try:
            annotation = GnssScenarioAnnotation(
                recording_id=recording_id,
                window_index=_required_int(row["window_index"], "window_index"),
                window_start_offset_ns=_required_int(
                    row["window_start_offset_ns"], "window_start_offset_ns"
                ),
                window_end_offset_ns=end,
                motion_label=_required_string(row["motion_label"], "motion_label"),
                validity_tags=tuple(cast(list[str], row["validity_tags"])),
                quality_tags=tuple(cast(list[str], row["quality_tags"])),
                anomaly_candidate=cast(bool, row["anomaly_candidate"]),
                anomaly_reasons=tuple(cast(list[str], row["anomaly_reasons"])),
                confidence=cast(float, _optional_float(row["confidence"], "confidence")),
                schema_version="indus_ad_gnss_annotations_v1",
                run_id=_required_string(row["run_id"], "run_id"),
            )
        except (TypeError, ValueError) as error:
            raise IndusAdAnnotationError("invalid GNSS weak-label row") from error
        identity = (annotation.recording_id, annotation.window_index)
        if identity in identities:
            raise IndusAdAnnotationError("duplicate GNSS annotation window")
        identities.add(identity)
        result.append(annotation)
    return tuple(result)
