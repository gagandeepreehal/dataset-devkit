"""Deterministic Parquet extensions for selected INDUS-AD weak labels."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from dataset_devkit.config import GlobalConfigV21
from dataset_devkit.indus_ad_contract import camera_annotation_schema as source_camera_schema
from dataset_devkit.indus_ad_join import JoinedIndusAdAnnotations
from dataset_devkit.indus_ad_models import (
    CameraAnnotation,
    DenseAssetRef,
    GnssScenarioAnnotation,
)
from dataset_devkit.provenance import IndusAdSourceFingerprint


def camera_annotation_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source_digest", pa.string(), nullable=False),
            pa.field("scene_token", pa.string(), nullable=False),
            pa.field("sample_token", pa.string(), nullable=False),
            pa.field("sample_data_token", pa.string(), nullable=False),
            *source_camera_schema(),
            pa.field("weak_label_provenance", pa.string(), nullable=False),
        ]
    )


def gnss_annotation_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source_digest", pa.string(), nullable=False),
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
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("scene_tokens", pa.list_(pa.string()), nullable=False),
            pa.field("sample_tokens", pa.list_(pa.string()), nullable=False),
            pa.field("weak_label_provenance", pa.string(), nullable=False),
        ]
    )


def dense_asset_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source_digest", pa.string(), nullable=False),
            pa.field("scene_token", pa.string(), nullable=False),
            pa.field("sample_data_token", pa.string(), nullable=False),
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("camera_name", pa.string(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("recording_offset_ns", pa.int64(), nullable=False),
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("kind", pa.string(), nullable=False),
            pa.field("dtype", pa.string(), nullable=False),
            pa.field("width", pa.int64(), nullable=False),
            pa.field("height", pa.int64(), nullable=False),
            pa.field("member_path", pa.string(), nullable=False),
            pa.field("payload_bytes", pa.int64(), nullable=False),
            pa.field("payload_sha256", pa.string(), nullable=False),
            pa.field("shard_path", pa.string(), nullable=False),
            pa.field("materialized", pa.bool_(), nullable=False),
        ]
    )


def _camera_row(item: object) -> dict[str, object]:
    from dataset_devkit.indus_ad_join import JoinedCameraAnnotation

    if not isinstance(item, JoinedCameraAnnotation):
        raise TypeError("expected joined camera annotation")
    annotation = item.annotation
    row: dict[str, object] = {
        "source_digest": item.source_digest,
        "scene_token": item.scene_token,
        "sample_token": item.sample_token,
        "sample_data_token": item.sample_data_token,
        "recording_id": annotation.recording_id,
        "camera_name": annotation.camera_name,
        "selected_time_of_day_ns": annotation.selected_time_of_day_ns,
        "recording_offset_ns": annotation.recording_offset_ns,
        "frame_index": annotation.frame_index,
        "frame_id": annotation.frame_id,
        "sample_kind": annotation.sample_kind,
        "window_id": annotation.window_id,
        "decoded_width": annotation.decoded_width,
        "decoded_height": annotation.decoded_height,
        "semantic_fractions": [asdict(value) for value in annotation.semantic_fractions],
        "weather_label": annotation.weather.label,
        "weather_confidence": annotation.weather.confidence,
        "weather_margin": annotation.weather.margin,
        "weather_uncertain": annotation.weather.uncertain,
        "weak_label_provenance": annotation.provenance,
    }
    row.update({f"{name}_fraction": value for name, value in annotation.category_fractions.items()})
    row.update({f"depth_{name}": value for name, value in annotation.depth_summary_m.items()})
    row.update(
        {
            f"weather_{name}_probability": value
            for name, value in annotation.weather.probabilities.items()
        }
    )
    row.update(annotation.image_quality.values)
    return row


def _assert_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("INDUS-AD extension contains a non-finite value")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite(nested)


def _parquet_bytes(rows: list[dict[str, object]], schema: pa.Schema) -> bytes:
    for row in rows:
        _assert_finite(row)
    table = pa.Table.from_pylist(rows, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        write_statistics=False,
        version="2.6",
    )
    return cast(bytes, sink.getvalue().to_pybytes())


@dataclass(frozen=True, slots=True)
class SerializedIndusAdExtensions:
    contract: dict[str, object]
    camera_annotations: bytes
    gnss_annotations: bytes
    dense_assets: bytes


@dataclass(frozen=True, slots=True)
class CameraAnnotationRecord:
    annotation: CameraAnnotation
    source_digest: str
    scene_token: str
    sample_token: str
    sample_data_token: str

    @property
    def frame_id(self) -> str:
        return self.annotation.frame_id

    @property
    def recording_id(self) -> str:
        return self.annotation.recording_id

    @property
    def camera_name(self) -> str:
        return self.annotation.camera_name


@dataclass(frozen=True, slots=True)
class GnssAnnotationRecord:
    annotation: GnssScenarioAnnotation
    source_digest: str
    scene_tokens: tuple[str, ...]
    sample_tokens: tuple[str, ...]

    @property
    def recording_id(self) -> str:
        return self.annotation.recording_id

    @property
    def window_index(self) -> int:
        return self.annotation.window_index


@dataclass(frozen=True, slots=True)
class DenseAssetRecord:
    ref: DenseAssetRef
    source_digest: str
    scene_token: str
    sample_data_token: str
    materialized: bool

    @property
    def frame_id(self) -> str:
        return self.ref.frame_id

    @property
    def kind(self) -> str:
        return self.ref.kind


def serialize_indus_ad_extensions(
    annotations: JoinedIndusAdAnnotations,
    config: GlobalConfigV21,
    fingerprints: tuple[IndusAdSourceFingerprint, ...],
) -> SerializedIndusAdExtensions:
    """Serialize sorted extension rows and their immutable source contract."""
    if not fingerprints:
        raise ValueError("integrated export requires source fingerprints")
    schema_digests = {item.annotation_schema_sha256 for item in fingerprints}
    if (
        {item.revision for item in fingerprints} != {config.source.revision}
        or {item.repo_id for item in fingerprints} != {config.source.repo_id}
        or {item.contract for item in fingerprints} != {config.source.contract}
        or len(schema_digests) != 1
    ):
        raise ValueError("integrated fingerprints differ from the resolved source contract")
    camera_rows = [
        _camera_row(item)
        for item in sorted(
            annotations.camera,
            key=lambda value: (value.sample_data_token, value.annotation.frame_id),
        )
    ]
    selected_layers = set(config.source.annotation_layers)
    for row in camera_rows:
        if "semantic_fractions" not in selected_layers:
            row["semantic_fractions"] = []
            for category in (
                "drivable",
                "lane_marking",
                "person_rider",
                "two_wheeler",
                "four_wheeler",
                "building",
                "vegetation",
                "sky",
            ):
                row[f"{category}_fraction"] = None
        if "depth_summary" not in selected_layers:
            for key in ("depth_p10", "depth_p50", "depth_p90"):
                row[key] = None
        if "weather" not in selected_layers:
            for key in tuple(row):
                if key.startswith("weather_"):
                    row[key] = None
    if "camera_scalar" not in selected_layers:
        camera_rows = []
    gnss_rows = [
        {
            "source_digest": item.source_digest,
            "recording_id": item.annotation.recording_id,
            "window_index": item.annotation.window_index,
            "window_start_offset_ns": item.annotation.window_start_offset_ns,
            "window_end_offset_ns": item.annotation.window_end_offset_ns,
            "motion_label": item.annotation.motion_label,
            "validity_tags": list(item.annotation.validity_tags),
            "quality_tags": list(item.annotation.quality_tags),
            "anomaly_candidate": item.annotation.anomaly_candidate,
            "anomaly_reasons": list(item.annotation.anomaly_reasons),
            "confidence": item.annotation.confidence,
            "schema_version": item.annotation.schema_version,
            "run_id": item.annotation.run_id,
            "scene_tokens": list(item.scene_tokens),
            "sample_tokens": list(item.sample_tokens),
            "weak_label_provenance": item.annotation.provenance,
        }
        for item in sorted(
            annotations.gnss,
            key=lambda value: (
                value.annotation.recording_id,
                value.annotation.window_index,
            ),
        )
    ]
    dense_rows = [
        {
            "source_digest": item.source_digest,
            "scene_token": item.scene_token,
            "sample_data_token": item.sample_data_token,
            **asdict(item.ref),
            "materialized": item.materialized_path is not None,
        }
        for item in sorted(
            annotations.dense,
            key=lambda value: (value.sample_data_token, value.ref.frame_id, value.ref.kind),
        )
        if "camera_scalar" in selected_layers
    ]
    contract = {
        "schema_version": 1,
        "contract": config.source.contract,
        "source_repo_id": config.source.repo_id,
        "source_revision": config.source.revision,
        "annotation_schema_sha256": next(iter(schema_digests)),
        "selected_layers": list(config.source.annotation_layers),
        "camera_rows": len(camera_rows),
        "gnss_rows": len(gnss_rows),
        "dense_rows": len(dense_rows),
        "join_audit": {
            item.recording_id: asdict(item)
            for item in sorted(annotations.audits, key=lambda value: value.recording_id)
        },
        "weak_label_provenance": "machine_generated_weak_pseudo_label",
    }
    return SerializedIndusAdExtensions(
        contract,
        _parquet_bytes(camera_rows, camera_annotation_schema()),
        _parquet_bytes(gnss_rows, gnss_annotation_schema()),
        _parquet_bytes(dense_rows, dense_asset_schema()),
    )
