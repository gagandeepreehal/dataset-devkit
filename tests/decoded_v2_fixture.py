from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

from dataset_devkit.config import (
    DEFAULT_DECODED_CAMERAS,
    CameraName,
    DecodedHfSourceConfig,
)

_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class DecodedFixture:
    root: Path
    revision: str
    recording_id: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_video(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=1)
        stream.width = 4
        stream.height = 4
        stream.pix_fmt = "yuv420p"
        for frame_index in range(2):
            array = np.full((4, 4, 3), color + frame_index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = frame_index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows),
        path,
        compression="zstd",
        write_statistics=False,
    )


def write_tiny_videos_and_tables(
    root: Path, recording_id: str, *, include_tilted: bool = False
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    cameras = (
        (*DEFAULT_DECODED_CAMERAS, "cam_front_tilted")
        if include_tilted
        else DEFAULT_DECODED_CAMERAS
    )
    videos: dict[str, dict[str, object]] = {}
    for camera_index, camera in enumerate(cameras):
        video_path = root / "data" / "videos" / camera / "bucket=fixture" / f"{recording_id}.mp4"
        _write_video(video_path, 10 + camera_index)
        videos[camera] = _artifact(root, video_path)

    camera_frames_path = (
        root
        / "data"
        / "tables"
        / "camera_frames"
        / "bucket=fixture"
        / f"part-{recording_id}.parquet"
    )
    _write_table(
        camera_frames_path,
        [
            {
                "recording_id": recording_id,
                "camera_name": camera,
                "frame_index": frame_index,
                "recording_offset_ns": frame_index * 1_000_000_000,
                "width": 4,
                "height": 4,
            }
            for camera in cameras
            for frame_index in range(2)
        ],
    )

    calibration_path = (
        root
        / "data"
        / "tables"
        / "calibration"
        / "bucket=fixture"
        / f"part-{recording_id}.parquet"
    )
    _write_table(
        calibration_path,
        [
            {
                "recording_id": recording_id,
                "camera_name": camera,
                "width": 4,
                "height": 4,
                "focal_length_x": 2.0,
                "focal_length_y": 2.0,
                "optical_center_x": 2.0,
                "optical_center_y": 2.0,
                "skew": 0.0,
                "rmse": 0.0,
                "distortion_coeffs": [],
                "rotation_vector": [0.0, 0.0, 0.0],
                "translation_vector": [0.0, 0.0, 0.0],
            }
            for camera in cameras
        ],
    )

    gnss_path = (
        root
        / "data"
        / "tables"
        / "gnss"
        / "bucket=fixture"
        / f"part-{recording_id}.parquet"
    )
    _write_table(
        gnss_path,
        [
            {
                "recording_id": recording_id,
                "recording_offset_ns": offset,
                "is_valid": True,
                "local_enu": {
                    "local_east_m": float(index * 10),
                    "local_north_m": float(index * 4),
                    "local_up_m": float(index) / 2.0,
                },
                "orientation": {"roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.1},
                "enu": {"east_m": 8_000_000.0 + index * 10, "north_m": 1_500_000.0 + index * 4},
                "lat_lon_ht": {
                    "latitude_deg": _TO_WGS84.transform(
                        8_000_000.0 + index * 10, 1_500_000.0 + index * 4
                    )[1],
                    "longitude_deg": _TO_WGS84.transform(
                        8_000_000.0 + index * 10, 1_500_000.0 + index * 4
                    )[0],
                    "height_m": 900.25 + index / 2.0,
                },
                "position_uncertainty": {"sigma_m": 0.1},
                "orientation_uncertainty": {"variance": 0.01},
                "quality": {"fix": "rtk", "satellites": 18, "hdop": 0.7},
                "local_enu_origin_offset_ns": 0,
                "published_horizontal_resolution_m": 1.0,
            }
            for index, offset in enumerate((0, 1_000_000_000))
        ],
    )

    artifacts = {
        "camera_frames": [_artifact(root, camera_frames_path)],
        "calibration": [_artifact(root, calibration_path)],
        "gnss": [_artifact(root, gnss_path)],
    }
    return videos, artifacts


def write_recordings_parquet(
    root: Path,
    recording_id: str,
    videos: dict[str, dict[str, object]],
    artifacts: dict[str, list[dict[str, object]]],
) -> Path:
    row: dict[str, object] = {
        "recording_id": recording_id,
        "split": "train",
        "default_cameras": list(DEFAULT_DECODED_CAMERAS),
        "available_cameras": sorted(videos),
        "video_paths": {key: value["path"] for key, value in videos.items()},
        "video_sizes": {key: value["size"] for key, value in videos.items()},
        "video_sha256": {key: value["sha256"] for key, value in videos.items()},
        "frame_counts": {key: 2 for key in videos},
        "duration_ns": 1_000_000_000,
        "start_time_of_day_ns": 10_000_000_000,
        "end_time_of_day_ns": 11_000_000_000,
        "start_offset_ns": 0,
        "end_offset_ns": 1_000_000_000,
        "artifact_paths": {
            key: [item["path"] for item in values] for key, values in artifacts.items()
        },
        "artifact_sizes": {
            key: [item["size"] for item in values] for key, values in artifacts.items()
        },
        "artifact_sha256": {
            key: [item["sha256"] for item in values] for key, values in artifacts.items()
        },
        "has_gnss": True,
        "has_calibration": True,
        "has_camera_labels": False,
        "has_semantic": False,
        "has_depth": False,
        "published_horizontal_resolution_m": 1.0,
    }
    for camera in (*DEFAULT_DECODED_CAMERAS, "cam_front_tilted"):
        row[f"has_{camera}"] = camera in videos
    path = root / "data" / "recordings.parquet"
    _write_table(path, [row])
    return path


def write_public_manifests(
    root: Path,
    videos: dict[str, dict[str, object]],
    artifacts: dict[str, list[dict[str, object]]],
) -> None:
    manifest_root = root / "data" / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    output_rows = [*videos.values()]
    for values in artifacts.values():
        output_rows.extend(values)
    _write_table(
        manifest_root / "output-files.parquet",
        sorted(output_rows, key=lambda item: str(item["path"])),
    )
    (manifest_root / "processing-config.json").write_text(
        json.dumps(
            {
                "global_horizontal_resolution_m": 1.0,
                "local_enu_precision": "retained",
                "run_id": "fixture-run",
                "temporal_config_sha256": "3" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (manifest_root / "model-receipt.json").write_text(
        json.dumps(
            {"model_lock_sha256": "1" * 64, "privacy_config_sha256": "2" * 64},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_table(
        manifest_root / "schema-audit.parquet",
        [
            {
                "audit_json": json.dumps(
                    {
                        "absolute_dates_removed": True,
                        "exact_recording_offsets_retained": True,
                        "published_horizontal_resolution_m": 1.0,
                    },
                    sort_keys=True,
                )
            }
        ],
    )


def write_decoded_fixture(root: Path, *, include_tilted: bool = False) -> DecodedFixture:
    recording_id = "recording-code"
    revision = "0123456789abcdef0123456789abcdef01234567"
    videos, artifacts = write_tiny_videos_and_tables(
        root, recording_id, include_tilted=include_tilted
    )
    write_recordings_parquet(root, recording_id, videos, artifacts)
    write_public_manifests(root, videos, artifacts)
    return DecodedFixture(root, revision, recording_id)


def rewrite_as_anonymizer_layout(fixture: DecodedFixture) -> None:
    """Rewrite fixture metadata to the production anonymizer's sharded schemas."""
    catalog_path = fixture.root / "data/recordings.parquet"
    catalog_row = pq.ParquetFile(catalog_path).read().to_pylist()[0]
    simple_frame_path = Path(catalog_row["artifact_paths"]["camera_frames"][0])
    simple_frames = pq.ParquetFile(fixture.root / simple_frame_path).read().to_pylist()
    simple_calibration_path = Path(catalog_row["artifact_paths"]["calibration"][0])
    simple_calibrations = pq.ParquetFile(
        fixture.root / simple_calibration_path
    ).read().to_pylist()
    simple_gnss_path = Path(catalog_row["artifact_paths"]["gnss"][0])
    simple_gnss = pq.ParquetFile(fixture.root / simple_gnss_path).read().to_pylist()

    frame_artifacts: list[dict[str, object]] = []
    for camera_index, camera in enumerate(DEFAULT_DECODED_CAMERAS):
        path = (
            fixture.root
            / "data/tables/camera-frames"
            / f"camera={camera}"
            / "bucket=fixture"
            / f"part-{fixture.recording_id}.parquet"
        )
        rows = []
        for source in simple_frames:
            if source["camera_name"] != camera:
                continue
            packet_index = int(source["frame_index"])
            batch_offset = int(source["recording_offset_ns"])
            camera_offset = (
                batch_offset + camera_index * 10
                if packet_index == 0
                else batch_offset - camera_index * 10
            )
            rows.append(
                {
                    "schema_version": 2,
                    "recording_id": fixture.recording_id,
                    "camera_index": camera_index,
                    "source_camera_name": camera,
                    "camera_time_of_day_ns": 10_000_000_000 + camera_offset,
                    "camera_time_offset_ns": camera_offset,
                    "frame_id": packet_index + 100,
                    "rec_frame_id": packet_index + 200,
                    "rec_time_of_day_ns": 10_000_000_000 + batch_offset,
                    "rec_time_offset_ns": batch_offset,
                    "batch_time_of_day_ns": 10_000_000_000 + batch_offset,
                    "batch_time_offset_ns": batch_offset,
                    "declared_camera_count": len(DEFAULT_DECODED_CAMERAS),
                    "full_batch": True,
                    "log_time_of_day_ns": 10_000_000_000 + batch_offset,
                    "log_time_offset_ns": batch_offset,
                    "publish_time_of_day_ns": 10_000_000_000 + batch_offset,
                    "publish_time_offset_ns": batch_offset,
                    "channel": "rec_cameras",
                    "schema_hash": "a" * 64,
                    "sequence": packet_index,
                    "message_index": packet_index,
                    "format": "h265",
                    "width": source["width"],
                    "height": source["height"],
                    "file_name": catalog_row["video_paths"][camera],
                    "discontinuity_interval_id": 0,
                    "packet_index": packet_index,
                    "relative_pts_ns": batch_offset,
                    "relative_dts_ns": batch_offset,
                    "duration_ns": 1_000_000_000,
                    "keyframe": packet_index == 0,
                    "access_unit_sha256": "b" * 64,
                    "nal_payload_sha256": "c" * 64,
                }
            )
        _write_table(path, rows)
        frame_artifacts.append(_artifact(fixture.root, path))

    calibration_path = (
        fixture.root
        / "data/tables/calibrations/bucket=fixture"
        / f"part-{fixture.recording_id}.parquet"
    )
    production_calibrations = []
    for camera_index, source in enumerate(simple_calibrations):
        intrinsic = {
            key: source[key]
            for key in (
                "width",
                "height",
                "focal_length_x",
                "focal_length_y",
                "optical_center_x",
                "optical_center_y",
                "skew",
                "rmse",
                "distortion_coeffs",
            )
        }
        extrinsic = {
            key: source[key]
            for key in ("rotation_vector", "translation_vector")
        }
        production_calibrations.append(
            {
                "recording_id": fixture.recording_id,
                "camera": source["camera_name"],
                "camera_index": camera_index,
                "message_index": 0,
                "time_of_day_ns": 10_000_000_000,
                "time_offset_ns": 0,
                "intrinsic_json": json.dumps(intrinsic, sort_keys=True),
                "extrinsic_json": json.dumps(extrinsic, sort_keys=True),
            }
        )
    _write_table(calibration_path, production_calibrations)

    gnss_path = (
        fixture.root
        / "data/tables/gnss/bucket=fixture"
        / f"production-{fixture.recording_id}.parquet"
    )
    production_gnss = []
    for sequence, source in enumerate(simple_gnss):
        offset = int(source["recording_offset_ns"])
        fields = {
            key: value
            for key, value in source.items()
            if key not in {"recording_id", "recording_offset_ns"}
        }
        production_gnss.append(
            {
                "recording_id": fixture.recording_id,
                "log_time_of_day_ns": 10_000_000_000 + offset,
                "log_time_offset_ns": offset,
                "publish_time_of_day_ns": 10_000_000_000 + offset,
                "publish_time_offset_ns": offset,
                "channel": "gnss",
                "schema_hash": "d" * 64,
                "sequence": sequence,
                "fields_json": json.dumps(fields, sort_keys=True),
            }
        )
    _write_table(gnss_path, production_gnss)

    artifacts = {
        "camera_frames": frame_artifacts,
        "calibration": [_artifact(fixture.root, calibration_path)],
        "gnss": [_artifact(fixture.root, gnss_path)],
    }
    catalog_row["artifact_paths"] = {
        kind: [item["path"] for item in values] for kind, values in artifacts.items()
    }
    catalog_row["artifact_sizes"] = {
        kind: [item["size"] for item in values] for kind, values in artifacts.items()
    }
    catalog_row["artifact_sha256"] = {
        kind: [item["sha256"] for item in values] for kind, values in artifacts.items()
    }
    _write_table(catalog_path, [catalog_row])

    output_path = fixture.root / "data/manifests/output-files.parquet"
    video_paths = set(catalog_row["video_paths"].values())
    video_rows = [
        row
        for row in pq.ParquetFile(output_path).read().to_pylist()
        if row["path"] in video_paths
    ]
    artifact_rows = [item for values in artifacts.values() for item in values]
    _write_table(
        output_path,
        sorted([*video_rows, *artifact_rows], key=lambda item: str(item["path"])),
    )


def decoded_source_config(
    revision: str,
    recording_ids: list[str],
    *,
    cameras: list[CameraName] | None = None,
) -> DecodedHfSourceConfig:
    return DecodedHfSourceConfig(
        type="decoded_hf",
        repo_id="owner/dataset",
        revision=revision,
        recordings_path="data/recordings.parquet",
        recording_ids=recording_ids,
        splits=["train"],
        cameras=list(DEFAULT_DECODED_CAMERAS) if cameras is None else cameras,
    )
