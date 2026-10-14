from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from dataset_devkit.config import DEFAULT_DECODED_CAMERAS, IndusAdHfSourceConfig
from dataset_devkit.indus_ad_contract import (
    CAMERA_ANNOTATION_VIEWER_PATH,
    METADATA_PATH,
    VIDEO_INTEGRITY_PATH,
    VIDEO_PRIVACY_PATH,
    VIDEO_VIEWER_PATH,
    camera_annotation_schema,
    dense_index_schema,
    video_viewer_schema,
)
from dataset_devkit.indus_ad_models import RepositoryFile


@dataclass(frozen=True)
class IndusAdFixture:
    root: Path
    source: IndusAdHfSourceConfig
    tree: tuple[RepositoryFile, ...]
    recording_id: str
    duration_ns: int
    camera_annotations: Path
    dense_index: Path


def _write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        write_statistics=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_tree(root: Path) -> tuple[RepositoryFile, ...]:
    items = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        items.append(
            RepositoryFile(
                path.relative_to(root).as_posix(),
                len(data),
                hashlib.sha1(data, usedforsecurity=False).hexdigest(),
                hashlib.sha256(data).hexdigest(),
            )
        )
    return tuple(items)


def _camera_annotation_row(recording_id: str) -> dict[str, object]:
    row: dict[str, object] = {
        "recording_id": recording_id,
        "camera_name": "cam_front",
        "selected_time_of_day_ns": 10_000_000_000,
        "recording_offset_ns": 0,
        "frame_index": 0,
        "frame_id": "frame-fixture",
        "sample_kind": "window_midpoint",
        "window_id": "window-fixture",
        "decoded_width": 4,
        "decoded_height": 4,
        "semantic_fractions": [{"class_id": 13, "fraction": 0.25}],
        "weather_label": "clear",
        "weather_uncertain": False,
    }
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
        "weather_confidence",
        "weather_margin",
        "luminance_mean",
        "luminance_p05",
        "luminance_p95",
        "contrast_std",
        "saturation_mean",
        "laplacian_variance",
        "dark_fraction",
        "overexposed_fraction",
    ):
        row[name] = 0.5
    row["weather_clear_probability"] = 1.0
    for name in (
        "weather_overcast_probability",
        "weather_rain_probability",
        "weather_fog_probability",
        "weather_low_light_probability",
        "weather_uncertain_mixed_probability",
    ):
        row[name] = 0.0
    return row


def _write_tar(path: Path, member_path: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo(member_path)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def write_indus_ad_fixture(root: Path) -> IndusAdFixture:
    recording_id = "recording-code"
    revision = "2" * 40
    video_revision = "4" * 40
    duration_ns = 1_000_000_000
    repo_id = "owner/dataset"
    all_cameras = (*DEFAULT_DECODED_CAMERAS, "cam_front_tilted")

    metadata: dict[str, object] = {"recording_id": recording_id}
    integrity: dict[str, object] = {"recording_id": recording_id}
    viewer: dict[str, object] = {"recording_id": recording_id}
    for camera in all_cameras:
        file_name = None
        if camera != "cam_front_tilted":
            file_name = f"videos/{camera}/aa/{recording_id}.mp4"
            path = root / "data" / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"privacy-video:{camera}".encode())
            integrity[f"{camera}_video_byte_size"] = path.stat().st_size
            integrity[f"{camera}_video_sha256"] = _sha256(path)
            viewer[f"{camera}_video"] = {
                "bytes": None,
                "path": (
                    f"hf://datasets/{repo_id}@{video_revision}/data/{file_name}"
                ),
            }
            status = "face_plate_blurred"
        else:
            integrity[f"{camera}_video_byte_size"] = None
            integrity[f"{camera}_video_sha256"] = None
            viewer[f"{camera}_video"] = None
            status = "missing"
        metadata[f"{camera}_video_file_name"] = file_name
        metadata[f"{camera}_video_privacy_status"] = status
        integrity[f"{camera}_video_file_name"] = file_name
        viewer[f"{camera}_video_privacy_status"] = status

    metadata_path = root / METADATA_PATH
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    integrity_path = root / VIDEO_INTEGRITY_PATH
    integrity_path.parent.mkdir(parents=True, exist_ok=True)
    integrity_path.write_text(json.dumps(integrity, sort_keys=True) + "\n", encoding="utf-8")
    _write_table(root / VIDEO_VIEWER_PATH, [viewer], video_viewer_schema())

    annotation_path = root / CAMERA_ANNOTATION_VIEWER_PATH
    _write_table(
        annotation_path,
        [_camera_annotation_row(recording_id)],
        camera_annotation_schema(),
    )

    for camera in DEFAULT_DECODED_CAMERAS:
        table_path = (
            root
            / "data/tables/camera-frames"
            / f"camera={camera}"
            / "bucket=aa"
            / f"part-{recording_id}.parquet"
        )
        _write_table(
            table_path,
            [{"recording_id": recording_id, "recording_offset_ns": 0}],
            pa.schema(
                [
                    pa.field("recording_id", pa.string()),
                    pa.field("recording_offset_ns", pa.int64()),
                ]
            ),
        )
    for family in ("calibrations", "gnss"):
        table_path = (
            root
            / f"data/tables/{family}"
            / "bucket=aa"
            / f"part-{recording_id}.parquet"
        )
        _write_table(
            table_path,
            [{"recording_id": recording_id}],
            pa.schema([pa.field("recording_id", pa.string())]),
        )

    semantic_buffer = io.BytesIO()
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(
        semantic_buffer, format="PNG"
    )
    semantic_payload = semantic_buffer.getvalue()
    depth_buffer = io.BytesIO()
    np.save(depth_buffer, np.zeros((4, 4), dtype=np.float32), allow_pickle=False)
    depth_payload = depth_buffer.getvalue()
    semantic_shard = "data/annotations/camera/semantic/shard-fixture.tar"
    depth_shard = "data/annotations/camera/depth/shard-fixture.tar"
    semantic_member = f"{recording_id}/cam_front/frame-fixture.png"
    depth_member = f"{recording_id}/cam_front/frame-fixture.npy"
    _write_tar(root / semantic_shard, semantic_member, semantic_payload)
    _write_tar(root / depth_shard, depth_member, depth_payload)
    dense_rows = [
        {
            "artifact_kind": "semantic",
            "camera_name": "cam_front",
            "dtype": "uint8",
            "frame_index": 0,
            "height": 4,
            "member_path": semantic_member,
            "payload_bytes": len(semantic_payload),
            "payload_sha256": hashlib.sha256(semantic_payload).hexdigest(),
            "recording_id": recording_id,
            "recording_offset_ns": 0,
            "shard_path": semantic_shard,
            "width": 4,
        },
        {
            "artifact_kind": "depth",
            "camera_name": "cam_front",
            "dtype": "float32",
            "frame_index": 0,
            "height": 4,
            "member_path": depth_member,
            "payload_bytes": len(depth_payload),
            "payload_sha256": hashlib.sha256(depth_payload).hexdigest(),
            "recording_id": recording_id,
            "recording_offset_ns": 0,
            "shard_path": depth_shard,
            "width": 4,
        },
    ]
    dense_path = root / "data/annotations/camera/dense-index/part-shard-fixture.parquet"
    _write_table(dense_path, dense_rows, dense_index_schema())

    bound_paths = (METADATA_PATH, VIDEO_INTEGRITY_PATH, VIDEO_VIEWER_PATH)
    privacy = {
        "schema_version": 2,
        "repo_id": repo_id,
        "recording_count": 1,
        "video_count": 6,
        "face_plate_blurred_video_count": 6,
        "missing_camera_slot_count": 1,
        "not_face_plate_blurred_video_count": 0,
        "status_values": ["face_plate_blurred", "missing"],
        "video_content_revision": video_revision,
        "files": [
            {
                "path": path,
                "bytes": (root / path).stat().st_size,
                "sha256": _sha256(root / path),
            }
            for path in bound_paths
        ],
    }
    privacy_path = root / VIDEO_PRIVACY_PATH
    privacy_path.write_text(json.dumps(privacy, sort_keys=True), encoding="utf-8")

    source = IndusAdHfSourceConfig(
        type="indus_ad_hf",
        contract="indus_ad_integrated_v1",
        repo_id=repo_id,
        revision=revision,
        recording_ids=[recording_id],
        splits=["train"],
        cameras=list(DEFAULT_DECODED_CAMERAS),
        annotation_layers=[
            "camera_scalar",
            "semantic_fractions",
            "depth_summary",
            "weather",
        ],
    )
    return IndusAdFixture(
        root,
        source,
        _repo_tree(root),
        recording_id,
        duration_ns,
        annotation_path,
        dense_path,
    )


def refreshed_tree(fixture: IndusAdFixture) -> IndusAdFixture:
    return replace(fixture, tree=_repo_tree(fixture.root))
