from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_devkit.config import DEFAULT_DECODED_CAMERAS
from dataset_devkit.decoded_manifest import (
    DecodedManifestError,
    PrivacyContract,
    parse_decoded_control_plane,
)
from decoded_v2_fixture import (
    decoded_source_config,
    rewrite_as_anonymizer_layout,
    write_decoded_fixture,
)


def _rewrite_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(
        pa.Table.from_pylist(rows),
        path,
        compression="zstd",
        write_statistics=False,
    )


def _mutate_catalog(root: Path, mutation: Callable[[list[dict[str, object]]], None]) -> None:
    path = root / "data/recordings.parquet"
    rows = pq.ParquetFile(path).read().to_pylist()
    mutation(rows)
    _rewrite_rows(path, rows)


def _mutate_manifest(root: Path, manifest: str, key: str, value: object) -> None:
    path = root / "data/manifests" / manifest
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[key] = value
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return
    rows = pq.ParquetFile(path).read().to_pylist()
    payload = json.loads(rows[0]["audit_json"])
    payload[key] = value
    rows[0]["audit_json"] = json.dumps(payload, sort_keys=True)
    _rewrite_rows(path, rows)


def test_control_plane_filters_before_payload_selection(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    source = decoded_source_config(fixture.revision, [fixture.recording_id])

    plan = parse_decoded_control_plane(fixture.root, source)

    assert [item.recording_id for item in plan.recordings] == [fixture.recording_id]
    assert plan.privacy == PrivacyContract(
        model_lock_sha256="1" * 64,
        privacy_config_sha256="2" * 64,
        absolute_dates_removed=True,
        exact_recording_offsets_retained=True,
        local_enu_precision="retained",
        global_horizontal_resolution_m=1.0,
    )
    assert {video.camera for video in plan.recordings[0].videos} == set(
        DEFAULT_DECODED_CAMERAS
    )
    assert plan.recordings[0].artifact("gnss").path.endswith(".parquet")


def test_control_plane_selects_anonymizer_camera_shards(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    rewrite_as_anonymizer_layout(fixture)

    plan = parse_decoded_control_plane(
        fixture.root,
        decoded_source_config(fixture.revision, [fixture.recording_id]),
    )

    frame_shards = plan.recordings[0].artifacts_of("camera_frames")
    assert len(frame_shards) == len(DEFAULT_DECODED_CAMERAS)
    assert all("/camera=" in item.path for item in frame_shards)


@pytest.mark.parametrize(
    ("manifest", "key", "value"),
    [
        ("processing-config.json", "global_horizontal_resolution_m", 10.0),
        ("processing-config.json", "local_enu_precision", "rounded"),
        ("schema-audit.parquet", "absolute_dates_removed", False),
        ("schema-audit.parquet", "exact_recording_offsets_retained", False),
        ("model-receipt.json", "model_lock_sha256", "invalid"),
    ],
)
def test_control_plane_rejects_privacy_contract_drift(
    tmp_path: Path, manifest: str, key: str, value: object
) -> None:
    fixture = write_decoded_fixture(tmp_path)
    _mutate_manifest(fixture.root, manifest, key, value)

    with pytest.raises(DecodedManifestError):
        parse_decoded_control_plane(
            fixture.root,
            decoded_source_config(fixture.revision, [fixture.recording_id]),
        )


def test_control_plane_rejects_duplicate_recording_ids(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    _mutate_catalog(fixture.root, lambda rows: rows.append(dict(rows[0])))

    with pytest.raises(DecodedManifestError, match="duplicate recording"):
        parse_decoded_control_plane(
            fixture.root,
            decoded_source_config(fixture.revision, [fixture.recording_id]),
        )


def test_control_plane_rejects_catalog_output_digest_mismatch(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        rows[0]["video_sha256"]["cam_front"] = "f" * 64  # type: ignore[index]

    _mutate_catalog(fixture.root, mutate)
    with pytest.raises(DecodedManifestError, match="output manifest"):
        parse_decoded_control_plane(
            fixture.root,
            decoded_source_config(fixture.revision, [fixture.recording_id]),
        )


@pytest.mark.parametrize("case", ["missing_camera", "unknown_camera", "bad_size", "arrays"])
def test_control_plane_rejects_malformed_catalog_rows(tmp_path: Path, case: str) -> None:
    fixture = write_decoded_fixture(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        row = rows[0]
        if case == "missing_camera":
            cast(list[str], row["available_cameras"]).remove("cam_rear_right")
            for key in ("video_paths", "video_sizes", "video_sha256", "frame_counts"):
                del cast(dict[str, object], row[key])["cam_rear_right"]
            row["has_cam_rear_right"] = False
        elif case == "unknown_camera":
            cast(list[str], row["available_cameras"]).append("cam_unknown")
            cast(dict[str, object], row["video_paths"])["cam_unknown"] = (
                "data/videos/cam_unknown/x.mp4"
            )
            cast(dict[str, object], row["video_sizes"])["cam_unknown"] = 1
            cast(dict[str, object], row["video_sha256"])["cam_unknown"] = "a" * 64
            cast(dict[str, object], row["frame_counts"])["cam_unknown"] = 1
        elif case == "bad_size":
            row["video_sizes"]["cam_front"] = "large"  # type: ignore[index]
        else:
            row["artifact_sizes"]["gnss"] = []  # type: ignore[index]

    _mutate_catalog(fixture.root, mutate)
    with pytest.raises(DecodedManifestError):
        parse_decoded_control_plane(
            fixture.root,
            decoded_source_config(fixture.revision, [fixture.recording_id]),
        )


def test_control_plane_rejects_unmatched_selection(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    source = decoded_source_config(fixture.revision, [fixture.recording_id]).model_copy(
        update={"splits": ["test"]}
    )

    with pytest.raises(DecodedManifestError, match="selection"):
        parse_decoded_control_plane(fixture.root, source)


def test_control_plane_rejects_date_token_in_selected_path(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        rows[0]["video_paths"]["cam_front"] = (  # type: ignore[index]
            "data/videos/cam_front/date=2025-01-01/recording-code.mp4"
        )

    _mutate_catalog(fixture.root, mutate)
    with pytest.raises(DecodedManifestError, match="path"):
        parse_decoded_control_plane(
            fixture.root,
            decoded_source_config(fixture.revision, [fixture.recording_id]),
        )
