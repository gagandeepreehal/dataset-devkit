from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.features import compute_recording_features
from dataset_devkit.scene_models import RecordingSceneResult
from dataset_devkit.scenes import build_recording_scenes, validate_scene_graph
from test_scenes import SOURCE, _annotation_config, _annotations, _camera, _config, _report


def _quaternion_from_rpy(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _motion_graph(
    tmp_path: Path,
    base: GlobalConfig,
    positions: tuple[float, ...],
    yaws_deg: tuple[float, ...],
) -> tuple[RecordingSceneResult, GlobalConfig]:
    timestamps = tuple(index * 1_000_000_000 for index in range(len(positions)))
    report = _report(tmp_path, timestamps)
    audits = []
    for position, yaw, audit in zip(positions, yaws_deg, report.final_candidates, strict=True):
        cameras = tuple(
            replace(
                camera,
                ego_pose=replace(
                    camera.ego_pose,
                    translation_xyz_m=(position, 0.0, 0.0),
                    rotation_wxyz=_quaternion_from_rpy(0.0, 0.0, math.radians(yaw)),
                ),
            )
            for camera in audit.samples
        )
        audits.append(replace(audit, samples=cameras))
    report = replace(report, sample_audits=tuple(audits), final_candidates=tuple(audits))
    config = _config(base, max_duration_s=max(2, len(positions)), max_sample_gap_ms=2000)
    return build_recording_scenes(report, SOURCE, config), config


def test_trajectory_uses_real_irregular_camera_timestamps(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0, 1_000_000_000, 3_000_000_000))
    audits = []
    for index, audit in enumerate(report.final_candidates):
        cameras = tuple(
            replace(
                camera,
                ego_pose=replace(camera.ego_pose, translation_xyz_m=(index * 10.0, 0.0, 0.0)),
            )
            for camera in audit.samples
        )
        audits.append(replace(audit, samples=cameras))
    report = replace(report, sample_audits=tuple(audits), final_candidates=tuple(audits))
    config = _config(config_factory(), max_duration_s=5, max_sample_gap_ms=5000)
    config = config.model_copy(
        update={
            "tags": config.tags.model_copy(
                update={"reference_camera_channel": "front", "reference_camera_policy": "require"}
            )
        }
    )
    graph = build_recording_scenes(report, SOURCE, config)

    feature = compute_recording_features(graph, config.tags).scenes[0]

    assert feature.total_distance_m == pytest.approx(20.0)
    assert feature.duration_s == pytest.approx(3.0)
    assert feature.mean_speed_mps == pytest.approx(7.5)
    assert feature.time_weighted_speed_mps == pytest.approx(20.0 / 3.0)
    assert feature.segment_speeds_mps == pytest.approx((10.0, 5.0))


def test_sync_metrics_include_grid_and_every_camera_independent_of_reference(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0, 1_000_000_000))
    audits = []
    for audit in report.final_candidates:
        logical = audit.grid_target_timestamp_ns
        cameras = (
            replace(
                _camera(tmp_path, logical, "front", logical + 10_000_000),
                batch_timestamp_ns=logical + 5_000_000,
            ),
            replace(
                _camera(tmp_path, logical, "rear", logical + 20_000_000),
                batch_timestamp_ns=logical + 5_000_000,
            ),
        )
        audits.append(
            replace(
                audit,
                batch_timestamp_ns=logical + 5_000_000,
                camera_timestamps=tuple(
                    (camera.camera_name, camera.camera_timestamp_ns) for camera in cameras
                ),
                samples=cameras,
            )
        )
    audits_tuple = tuple(audits)
    report = replace(report, sample_audits=audits_tuple, final_candidates=audits_tuple)
    config = _config(config_factory())
    graph = build_recording_scenes(report, SOURCE, config)

    front = compute_recording_features(graph, config.tags).scenes[0]
    rear_config = config.tags.model_copy(update={"reference_camera_channel": "rear"})
    rear = compute_recording_features(graph, rear_config).scenes[0]

    assert front.max_abs_sync_error_ms == 20.0
    assert front.mean_abs_sync_error_ms == pytest.approx(70.0 / 6.0)
    assert (front.max_abs_sync_error_ms, front.mean_abs_sync_error_ms) == (
        rear.max_abs_sync_error_ms,
        rear.mean_abs_sync_error_ms,
    )


def test_one_sample_scene_has_finite_stationary_metrics(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    annotations = _annotations(
        tmp_path / "annotations.jsonl",
        [{"blob_path": SOURCE.blob_path, "timestamp_ns": 0, "labels": ["category/demo"]}],
    )
    config = _annotation_config(
        config_factory(), mode="annotation_only", tolerance_ms=1, before_s=0, after_s=0
    )
    graph = build_recording_scenes(
        _report(tmp_path / "recording", (0,)), SOURCE, config, annotations_path=annotations
    )

    feature = compute_recording_features(graph, config.tags).scenes[0]

    assert feature.duration_s == 0.0
    assert feature.total_distance_m == 0.0
    assert feature.computed_tags == (
        "camera_coverage_complete",
        "gnss_valid",
        "stationary",
        "straight",
    )
    assert feature.human_labels == ("category/demo",)


def test_task5_feature_evidence_is_sealed_against_mutation(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _config(config_factory())
    graph = build_recording_scenes(_report(tmp_path, (0, 1)), SOURCE, config)
    source_sample = graph.source_samples[0]
    mutated = replace(
        graph,
        source_samples=(
            replace(
                source_sample,
                expected_channels=(*source_sample.expected_channels, "zfabricated"),
            ),
            *graph.source_samples[1:],
        ),
    )

    with pytest.raises(StructuralExtractionError, match="feature evidence seal"):
        validate_scene_graph(mutated)


def test_yaw_unwrap_and_enu_direction_tags(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    graph, config = _motion_graph(tmp_path / "wrap", config_factory(), (0.0, 1.0), (179, -179))
    wrap = compute_recording_features(graph, config.tags).scenes[0]
    assert math.degrees(wrap.signed_heading_changes_rad[0]) == pytest.approx(2.0)

    left_graph, left_config = _motion_graph(
        tmp_path / "left", config_factory(), (0.0, 1.0), (0, 45)
    )
    right_graph, right_config = _motion_graph(
        tmp_path / "right", config_factory(), (0.0, 1.0), (0, -45)
    )
    left_tags = compute_recording_features(left_graph, left_config.tags).scenes[0].computed_tags
    right_tags = compute_recording_features(right_graph, right_config.tags).scenes[0].computed_tags
    assert "left_turn" in left_tags
    assert "right_turn" in right_tags


def test_mixed_roll_pitch_quaternion_uses_wxyz_yaw_convention(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    graph, config = _motion_graph(tmp_path, config_factory(), (0.0, 1.0), (0, 0))
    second = graph.sample_data[1]
    quaternion = _quaternion_from_rpy(math.radians(20), math.radians(-15), math.radians(30))
    mutated_data = replace(second, ego_pose=replace(second.ego_pose, rotation_wxyz=quaternion))
    # Rebuilding through Task 5 is required because direct graph mutation is sealed.
    report = _report(tmp_path / "rebuilt", (0, 1_000_000_000))
    audits = []
    for index, audit in enumerate(report.final_candidates):
        rotation = (1.0, 0.0, 0.0, 0.0) if index == 0 else quaternion
        audits.append(
            replace(
                audit,
                samples=tuple(
                    replace(camera, ego_pose=replace(camera.ego_pose, rotation_wxyz=rotation))
                    for camera in audit.samples
                ),
            )
        )
    report = replace(report, sample_audits=tuple(audits), final_candidates=tuple(audits))
    rebuilt = build_recording_scenes(report, SOURCE, config)
    feature = compute_recording_features(rebuilt, config.tags).scenes[0]
    assert math.degrees(feature.headings_rad[-1]) == pytest.approx(30.0)
    assert mutated_data.ego_pose.rotation_wxyz == quaternion


@pytest.mark.parametrize(
    ("heading_deg", "expected"),
    [(5.0, "straight"), (10.0, "left_curvature"), (45.0, "left_turn")],
)
def test_direction_tag_threshold_equalities_are_exclusive(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    heading_deg: float,
    expected: str,
) -> None:
    graph, config = _motion_graph(
        tmp_path / expected, config_factory(), (0.0, 1.0), (0.0, heading_deg)
    )
    tags = compute_recording_features(graph, config.tags).scenes[0].computed_tags
    directional = set(tags) & {
        "straight", "left_curvature", "right_curvature", "left_turn", "right_turn"
    }
    assert directional == {expected}


def test_zero_distance_curvature_and_start_stop_transitions(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    graph, config = _motion_graph(
        tmp_path, config_factory(), (0.0, 1.0, 1.0, 2.0), (0.0, 10.0, 20.0, 30.0)
    )
    feature = compute_recording_features(graph, config.tags).scenes[0]
    assert feature.signed_curvatures_rad_per_m[1] == 0.0
    assert feature.segment_stationary == (False, True, False)
    assert feature.moving_to_stationary_count == 1
    assert feature.stationary_to_moving_count == 1
    assert feature.stationary_duration_s == 1.0
    assert {"moving", "stopping", "starting"} <= set(feature.computed_tags)


def test_reference_require_fallback_coverage_and_gnss_ratio(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0, 1_000_000_000))
    first, second = report.final_candidates
    invalid_front = second.samples[0]
    invalid_interpolation = replace(
        invalid_front.ego_pose.interpolation, source_validity=(True, False)
    )
    invalid_front = replace(
        invalid_front,
        ego_pose=replace(invalid_front.ego_pose, interpolation=invalid_interpolation),
    )
    second = replace(
        second,
        camera_timestamps=(("front", invalid_front.camera_timestamp_ns),),
        samples=(invalid_front,),
    )
    report = replace(
        report,
        sample_audits=(first, second),
        final_candidates=(first, second),
    )
    config = _config(config_factory())
    graph = build_recording_scenes(report, SOURCE, config)
    fallback_config = config.tags.model_copy(
        update={
            "reference_camera_channel": "side",
            "reference_camera_policy": "lexicographic_fallback",
        }
    )
    feature = compute_recording_features(graph, fallback_config).scenes[0]
    assert feature.reference_channels_used == ("front", "front")
    assert feature.reference_fallback_count == 2
    assert feature.camera_coverage_ratio == 0.75
    assert {item.channel: item.ratio for item in feature.camera_coverage_by_channel} == {
        "front": 1.0,
        "rear": 0.5,
    }
    assert feature.source_gnss_valid_ratio == 0.5
    required = fallback_config.model_copy(update={"reference_camera_policy": "require"})
    with pytest.raises(StructuralExtractionError, match="reference camera"):
        compute_recording_features(graph, required)


def test_invalid_quaternion_and_nonincreasing_real_timestamp_structural_fail(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    graph, config = _motion_graph(tmp_path / "quaternion", config_factory(), (0.0, 1.0), (0, 0))
    report = _report(tmp_path / "zero", (0, 1_000_000_000))
    audits = []
    for audit in report.final_candidates:
        audits.append(
            replace(
                audit,
                samples=tuple(
                    replace(
                        camera,
                        ego_pose=replace(camera.ego_pose, rotation_wxyz=(0.0, 0.0, 0.0, 0.0)),
                    )
                    for camera in audit.samples
                ),
            )
        )
    report = replace(report, sample_audits=tuple(audits), final_candidates=tuple(audits))
    zero_graph = build_recording_scenes(report, SOURCE, config)
    with pytest.raises(StructuralExtractionError, match="quaternion"):
        compute_recording_features(zero_graph, config.tags)

    same_report = _report(tmp_path / "timestamp", (0, 1_000_000_000))
    same_audits = []
    for audit in same_report.final_candidates:
        cameras = tuple(
            _camera(tmp_path / "timestamp-new", audit.grid_target_timestamp_ns, channel, 123)
            for channel in ("front", "rear")
        )
        same_audits.append(
            replace(
                audit,
                camera_timestamps=tuple(
                    (item.camera_name, item.camera_timestamp_ns) for item in cameras
                ),
                samples=cameras,
            )
        )
    same_report = replace(
        same_report,
        sample_audits=tuple(same_audits),
        final_candidates=tuple(same_audits),
    )
    with pytest.raises(StructuralExtractionError, match="timestamps are not monotonic"):
        build_recording_scenes(same_report, SOURCE, config)


def test_nonfinite_pose_quaternion_and_timestamp_are_structural(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _config(config_factory())
    report = _report(tmp_path / "quaternion", (0, 1))
    camera = report.final_candidates[0].samples[0]
    bad_camera = replace(
        camera,
        ego_pose=replace(camera.ego_pose, rotation_wxyz=(math.nan, 0.0, 0.0, 0.0)),
    )
    bad_audit = replace(
        report.final_candidates[0],
        samples=(bad_camera, *report.final_candidates[0].samples[1:]),
    )
    bad_report = replace(report, sample_audits=(bad_audit,), final_candidates=(bad_audit,))
    with pytest.raises(StructuralExtractionError, match="non-finite"):
        build_recording_scenes(bad_report, SOURCE, config)

    timestamp: Any = math.nan
    report = _report(tmp_path / "timestamp", (0, 1))
    audit = report.final_candidates[0]
    camera = audit.samples[0]
    interpolation = replace(camera.ego_pose.interpolation, timestamp_ns=timestamp)
    pose = replace(camera.ego_pose, timestamp_ns=timestamp, interpolation=interpolation)
    staged = replace(camera.staged_image, timestamp_ns=timestamp)
    camera = replace(
        camera,
        camera_timestamp_ns=timestamp,
        staged_image=staged,
        ego_pose=pose,
    )
    audit = replace(
        audit,
        camera_timestamps=(("front", timestamp), audit.camera_timestamps[1]),
        samples=(camera, audit.samples[1]),
    )
    report = replace(report, sample_audits=(audit,), final_candidates=(audit,))
    with pytest.raises(StructuralExtractionError, match="timestamp"):
        build_recording_scenes(report, SOURCE, config)
