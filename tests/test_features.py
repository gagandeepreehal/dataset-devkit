from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.features import compute_recording_features
from dataset_devkit.scenes import build_recording_scenes, validate_scene_graph
from test_scenes import SOURCE, _annotation_config, _annotations, _config, _report


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
    assert feature.mean_speed_mps == pytest.approx(20.0 / 3.0)
    assert feature.segment_speeds_mps == pytest.approx((10.0, 5.0))


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
