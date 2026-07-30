from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image

from dataset_devkit import scenes as scenes_module
from dataset_devkit.annotations import ParsedAnnotation
from dataset_devkit.config import GlobalConfig, InvalidationRulesConfig
from dataset_devkit.extraction.camera import DecoderOutput
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import (
    CameraCalibration,
    CameraExtrinsic,
    CameraIntrinsic,
    EgoPose,
    ExtractedCameraSample,
    GnssInterpolation,
)
from dataset_devkit.extraction.service import RecordingExtractor
from dataset_devkit.extraction.staging import stage_jpeg
from dataset_devkit.provenance import SourceFingerprint, canonical_json
from dataset_devkit.scenes import build_recording_scenes, validate_scene_graph
from dataset_devkit.validity import (
    INVALIDITY_CODES,
    LogicalSampleAudit,
    ValidityReport,
    evaluate_validity,
)
from mcap_fixture import camera_message, write_mcap

SOURCE = SourceFingerprint(
    "https://example.blob.core.windows.net",
    "recordings",
    "mcap-h265/day/a.mcap",
    '"etag"',
    123,
)


class _DeterministicDecoder:
    def decode(self, payload: bytes, pts: int, time_base: Fraction) -> list[DecoderOutput]:
        return [DecoderOutput(pts, Image.new("RGB", (4, 3), (1, 2, 3)))]

    def flush(self) -> list[DecoderOutput]:
        return []

    def close(self) -> None:
        pass


def _camera(tmp_path: Path, logical: int, channel: str, real: int) -> ExtractedCameraSample:
    camera_index = 0 if channel == "front" else 1
    staged = stage_jpeg(
        tmp_path / "scene-staging",
        "recording",
        camera_index,
        channel,
        real,
        Image.new("RGB", (4, 3), (camera_index, 2, 3)),
        (4, 3),
        batch_ordinal=logical,
    )
    interpolation = GnssInterpolation(real, True, None, None, 0.0, 0, 0)
    pose = EgoPose(real, True, (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0), interpolation)
    calibration = CameraCalibration(
        CameraIntrinsic(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, (), 4.0, 3.0),
        CameraExtrinsic((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    return ExtractedCameraSample(
        logical,
        logical,
        real,
        camera_index,
        channel,
        staged,
        pose,
        calibration,
    )


def _report(tmp_path: Path, timestamps: tuple[int, ...]) -> ValidityReport:
    tmp_path.mkdir(parents=True, exist_ok=True)
    audits = tuple(
        LogicalSampleAudit(
            timestamp,
            timestamp,
            (("front", timestamp + 10), ("rear", timestamp + 20)),
            (
                _camera(tmp_path, timestamp, "front", timestamp + 10),
                _camera(tmp_path, timestamp, "rear", timestamp + 20),
            ),
            (),
            True,
        )
        for timestamp in timestamps
    )
    return ValidityReport(
        tmp_path / "local.mcap", (), (), audits, audits, (), True, "retain_for_audit"
    )


def _config(base: GlobalConfig, **scene_changes: object) -> GlobalConfig:
    decimal_fields = {
        "min_duration_s",
        "max_duration_s",
        "max_sample_gap_ms",
        "skip_between_scenes_s",
    }
    scene_changes = {
        key: Decimal(str(value)) if key in decimal_fields else value
        for key, value in scene_changes.items()
    }
    return base.model_copy(
        update={
            "scenes": base.scenes.model_copy(
                update={
                    "mode": "automatic",
                    "dataset_namespace": UUID("8d55f58b-4a7b-5a9a-a95a-a3989610795b"),
                    "min_duration_s": Decimal("0"),
                    "max_duration_s": Decimal("2"),
                    "min_samples": 1,
                    "max_sample_gap_ms": Decimal("1000"),
                    "skip_between_scenes_s": Decimal("0"),
                    **scene_changes,
                }
            )
        }
    )


def _annotations(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    return path


def _annotation_config(
    base: GlobalConfig, *, mode: str, tolerance_ms: float, before_s: float, after_s: float
) -> GlobalConfig:
    configured = _config(base, mode=mode)
    return configured.model_copy(
        update={
            "annotations": base.annotations.model_copy(
                update={
                    "match_tolerance_ms": Decimal(str(tolerance_ms)),
                    "before_s": Decimal(str(before_s)),
                    "after_s": Decimal(str(after_s)),
                }
            )
        }
    )


def test_gap_and_max_duration_equalities_stay_while_exceeding_values_split(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _config(config_factory())
    report = _report(tmp_path, (0, 1_000_000_000, 2_000_000_000, 3_000_000_001))

    result = build_recording_scenes(report, SOURCE, config)

    assert [(scene.first_timestamp_ns, scene.last_timestamp_ns) for scene in result.scenes] == [
        (0, 2_000_000_000),
        (3_000_000_001, 3_000_000_001),
    ]


def test_minimum_duration_count_skip_and_leftover_reasons(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _config(
        config_factory(),
        min_duration_s=1.0,
        max_duration_s=1.0,
        min_samples=2,
        skip_between_scenes_s=1.0,
    )
    report = _report(
        tmp_path, (0, 1_000_000_000, 1_500_000_000, 2_000_000_000, 3_000_000_000, 4_500_000_000)
    )

    result = build_recording_scenes(report, SOURCE, config)

    assert [(scene.first_timestamp_ns, scene.last_timestamp_ns) for scene in result.scenes] == [
        (0, 1_000_000_000),
        (2_000_000_000, 3_000_000_000),
    ]
    assert [(item.timestamp_ns, item.reason) for item in result.unassigned] == [
        (1_500_000_000, "inter_scene_skip"),
        (4_500_000_000, "candidate_too_short"),
    ]


def test_multicamera_tokens_real_timestamps_chains_and_determinism(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _config(config_factory())
    report = _report(tmp_path, (0, 1_000_000_000))
    first = build_recording_scenes(report, SOURCE, config)
    second = build_recording_scenes(report, SOURCE, config)

    assert canonical_json(first.to_dict()) == canonical_json(second.to_dict())
    assert len(first.samples) == 2
    assert [item.timestamp_ns for item in first.sample_data if item.channel == "front"] == [
        10,
        1_000_000_010,
    ]
    assert [item.timestamp_ns for item in first.sample_data if item.channel == "rear"] == [
        20,
        1_000_000_020,
    ]
    assert first.samples[0].prev == "" and first.samples[-1].next == ""
    for channel in ("front", "rear"):
        chain = [item for item in first.sample_data if item.channel == channel]
        assert chain[0].prev == "" and chain[-1].next == ""
        assert chain[0].next == chain[1].token and chain[1].prev == chain[0].token


def test_duplicate_final_logical_timestamp_structural_fails(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0, 1))
    report = replace(report, final_candidates=(report.final_candidates[0],) * 2)
    with pytest.raises(StructuralExtractionError, match="duplicate logical timestamp"):
        build_recording_scenes(report, SOURCE, _config(config_factory()))


def test_staged_image_camera_name_mismatch_structural_fails(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0,))
    audit = report.final_candidates[0]
    camera = audit.samples[0]
    bad_camera = replace(
        camera, staged_image=replace(camera.staged_image, camera_name="wrong-camera")
    )
    bad_audit = replace(audit, samples=(bad_camera, *audit.samples[1:]))
    bad_report = replace(report, sample_audits=(bad_audit,), final_candidates=(bad_audit,))

    with pytest.raises(StructuralExtractionError, match="staged image.*identity"):
        build_recording_scenes(bad_report, SOURCE, _config(config_factory()))


def test_staged_image_camera_index_mismatch_structural_fails(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0,))
    audit = report.final_candidates[0]
    camera = audit.samples[0]
    bad_camera = replace(camera, staged_image=replace(camera.staged_image, camera_index=99))
    bad_audit = replace(audit, samples=(bad_camera, *audit.samples[1:]))
    bad_report = replace(report, sample_audits=(bad_audit,), final_candidates=(bad_audit,))

    with pytest.raises(StructuralExtractionError, match="staged image.*identity"):
        build_recording_scenes(bad_report, SOURCE, _config(config_factory()))


def test_staged_image_owned_inode_and_calibration_dimensions_are_verified(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0,))
    audit = report.final_candidates[0]
    camera = audit.samples[0]

    bad_staged = replace(camera.staged_image, inode=(camera.staged_image.inode or 0) + 1)
    bad_camera = replace(camera, staged_image=bad_staged)
    bad_audit = replace(audit, samples=(bad_camera, *audit.samples[1:]))
    bad_report = replace(report, sample_audits=(bad_audit,), final_candidates=(bad_audit,))
    with pytest.raises(StructuralExtractionError, match="staged image.*ownership"):
        build_recording_scenes(bad_report, SOURCE, _config(config_factory()))

    assert camera.calibration is not None
    bad_intrinsic = replace(camera.calibration.intrinsic, width=5.0)
    bad_calibration = replace(camera.calibration, intrinsic=bad_intrinsic)
    bad_camera = replace(camera, calibration=bad_calibration)
    bad_audit = replace(audit, samples=(bad_camera, *audit.samples[1:]))
    bad_report = replace(report, sample_audits=(bad_audit,), final_candidates=(bad_audit,))
    with pytest.raises(StructuralExtractionError, match="staged image.*dimensions"):
        build_recording_scenes(bad_report, SOURCE, _config(config_factory()))


def test_ego_pose_interpolation_identity_is_verified_at_scene_boundary(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    report = _report(tmp_path, (0,))
    audit = report.final_candidates[0]
    camera = audit.samples[0]
    bad_interpolation = replace(
        camera.ego_pose.interpolation,
        timestamp_ns=camera.ego_pose.interpolation.timestamp_ns + 1,
    )
    bad_pose = replace(camera.ego_pose, interpolation=bad_interpolation)
    bad_camera = replace(camera, ego_pose=bad_pose)
    bad_audit = replace(audit, samples=(bad_camera, *audit.samples[1:]))
    bad_report = replace(report, sample_audits=(bad_audit,), final_candidates=(bad_audit,))

    with pytest.raises(StructuralExtractionError, match="ego-pose.*identity"):
        build_recording_scenes(bad_report, SOURCE, _config(config_factory()))


def test_valid_multicamera_staged_identity_reaches_sample_data(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    result = build_recording_scenes(_report(tmp_path, (0,)), SOURCE, _config(config_factory()))

    assert [(item.channel, item.staged_image.path.name) for item in result.sample_data] == [
        ("front", "000000000-000-front-10.jpg"),
        ("rear", "000000000-001-rear-20.jpg"),
    ]
    assert all(item.calibration is not None for item in result.sample_data)


def test_validator_rejects_broken_or_cyclic_links(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    result = build_recording_scenes(
        _report(tmp_path, (0, 1_000_000_000)),
        SOURCE,
        _config(config_factory()),
    )
    broken = replace(
        result,
        samples=(replace(result.samples[0], next=result.samples[0].token), result.samples[1]),
    )
    with pytest.raises(StructuralExtractionError, match="sample chain"):
        validate_scene_graph(broken)


def test_validator_rejects_symmetric_sample_chain_crossing_scenes(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    result = build_recording_scenes(
        _report(tmp_path, (0, 1_000_000_000, 3_000_000_000, 4_000_000_000)),
        SOURCE,
        _config(config_factory()),
    )
    assert len(result.scenes) == 2
    samples = list(result.samples)
    first_end = next(
        i for i, item in enumerate(samples) if item.token == result.scenes[0].last_sample_token
    )
    second_start = next(
        i for i, item in enumerate(samples) if item.token == result.scenes[1].first_sample_token
    )
    samples[first_end] = replace(samples[first_end], next=samples[second_start].token)
    samples[second_start] = replace(samples[second_start], prev=samples[first_end].token)

    with pytest.raises(StructuralExtractionError, match="cross-scene|endpoints"):
        validate_scene_graph(replace(result, samples=tuple(samples)))


def test_nearest_annotation_match_uses_earlier_tie_and_exact_tolerance(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    annotation_path = _annotations(
        tmp_path / "annotations.jsonl",
        [
            {"blob_path": SOURCE.blob_path, "timestamp_ns": 1_000_000_000, "labels": ["tie"]},
            {"blob_path": SOURCE.blob_path, "timestamp_ns": 3_000_000_001, "labels": ["late"]},
            {"blob_path": "mcap-h265/day/other.mcap", "timestamp_ns": 0, "labels": ["other"]},
        ],
    )
    config = _annotation_config(
        config_factory(),
        mode="annotation_only",
        tolerance_ms=1000.0,
        before_s=0.0,
        after_s=0.0,
    )

    result = build_recording_scenes(
        _report(tmp_path, (0, 2_000_000_000)),
        SOURCE,
        config,
        annotations_path=annotation_path,
    )

    assert [
        (
            item.line_number,
            item.matched,
            item.sample_timestamp_ns,
            item.signed_error_ns,
            item.reason,
        )
        for item in result.annotation_matches
    ] == [
        (1, True, 0, -1_000_000_000, "matched"),
        (2, False, None, None, "outside_tolerance"),
        (3, False, None, None, "different_recording"),
    ]
    assert len(result.scenes) == 1 and result.scenes[0].labels == ("tie",)


def test_annotation_windows_clip_to_runs_and_merge_overlap_with_lineage(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    annotation_path = _annotations(
        tmp_path / "annotations.jsonl",
        [
            {
                "blob_path": SOURCE.blob_path,
                "timestamp_ns": 1_000_000_000,
                "labels": ["turn", "rain"],
            },
            {
                "blob_path": SOURCE.blob_path,
                "timestamp_ns": 2_000_000_000,
                "labels": ["rain", "pedestrian"],
            },
            {"blob_path": SOURCE.blob_path, "timestamp_ns": 5_000_000_000, "labels": ["later"]},
        ],
    )
    config = _annotation_config(
        config_factory(),
        mode="annotation_only",
        tolerance_ms=0.0,
        before_s=2.0,
        after_s=2.0,
    )
    report = _report(tmp_path, (0, 1_000_000_000, 2_000_000_000, 5_000_000_000, 6_000_000_000))

    result = build_recording_scenes(report, SOURCE, config, annotations_path=annotation_path)

    assert len(result.annotation_windows) == 2
    first = result.annotation_windows[0]
    assert (first.first_sample_timestamp_ns, first.last_sample_timestamp_ns) == (0, 2_000_000_000)
    assert first.labels == ("turn", "rain", "pedestrian")
    assert len(first.annotation_tokens) == 2
    assert result.scenes[0].annotation_window_ref == first.token
    second = result.annotation_windows[1]
    assert (second.first_sample_timestamp_ns, second.last_sample_timestamp_ns) == (
        5_000_000_000,
        6_000_000_000,
    )
    assert all(scene.nbr_samples >= 1 for scene in result.scenes)


def test_merged_annotation_lineage_and_labels_follow_source_lines_not_time(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    annotation_path = _annotations(
        tmp_path / "reverse-annotations.jsonl",
        [
            {
                "blob_path": SOURCE.blob_path,
                "timestamp_ns": 2_000_000_000,
                "labels": ["zebra", "alpha"],
            },
            {
                "blob_path": SOURCE.blob_path,
                "timestamp_ns": 1_000_000_000,
                "labels": ["alpha", "beta"],
            },
        ],
    )
    config = _annotation_config(
        config_factory(),
        mode="annotation_only",
        tolerance_ms=0.0,
        before_s=1.0,
        after_s=1.0,
    )
    result = build_recording_scenes(
        _report(tmp_path, (0, 1_000_000_000, 2_000_000_000, 3_000_000_000)),
        SOURCE,
        config,
        annotations_path=annotation_path,
    )

    assert len(result.annotation_windows) == 1
    window = result.annotation_windows[0]
    assert window.annotation_tokens == tuple(item.token for item in result.annotations)
    assert window.labels == ("zebra", "alpha", "beta")


def test_hybrid_annotation_range_splits_automatic_runs_without_sample_reuse(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    annotation_path = _annotations(
        tmp_path / "annotations.jsonl",
        [{"blob_path": SOURCE.blob_path, "timestamp_ns": 2_000_000_000, "labels": ["event"]}],
    )
    config = _annotation_config(
        config_factory(),
        mode="hybrid",
        tolerance_ms=0.0,
        before_s=0.0,
        after_s=0.0,
    )
    config = config.model_copy(
        update={
            "scenes": config.scenes.model_copy(
                update={
                    "min_duration_s": Decimal("1"),
                    "max_duration_s": Decimal("10"),
                    "min_samples": 2,
                    "max_sample_gap_ms": Decimal("2000"),
                }
            )
        }
    )
    result = build_recording_scenes(
        _report(tmp_path, (0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000)),
        SOURCE,
        config,
        annotations_path=annotation_path,
    )

    observed = [
        (scene.kind, scene.first_timestamp_ns, scene.last_timestamp_ns) for scene in result.scenes
    ]
    assert observed == [
        ("automatic", 0, 1_000_000_000),
        ("annotation", 2_000_000_000, 2_000_000_000),
        ("automatic", 3_000_000_000, 4_000_000_000),
    ]
    assert len({sample.timestamp_ns for sample in result.samples}) == 5


def test_annotation_only_and_automatic_have_explicit_opposite_coverage(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    annotation_path = _annotations(
        tmp_path / "annotations.jsonl",
        [{"blob_path": SOURCE.blob_path, "timestamp_ns": 1_000_000_000, "labels": ["human"]}],
    )
    report = _report(tmp_path, (0, 1_000_000_000, 2_000_000_000))
    annotation_only = _annotation_config(
        config_factory(),
        mode="annotation_only",
        tolerance_ms=0.0,
        before_s=0.0,
        after_s=0.0,
    )
    automatic = _annotation_config(
        config_factory(),
        mode="automatic",
        tolerance_ms=0.0,
        before_s=0.0,
        after_s=0.0,
    )
    automatic = automatic.model_copy(
        update={
            "scenes": automatic.scenes.model_copy(
                update={
                    "min_duration_s": Decimal("0"),
                    "min_samples": 1,
                    "max_duration_s": Decimal("10"),
                    "max_sample_gap_ms": Decimal("1000"),
                }
            )
        }
    )

    only = build_recording_scenes(report, SOURCE, annotation_only, annotations_path=annotation_path)
    auto = build_recording_scenes(report, SOURCE, automatic, annotations_path=annotation_path)

    assert [sample.timestamp_ns for sample in only.samples] == [1_000_000_000]
    assert [item.timestamp_ns for item in only.unassigned] == [0, 2_000_000_000]
    assert len(auto.scenes) == 1 and auto.scenes[0].kind == "automatic"
    assert auto.scenes[0].labels == () and auto.scenes[0].annotation_refs == ()


def test_uuid_tokens_change_for_namespace_source_and_scene_config_not_local_staging(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _config(config_factory())
    report = _report(tmp_path / "one", (0, 1_000_000_000))
    base = build_recording_scenes(report, SOURCE, config)
    moved = _report(tmp_path / "two", (0, 1_000_000_000))
    same = build_recording_scenes(moved, SOURCE, config)
    changed_namespace = config.model_copy(
        update={
            "scenes": config.scenes.model_copy(
                update={"dataset_namespace": UUID("11111111-1111-5111-8111-111111111111")}
            )
        }
    )
    changed_config = config.model_copy(
        update={"scenes": config.scenes.model_copy(update={"max_duration_s": Decimal("3")})}
    )
    changed_source = replace(SOURCE, etag='"other"')

    assert [scene.token for scene in base.scenes] == [scene.token for scene in same.scenes]
    assert (
        base.scenes[0].token
        != build_recording_scenes(report, SOURCE, changed_namespace).scenes[0].token
    )
    assert (
        base.scenes[0].token
        != build_recording_scenes(report, SOURCE, changed_config).scenes[0].token
    )
    assert (
        base.scenes[0].token
        != build_recording_scenes(report, changed_source, config).scenes[0].token
    )


def test_actual_synthetic_extraction_to_validity_to_scene_graph_boundary(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    mcap_path = tmp_path / "actual.mcap"
    write_mcap(
        mcap_path,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
            camera_message(1_500_000_000, (1_500_000_010, 1_500_000_020)),
        ),
    )
    extraction = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=Fraction(2, 1),
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=_DeterministicDecoder,
    ).extract(mcap_path)
    base = config_factory()
    validity_config = base.model_copy(
        update={
            "frame_validity": base.frame_validity.model_copy(
                update={
                    "required_cameras": ["front", "rear"],
                    "invalidate_on": InvalidationRulesConfig.model_validate(
                        {code: False for code in INVALIDITY_CODES}
                    ),
                }
            )
        }
    )
    report = evaluate_validity(extraction, validity_config)
    scene_config = _config(
        base, min_duration_s=0.1, max_duration_s=1.0, min_samples=2, max_sample_gap_ms=500.0
    )

    result = build_recording_scenes(report, SOURCE, scene_config)

    assert len(report.final_candidates) == 2
    assert len(result.scenes) == 1 and result.scenes[0].nbr_samples == 2
    assert len(result.sample_data) == 4
    assert all(item.calibration is not None for item in result.sample_data)
    assert {item.timestamp_ns for item in result.sample_data} == {
        1_000_000_010,
        1_000_000_020,
        1_500_000_010,
        1_500_000_020,
    }


def test_actual_task4_invalid_audit_and_grid_miss_never_enter_scenes(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    mcap_path = tmp_path / "invalid.mcap"
    write_mcap(
        mcap_path,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
            camera_message(2_000_000_000, (2_000_000_010, 2_000_000_020)),
        ),
    )
    extraction = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=Fraction(2, 1),
        tolerance_ns=0,
        staging_root=tmp_path / "staging-invalid",
        decoder_factory=_DeterministicDecoder,
    ).extract(mcap_path)
    base = config_factory()
    validity_config = base.model_copy(
        update={
            "frame_validity": base.frame_validity.model_copy(
                update={
                    "required_cameras": ["front", "rear", "missing-camera"],
                    "invalidate_on": InvalidationRulesConfig.model_validate(
                        {
                            code: code in {"missing_required_camera", "grid_miss"}
                            for code in INVALIDITY_CODES
                        }
                    ),
                }
            )
        }
    )
    report = evaluate_validity(extraction, validity_config)

    result = build_recording_scenes(report, SOURCE, _config(base))

    assert report.audit_only_samples and not report.final_candidates
    assert any(item.batch_timestamp_ns is None for item in report.grid_audits)
    assert result.scenes == ()
    assert result.samples == ()
    assert result.sample_data == ()


def test_graph_validator_seals_source_coverage_and_expected_camera_channels(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    result = build_recording_scenes(
        _report(tmp_path, (0, 1_000_000_000)), SOURCE, _config(config_factory())
    )
    assert [(item.timestamp_ns, item.expected_channels) for item in result.source_samples] == [
        (0, ("front", "rear")),
        (1_000_000_000, ("front", "rear")),
    ]

    with pytest.raises(StructuralExtractionError, match="sample_data.*coverage|channel"):
        validate_scene_graph(replace(result, sample_data=()))
    with pytest.raises(StructuralExtractionError, match="sample_data.*coverage|channel"):
        validate_scene_graph(replace(result, sample_data=result.sample_data[1:]))

    original = result.sample_data[0]
    duplicate_channel = replace(original, token="00000000-0000-5000-8000-000000000001")
    with pytest.raises(StructuralExtractionError, match="duplicate.*channel|coverage"):
        validate_scene_graph(replace(result, sample_data=(*result.sample_data, duplicate_channel)))

    extra_channel = replace(
        original,
        token="00000000-0000-5000-8000-000000000002",
        channel="unexpected",
        prev="",
        next="",
    )
    with pytest.raises(StructuralExtractionError, match="coverage|extra|inconsistent"):
        validate_scene_graph(replace(result, sample_data=(*result.sample_data, extra_channel)))

    with pytest.raises(StructuralExtractionError, match="token collision"):
        validate_scene_graph(replace(result, sample_data=(*result.sample_data, original)))

    with pytest.raises(StructuralExtractionError, match="coverage"):
        validate_scene_graph(replace(result, source_samples=result.source_samples[:-1]))


def test_graph_validator_rejects_duplicate_unassigned_timestamps(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _config(config_factory(), min_duration_s=Decimal("1"), min_samples=2)
    result = build_recording_scenes(_report(tmp_path, (0,)), SOURCE, config)
    assert len(result.unassigned) == 1

    with pytest.raises(StructuralExtractionError, match="duplicate unassigned"):
        validate_scene_graph(replace(result, unassigned=(*result.unassigned, result.unassigned[0])))


def test_scene_boundary_rejects_forged_unsafe_channel_and_filename(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    result = build_recording_scenes(_report(tmp_path, (0,)), SOURCE, _config(config_factory()))
    original = result.sample_data[0]
    with pytest.raises(StructuralExtractionError, match="channel"):
        validate_scene_graph(
            replace(
                result,
                sample_data=(replace(original, channel="../escape"), *result.sample_data[1:]),
            )
        )
    with pytest.raises(StructuralExtractionError, match="filename"):
        validate_scene_graph(
            replace(
                result,
                sample_data=(
                    replace(original, filename="samples/digest/front/../../escape.jpg"),
                    *result.sample_data[1:],
                ),
            )
        )


def test_validator_seals_annotation_matches_windows_and_build_config(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    annotation_path = _annotations(
        tmp_path / "sealed.jsonl",
        [{"blob_path": SOURCE.blob_path, "timestamp_ns": 0, "labels": ["sealed"]}],
    )
    config = _annotation_config(
        config_factory(), mode="annotation_only", tolerance_ms=0, before_s=0, after_s=0
    )
    result = build_recording_scenes(
        _report(tmp_path, (0, 1_000_000_000)),
        SOURCE,
        config,
        annotations_path=annotation_path,
    )
    match = result.annotation_matches[0]
    with pytest.raises(StructuralExtractionError, match="match decision"):
        validate_scene_graph(
            replace(result, annotation_matches=(replace(match, signed_error_ns=1),))
        )
    window = result.annotation_windows[0]
    with pytest.raises(StructuralExtractionError, match="window derivation"):
        validate_scene_graph(
            replace(result, annotation_windows=(replace(window, last_timestamp_ns=1),))
        )
    with pytest.raises(StructuralExtractionError, match="build configuration"):
        validate_scene_graph(replace(result, annotation_before_ns=1))


def test_annotation_index_is_precomputed_and_large_matching_is_stable(
    config_factory: Callable[[], GlobalConfig],
) -> None:
    count = 10_000
    audits = tuple(
        LogicalSampleAudit(index * 10, index * 10, (), (), (), True) for index in range(count)
    )
    index = scenes_module._SampleIndex.build(audits, 10)
    assert len(index.runs) == 1
    assert index.nearest_position(55) == 5
    parsed = tuple(
        ParsedAnnotation(
            line_number + 1,
            SOURCE.blob_path,
            line_number * 10,
            ("event",),
        )
        for line_number in range(count)
    )
    config = _annotation_config(
        config_factory(), mode="annotation_only", tolerance_ms=0, before_s=0, after_s=0
    )
    records, matches, windows, samples_by_window = scenes_module._annotation_state(
        parsed, SOURCE, index, config
    )
    assert len(records) == len(matches) == len(windows) == len(samples_by_window) == count
