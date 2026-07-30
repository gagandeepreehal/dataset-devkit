from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dataset_devkit.config import GlobalConfig, InvalidationRulesConfig
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import interpolate_gnss
from dataset_devkit.extraction.grid import GridMiss, GridSelection, SelectedGridEntry
from dataset_devkit.extraction.models import (
    CameraCalibration,
    CameraExtrinsic,
    CameraIntrinsic,
    EgoPose,
    ExtractedCameraSample,
    GnssSample,
    RawCameraBatch,
    RawCameraFrame,
    RecordingExtractionResult,
    StagedImage,
)
from dataset_devkit.validity import INVALIDITY_CODES, evaluate_validity


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        CameraIntrinsic(1, 1, 1, 1, 0, 0, (), 4, 3),
        CameraExtrinsic((0, 0, 0), (0, 0, 0)),
    )


def _gnss(timestamp: int, *, valid: bool, sigma: float, variance: float) -> GnssSample:
    return GnssSample(
        timestamp,
        timestamp,
        valid,
        0,
        0,
        0,
        0,
        0,
        0,
        {
            "east_sigma_m": sigma,
            "north_sigma_m": sigma - 0.1,
            "up_sigma_m": sigma - 0.2,
        },
        {"roll_variance": variance - 0.2, "pitch_variance": variance - 0.1,
         "yaw_variance": variance, "label": "preserved"},
    )


def _result(tmp_path: Path) -> RecordingExtractionResult:
    staging = tmp_path / "staging" / "recording-owned"
    staging.mkdir(parents=True)
    calibration = _calibration()
    batches = (
        RawCameraBatch(
            100, 100, 1, 1, "hevc", 4, 3,
            (
                RawCameraFrame(0, "front", 100, calibration),
                RawCameraFrame(1, "rear", 100, calibration),
            ),
        ),
        RawCameraBatch(
            2_000_000_100, 2_000_000_100, 2, 2, "hevc", 4, 3,
            (
                RawCameraFrame(0, "front", 100, calibration),
                RawCameraFrame(1, "rear", 2_000_000_100, calibration),
                RawCameraFrame(2, "extra", 2_000_000_100, calibration),
            ),
        ),
    )
    gnss = (
        _gnss(0, valid=False, sigma=1.2, variance=0.5),
        _gnss(4_000_000_000, valid=True, sigma=1.2, variance=0.5),
    )
    samples: list[ExtractedCameraSample] = []
    poses: dict[int, EgoPose] = {}
    for index, name, timestamp in ((0, "front", 100), (1, "rear", 2_000_000_100),
                                   (2, "extra", 2_000_000_100)):
        interpolation = interpolate_gnss(gnss, timestamp)
        pose = EgoPose(
            timestamp,
            True,
            (interpolation.projected_x_m or 0, interpolation.projected_y_m or 0,
             interpolation.height_m or 0),
            interpolation.quaternion_wxyz,
            interpolation,
        )
        image_path = staging / f"{index}-{name}.jpg"
        image_path.write_bytes(b"owned")
        image_stat = image_path.stat()
        staged = StagedImage(index, name, timestamp, image_path, 4, 3,
                             image_stat.st_dev, image_stat.st_ino)
        samples.append(
            ExtractedCameraSample(2_000_000_000, 2_000_000_100, timestamp,
                                  index, name, staged, pose)
        )
        poses[timestamp] = pose
    return RecordingExtractionResult(
        tmp_path / "recording.mcap",
        staging,
        batches,
        gnss,
        GridSelection(
            (SelectedGridEntry(2_000_000_000, 2_000_000_100, 100, 100),),
            (GridMiss(3_000_000_000),),
            (100,),
        ),
        tuple(samples),
        poses,
        (),
    )


def _configured(config: GlobalConfig, **toggles: bool) -> GlobalConfig:
    rules: dict[str, bool] = {code: False for code in INVALIDITY_CODES}
    rules.update(toggles)
    return config.model_copy(
        update={
            "frame_validity": config.frame_validity.model_copy(
                update={
                    "required_cameras": ["front", "rear", "side"],
                    "invalidate_on": InvalidationRulesConfig.model_validate(rules),
                }
            )
        }
    )


def test_observation_first_engine_retains_every_reason_and_raw_measurement(
    tmp_path: Path, config_factory: object
) -> None:
    base = config_factory()  # type: ignore[operator]
    config = base.model_copy(
        update={
            "frame_validity": base.frame_validity.model_copy(
                update={"required_cameras": ["front", "rear", "side"]}
            )
        }
    )
    result = _result(tmp_path)
    report = evaluate_validity(result, config)

    assert {observation.code for observation in report.observations} == set(INVALIDITY_CODES)
    assert all(observation.enabled_as_invalidator for observation in report.observations)
    assert len(report.sample_audits) == 1
    audit = report.sample_audits[0]
    assert audit.grid_target_timestamp_ns == 2_000_000_000
    assert audit.batch_timestamp_ns == 2_000_000_100
    assert audit.camera_timestamps == (
        ("front", 100), ("rear", 2_000_000_100), ("extra", 2_000_000_100)
    )
    assert not audit.valid
    assert audit.samples[2].camera_name == "extra"
    sigma = next(item for item in report.observations if item.code == "position_sigma_exceeded")
    assert sigma.measured_values == {
        "east_sigma_m": pytest.approx(1.2),
        "north_sigma_m": pytest.approx(1.1),
        "up_sigma_m": pytest.approx(1.0),
    }
    assert sigma.threshold == 0.5
    orientation = next(
        item for item in report.observations if item.code == "orientation_variance_exceeded"
    )
    assert orientation.measured_values["maximum_variance"] == pytest.approx(0.5)
    assert orientation.details["orientation_uncertainty"]["label"] == "preserved"
    sync = next(item for item in report.observations if item.code == "gnss_sync_gap_exceeded")
    assert sync.measured_values["maximum_endpoint_gap_ns"] == 3_999_999_900
    assert not report.final_candidates
    assert report.audit_only_samples == report.sample_audits
    assert all(sample.staged_image.path.is_file() for sample in result.samples)
    assert len(report.grid_audits) == 2
    assert report.grid_audits[1].batch_timestamp_ns is None


@pytest.mark.parametrize("enabled_code", INVALIDITY_CODES)
def test_each_invalidator_toggle_is_independent(
    tmp_path: Path, config_factory: object, enabled_code: str
) -> None:
    config = _configured(config_factory(), **{enabled_code: True})  # type: ignore[operator]
    report = evaluate_validity(_result(tmp_path / enabled_code), config)

    assert {
        observation.code
        for observation in report.observations
        if observation.enabled_as_invalidator
    } == {enabled_code}
    assert {observation.code for observation in report.observations} == set(INVALIDITY_CODES)


@pytest.mark.parametrize(
    ("sigma", "variance", "gap_ns", "expected"),
    [
        (0.5, 0.1, 30_000_000, set()),
        (0.500001, 0.100001, 30_000_001,
         {"position_sigma_exceeded", "orientation_variance_exceeded",
          "gnss_sync_gap_exceeded"}),
    ],
)
def test_thresholds_use_strict_exceed_not_equality(
    tmp_path: Path,
    config_factory: object,
    sigma: float,
    variance: float,
    gap_ns: int,
    expected: set[str],
) -> None:
    result = _result(tmp_path)
    before = _gnss(100 - gap_ns, valid=True, sigma=sigma, variance=variance)
    after = _gnss(100 + gap_ns, valid=True, sigma=sigma, variance=variance)
    interpolation = interpolate_gnss((before, after), 100)
    pose = replace(result.samples[0].ego_pose, interpolation=interpolation)
    sample = replace(result.samples[0], ego_pose=pose)
    second_batch = replace(result.camera_batches[1], frames=(result.camera_batches[1].frames[0],))
    result = replace(
        result,
        camera_batches=(result.camera_batches[0], second_batch),
        samples=(sample,),
        ego_poses_by_timestamp={100: pose},
        gnss_samples=(before, after),
    )
    config = _configured(
        config_factory(),  # type: ignore[operator]
        position_sigma_exceeded=True,
        orientation_variance_exceeded=True,
        gnss_sync_gap_exceeded=True,
    )

    observed = {item.code for item in evaluate_validity(result, config).observations}

    assert observed & {
        "position_sigma_exceeded", "orientation_variance_exceeded",
        "gnss_sync_gap_exceeded"
    } == expected


def test_drop_removes_only_owned_invalid_images_and_retains_audit_reasons(
    tmp_path: Path, config_factory: object
) -> None:
    result = _result(tmp_path)
    external = tmp_path / "external.jpg"
    external.write_bytes(b"external")
    prior = result.staging_root / "prior.jpg"
    prior.write_bytes(b"prior")
    config = config_factory().model_copy(  # type: ignore[operator]
        update={
            "frame_validity": config_factory().frame_validity.model_copy(  # type: ignore[operator]
                update={"invalid_sample_policy": "drop"}
            )
        }
    )

    report = evaluate_validity(result, config)

    assert not report.audit_only_samples
    assert all(not audit.samples for audit in report.sample_audits if not audit.valid)
    assert report.observations
    assert all(not sample.staged_image.path.exists() for sample in result.samples)
    assert external.read_bytes() == b"external"
    assert prior.read_bytes() == b"prior"


def test_inconsistent_final_reference_is_mandatory_structural_failure(
    tmp_path: Path, config_factory: object
) -> None:
    result = _result(tmp_path)
    wrong = replace(result.samples[0].staged_image, camera_name="not-front")
    result = replace(
        result,
        samples=(replace(result.samples[0], staged_image=wrong), *result.samples[1:]),
    )

    with pytest.raises(StructuralExtractionError, match="reference|inconsistent"):
        evaluate_validity(result, config_factory())  # type: ignore[operator]


def test_drop_refuses_ancestor_symlink_without_deleting_owned_inodes(
    tmp_path: Path, config_factory: object
) -> None:
    result = _result(tmp_path)
    original_paths = tuple(sample.staged_image.path for sample in result.samples)
    moved_root = tmp_path / "moved-staging"
    (tmp_path / "staging").rename(moved_root)
    (tmp_path / "staging").symlink_to(moved_root, target_is_directory=True)
    config = config_factory()  # type: ignore[operator]
    config = config.model_copy(
        update={
            "frame_validity": config.frame_validity.model_copy(
                update={"invalid_sample_policy": "drop"}
            )
        }
    )

    with pytest.raises(StructuralExtractionError, match="ancestor|symlink"):
        evaluate_validity(result, config)

    assert all(
        (moved_root / "recording-owned" / path.name).is_file()
        for path in original_paths
    )
