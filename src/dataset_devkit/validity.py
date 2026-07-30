"""Observation-first validity policy over immutable extraction results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import parse_numeric_uncertainty_leaf
from dataset_devkit.extraction.models import ExtractedCameraSample, RecordingExtractionResult
from dataset_devkit.extraction.staging import (
    remove_owned_staged_images,
    verify_owned_staged_images,
)

type InvalidityCode = Literal[
    "gnss_source_invalid",
    "position_sigma_exceeded",
    "orientation_variance_exceeded",
    "gnss_sync_gap_exceeded",
    "camera_timestamp_non_monotonic",
    "camera_timestamp_gap_exceeded",
    "missing_required_camera",
    "grid_miss",
]
type InvalidityScope = Literal["recording", "grid", "sample", "camera", "pose"]

INVALIDITY_CODES: tuple[InvalidityCode, ...] = (
    "gnss_source_invalid",
    "position_sigma_exceeded",
    "orientation_variance_exceeded",
    "gnss_sync_gap_exceeded",
    "camera_timestamp_non_monotonic",
    "camera_timestamp_gap_exceeded",
    "missing_required_camera",
    "grid_miss",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class InvalidityObservation:
    code: InvalidityCode
    scope: InvalidityScope
    measured_values: Mapping[str, int | float] = field(default_factory=dict)
    threshold: int | float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    grid_target_timestamp_ns: int | None = None
    batch_timestamp_ns: int | None = None
    camera_timestamp_ns: int | None = None
    camera_name: str | None = None
    enabled_as_invalidator: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "measured_values", _freeze(self.measured_values))
        object.__setattr__(self, "details", _freeze(self.details))


@dataclass(frozen=True)
class LogicalSampleAudit:
    grid_target_timestamp_ns: int
    batch_timestamp_ns: int
    camera_timestamps: tuple[tuple[str, int], ...]
    samples: tuple[ExtractedCameraSample, ...]
    observations: tuple[InvalidityObservation, ...]
    valid: bool


@dataclass(frozen=True)
class GridAuditRecord:
    grid_target_timestamp_ns: int
    batch_timestamp_ns: int | None
    camera_timestamps: tuple[tuple[str, int], ...]
    observations: tuple[InvalidityObservation, ...]
    valid: bool


@dataclass(frozen=True)
class ValidityReport:
    source_path: Path
    observations: tuple[InvalidityObservation, ...]
    grid_audits: tuple[GridAuditRecord, ...]
    sample_audits: tuple[LogicalSampleAudit, ...]
    final_candidates: tuple[LogicalSampleAudit, ...]
    audit_only_samples: tuple[LogicalSampleAudit, ...]
    valid: bool
    invalid_sample_policy: Literal["retain_for_audit", "drop"]


def _enabled(config: GlobalConfig, code: InvalidityCode) -> bool:
    return cast(bool, getattr(config.frame_validity.invalidate_on, code))


def _observation(
    config: GlobalConfig,
    code: InvalidityCode,
    scope: InvalidityScope,
    **kwargs: Any,
) -> InvalidityObservation:
    return InvalidityObservation(
        code=code,
        scope=scope,
        enabled_as_invalidator=_enabled(config, code),
        **kwargs,
    )


def _numeric_orientation_values(value: Mapping[str, Any]) -> dict[str, float]:
    found: dict[str, float] = {}

    def visit(item: object, prefix: str) -> None:
        number = parse_numeric_uncertainty_leaf(item)
        if number is not None:
            found[prefix] = number
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                name = f"{prefix}.{key}" if prefix else str(key)
                visit(item[key], name)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for index, nested in enumerate(item):
                visit(nested, f"{prefix}[{index}]")

    visit(value, "")
    return found


def _assert_sample_reference(
    result: RecordingExtractionResult, sample: ExtractedCameraSample
) -> None:
    if sample.batch_timestamp_ns not in {
        entry.batch_timestamp_ns for entry in result.selected_grid.entries
    }:
        raise StructuralExtractionError("sample references an unselected camera batch")
    matching_entry = next(
        entry for entry in result.selected_grid.entries
        if entry.batch_timestamp_ns == sample.batch_timestamp_ns
    )
    if matching_entry.target_timestamp_ns != sample.grid_target_timestamp_ns:
        raise StructuralExtractionError("sample grid target reference is inconsistent")
    matching_batch = next(
        (batch for batch in result.camera_batches
         if batch.rec_timestamp_ns == sample.batch_timestamp_ns),
        None,
    )
    if matching_batch is None or not any(
        frame.camera_index == sample.camera_index
        and frame.camera_name == sample.camera_name
        and frame.camera_timestamp_ns == sample.camera_timestamp_ns
        for frame in matching_batch.frames
    ):
        raise StructuralExtractionError("sample camera reference is inconsistent")
    staged = sample.staged_image
    if (
        staged.camera_index != sample.camera_index
        or staged.camera_name != sample.camera_name
        or staged.timestamp_ns != sample.camera_timestamp_ns
        or (staged.width, staged.height) != (matching_batch.width, matching_batch.height)
    ):
        raise StructuralExtractionError("staged image reference is inconsistent")
    if staged.path.parent != result.staging_root or staged.device is None or staged.inode is None:
        raise StructuralExtractionError("staged image is not owned by this extraction invocation")
    pose = sample.ego_pose
    if pose.timestamp_ns != sample.camera_timestamp_ns:
        raise StructuralExtractionError("pose timestamp reference is inconsistent")
    numbers: tuple[float, ...] = ()
    if pose.translation_xyz_m is not None:
        numbers += pose.translation_xyz_m
    if pose.rotation_wxyz is not None:
        numbers += pose.rotation_wxyz
    if not all(math.isfinite(number) for number in numbers):
        raise StructuralExtractionError("sample pose contains non-finite values")
    if pose.available != pose.interpolation.available:
        raise StructuralExtractionError("pose availability reference is inconsistent")
    if pose.available and (pose.translation_xyz_m is None or pose.rotation_wxyz is None):
        raise StructuralExtractionError("available pose lacks translation or rotation")
    if not pose.available and (
        pose.translation_xyz_m is not None or pose.rotation_wxyz is not None
    ):
        raise StructuralExtractionError("unavailable pose contains final pose values")


def _validate_result(result: RecordingExtractionResult) -> None:
    batches = {batch.rec_timestamp_ns: batch for batch in result.camera_batches}
    if len(batches) != len(result.camera_batches):
        raise StructuralExtractionError("recording contains duplicate camera batch references")
    selected_batches = [entry.batch_timestamp_ns for entry in result.selected_grid.entries]
    selected_targets = [entry.target_timestamp_ns for entry in result.selected_grid.entries]
    if len(set(selected_batches)) != len(selected_batches):
        raise StructuralExtractionError("selected grid contains duplicate batch references")
    if len(set(selected_targets)) != len(selected_targets):
        raise StructuralExtractionError("selected grid contains duplicate target references")
    samples_by_batch: dict[int, list[ExtractedCameraSample]] = {}
    for sample in result.samples:
        _assert_sample_reference(result, sample)
        samples_by_batch.setdefault(sample.batch_timestamp_ns, []).append(sample)
        mapped_pose = result.ego_poses_by_timestamp.get(sample.camera_timestamp_ns)
        if mapped_pose != sample.ego_pose:
            raise StructuralExtractionError("sample pose map reference is inconsistent")
    verify_owned_staged_images(
        result.staging_root, tuple(sample.staged_image for sample in result.samples)
    )
    for timestamp, pose in result.ego_poses_by_timestamp.items():
        if timestamp != pose.timestamp_ns:
            raise StructuralExtractionError("pose map key is inconsistent")
    if set(result.ego_poses_by_timestamp) != {
        sample.camera_timestamp_ns for sample in result.samples
    }:
        raise StructuralExtractionError("pose map contains missing or unreferenced poses")
    for batch_timestamp in selected_batches:
        batch = batches.get(batch_timestamp)
        if batch is None:
            raise StructuralExtractionError("selected grid references a missing camera batch")
        expected = {
            (frame.camera_index, frame.camera_name, frame.camera_timestamp_ns)
            for frame in batch.frames
        }
        actual = {
            (sample.camera_index, sample.camera_name, sample.camera_timestamp_ns)
            for sample in samples_by_batch.get(batch_timestamp, ())
        }
        if expected != actual or len(actual) != len(samples_by_batch.get(batch_timestamp, ())):
            raise StructuralExtractionError("selected batch sample references are inconsistent")


def _camera_timeline_observations(
    result: RecordingExtractionResult, config: GlobalConfig
) -> tuple[tuple[InvalidityObservation, ...], dict[int, list[InvalidityObservation]]]:
    all_observations: list[InvalidityObservation] = []
    by_batch: dict[int, list[InvalidityObservation]] = {}
    previous: dict[str, int] = {}
    target_by_batch = {
        entry.batch_timestamp_ns: entry.target_timestamp_ns
        for entry in result.selected_grid.entries
    }
    gap_threshold_ns = int(config.frame_validity.camera_timestamp_gap_max_ms * 1_000_000)
    for batch in result.camera_batches:
        for frame in batch.frames:
            earlier = previous.get(frame.camera_name)
            previous[frame.camera_name] = frame.camera_timestamp_ns
            if earlier is None:
                continue
            delta = frame.camera_timestamp_ns - earlier
            common = {
                "measured_values": {
                    "previous_timestamp_ns": earlier,
                    "current_timestamp_ns": frame.camera_timestamp_ns,
                    "delta_ns": delta,
                },
                "details": {"camera_index": frame.camera_index},
                "grid_target_timestamp_ns": target_by_batch.get(batch.rec_timestamp_ns),
                "batch_timestamp_ns": batch.rec_timestamp_ns,
                "camera_timestamp_ns": frame.camera_timestamp_ns,
                "camera_name": frame.camera_name,
            }
            if delta <= 0:
                observation = _observation(
                    config, "camera_timestamp_non_monotonic", "camera",
                    threshold=0, **common,
                )
                all_observations.append(observation)
                by_batch.setdefault(batch.rec_timestamp_ns, []).append(observation)
            if delta > gap_threshold_ns:
                observation = _observation(
                    config, "camera_timestamp_gap_exceeded", "camera",
                    threshold=gap_threshold_ns, **common,
                )
                all_observations.append(observation)
                by_batch.setdefault(batch.rec_timestamp_ns, []).append(observation)
    return tuple(all_observations), by_batch


def _pose_observations(
    sample: ExtractedCameraSample, config: GlobalConfig
) -> list[InvalidityObservation]:
    interpolation = sample.ego_pose.interpolation
    common = {
        "grid_target_timestamp_ns": sample.grid_target_timestamp_ns,
        "batch_timestamp_ns": sample.batch_timestamp_ns,
        "camera_timestamp_ns": sample.camera_timestamp_ns,
        "camera_name": sample.camera_name,
    }
    observed: list[InvalidityObservation] = []
    endpoint_details = {
        "interpolation_fraction": interpolation.fraction,
        "sync_gap_before_ns": interpolation.sync_gap_before_ns,
        "sync_gap_after_ns": interpolation.sync_gap_after_ns,
        "before_timestamp_ns": (
            None if interpolation.before is None else interpolation.before.timestamp_ns
        ),
        "after_timestamp_ns": (
            None if interpolation.after is None else interpolation.after.timestamp_ns
        ),
    }
    if not interpolation.available or (
        interpolation.source_validity is not None
        and not all(interpolation.source_validity)
    ):
        before_valid = (
            None if interpolation.before is None else interpolation.before.is_valid
        )
        after_valid = None if interpolation.after is None else interpolation.after.is_valid
        observed.append(
            _observation(
                config, "gnss_source_invalid", "pose",
                measured_values={
                    "interpolation_available": int(interpolation.available),
                    "before_valid": -1 if before_valid is None else int(before_valid),
                    "after_valid": -1 if after_valid is None else int(after_valid),
                },
                details={
                    **endpoint_details,
                    "before_position_uncertainty": (
                        {} if interpolation.before is None
                        else interpolation.before.position_uncertainty
                    ),
                    "after_position_uncertainty": (
                        {} if interpolation.after is None
                        else interpolation.after.position_uncertainty
                    ),
                    "before_orientation_uncertainty": (
                        {} if interpolation.before is None
                        else interpolation.before.orientation_uncertainty
                    ),
                    "after_orientation_uncertainty": (
                        {} if interpolation.after is None
                        else interpolation.after.orientation_uncertainty
                    ),
                },
                **common,
            )
        )
    position = {
        key: float(interpolation.position_uncertainty[key])
        for key in ("east_sigma_m", "north_sigma_m", "up_sigma_m")
        if key in interpolation.position_uncertainty
    }
    if any(value > config.gnss.position_sigma_max_m for value in position.values()):
        observed.append(
            _observation(
                config, "position_sigma_exceeded", "pose",
                measured_values=position,
                threshold=config.gnss.position_sigma_max_m,
                details={
                    **endpoint_details,
                    "before_position_uncertainty": (
                        {} if interpolation.before is None
                        else interpolation.before.position_uncertainty
                    ),
                    "after_position_uncertainty": (
                        {} if interpolation.after is None
                        else interpolation.after.position_uncertainty
                    ),
                    "interpolated_position_uncertainty": (
                        interpolation.position_uncertainty
                    ),
                },
                **common,
            )
        )
    variances = _numeric_orientation_values(interpolation.orientation_uncertainty)
    maximum = max(variances.values(), default=None)
    if maximum is not None and maximum > config.gnss.orientation_variance_max:
        orientation_details = dict(interpolation.orientation_uncertainty)
        if interpolation.before is not None and interpolation.after is not None:
            for key, value in interpolation.before.orientation_uncertainty.items():
                if interpolation.after.orientation_uncertainty.get(key) == value:
                    orientation_details.setdefault(key, value)
        observed.append(
            _observation(
                config, "orientation_variance_exceeded", "pose",
                measured_values={**variances, "maximum_variance": maximum},
                threshold=config.gnss.orientation_variance_max,
                details={
                    **endpoint_details,
                    "maximum_variance_path": min(
                        path for path, value in variances.items() if value == maximum
                    ),
                    "numeric_orientation_paths": tuple(sorted(variances)),
                    "uninterpolated_orientation_paths": (
                        interpolation.orientation_uncertainty_uninterpolated_paths
                    ),
                    "orientation_uncertainty": orientation_details,
                    "interpolated_orientation_uncertainty": (
                        interpolation.orientation_uncertainty
                    ),
                    "before_orientation_uncertainty": (
                        {} if interpolation.before is None
                        else interpolation.before.orientation_uncertainty
                    ),
                    "after_orientation_uncertainty": (
                        {} if interpolation.after is None
                        else interpolation.after.orientation_uncertainty
                    ),
                },
                **common,
            )
        )
    gaps = tuple(
        value for value in (interpolation.sync_gap_before_ns, interpolation.sync_gap_after_ns)
        if value is not None
    )
    threshold_ns = int(config.gnss.sync_gap_max_ms * 1_000_000)
    maximum_gap = max(gaps, default=None)
    if maximum_gap is not None and maximum_gap > threshold_ns:
        observed.append(
            _observation(
                config, "gnss_sync_gap_exceeded", "pose",
                measured_values={
                    "before_gap_ns": interpolation.sync_gap_before_ns or 0,
                    "after_gap_ns": interpolation.sync_gap_after_ns or 0,
                    "maximum_endpoint_gap_ns": maximum_gap,
                },
                threshold=threshold_ns,
                details=endpoint_details,
                **common,
            )
        )
    return observed


def evaluate_validity(
    result: RecordingExtractionResult, config: GlobalConfig
) -> ValidityReport:
    """Observe every policy reason, then derive sample and recording validity."""
    _validate_result(result)
    samples_by_batch: dict[int, list[ExtractedCameraSample]] = {}
    for sample in result.samples:
        samples_by_batch.setdefault(sample.batch_timestamp_ns, []).append(sample)
    timeline_observations, timeline_by_batch = _camera_timeline_observations(result, config)
    all_observations: list[InvalidityObservation] = list(timeline_observations)
    sample_audits: list[LogicalSampleAudit] = []
    grid_audits: list[GridAuditRecord] = []
    required = tuple(config.frame_validity.required_cameras)
    for entry in result.selected_grid.entries:
        samples = tuple(samples_by_batch.get(entry.batch_timestamp_ns, ()))
        observations = list(timeline_by_batch.get(entry.batch_timestamp_ns, ()))
        timeline_count = len(observations)
        present = {sample.camera_name for sample in samples}
        missing = tuple(name for name in required if name not in present)
        if missing:
            observations.append(
                _observation(
                    config, "missing_required_camera", "sample",
                    measured_values={"missing_count": len(missing)},
                    details={"required_cameras": required, "missing_cameras": missing,
                             "present_cameras": tuple(sample.camera_name for sample in samples)},
                    grid_target_timestamp_ns=entry.target_timestamp_ns,
                    batch_timestamp_ns=entry.batch_timestamp_ns,
                )
            )
        for sample in samples:
            observations.extend(_pose_observations(sample, config))
        valid = not any(item.enabled_as_invalidator for item in observations)
        cameras = tuple((sample.camera_name, sample.camera_timestamp_ns) for sample in samples)
        audit = LogicalSampleAudit(
            entry.target_timestamp_ns,
            entry.batch_timestamp_ns,
            cameras,
            samples,
            tuple(observations),
            valid,
        )
        sample_audits.append(audit)
        grid_audits.append(
            GridAuditRecord(entry.target_timestamp_ns, entry.batch_timestamp_ns, cameras,
                            tuple(observations), valid)
        )
        all_observations.extend(observations[timeline_count:])
    for miss in result.selected_grid.misses:
        observation = _observation(
            config,
            "grid_miss",
            "grid",
            measured_values={"target_timestamp_ns": miss.target_timestamp_ns},
            grid_target_timestamp_ns=miss.target_timestamp_ns,
        )
        grid_audits.append(
            GridAuditRecord(
                miss.target_timestamp_ns,
                None,
                (),
                (observation,),
                not observation.enabled_as_invalidator,
            )
        )
        all_observations.append(observation)
    final = tuple(audit for audit in sample_audits if audit.valid)
    invalid = tuple(audit for audit in sample_audits if not audit.valid)
    if config.frame_validity.invalid_sample_policy == "drop":
        remove_owned_staged_images(
            result.staging_root,
            tuple(sample.staged_image for audit in invalid for sample in audit.samples),
        )
        audit_only: tuple[LogicalSampleAudit, ...] = ()
        returned_sample_audits = tuple(
            audit if audit.valid else replace(audit, samples=()) for audit in sample_audits
        )
    else:
        audit_only = invalid
        returned_sample_audits = tuple(sample_audits)
    return ValidityReport(
        result.source_path,
        tuple(all_observations),
        tuple(grid_audits),
        returned_sample_audits,
        final,
        audit_only,
        not any(item.enabled_as_invalidator for item in all_observations),
        config.frame_validity.invalid_sample_policy,
    )
