"""Deterministic per-recording scene construction and graph validation."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid5

from dataset_devkit.annotations import ParsedAnnotation, parse_annotations
from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import ExtractedCameraSample
from dataset_devkit.extraction.staging import verify_staged_image_identity
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.provenance import SourceFingerprint, canonical_json
from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_mcap_path
from dataset_devkit.scene_models import (
    AnnotationMatch,
    AnnotationRecord,
    AnnotationWindow,
    RecordingSceneResult,
    SampleDataRecord,
    SampleRecord,
    SceneRecord,
    SourceSampleRecord,
    UnassignedSample,
)
from dataset_devkit.validity import LogicalSampleAudit, ValidityReport


def _token(namespace: UUID, kind: str, identity: object) -> str:
    return str(uuid5(namespace, f"dataset-devkit/{kind}/{canonical_json(identity)}"))


def _evidence_jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _evidence_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _evidence_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_evidence_jsonable(item) for item in value]
    return value


def _feature_evidence_token(
    namespace: UUID,
    source: SourceFingerprint,
    source_samples: tuple[SourceSampleRecord, ...],
    sample_data: tuple[SampleDataRecord, ...],
) -> str:
    return _token(
        namespace,
        "feature-evidence",
        [
            source.to_dict(),
            [
                [
                    item.timestamp_ns,
                    item.batch_timestamp_ns,
                    item.expected_channels,
                    item.present_channels,
                    item.source_gnss_valid,
                    item.grid_signed_sync_error_ns,
                ]
                for item in source_samples
            ],
            [
                [
                    item.token,
                    item.timestamp_ns,
                    item.grid_signed_sync_error_ns,
                    item.camera_signed_sync_error_ns,
                    item.gnss_source_validity,
                    _evidence_jsonable(item.ego_pose),
                ]
                for item in sample_data
            ],
        ],
    )


def _validate_input(
    report: ValidityReport, source: SourceFingerprint
) -> tuple[LogicalSampleAudit, ...]:
    try:
        validate_repo_mcap_path(source.repo_path)
    except RepositoryPathError as error:
        raise StructuralExtractionError(
            "source fingerprint has an invalid repository MCAP path"
        ) from error
    samples = report.final_candidates
    timestamps = tuple(item.grid_target_timestamp_ns for item in samples)
    if len(timestamps) != len(set(timestamps)):
        raise StructuralExtractionError("duplicate logical timestamp in final candidates")
    if any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise StructuralExtractionError("final candidate logical timestamps are not ordered")
    audit_ids = {id(item) for item in report.sample_audits}
    if any(id(item) not in audit_ids or not item.valid for item in samples):
        raise StructuralExtractionError("final candidate is not a final valid audit sample")
    for audit in samples:
        if (
            not isinstance(audit.grid_target_timestamp_ns, int)
            or isinstance(audit.grid_target_timestamp_ns, bool)
            or not isinstance(audit.batch_timestamp_ns, int)
            or isinstance(audit.batch_timestamp_ns, bool)
        ):
            raise StructuralExtractionError("logical or batch timestamp is not an integer")
        if not audit.samples:
            raise StructuralExtractionError("final candidate has no staged camera samples")
        cameras = tuple((item.camera_name, item.camera_timestamp_ns) for item in audit.samples)
        if cameras != audit.camera_timestamps or len(cameras) != len(set(cameras)):
            raise StructuralExtractionError("final candidate camera references are inconsistent")
        for camera in audit.samples:
            if (
                not isinstance(camera.camera_timestamp_ns, int)
                or isinstance(camera.camera_timestamp_ns, bool)
            ):
                raise StructuralExtractionError("camera timestamp is not an integer")
            try:
                validate_safe_segment(camera.camera_name)
            except ValueError as error:
                raise StructuralExtractionError(
                    "camera channel is not a safe path segment"
                ) from error
            if camera.grid_target_timestamp_ns != audit.grid_target_timestamp_ns:
                raise StructuralExtractionError("broken staged frame logical timestamp reference")
            if camera.batch_timestamp_ns != audit.batch_timestamp_ns:
                raise StructuralExtractionError("broken staged frame batch timestamp reference")
            staged = camera.staged_image
            if (
                staged.camera_index != camera.camera_index
                or staged.camera_name != camera.camera_name
                or staged.timestamp_ns != camera.camera_timestamp_ns
            ):
                raise StructuralExtractionError("staged image camera identity is inconsistent")
            verify_staged_image_identity(staged)
            if staged.width <= 0 or staged.height <= 0:
                raise StructuralExtractionError("staged image dimensions are invalid")
            if camera.ego_pose.timestamp_ns != camera.camera_timestamp_ns:
                raise StructuralExtractionError("broken ego-pose timestamp reference")
            pose = camera.ego_pose
            if (
                pose.interpolation.timestamp_ns != pose.timestamp_ns
                or pose.available != pose.interpolation.available
                or (
                    pose.available
                    and (pose.translation_xyz_m is None or pose.rotation_wxyz is None)
                )
                or (
                    not pose.available
                    and (pose.translation_xyz_m is not None or pose.rotation_wxyz is not None)
                )
            ):
                raise StructuralExtractionError("ego-pose interpolation identity is inconsistent")
            pose_numbers = (
                *(camera.ego_pose.translation_xyz_m or ()),
                *(camera.ego_pose.rotation_wxyz or ()),
            )
            if not all(math.isfinite(value) for value in pose_numbers):
                raise StructuralExtractionError("sample pose contains non-finite values")
            if camera.calibration is not None:
                calibration = camera.calibration
                if (
                    calibration.intrinsic.width != staged.width
                    or calibration.intrinsic.height != staged.height
                ):
                    raise StructuralExtractionError(
                        "staged image calibration dimensions are inconsistent"
                    )
                calibration_numbers = (
                    calibration.intrinsic.focal_length_x,
                    calibration.intrinsic.focal_length_y,
                    calibration.intrinsic.optical_center_x,
                    calibration.intrinsic.optical_center_y,
                    calibration.intrinsic.rmse,
                    calibration.intrinsic.skew,
                    calibration.intrinsic.width,
                    calibration.intrinsic.height,
                    *calibration.intrinsic.distortion_coeffs,
                    *calibration.extrinsic.rotation_vector,
                    *calibration.extrinsic.translation_vector,
                )
                if not all(math.isfinite(value) for value in calibration_numbers):
                    raise StructuralExtractionError("sample calibration contains non-finite values")
    if report.source_path.name and not source.repo_path:
        raise StructuralExtractionError("source fingerprint lacks exact repository path")
    return samples


@dataclass(frozen=True)
class _WindowCandidate:
    annotation: AnnotationRecord
    run_id: int
    start_index: int
    end_index: int
    first_ns: int
    last_ns: int


@dataclass(frozen=True)
class _SampleIndex:
    samples: tuple[LogicalSampleAudit, ...]
    timestamps: tuple[int, ...]
    runs: tuple[tuple[int, int], ...]
    run_id_by_position: tuple[int, ...]

    @classmethod
    def build(cls, samples: Sequence[LogicalSampleAudit], max_gap_ns: int) -> _SampleIndex:
        ordered = tuple(samples)
        timestamps = tuple(item.grid_target_timestamp_ns for item in ordered)
        if not ordered:
            return cls((), (), (), ())
        starts = [0]
        for position in range(1, len(ordered)):
            if timestamps[position] - timestamps[position - 1] > max_gap_ns:
                starts.append(position)
        ends = [*starts[1:], len(ordered)]
        runs = tuple(zip(starts, ends, strict=True))
        run_ids = [0] * len(ordered)
        for run_id, (start, end) in enumerate(runs):
            run_ids[start:end] = [run_id] * (end - start)
        return cls(ordered, timestamps, runs, tuple(run_ids))

    def nearest_position(self, timestamp_ns: int) -> int:
        right = bisect_left(self.timestamps, timestamp_ns)
        if right == 0:
            return 0
        if right == len(self.timestamps):
            return right - 1
        left = right - 1
        left_error = timestamp_ns - self.timestamps[left]
        right_error = self.timestamps[right] - timestamp_ns
        return left if left_error <= right_error else right

    def run_timestamps(self, run_id: int) -> tuple[int, ...]:
        start, end = self.runs[run_id]
        return self.timestamps[start:end]

    def materialize_run_range(
        self, run_id: int, start_index: int, end_index: int
    ) -> tuple[LogicalSampleAudit, ...]:
        run_start, run_end = self.runs[run_id]
        if not (0 <= start_index < end_index <= run_end - run_start):
            raise StructuralExtractionError("annotation window index range is invalid")
        return self.samples[run_start + start_index : run_start + end_index]


def _annotation_state(
    parsed: Sequence[ParsedAnnotation],
    source: SourceFingerprint,
    index: _SampleIndex,
    config: GlobalConfig,
) -> tuple[
    tuple[AnnotationRecord, ...],
    tuple[AnnotationMatch, ...],
    tuple[AnnotationWindow, ...],
    dict[str, tuple[LogicalSampleAudit, ...]],
]:
    namespace = config.scenes.dataset_namespace
    source_identity = source.to_dict()
    records = tuple(
        AnnotationRecord(
            _token(
                namespace,
                "annotation",
                [source_identity, item.line_number, item.repo_path, item.timestamp_ns, item.labels],
            ),
            item.line_number,
            item.repo_path,
            item.timestamp_ns,
            item.labels,
        )
        for item in parsed
    )
    matches: list[AnnotationMatch] = []
    candidates: list[_WindowCandidate] = []
    for parsed_item, record in zip(parsed, records, strict=True):
        if parsed_item.repo_path != source.repo_path:
            matches.append(
                AnnotationMatch(
                    record.token, record.line_number, False, None, None, None, "different_recording"
                )
            )
            continue
        if not index.samples:
            matches.append(
                AnnotationMatch(
                    record.token, record.line_number, False, None, None, None, "no_valid_samples"
                )
            )
            continue
        nearest_position = index.nearest_position(parsed_item.timestamp_ns)
        nearest = index.samples[nearest_position]
        signed = nearest.grid_target_timestamp_ns - parsed_item.timestamp_ns
        absolute = abs(signed)
        if absolute > config.annotations.match_tolerance_ns:
            matches.append(
                AnnotationMatch(
                    record.token,
                    record.line_number,
                    False,
                    None,
                    None,
                    None,
                    "outside_tolerance",
                )
            )
            continue
        anchor = nearest.grid_target_timestamp_ns
        matches.append(
            AnnotationMatch(
                record.token, record.line_number, True, anchor, signed, absolute, "matched"
            )
        )
        run_id = index.run_id_by_position[nearest_position]
        run_start, run_end = index.runs[run_id]
        requested_first = anchor - config.annotations.before_ns
        requested_last = anchor + config.annotations.after_ns
        first_ns = max(requested_first, index.timestamps[run_start])
        last_ns = min(requested_last, index.timestamps[run_end - 1])
        start_index = bisect_left(index.timestamps, first_ns, run_start, run_end) - run_start
        end_index = bisect_right(index.timestamps, last_ns, run_start, run_end) - run_start
        candidates.append(
            _WindowCandidate(
                record,
                run_id,
                start_index,
                end_index,
                first_ns,
                last_ns,
            )
        )

    candidates.sort(
        key=lambda item: (
            item.run_id,
            item.start_index,
            item.end_index,
            item.annotation.line_number,
        )
    )
    merged: list[list[_WindowCandidate]] = []
    merged_end_index: list[int] = []
    merged_last_ns: list[int] = []
    for candidate in candidates:
        if (
            not merged
            or candidate.run_id != merged[-1][0].run_id
            or candidate.first_ns > merged_last_ns[-1]
        ):
            merged.append([candidate])
            merged_end_index.append(candidate.end_index)
            merged_last_ns.append(candidate.last_ns)
        else:
            merged[-1].append(candidate)
            merged_end_index[-1] = max(merged_end_index[-1], candidate.end_index)
            merged_last_ns[-1] = max(merged_last_ns[-1], candidate.last_ns)
    windows: list[AnnotationWindow] = []
    samples_by_window: dict[str, tuple[LogicalSampleAudit, ...]] = {}
    for group_index, group in enumerate(merged):
        lineage = sorted(group, key=lambda item: item.annotation.line_number)
        first_ns = min(item.first_ns for item in group)
        last_ns = max(item.last_ns for item in group)
        start_index = group[0].start_index
        end_index = merged_end_index[group_index]
        ordered_samples = index.materialize_run_range(group[0].run_id, start_index, end_index)
        annotation_tokens = tuple(item.annotation.token for item in lineage)
        labels = tuple(dict.fromkeys(label for item in lineage for label in item.annotation.labels))
        window_token = _token(
            namespace,
            "annotation-window",
            [
                source_identity,
                annotation_tokens,
                first_ns,
                last_ns,
                ordered_samples[0].grid_target_timestamp_ns,
                ordered_samples[-1].grid_target_timestamp_ns,
            ],
        )
        windows.append(
            AnnotationWindow(
                window_token,
                annotation_tokens,
                first_ns,
                last_ns,
                ordered_samples[0].grid_target_timestamp_ns,
                ordered_samples[-1].grid_target_timestamp_ns,
                labels,
            )
        )
        samples_by_window[window_token] = ordered_samples
    return records, tuple(matches), tuple(windows), samples_by_window


@dataclass(frozen=True)
class _SceneCandidate:
    kind: str
    samples: tuple[LogicalSampleAudit, ...]
    labels: tuple[str, ...] = ()
    annotation_refs: tuple[str, ...] = ()
    window_token: str = ""


@dataclass(frozen=True)
class _BuildSettings:
    mode: str
    min_duration_ns: int
    max_duration_ns: int
    min_samples: int
    max_sample_gap_ns: int
    skip_between_scenes_ns: int
    annotation_match_tolerance_ns: int
    annotation_before_ns: int
    annotation_after_ns: int
    annotation_window_merge_semantics: str = "same_run_overlap_or_touch_v1"

    @classmethod
    def from_config(cls, config: GlobalConfig) -> _BuildSettings:
        return cls(
            config.scenes.mode,
            config.scenes.min_duration_ns,
            config.scenes.max_duration_ns,
            config.scenes.min_samples,
            config.scenes.max_sample_gap_ns,
            config.scenes.skip_between_scenes_ns,
            config.annotations.match_tolerance_ns,
            config.annotations.before_ns,
            config.annotations.after_ns,
        )

    @classmethod
    def from_result(cls, result: RecordingSceneResult) -> _BuildSettings:
        return cls(
            result.build_mode,
            result.min_scene_duration_ns,
            result.max_scene_duration_ns,
            result.min_scene_samples,
            result.max_sample_gap_ns,
            result.inter_scene_skip_ns,
            result.annotation_match_tolerance_ns,
            result.annotation_before_ns,
            result.annotation_after_ns,
            result.annotation_window_merge_semantics,
        )

    def identity(self) -> list[object]:
        return [
            self.mode,
            self.min_duration_ns,
            self.max_duration_ns,
            self.min_samples,
            self.max_sample_gap_ns,
            self.skip_between_scenes_ns,
            self.annotation_match_tolerance_ns,
            self.annotation_before_ns,
            self.annotation_after_ns,
            self.annotation_window_merge_semantics,
        ]


@dataclass(frozen=True)
class _AutomaticPartition:
    timestamps: tuple[int, ...]


@dataclass(frozen=True)
class _ExpectedScenePartition:
    kind: str
    timestamps: tuple[int, ...]
    labels: tuple[str, ...] = ()
    annotation_refs: tuple[str, ...] = ()
    window_token: str = ""


def _automatic_partition(
    runs: Iterable[Sequence[int]], settings: _BuildSettings
) -> tuple[tuple[_AutomaticPartition, ...], tuple[UnassignedSample, ...]]:
    kept: list[_AutomaticPartition] = []
    unassigned: list[UnassignedSample] = []
    for run in runs:
        index = 0
        previous_kept_end: int | None = None
        while index < len(run):
            if previous_kept_end is not None:
                while (
                    index < len(run)
                    and run[index] - previous_kept_end < settings.skip_between_scenes_ns
                ):
                    unassigned.append(
                        UnassignedSample(
                            run[index],
                            "inter_scene_skip",
                            "sample falls before the inclusive next-start boundary",
                        )
                    )
                    index += 1
                if index == len(run):
                    break
            start = index
            start_timestamp = run[start]
            index += 1
            while index < len(run) and run[index] - start_timestamp <= settings.max_duration_ns:
                index += 1
            candidate = tuple(run[start:index])
            duration = candidate[-1] - start_timestamp
            if duration >= settings.min_duration_ns and len(candidate) >= settings.min_samples:
                kept.append(_AutomaticPartition(candidate))
                previous_kept_end = candidate[-1]
            else:
                for timestamp in candidate:
                    unassigned.append(
                        UnassignedSample(
                            timestamp,
                            "candidate_too_short",
                            f"duration_ns={duration}, sample_count={len(candidate)}",
                        )
                    )
    return tuple(kept), tuple(unassigned)


def _exclude_annotation_ranges(
    runs: Sequence[Sequence[int]], excluded: set[int]
) -> tuple[tuple[int, ...], ...]:
    """Remove annotation samples while preserving each excluded range as a hard boundary."""
    output: list[tuple[int, ...]] = []
    for run in runs:
        current: list[int] = []
        for timestamp in run:
            if timestamp in excluded:
                if current:
                    output.append(tuple(current))
                    current = []
            else:
                current.append(timestamp)
        if current:
            output.append(tuple(current))
    return tuple(output)


def _materialize(
    candidates: Sequence[_SceneCandidate],
    source: SourceFingerprint,
    settings: _BuildSettings,
    namespace: UUID,
) -> tuple[tuple[SceneRecord, ...], tuple[SampleRecord, ...], tuple[SampleDataRecord, ...]]:
    scenes: list[SceneRecord] = []
    samples: list[SampleRecord] = []
    sample_data: list[SampleDataRecord] = []
    source_identity = source.to_dict()
    settings_identity = settings.identity()
    for ordinal, candidate in enumerate(candidates):
        logical_timestamps = tuple(item.grid_target_timestamp_ns for item in candidate.samples)
        scene_token = _token(
            namespace,
            "scene",
            [
                source_identity,
                candidate.kind,
                candidate.window_token,
                logical_timestamps,
                settings_identity,
            ],
        )
        scene_sample_tokens = tuple(
            _token(namespace, "sample", [source_identity, scene_token, timestamp])
            for timestamp in logical_timestamps
        )
        for index, (audit, sample_token) in enumerate(
            zip(candidate.samples, scene_sample_tokens, strict=True)
        ):
            samples.append(
                SampleRecord(
                    sample_token,
                    scene_token,
                    audit.grid_target_timestamp_ns,
                    audit.grid_target_timestamp_ns,
                    audit.batch_timestamp_ns,
                    "" if index == 0 else scene_sample_tokens[index - 1],
                    "" if index == len(scene_sample_tokens) - 1 else scene_sample_tokens[index + 1],
                )
            )
        by_channel: dict[str, list[tuple[int, ExtractedCameraSample]]] = defaultdict(list)
        for sample_index, audit in enumerate(candidate.samples):
            for camera in audit.samples:
                by_channel[camera.camera_name].append((sample_index, camera))
        for channel in sorted(by_channel):
            channel_items = by_channel[channel]
            data_tokens = tuple(
                _token(
                    namespace,
                    "sample-data",
                    [
                        source_identity,
                        scene_token,
                        scene_sample_tokens[sample_index],
                        channel,
                        camera.camera_timestamp_ns,
                        camera.camera_index,
                        data_index,
                    ],
                )
                for data_index, (sample_index, camera) in enumerate(channel_items)
            )
            for data_index, ((sample_index, camera), data_token) in enumerate(
                zip(channel_items, data_tokens, strict=True)
            ):
                relative = PurePosixPath("samples") / source.digest / channel / f"{data_token}.jpg"
                if relative.is_absolute() or ".." in relative.parts:
                    raise StructuralExtractionError("sample_data filename is unsafe")
                sample_data.append(
                    SampleDataRecord(
                        data_token,
                        scene_sample_tokens[sample_index],
                        scene_token,
                        channel,
                        camera.camera_index,
                        data_index,
                        camera.camera_timestamp_ns,
                        relative.as_posix(),
                        camera.staged_image,
                        camera.calibration,
                        camera.ego_pose,
                        "" if data_index == 0 else data_tokens[data_index - 1],
                        "" if data_index == len(data_tokens) - 1 else data_tokens[data_index + 1],
                        camera.batch_timestamp_ns - camera.grid_target_timestamp_ns,
                        camera.camera_timestamp_ns - camera.grid_target_timestamp_ns,
                        tuple(camera.ego_pose.interpolation.source_validity or ()),
                    )
                )
        scenes.append(
            SceneRecord(
                scene_token,
                f"{source.digest[:12]}-{ordinal:06d}",
                ordinal,
                "annotation" if candidate.kind == "annotation" else "automatic",
                scene_sample_tokens[0],
                scene_sample_tokens[-1],
                len(scene_sample_tokens),
                logical_timestamps[0],
                logical_timestamps[-1],
                candidate.labels,
                candidate.annotation_refs,
                candidate.window_token,
                source.repo_path,
            )
        )
    return tuple(scenes), tuple(samples), tuple(sample_data)


def build_recording_scenes(
    report: ValidityReport,
    source: SourceFingerprint,
    config: GlobalConfig,
    *,
    annotations_path: Path | None = None,
) -> RecordingSceneResult:
    """Build and validate one deterministic scene graph from final valid candidates only."""
    final_samples = _validate_input(report, source)
    settings = _BuildSettings.from_config(config)
    index = _SampleIndex.build(final_samples, settings.max_sample_gap_ns)
    timestamp_runs = tuple(index.timestamps[start:end] for start, end in index.runs)
    path = config.annotations.path if annotations_path is None else annotations_path
    if not path.is_file():
        raise StructuralExtractionError(f"annotation JSONL is not a file: {path}")
    parsed = parse_annotations(path)
    annotations, matches, windows, samples_by_window = _annotation_state(
        parsed, source, index, config
    )
    annotation_candidates = tuple(
        _SceneCandidate(
            "annotation",
            samples_by_window[window.token],
            window.labels,
            window.annotation_tokens,
            window.token,
        )
        for window in windows
    )
    annotation_timestamps = {
        sample.grid_target_timestamp_ns
        for candidate in annotation_candidates
        for sample in candidate.samples
    }
    unassigned: list[UnassignedSample] = []
    if config.scenes.mode == "annotation_only":
        candidates = annotation_candidates
        assigned = annotation_timestamps
        unassigned.extend(
            UnassignedSample(
                item.grid_target_timestamp_ns,
                "annotation_mode_excluded",
                "valid sample is outside all matched annotation windows",
            )
            for item in final_samples
            if item.grid_target_timestamp_ns not in assigned
        )
    else:
        automatic_runs = (
            timestamp_runs
            if config.scenes.mode == "automatic"
            else _exclude_annotation_ranges(timestamp_runs, annotation_timestamps)
        )
        automatic_partitions, automatic_unassigned = _automatic_partition(automatic_runs, settings)
        audits_by_timestamp = {item.grid_target_timestamp_ns: item for item in final_samples}
        automatic = tuple(
            _SceneCandidate(
                "automatic",
                tuple(audits_by_timestamp[timestamp] for timestamp in partition.timestamps),
            )
            for partition in automatic_partitions
        )
        unassigned.extend(automatic_unassigned)
        if config.scenes.mode == "hybrid":
            candidates = (*annotation_candidates, *automatic)
        else:
            candidates = automatic
    ordered_candidates = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.samples[0].grid_target_timestamp_ns,
                0 if item.kind == "annotation" else 1,
                item.window_token,
            ),
        )
    )
    scenes, samples, sample_data = _materialize(
        ordered_candidates, source, settings, config.scenes.dataset_namespace
    )
    observed_channels = tuple(
        sorted({camera.camera_name for item in final_samples for camera in item.samples})
    )
    expected_channels = tuple(
        sorted(set(config.frame_validity.required_cameras) | set(observed_channels))
    )
    source_samples = tuple(
        SourceSampleRecord(
            item.grid_target_timestamp_ns,
            item.batch_timestamp_ns,
            expected_channels,
            index.run_id_by_position[position],
            tuple(sorted(camera.camera_name for camera in item.samples)),
            all(
                all(camera.ego_pose.interpolation.source_validity or ())
                for camera in item.samples
            ),
            item.batch_timestamp_ns - item.grid_target_timestamp_ns,
        )
        for position, item in enumerate(final_samples)
    )
    result = RecordingSceneResult(
        source,
        source_samples,
        scenes,
        samples,
        sample_data,
        annotations,
        matches,
        windows,
        tuple(sorted(unassigned, key=lambda item: item.timestamp_ns)),
        config.scenes.mode,
        config.scenes.min_duration_ns,
        config.scenes.max_duration_ns,
        config.scenes.min_samples,
        config.annotations.match_tolerance_ns,
        config.annotations.before_ns,
        config.annotations.after_ns,
        config.scenes.max_sample_gap_ns,
        config.scenes.skip_between_scenes_ns,
        "same_run_overlap_or_touch_v1",
        config.scenes.dataset_namespace,
        _token(
            config.scenes.dataset_namespace,
            "scene-build-config",
            settings.identity(),
        ),
        _feature_evidence_token(
            config.scenes.dataset_namespace,
            source,
            source_samples,
            sample_data,
        ),
    )
    validate_scene_graph(result)
    return result


def _validate_chain(
    records: Sequence[SampleRecord] | Sequence[SampleDataRecord], label: str
) -> None:
    by_token = {item.token: item for item in records}
    if len(by_token) != len(records):
        raise StructuralExtractionError(f"duplicate {label} token")
    for item in records:
        if item.prev:
            previous = by_token.get(item.prev)
            if previous is None or previous.next != item.token:
                raise StructuralExtractionError(f"{label} chain has broken prev/next symmetry")
        if item.next:
            following = by_token.get(item.next)
            if following is None or following.prev != item.token:
                raise StructuralExtractionError(f"{label} chain has broken prev/next symmetry")
    state: dict[str, int] = {}
    for item in records:
        if state.get(item.token) == 2:
            continue
        path: list[str] = []
        current = item
        while True:
            status = state.get(current.token, 0)
            if status == 1:
                raise StructuralExtractionError(f"{label} chain contains a cycle")
            if status == 2:
                break
            state[current.token] = 1
            path.append(current.token)
            if not current.next:
                break
            current = by_token[current.next]
        for token in path:
            state[token] = 2


def validate_scene_graph(result: RecordingSceneResult) -> None:
    """Structural validation for scene/sample/sample-data and annotation references."""
    all_tokens = [item.token for item in result.scenes]
    all_tokens += [item.token for item in result.samples]
    all_tokens += [item.token for item in result.sample_data]
    all_tokens += [item.token for item in result.annotations]
    all_tokens += [item.token for item in result.annotation_windows]
    if len(all_tokens) != len(set(all_tokens)):
        raise StructuralExtractionError("scene graph contains a token collision")
    scenes = {item.token: item for item in result.scenes}
    scene_ordinals = {item.token: item.ordinal for item in result.scenes}
    canonical_collections: tuple[tuple[str, bool], ...] = (
        (
            "source_samples",
            result.source_samples
            == tuple(sorted(result.source_samples, key=lambda item: item.timestamp_ns)),
        ),
        (
            "scenes",
            result.scenes == tuple(sorted(result.scenes, key=lambda item: item.ordinal)),
        ),
        (
            "annotations",
            result.annotations
            == tuple(sorted(result.annotations, key=lambda item: item.line_number)),
        ),
        (
            "annotation_matches",
            result.annotation_matches
            == tuple(sorted(result.annotation_matches, key=lambda item: item.line_number)),
        ),
        (
            "annotation_windows",
            result.annotation_windows
            == tuple(
                sorted(
                    result.annotation_windows,
                    key=lambda item: (item.first_sample_timestamp_ns, item.token),
                )
            ),
        ),
        (
            "unassigned",
            result.unassigned
            == tuple(sorted(result.unassigned, key=lambda item: item.timestamp_ns)),
        ),
    )
    for label, is_canonical in canonical_collections:
        if not is_canonical:
            raise StructuralExtractionError(
                f"scene graph {label} is not in canonical global tuple order"
            )
    samples = {item.token: item for item in result.samples}
    source_by_timestamp = {item.timestamp_ns: item for item in result.source_samples}
    if len(source_by_timestamp) != len(result.source_samples):
        raise StructuralExtractionError("source coverage contains duplicate logical timestamps")
    source_timestamps = tuple(item.timestamp_ns for item in result.source_samples)
    if any(
        current <= previous
        for previous, current in zip(source_timestamps, source_timestamps[1:], strict=False)
    ):
        raise StructuralExtractionError("source coverage timestamps are not strictly ordered")
    for source_record in result.source_samples:
        if not source_record.expected_channels or source_record.expected_channels != tuple(
            sorted(set(source_record.expected_channels))
        ):
            raise StructuralExtractionError(
                "source sample expected channel coverage is empty, duplicated, or noncanonical"
            )
        if source_record.present_channels != tuple(sorted(set(source_record.present_channels))):
            raise StructuralExtractionError(
                "source sample present channel evidence is noncanonical"
            )
        if not set(source_record.present_channels) <= set(source_record.expected_channels):
            raise StructuralExtractionError(
                "source sample present channels exceed expected coverage"
            )
        if (
            source_record.grid_signed_sync_error_ns
            != source_record.batch_timestamp_ns - source_record.timestamp_ns
        ):
            raise StructuralExtractionError("source sample grid sync evidence is inconsistent")
    settings = _BuildSettings.from_result(result)
    if (
        settings.mode not in {"automatic", "annotation_only", "hybrid"}
        or settings.min_duration_ns <= 0
        or settings.max_duration_ns < settings.min_duration_ns
        or settings.min_samples < 1
        or min(
            settings.max_sample_gap_ns,
            settings.skip_between_scenes_ns,
            settings.annotation_match_tolerance_ns,
            settings.annotation_before_ns,
            settings.annotation_after_ns,
        )
        < 0
        or settings.annotation_window_merge_semantics != "same_run_overlap_or_touch_v1"
    ):
        raise StructuralExtractionError("scene graph build configuration is invalid")
    expected_run_id = 0
    for previous, current in zip(result.source_samples, result.source_samples[1:], strict=False):
        if current.timestamp_ns - previous.timestamp_ns > result.max_sample_gap_ns:
            expected_run_id += 1
        if current.valid_run_id != expected_run_id:
            raise StructuralExtractionError("source valid-run identity is inconsistent")
    if result.source_samples and result.source_samples[0].valid_run_id != 0:
        raise StructuralExtractionError("source valid-run identity is inconsistent")
    source_run_lists: dict[int, list[int]] = defaultdict(list)
    source_positions: dict[int, tuple[int, int]] = {}
    for source_record in result.source_samples:
        run_values = source_run_lists[source_record.valid_run_id]
        source_positions[source_record.timestamp_ns] = (
            source_record.valid_run_id,
            len(run_values),
        )
        run_values.append(source_record.timestamp_ns)
    source_runs = {run_id: tuple(values) for run_id, values in source_run_lists.items()}
    annotations = {item.token: item for item in result.annotations}
    windows = {item.token: item for item in result.annotation_windows}
    for annotation in result.annotations:
        expected_annotation_token = _token(
            result.dataset_namespace,
            "annotation",
            [
                result.source.to_dict(),
                annotation.line_number,
                annotation.repo_path,
                annotation.timestamp_ns,
                annotation.labels,
            ],
        )
        if annotation.token != expected_annotation_token:
            raise StructuralExtractionError("annotation token identity is inconsistent")
    expected_config_token = _token(
        result.dataset_namespace,
        "scene-build-config",
        settings.identity(),
    )
    if result.build_config_token != expected_config_token:
        raise StructuralExtractionError("scene graph build configuration is inconsistent")
    if len(result.annotation_matches) != len(result.annotations):
        raise StructuralExtractionError("annotation match coverage is incomplete or duplicated")
    matches_by_annotation = {item.annotation_token: item for item in result.annotation_matches}
    if len(matches_by_annotation) != len(result.annotation_matches) or set(
        matches_by_annotation
    ) != set(annotations):
        raise StructuralExtractionError("annotation match coverage is incomplete or duplicated")
    timestamp_values = tuple(source_by_timestamp)
    for annotation in result.annotations:
        match = matches_by_annotation[annotation.token]
        expected_match: tuple[bool, int | None, int | None, int | None, str]
        if match.line_number != annotation.line_number:
            raise StructuralExtractionError("annotation match line identity is inconsistent")
        if annotation.repo_path != result.source.repo_path:
            expected_match = (False, None, None, None, "different_recording")
        elif not timestamp_values:
            expected_match = (False, None, None, None, "no_valid_samples")
        else:
            right = bisect_left(timestamp_values, annotation.timestamp_ns)
            if right == 0:
                nearest = timestamp_values[0]
            elif right == len(timestamp_values):
                nearest = timestamp_values[-1]
            else:
                before = timestamp_values[right - 1]
                after = timestamp_values[right]
                nearest = (
                    before
                    if annotation.timestamp_ns - before <= after - annotation.timestamp_ns
                    else after
                )
            signed = nearest - annotation.timestamp_ns
            absolute = abs(signed)
            expected_match = (
                (True, nearest, signed, absolute, "matched")
                if absolute <= result.annotation_match_tolerance_ns
                else (False, None, None, None, "outside_tolerance")
            )
        actual_match = (
            match.matched,
            match.sample_timestamp_ns,
            match.signed_error_ns,
            match.absolute_error_ns,
            match.reason,
        )
        if actual_match != expected_match:
            raise StructuralExtractionError("annotation match decision is inconsistent")

    expected_window_candidates: list[_WindowCandidate] = []
    for annotation in result.annotations:
        match = matches_by_annotation[annotation.token]
        if not match.matched:
            continue
        assert match.sample_timestamp_ns is not None
        run_id, anchor_index = source_positions[match.sample_timestamp_ns]
        run_timestamps = source_runs[run_id]
        requested_first = match.sample_timestamp_ns - result.annotation_before_ns
        requested_last = match.sample_timestamp_ns + result.annotation_after_ns
        first_ns = max(requested_first, run_timestamps[0])
        last_ns = min(requested_last, run_timestamps[-1])
        first_position = bisect_left(run_timestamps, first_ns)
        last_position = bisect_right(run_timestamps, last_ns)
        expected_window_candidates.append(
            _WindowCandidate(
                annotation,
                run_id,
                first_position,
                last_position,
                first_ns,
                last_ns,
            )
        )
        if not (first_position <= anchor_index < last_position):
            raise StructuralExtractionError("annotation match anchor is outside its window")
    expected_window_candidates.sort(
        key=lambda item: (
            item.run_id,
            item.start_index,
            item.end_index,
            item.annotation.line_number,
        )
    )
    expected_groups: list[list[_WindowCandidate]] = []
    expected_group_end_index: list[int] = []
    expected_group_last_ns: list[int] = []
    for candidate in expected_window_candidates:
        if (
            not expected_groups
            or candidate.run_id != expected_groups[-1][0].run_id
            or candidate.first_ns > expected_group_last_ns[-1]
        ):
            expected_groups.append([candidate])
            expected_group_end_index.append(candidate.end_index)
            expected_group_last_ns.append(candidate.last_ns)
        else:
            expected_groups[-1].append(candidate)
            expected_group_end_index[-1] = max(expected_group_end_index[-1], candidate.end_index)
            expected_group_last_ns[-1] = max(expected_group_last_ns[-1], candidate.last_ns)
    if len(expected_groups) != len(result.annotation_windows):
        raise StructuralExtractionError("annotation window coverage is inconsistent")
    expected_samples_by_window: dict[str, tuple[int, ...]] = {}
    for group_index, (window, group) in enumerate(
        zip(result.annotation_windows, expected_groups, strict=True)
    ):
        lineage = sorted(group, key=lambda item: item.annotation.line_number)
        expected_tokens = tuple(item.annotation.token for item in lineage)
        expected_labels = tuple(
            dict.fromkeys(label for item in lineage for label in item.annotation.labels)
        )
        sample_timestamps = source_runs[group[0].run_id][
            group[0].start_index : expected_group_end_index[group_index]
        ]
        expected_first = min(item.first_ns for item in group)
        expected_last = expected_group_last_ns[group_index]
        expected_window_token = _token(
            result.dataset_namespace,
            "annotation-window",
            [
                result.source.to_dict(),
                expected_tokens,
                expected_first,
                expected_last,
                sample_timestamps[0],
                sample_timestamps[-1],
            ],
        )
        if (
            window.token != expected_window_token
            or window.annotation_tokens != expected_tokens
            or window.first_timestamp_ns != expected_first
            or window.last_timestamp_ns != expected_last
            or window.first_sample_timestamp_ns != sample_timestamps[0]
            or window.last_sample_timestamp_ns != sample_timestamps[-1]
            or window.labels != expected_labels
        ):
            raise StructuralExtractionError("annotation window derivation is inconsistent")
        expected_samples_by_window[window.token] = sample_timestamps

    annotation_partitions = tuple(
        _ExpectedScenePartition(
            "annotation",
            expected_samples_by_window[window.token],
            window.labels,
            window.annotation_tokens,
            window.token,
        )
        for window in result.annotation_windows
    )
    annotation_timestamps = {
        timestamp for partition in annotation_partitions for timestamp in partition.timestamps
    }
    timestamp_runs = tuple(source_runs[run_id] for run_id in sorted(source_runs))
    if settings.mode == "annotation_only":
        expected_partitions = annotation_partitions
        expected_unassigned = tuple(
            UnassignedSample(
                timestamp,
                "annotation_mode_excluded",
                "valid sample is outside all matched annotation windows",
            )
            for timestamp in source_timestamps
            if timestamp not in annotation_timestamps
        )
    else:
        automatic_runs = (
            timestamp_runs
            if settings.mode == "automatic"
            else _exclude_annotation_ranges(timestamp_runs, annotation_timestamps)
        )
        automatic_partitions, automatic_unassigned = _automatic_partition(automatic_runs, settings)
        expected_automatic = tuple(
            _ExpectedScenePartition("automatic", item.timestamps) for item in automatic_partitions
        )
        expected_partitions = (
            (*annotation_partitions, *expected_automatic)
            if settings.mode == "hybrid"
            else expected_automatic
        )
        expected_unassigned = automatic_unassigned
    expected_partitions = tuple(
        sorted(
            expected_partitions,
            key=lambda item: (
                item.timestamps[0],
                0 if item.kind == "annotation" else 1,
                item.window_token,
            ),
        )
    )
    if tuple(result.unassigned) != tuple(
        sorted(expected_unassigned, key=lambda item: item.timestamp_ns)
    ):
        raise StructuralExtractionError("scene graph unassigned derivation is inconsistent")
    if len(result.scenes) != len(expected_partitions):
        raise StructuralExtractionError("scene graph partition count is inconsistent")
    samples_by_scene: dict[str, list[SampleRecord]] = defaultdict(list)
    for sample in result.samples:
        samples_by_scene[sample.scene_token].append(sample)
    for members in samples_by_scene.values():
        members.sort(key=lambda item: item.timestamp_ns)
    for expected_ordinal, (scene, expected_partition) in enumerate(
        zip(result.scenes, expected_partitions, strict=True)
    ):
        members = samples_by_scene.get(scene.token, [])
        if scene.source_repo_path != result.source.repo_path:
            raise StructuralExtractionError("scene source repository path is inconsistent")
        expected_scene_token = _token(
            result.dataset_namespace,
            "scene",
            [
                result.source.to_dict(),
                expected_partition.kind,
                expected_partition.window_token,
                expected_partition.timestamps,
                settings.identity(),
            ],
        )
        if (
            scene.token != expected_scene_token
            or scene.name != f"{result.source.digest[:12]}-{expected_ordinal:06d}"
            or scene.ordinal != expected_ordinal
            or scene.kind != expected_partition.kind
            or scene.labels != expected_partition.labels
            or scene.annotation_refs != expected_partition.annotation_refs
            or scene.annotation_window_ref != expected_partition.window_token
            or len(members) != scene.nbr_samples
            or not members
            or tuple(item.timestamp_ns for item in members) != expected_partition.timestamps
            or members[0].token != scene.first_sample_token
            or members[-1].token != scene.last_sample_token
            or members[0].timestamp_ns != scene.first_timestamp_ns
            or members[-1].timestamp_ns != scene.last_timestamp_ns
            or any(
                current.timestamp_ns <= previous.timestamp_ns
                for previous, current in zip(members, members[1:], strict=False)
            )
        ):
            raise StructuralExtractionError("scene sample endpoints/count/order are inconsistent")
        for index, member in enumerate(members):
            if (
                member.batch_timestamp_ns
                != source_by_timestamp[member.timestamp_ns].batch_timestamp_ns
            ):
                raise StructuralExtractionError("sample batch timestamp evidence is inconsistent")
            expected_sample_token = _token(
                result.dataset_namespace,
                "sample",
                [result.source.to_dict(), scene.token, member.timestamp_ns],
            )
            expected_prev = "" if index == 0 else members[index - 1].token
            expected_next = "" if index == len(members) - 1 else members[index + 1].token
            if (
                member.token != expected_sample_token
                or member.grid_timestamp_ns != member.timestamp_ns
                or member.prev != expected_prev
                or member.next != expected_next
            ):
                raise StructuralExtractionError("sample chain has cross-scene or endpoint links")
        if any(reference not in annotations for reference in scene.annotation_refs):
            raise StructuralExtractionError("scene has a foreign annotation reference")
        if scene.kind == "annotation":
            selected_window = windows.get(scene.annotation_window_ref)
            if (
                selected_window is None
                or scene.annotation_refs != selected_window.annotation_tokens
                or scene.labels != selected_window.labels
                or scene.first_timestamp_ns != selected_window.first_sample_timestamp_ns
                or scene.last_timestamp_ns != selected_window.last_sample_timestamp_ns
                or tuple(item.timestamp_ns for item in members)
                != expected_samples_by_window[selected_window.token]
            ):
                raise StructuralExtractionError("annotation scene has a broken window reference")
        elif scene.annotation_window_ref or scene.annotation_refs or scene.labels:
            raise StructuralExtractionError("automatic scene contains human annotation references")
    if any(item.scene_token not in scenes for item in result.samples):
        raise StructuralExtractionError("sample has a foreign scene reference")
    _validate_chain(result.samples, "sample")
    data_groups: dict[tuple[str, str], list[SampleDataRecord]] = defaultdict(list)
    data_by_sample: dict[str, list[SampleDataRecord]] = defaultdict(list)
    filenames: set[str] = set()
    for item in result.sample_data:
        try:
            validate_safe_segment(item.channel)
        except ValueError as error:
            raise StructuralExtractionError("sample_data channel is unsafe") from error
        expected_data_token = _token(
            result.dataset_namespace,
            "sample-data",
            [
                result.source.to_dict(),
                item.scene_token,
                item.sample_token,
                item.channel,
                item.timestamp_ns,
                item.camera_index,
                item.channel_ordinal,
            ],
        )
        expected_filename = (
            PurePosixPath("samples")
            / result.source.digest
            / item.channel
            / f"{expected_data_token}.jpg"
        ).as_posix()
        filename = PurePosixPath(item.filename)
        selected_sample = samples.get(item.sample_token)
        if selected_sample is None or selected_sample.scene_token != item.scene_token:
            raise StructuralExtractionError("sample_data has a foreign or cross-scene reference")
        if (
            item.token != expected_data_token
            or item.filename != expected_filename
            or filename.is_absolute()
            or ".." in filename.parts
            or len(filename.parts) != 4
            or filename.parts[:3] != ("samples", result.source.digest, item.channel)
        ):
            raise StructuralExtractionError("sample_data filename is unsafe or inconsistent")
        if item.filename in filenames:
            raise StructuralExtractionError("sample_data filename is duplicated")
        filenames.add(item.filename)
        if (
            item.staged_image.camera_name != item.channel
            or item.staged_image.camera_index != item.camera_index
            or item.staged_image.timestamp_ns != item.timestamp_ns
            or item.staged_image.path.suffix.lower() != ".jpg"
            or item.grid_signed_sync_error_ns
            != selected_sample.batch_timestamp_ns - selected_sample.grid_timestamp_ns
            or item.camera_signed_sync_error_ns
            != item.timestamp_ns - selected_sample.grid_timestamp_ns
            or item.gnss_source_validity
            != tuple(item.ego_pose.interpolation.source_validity or ())
        ):
            raise StructuralExtractionError("sample_data staged camera evidence is inconsistent")
        verify_staged_image_identity(item.staged_image)
        if item.ego_pose.timestamp_ns != item.timestamp_ns:
            raise StructuralExtractionError("sample_data does not preserve real pose timestamp")
        data_groups[(item.scene_token, item.channel)].append(item)
        data_by_sample[item.sample_token].append(item)
    for sample in result.samples:
        expected = source_by_timestamp.get(sample.timestamp_ns)
        if expected is None:
            raise StructuralExtractionError("assigned sample is outside expected source coverage")
        actual_channels = tuple(item.channel for item in data_by_sample[sample.token])
        if len(actual_channels) != len(set(actual_channels)):
            raise StructuralExtractionError("sample_data contains a duplicate sample channel")
        if set(actual_channels) != set(expected.present_channels):
            raise StructuralExtractionError("sample_data channel coverage is missing or extra")
        data = data_by_sample[sample.token]
        if expected.source_gnss_valid != all(
            all(item.gnss_source_validity) for item in data
        ):
            raise StructuralExtractionError("source sample GNSS validity evidence is inconsistent")
    for data_group in data_groups.values():
        ordered = sorted(data_group, key=lambda item: samples[item.sample_token].timestamp_ns)
        if any(
            current.timestamp_ns <= previous.timestamp_ns
            for previous, current in zip(ordered, ordered[1:], strict=False)
        ):
            raise StructuralExtractionError("sample_data timestamps are not monotonic")
        for index, item in enumerate(ordered):
            expected_prev = "" if index == 0 else ordered[index - 1].token
            expected_next = "" if index == len(ordered) - 1 else ordered[index + 1].token
            if item.prev != expected_prev or item.next != expected_next:
                raise StructuralExtractionError(
                    "sample_data chain has cross-scene, cross-channel, or endpoint links"
                )
            if item.channel_ordinal != index:
                raise StructuralExtractionError("sample_data channel ordinal is inconsistent")
        _validate_chain(data_group, "sample_data")
    for window in result.annotation_windows:
        if any(reference not in annotations for reference in window.annotation_tokens):
            raise StructuralExtractionError("annotation window has a foreign annotation reference")
        annotation_lineage = tuple(annotations[reference] for reference in window.annotation_tokens)
        if tuple(item.line_number for item in annotation_lineage) != tuple(
            sorted(item.line_number for item in annotation_lineage)
        ):
            raise StructuralExtractionError("annotation window lineage is not in source line order")
        lineage_labels = tuple(
            dict.fromkeys(label for annotation in annotation_lineage for label in annotation.labels)
        )
        if window.labels != lineage_labels:
            raise StructuralExtractionError("annotation window label lineage is inconsistent")
        if window.first_sample_timestamp_ns > window.last_sample_timestamp_ns:
            raise StructuralExtractionError("annotation window sample endpoints are reversed")
    assigned = [item.timestamp_ns for item in result.samples]
    if len(assigned) != len(set(assigned)):
        raise StructuralExtractionError("a logical sample is claimed by multiple scenes")
    unassigned_timestamps = [item.timestamp_ns for item in result.unassigned]
    if len(unassigned_timestamps) != len(set(unassigned_timestamps)):
        raise StructuralExtractionError("scene graph contains duplicate unassigned timestamps")
    assigned_set = set(assigned)
    unassigned_set = set(unassigned_timestamps)
    expected_set = set(source_by_timestamp)
    if assigned_set & unassigned_set or assigned_set | unassigned_set != expected_set:
        raise StructuralExtractionError("scene graph source coverage is inconsistent")
    annotation_scene_windows = [
        scene.annotation_window_ref for scene in result.scenes if scene.kind == "annotation"
    ]
    if result.build_mode == "automatic" and annotation_scene_windows:
        raise StructuralExtractionError("automatic mode contains annotation scenes")
    if result.build_mode in {"annotation_only", "hybrid"} and sorted(
        annotation_scene_windows
    ) != sorted(windows):
        raise StructuralExtractionError("annotation windows do not have exactly one scene")
    if result.samples != tuple(
        sorted(
            result.samples,
            key=lambda item: (
                scene_ordinals.get(item.scene_token, len(scenes)),
                item.timestamp_ns,
            ),
        )
    ):
        raise StructuralExtractionError(
            "scene graph samples is not in canonical global tuple order"
        )
    if result.sample_data != tuple(
        sorted(
            result.sample_data,
            key=lambda item: (
                scene_ordinals.get(item.scene_token, len(scenes)),
                item.channel,
                item.channel_ordinal,
            ),
        )
    ):
        raise StructuralExtractionError(
            "scene graph sample_data is not in canonical global tuple order"
        )
    if result.feature_evidence_token != _feature_evidence_token(
        result.dataset_namespace,
        result.source,
        result.source_samples,
        result.sample_data,
    ):
        raise StructuralExtractionError("scene graph feature evidence seal is inconsistent")
