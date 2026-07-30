"""Deterministic per-recording scene construction and graph validation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5

from dataset_devkit.annotations import ParsedAnnotation, parse_annotations
from dataset_devkit.blob_list import BlobListError, validate_blob_path
from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import ExtractedCameraSample
from dataset_devkit.provenance import SourceFingerprint, canonical_json
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


def _partition_runs(
    samples: Sequence[LogicalSampleAudit], max_gap_ns: int
) -> tuple[tuple[LogicalSampleAudit, ...], ...]:
    if not samples:
        return ()
    runs: list[list[LogicalSampleAudit]] = [[samples[0]]]
    for sample in samples[1:]:
        if sample.grid_target_timestamp_ns - runs[-1][-1].grid_target_timestamp_ns > max_gap_ns:
            runs.append([])
        runs[-1].append(sample)
    return tuple(tuple(run) for run in runs)


def _validate_input(
    report: ValidityReport, source: SourceFingerprint
) -> tuple[LogicalSampleAudit, ...]:
    try:
        validate_blob_path(source.blob_path)
    except BlobListError as error:
        raise StructuralExtractionError(
            "source fingerprint has an invalid exact blob path"
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
        if not audit.samples:
            raise StructuralExtractionError("final candidate has no staged camera samples")
        cameras = tuple((item.camera_name, item.camera_timestamp_ns) for item in audit.samples)
        if cameras != audit.camera_timestamps or len(cameras) != len(set(cameras)):
            raise StructuralExtractionError("final candidate camera references are inconsistent")
        for camera in audit.samples:
            if camera.grid_target_timestamp_ns != audit.grid_target_timestamp_ns:
                raise StructuralExtractionError("broken staged frame logical timestamp reference")
            if camera.batch_timestamp_ns != audit.batch_timestamp_ns:
                raise StructuralExtractionError("broken staged frame batch timestamp reference")
            if camera.staged_image.timestamp_ns != camera.camera_timestamp_ns:
                raise StructuralExtractionError("broken staged image timestamp reference")
            if camera.ego_pose.timestamp_ns != camera.camera_timestamp_ns:
                raise StructuralExtractionError("broken ego-pose timestamp reference")
            pose_numbers = (
                *(camera.ego_pose.translation_xyz_m or ()),
                *(camera.ego_pose.rotation_wxyz or ()),
            )
            if not all(math.isfinite(value) for value in pose_numbers):
                raise StructuralExtractionError("sample pose contains non-finite values")
            if camera.calibration is not None:
                calibration = camera.calibration
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
            if not camera.staged_image.path.is_file():
                raise StructuralExtractionError("broken staged image file reference")
    if report.source_path.name and not source.blob_path:
        raise StructuralExtractionError("source fingerprint lacks exact blob path")
    return samples


@dataclass(frozen=True)
class _WindowCandidate:
    annotation: AnnotationRecord
    first_ns: int
    last_ns: int
    samples: tuple[LogicalSampleAudit, ...]


def _annotation_state(
    parsed: Sequence[ParsedAnnotation],
    source: SourceFingerprint,
    samples: Sequence[LogicalSampleAudit],
    runs: Sequence[Sequence[LogicalSampleAudit]],
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
                [source_identity, item.line_number, item.blob_path, item.timestamp_ns, item.labels],
            ),
            item.line_number,
            item.blob_path,
            item.timestamp_ns,
            item.labels,
        )
        for item in parsed
    )
    run_by_timestamp = {
        sample.grid_target_timestamp_ns: tuple(run) for run in runs for sample in run
    }
    matches: list[AnnotationMatch] = []
    candidates: list[_WindowCandidate] = []
    for parsed_item, record in zip(parsed, records, strict=True):
        if parsed_item.blob_path != source.blob_path:
            matches.append(
                AnnotationMatch(
                    record.token, record.line_number, False, None, None, None, "different_recording"
                )
            )
            continue
        if not samples:
            matches.append(
                AnnotationMatch(
                    record.token, record.line_number, False, None, None, None, "no_valid_samples"
                )
            )
            continue
        nearest = min(
            samples,
            key=lambda item: (
                abs(item.grid_target_timestamp_ns - parsed_item.timestamp_ns),
                item.grid_target_timestamp_ns,
            ),
        )
        signed = nearest.grid_target_timestamp_ns - parsed_item.timestamp_ns
        absolute = abs(signed)
        if absolute > config.annotations.match_tolerance_ns:
            matches.append(
                AnnotationMatch(
                    record.token,
                    record.line_number,
                    False,
                    nearest.grid_target_timestamp_ns,
                    signed,
                    absolute,
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
        run = run_by_timestamp[anchor]
        requested_first = anchor - config.annotations.before_ns
        requested_last = anchor + config.annotations.after_ns
        first_ns = max(requested_first, run[0].grid_target_timestamp_ns)
        last_ns = min(requested_last, run[-1].grid_target_timestamp_ns)
        window_samples = tuple(
            item for item in run if first_ns <= item.grid_target_timestamp_ns <= last_ns
        )
        candidates.append(_WindowCandidate(record, first_ns, last_ns, window_samples))

    candidates.sort(key=lambda item: (item.first_ns, item.last_ns, item.annotation.line_number))
    merged: list[list[_WindowCandidate]] = []
    for candidate in candidates:
        if not merged or candidate.first_ns > max(item.last_ns for item in merged[-1]):
            merged.append([candidate])
        else:
            merged[-1].append(candidate)
    windows: list[AnnotationWindow] = []
    samples_by_window: dict[str, tuple[LogicalSampleAudit, ...]] = {}
    for group in merged:
        lineage = sorted(group, key=lambda item: item.annotation.line_number)
        first_ns = min(item.first_ns for item in group)
        last_ns = max(item.last_ns for item in group)
        grouped_samples = {
            sample.grid_target_timestamp_ns: sample for item in group for sample in item.samples
        }
        ordered_samples = tuple(grouped_samples[key] for key in sorted(grouped_samples))
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


def _automatic_candidates(
    runs: Iterable[Sequence[LogicalSampleAudit]], config: GlobalConfig
) -> tuple[tuple[_SceneCandidate, ...], tuple[UnassignedSample, ...]]:
    kept: list[_SceneCandidate] = []
    unassigned: list[UnassignedSample] = []
    for run in runs:
        index = 0
        previous_kept_end: int | None = None
        while index < len(run):
            if previous_kept_end is not None:
                while (
                    index < len(run)
                    and run[index].grid_target_timestamp_ns - previous_kept_end
                    < config.scenes.skip_between_scenes_ns
                ):
                    unassigned.append(
                        UnassignedSample(
                            run[index].grid_target_timestamp_ns,
                            "inter_scene_skip",
                            "sample falls before the inclusive next-start boundary",
                        )
                    )
                    index += 1
                if index == len(run):
                    break
            start = index
            start_timestamp = run[start].grid_target_timestamp_ns
            index += 1
            while (
                index < len(run)
                and run[index].grid_target_timestamp_ns - start_timestamp
                <= config.scenes.max_duration_ns
            ):
                index += 1
            candidate = tuple(run[start:index])
            duration = candidate[-1].grid_target_timestamp_ns - start_timestamp
            if (
                duration >= config.scenes.min_duration_ns
                and len(candidate) >= config.scenes.min_samples
            ):
                kept.append(_SceneCandidate("automatic", candidate))
                previous_kept_end = candidate[-1].grid_target_timestamp_ns
            else:
                for sample in candidate:
                    unassigned.append(
                        UnassignedSample(
                            sample.grid_target_timestamp_ns,
                            "candidate_too_short",
                            f"duration_ns={duration}, sample_count={len(candidate)}",
                        )
                    )
    return tuple(kept), tuple(unassigned)


def _exclude_annotation_ranges(
    runs: Sequence[Sequence[LogicalSampleAudit]], excluded: set[int]
) -> tuple[tuple[LogicalSampleAudit, ...], ...]:
    """Remove annotation samples while preserving each excluded range as a hard boundary."""
    output: list[tuple[LogicalSampleAudit, ...]] = []
    for run in runs:
        current: list[LogicalSampleAudit] = []
        for sample in run:
            if sample.grid_target_timestamp_ns in excluded:
                if current:
                    output.append(tuple(current))
                    current = []
            else:
                current.append(sample)
        if current:
            output.append(tuple(current))
    return tuple(output)


def _materialize(
    candidates: Sequence[_SceneCandidate],
    source: SourceFingerprint,
    config: GlobalConfig,
) -> tuple[tuple[SceneRecord, ...], tuple[SampleRecord, ...], tuple[SampleDataRecord, ...]]:
    scenes: list[SceneRecord] = []
    samples: list[SampleRecord] = []
    sample_data: list[SampleDataRecord] = []
    source_identity = source.to_dict()
    settings = {
        "min_duration_ns": config.scenes.min_duration_ns,
        "max_duration_ns": config.scenes.max_duration_ns,
        "min_samples": config.scenes.min_samples,
        "max_sample_gap_ns": config.scenes.max_sample_gap_ns,
        "skip_between_scenes_ns": config.scenes.skip_between_scenes_ns,
    }
    for ordinal, candidate in enumerate(candidates):
        logical_timestamps = tuple(item.grid_target_timestamp_ns for item in candidate.samples)
        scene_token = _token(
            config.scenes.dataset_namespace,
            "scene",
            [source_identity, candidate.kind, candidate.window_token, logical_timestamps, settings],
        )
        scene_sample_tokens = tuple(
            _token(
                config.scenes.dataset_namespace, "sample", [source_identity, scene_token, timestamp]
            )
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
                    config.scenes.dataset_namespace,
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
                suffix = camera.staged_image.path.suffix.lower() or ".jpg"
                sample_data.append(
                    SampleDataRecord(
                        data_token,
                        scene_sample_tokens[sample_index],
                        scene_token,
                        channel,
                        camera.camera_timestamp_ns,
                        f"samples/{source.digest}/{channel}/{data_token}{suffix}",
                        camera.staged_image.path,
                        camera.calibration,
                        camera.ego_pose,
                        "" if data_index == 0 else data_tokens[data_index - 1],
                        "" if data_index == len(data_tokens) - 1 else data_tokens[data_index + 1],
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
                source.blob_path,
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
    runs = _partition_runs(final_samples, config.scenes.max_sample_gap_ns)
    path = config.annotations.path if annotations_path is None else annotations_path
    if not path.is_file():
        raise StructuralExtractionError(f"annotation JSONL is not a file: {path}")
    parsed = parse_annotations(path)
    annotations, matches, windows, samples_by_window = _annotation_state(
        parsed, source, final_samples, runs, config
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
            runs
            if config.scenes.mode == "automatic"
            else _exclude_annotation_ranges(runs, annotation_timestamps)
        )
        automatic, automatic_unassigned = _automatic_candidates(automatic_runs, config)
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
    scenes, samples, sample_data = _materialize(ordered_candidates, source, config)
    source_samples = tuple(
        SourceSampleRecord(
            item.grid_target_timestamp_ns,
            tuple(camera.camera_name for camera in item.samples),
        )
        for item in final_samples
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
    for item in records:
        seen: set[str] = set()
        current = item
        while current.next:
            if current.token in seen:
                raise StructuralExtractionError(f"{label} chain contains a cycle")
            seen.add(current.token)
            current = by_token[current.next]


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
        if not source_record.expected_channels or len(source_record.expected_channels) != len(
            set(source_record.expected_channels)
        ):
            raise StructuralExtractionError(
                "source sample expected channel coverage is empty or duplicated"
            )
    annotations = {item.token: item for item in result.annotations}
    windows = {item.token: item for item in result.annotation_windows}
    for scene in result.scenes:
        members = sorted(
            (item for item in result.samples if item.scene_token == scene.token),
            key=lambda item: item.timestamp_ns,
        )
        if (
            len(members) != scene.nbr_samples
            or not members
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
            expected_prev = "" if index == 0 else members[index - 1].token
            expected_next = "" if index == len(members) - 1 else members[index + 1].token
            if member.prev != expected_prev or member.next != expected_next:
                raise StructuralExtractionError("sample chain has cross-scene or endpoint links")
        if any(reference not in annotations for reference in scene.annotation_refs):
            raise StructuralExtractionError("scene has a foreign annotation reference")
        if scene.kind == "annotation":
            window = windows.get(scene.annotation_window_ref)
            if (
                window is None
                or scene.annotation_refs != window.annotation_tokens
                or scene.labels != window.labels
                or scene.first_timestamp_ns != window.first_sample_timestamp_ns
                or scene.last_timestamp_ns != window.last_sample_timestamp_ns
            ):
                raise StructuralExtractionError("annotation scene has a broken window reference")
        elif scene.annotation_window_ref or scene.annotation_refs or scene.labels:
            raise StructuralExtractionError("automatic scene contains human annotation references")
    if any(item.scene_token not in scenes for item in result.samples):
        raise StructuralExtractionError("sample has a foreign scene reference")
    _validate_chain(result.samples, "sample")
    data_groups: dict[tuple[str, str], list[SampleDataRecord]] = defaultdict(list)
    data_by_sample: dict[str, list[SampleDataRecord]] = defaultdict(list)
    for item in result.sample_data:
        sample = samples.get(item.sample_token)
        if sample is None or sample.scene_token != item.scene_token:
            raise StructuralExtractionError("sample_data has a foreign or cross-scene reference")
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
        if set(actual_channels) != set(expected.expected_channels):
            raise StructuralExtractionError("sample_data channel coverage is missing or extra")
    for group in data_groups.values():
        ordered = sorted(group, key=lambda item: samples[item.sample_token].timestamp_ns)
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
        _validate_chain(group, "sample_data")
    for match in result.annotation_matches:
        if match.annotation_token not in annotations:
            raise StructuralExtractionError("annotation match has a foreign annotation reference")
        if match.matched and match.sample_timestamp_ns is None:
            raise StructuralExtractionError("matched annotation lacks a sample reference")
    for window in result.annotation_windows:
        if any(reference not in annotations for reference in window.annotation_tokens):
            raise StructuralExtractionError("annotation window has a foreign annotation reference")
        lineage = tuple(annotations[reference] for reference in window.annotation_tokens)
        if tuple(item.line_number for item in lineage) != tuple(
            sorted(item.line_number for item in lineage)
        ):
            raise StructuralExtractionError("annotation window lineage is not in source line order")
        expected_labels = tuple(
            dict.fromkeys(label for annotation in lineage for label in annotation.labels)
        )
        if window.labels != expected_labels:
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
